"""Etapa 3 Evolve: Gemini Embedding 2 + ChromaDB sin LangGraph.

Uso: check | smoke | index --muestra | index --completa | query | status
Seguridad: las credenciales se leen solo del entorno, sin guardar la clave.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

BASE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "corpus" / "chunks" / "chunks_para_embeddings.jsonl"
CHROMA_DIR = BASE / "data" / "chroma_etapa3"
REPORT_DIR = BASE / "informes" / "etapa3"
MODEL = "gemini-embedding-2"
DIM = 768
EXPECTED_SHA256 = "c296c5fe1612e8e07e1221cbbdd61f7f30dbf787b20d930f13520155d3733b42"
EXPECTED_COUNT = 437
COLLECTIONS = {"muestra": "evolve_contratos_muestra_v1", "completa": "evolve_contratos_completa_v1"}

RE_DNI = re.compile(r"\b(?:\d{8}[\s-]?[A-Za-z]|[XYZxyz]\d{7}[A-Za-z])\b")
RE_IBAN = re.compile(r"\bES\d{2}(?:[\s-]?\d{4}){5}\b", re.IGNORECASE)
RE_DOMICILIO = re.compile(r"(?m)^(\s*[•*-]?\s*(?:Domicilio(?: social)?|Fecha de nacimiento)\s*:\s*)[^\n]*")


def texto_para_api(texto: str) -> str:
    """Reduce identificadores de demostración antes de enviarlos al proveedor externo."""
    t = RE_IBAN.sub("[IBAN OMITIDO]", texto)
    t = RE_DNI.sub("[DOCUMENTO IDENTIDAD OMITIDO]", t)
    t = RE_DOMICILIO.sub(lambda m: m.group(1) + "[DATO OMITIDO]", t)
    return t


def formato_documento(registro: dict) -> str:
    m = registro["metadata"]
    titulo = f'{m["documento_id"]} - {m["seccion"]}'
    return f'title: {titulo} | text: {texto_para_api(registro["texto"])}'


def formato_pregunta(pregunta: str) -> str:
    return f"task: question answering | query: {pregunta}"


def leer_corpus(ruta: Path = CORPUS, verificar_sha: bool = True) -> list[dict]:
    if not ruta.is_file():
        raise ValueError(f"Falta el corpus derivado: {ruta}")
    datos = ruta.read_bytes()
    huella = hashlib.sha256(datos).hexdigest()
    if verificar_sha and huella != EXPECTED_SHA256:
        raise ValueError("El SHA256 del corpus no coincide con el archivo v4 aprobado. No indexar; revisar si fue modificado.")
    registros = []
    ids = set()
    for numero, linea in enumerate(datos.decode("utf-8-sig").splitlines(), 1):
        if not linea.strip():
            continue
        try:
            r = json.loads(linea)
            id_ = r["id"]
            texto = r["texto"]
            m = r["metadata"]
            if not isinstance(id_, str) or not isinstance(texto, str) or not texto.strip():
                raise ValueError("ID o texto vacío")
            if id_ in ids:
                raise ValueError("ID duplicado")
            ids.add(id_)
            for campo in ("documento_id", "seccion", "pagina_inicio", "pagina_fin", "sha256_texto"):
                if campo not in m:
                    raise ValueError("Falta metadato: " + campo)
            if m["documento_id"] not in ("DOC-001", "DOC-002", "NOR-001", "NOR-002"):
                raise ValueError("Documento no autorizado")
            if hashlib.sha256(texto.encode("utf-8")).hexdigest() != m["sha256_texto"]:
                raise ValueError("SHA256 de texto incorrecto")
            if not (1 <= int(m["pagina_inicio"]) <= int(m["pagina_fin"])):
                raise ValueError("Rango de páginas inválido")
            registros.append(r)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"Registro {numero}: {e}") from e
    if verificar_sha and len(registros) != EXPECTED_COUNT:
        raise ValueError(f"Cantidad inesperada: {len(registros)}; se esperan {EXPECTED_COUNT}")
    return registros


def seleccionar_muestra(registros: list[dict]) -> list[dict]:
    """Tres registros iniciales por documento y las cuatro secciones evaluables."""
    escogidos = {}
    por_doc = {d: [] for d in ("DOC-001", "DOC-002", "NOR-001", "NOR-002")}
    for r in registros:
        por_doc[r["metadata"]["documento_id"]].append(r)
    for docs in por_doc.values():
        for r in docs[:3]:
            escogidos[r["id"]] = r
    patrones = {
        "DOC-001": r"^PRIMERA\b",
        "DOC-002": r"^TERCERA\b",
        "NOR-001": r"^Artículo\s+15\.",
        "NOR-002": r"^Artículo\s+1\.",
    }
    for doc, patron in patrones.items():
        hallados = [r for r in por_doc[doc] if re.search(patron, r["metadata"]["seccion"], re.I)]
        if not hallados:
            raise ValueError(f"No se encuentra sección crítica para muestra: {doc} {patron}")
        escogidos[hallados[0]["id"]] = hallados[0]
    return [r for r in registros if r["id"] in escogidos]


def metadatos_chroma(r: dict) -> dict:
    m = r["metadata"]
    return {
        "documento_id": str(m["documento_id"]),
        "seccion": str(m["seccion"]),
        "pagina_inicio": int(m["pagina_inicio"]),
        "pagina_fin": int(m["pagina_fin"]),
        "tipo_documento": str(m.get("tipo_documento") or ""),
        "fuente": str(m.get("fuente") or ""),
        "url_oficial": str(m.get("url_oficial") or ""),
        "sha256_texto_original": str(m["sha256_texto"]),
    }


def vector_valido(v: list[float]) -> bool:
    return len(v) == DIM and all(isinstance(x, (float, int)) and math.isfinite(x) for x in v)


def clave_configurada() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def cliente_gemini():
    if not clave_configurada():
        raise RuntimeError("Falta GEMINI_API_KEY. Configúrala solo en la sesión de PowerShell; no la pegues en el chat.")
    from google import genai
    return genai.Client()


def embedding(client, texto: str) -> list[float]:
    """Una cadena por llamada: evita la agregación multimodal de Embedding 2."""
    from google.genai import types
    res = client.models.embed_content(
        model=MODEL,
        contents=texto,
        config=types.EmbedContentConfig(output_dimensionality=DIM),
    )
    if not res.embeddings or len(res.embeddings) != 1 or not res.embeddings[0].values:
        raise RuntimeError("Respuesta de Gemini vacía o de número inesperado de embeddings")
    v = list(res.embeddings[0].values)
    if not vector_valido(v):
        raise RuntimeError(f"Dimensión o datos inválidos en embedding; esperados {DIM}, recibidos {len(v)}")
    return v


def cliente_chroma():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def coleccion_compatible(db, tipo: str, crear: bool):
    nombre = COLLECTIONS[tipo]
    meta = {"modelo": MODEL, "dimension": DIM, "sha256_corpus": EXPECTED_SHA256, "formato": "qa_asimetrico_v1"}
    if crear:
        col = db.get_or_create_collection(name=nombre, metadata=meta)
    else:
        col = db.get_collection(name=nombre)
    if col.metadata is None or any(col.metadata.get(k) != v for k, v in meta.items()):
        raise ValueError(f"La colección {nombre} tiene metadatos incompatibles; no mezclar embeddings.")
    return col


def informe(tipo: str, n: int, meta: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORT_DIR / f"indexacion_{tipo}.json"
    info = {"actualizado_utc": datetime.now(timezone.utc).isoformat(), "coleccion": COLLECTIONS[tipo],
            "registros": n, "modelo": MODEL, "dimension": DIM, "sha256_corpus": EXPECTED_SHA256, **meta}
    ruta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"INFORME: {ruta.relative_to(BASE)}")


def ejecutar_indexacion(tipo: str, pausa: float):
    registros = leer_corpus()
    seleccion = seleccionar_muestra(registros) if tipo == "muestra" else registros
    # El usuario decidió explícitamente ejecutar la indexación: implica enviar textos depurados a Gemini.
    gemini = cliente_gemini()
    db = cliente_chroma()
    col = coleccion_compatible(db, tipo, crear=True)
    existentes = set(col.get(include=[])['ids'])
    extra = existentes - {r['id'] for r in seleccion}
    if extra:
        raise ValueError(f"Colección {tipo} contiene {len(extra)} IDs inesperados; se interrumpe.")
    print(f"COLECCIÓN: {COLLECTIONS[tipo]} | objetivo={len(seleccion)} | existentes={len(existentes)}")
    pendientes = [r for r in seleccion if r['id'] not in existentes]
    for i, r in enumerate(pendientes, 1):
        # Los fallos de cuota/SSL detienen sin corromper registros ya persistidos.
        v = embedding(gemini, formato_documento(r))
        col.add(ids=[r['id']], embeddings=[v], documents=[texto_para_api(r['texto'])], metadatas=[metadatos_chroma(r)])
        print(f"[{i}/{len(pendientes)}] {r['id']} ✓ | total persistido: {col.count()}", flush=True)
        if pausa and i < len(pendientes):
            time.sleep(pausa)
    n = col.count()
    informe(tipo, n, {"objetivo": len(seleccion), "completo": n == len(seleccion)})
    print("RESULTADO:", "INDEXACIÓN COMPLETA" if n == len(seleccion) else "INDEXACIÓN PARCIAL")


def consultar(tipo: str, pregunta: str, top_k: int):
    leer_corpus()
    db = cliente_chroma()
    col = coleccion_compatible(db, tipo, crear=False)
    cantidad = col.count()
    if cantidad == 0:
        raise ValueError("Colección vacía: indexa primero una muestra.")
    gemini = cliente_gemini()
    v = embedding(gemini, formato_pregunta(pregunta))
    resultados = col.query(query_embeddings=[v], n_results=min(top_k, cantidad),
                            include=["metadatas", "documents", "distances"])
    for i, (id_, meta, dist, contenido) in enumerate(zip(resultados['ids'][0], resultados['metadatas'][0],
                                                           resultados['distances'][0], resultados['documents'][0]), 1):
        pag = f"{meta['pagina_inicio']}" if meta['pagina_inicio'] == meta['pagina_fin'] else f"{meta['pagina_inicio']}-{meta['pagina_fin']}"
        print(f"{i}. {id_} | {meta['seccion']} | pág. PDF {pag} | distancia={dist:.4f}")
        print('   Extracto:', contenido.replace('\n',' ')[:250])
    print("NOTA: la distancia es comparativa; aún no es una puntuación de precisión del RAG.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="accion", required=True)
    subs.add_parser('check', help="Solo disco local, sin API ni costes")
    subs.add_parser('smoke', help="Una única llamada inocua a Gemini Embeddings")
    ip = subs.add_parser('index', help="Crea/continúa una colección persistente")
    grupo = ip.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--muestra', action='store_true')
    grupo.add_argument('--completa', action='store_true')
    ip.add_argument('--pausa', type=float, default=5.0, help="Segundos entre llamadas; por defecto 5")
    qp = subs.add_parser('query', help="Consulta con Gemini y Chroma")
    qp.add_argument('--coleccion', choices=['muestra','completa'], default='muestra')
    qp.add_argument('--texto', required=True)
    qp.add_argument('--top-k', type=int, default=5)
    subs.add_parser('status', help="Verifica las colecciones en disco sin llamar a Gemini")
    args = parser.parse_args()
    try:
        if args.accion == 'check':
            regs = leer_corpus()
            print("CORPUS: OK", len(regs), "fragmentos")
            print("DISTRIBUCIÓN:", dict(Counter(r['metadata']['documento_id'] for r in regs)))
            print("MUESTRA ESTRATIFICADA:", len(seleccionar_muestra(regs)), "fragmentos")
            print("MODELO:", MODEL, "| dimensión:", DIM)
            print("GEMINI_API_KEY:", "configurada (no se muestra)" if clave_configurada() else "pendiente")
            print("RESULTADO: PREFLIGHT LOCAL CORRECTO")
        elif args.accion == 'smoke':
            v = embedding(cliente_gemini(), "title: prueba técnica | text: El contrato laboral puede ser indefinido.")
            print(f"GEMINI OK | modelo={MODEL} | dimensiones={len(v)} | norm={math.sqrt(sum(x*x for x in v)):.4f}")
        elif args.accion == 'index':
            if args.pausa < 0: raise ValueError("--pausa debe ser >= 0")
            ejecutar_indexacion('muestra' if args.muestra else 'completa', args.pausa)
        elif args.accion == 'query':
            if args.top_k < 1: raise ValueError("--top-k debe ser >=1")
            consultar(args.coleccion, args.texto, args.top_k)
        elif args.accion == 'status':
            db = cliente_chroma()
            for tipo, nombre in COLLECTIONS.items():
                try:
                    col = coleccion_compatible(db, tipo, crear=False)
                    print(tipo, nombre, 'registros:', col.count())
                except Exception as e:
                    print(tipo, nombre, 'no disponible / incompatible:', str(e)[:150])
    except (Exception, KeyboardInterrupt) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        print("Los registros ya escritos en Chroma persisten. Puedes repetir el mismo comando.",file=sys.stderr)
        raise SystemExit(1)

if __name__ == '__main__':
    main()
