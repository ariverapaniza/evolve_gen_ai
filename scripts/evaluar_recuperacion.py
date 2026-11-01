"""Evolve Etapa 4: prueba trazable de recuperación Gemini Embeddings + ChromaDB.

Comandos:
    python scripts/evaluar_recuperacion.py check
    python scripts/evaluar_recuperacion.py evaluar --limite 3
    python scripts/evaluar_recuperacion.py evaluar --limite 12 --pausa 5
    python scripts/evaluar_recuperacion.py resumen

No se generan embeddings documentales nuevos. Cada pregunta evaluada genera UN
embedding de consulta mediante la API de Google. Los resultados se guardan en
informes/etapa4/evaluacion_recuperacion.json y pueden reanudarse.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
CASES = BASE / "evaluacion" / "casos_recuperacion.json"
REPORT = BASE / "informes" / "etapa4" / "evaluacion_recuperacion.json"
EXPECTED = 437
TOP_K = 5


def modulo_etapa3():
    # El script existente se importa, no se altera ni se sustituye.
    import etapa3_vectores as e3
    return e3


def leer_casos(ruta: Path = CASES) -> list[dict]:
    datos = json.loads(ruta.read_text(encoding="utf-8-sig"))
    casos = datos["casos"]
    ids = set()
    if not casos:
        raise ValueError("Banco de pruebas vacío")
    for c in casos:
        if c["id"] in ids or not c["pregunta"].strip():
            raise ValueError(f"Caso duplicado o sin pregunta: {c['id']}")
        ids.add(c["id"])
        if c["tipo"] not in {"simple", "multiple"}:
            raise ValueError(f"Tipo no admitido: {c['tipo']}")
        if len(c["esperados"]) != (1 if c["tipo"] == "simple" else 2):
            raise ValueError(f"Número de expectativas incorrecto en {c['id']}")
        if any(not x.get("documento_id") or not x.get("seccion_inicio") for x in c["esperados"]):
            raise ValueError(f"Expectativa incompleta en {c['id']}")
    return casos


def coincide(m: dict, esperado: dict) -> bool:
    return (str(m.get("documento_id", "")) == esperado["documento_id"]
            and str(m.get("seccion", "")).casefold().startswith(esperado["seccion_inicio"].casefold()))


def evaluar_resultados(caso: dict, resultados: list[dict]) -> dict:
    posiciones = []
    for exp in caso["esperados"]:
        pos = next((i for i, item in enumerate(resultados, 1)
                    if coincide(item["metadata"], exp)), None)
        posiciones.append(pos)
    if caso["tipo"] == "simple":
        p = posiciones[0]
        return {"posiciones": posiciones, "hit1": p == 1, "hit5": p is not None,
                "rr5": round(1 / p, 6) if p is not None else 0.0, "completo": p is not None}
    return {"posiciones": posiciones, "cobertura_fuentes": sum(p is not None for p in posiciones),
            "total_fuentes": len(posiciones), "completo": all(p is not None for p in posiciones)}


def validar_etiquetas_en_corpus(casos: list[dict], corpus: list[dict]) -> None:
    for c in casos:
        for esp in c["esperados"]:
            if not any(coincide(r["metadata"], esp) for r in corpus):
                raise ValueError(f"La etiqueta esperada de {c['id']} no existe en el corpus: {esp}")


def preflight():
    e3 = modulo_etapa3()
    corpus = e3.leer_corpus()
    casos = leer_casos()
    validar_etiquetas_en_corpus(casos, corpus)
    db = e3.cliente_chroma()
    col = e3.coleccion_compatible(db, "completa", crear=False)
    if col.count() != EXPECTED:
        raise ValueError(f"Se esperaban {EXPECTED} registros, pero hay {col.count()}")
    ids_chroma = set(col.get(include=[])["ids"])
    ids_corpus = {r["id"] for r in corpus}
    if ids_chroma != ids_corpus:
        raise ValueError(f"Chroma no coincide con corpus: faltan {len(ids_corpus - ids_chroma)}, sobran {len(ids_chroma - ids_corpus)} IDs")
    # Las dos estructuras deben haber sido indexadas con el mismo modelo y huella.
    return e3, col, casos


def huella_casos(casos: list[dict]) -> str:
    s = json.dumps(casos, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def cargar_reporte(casos: list[dict], e3, top_k: int = TOP_K, ruta: Path = REPORT) -> dict:
    base = {"modelo": e3.MODEL, "dimension": e3.DIM,
            "sha256_corpus": e3.EXPECTED_SHA256,
            "sha256_casos": huella_casos(casos), "coleccion": e3.COLLECTIONS["completa"],
            "total_registros": EXPECTED, "top_k": top_k}
    if ruta.is_file():
        antiguo = json.loads(ruta.read_text(encoding="utf-8"))
        if any(antiguo.get(campo) != valor for campo, valor in base.items()):
            raise ValueError("Informe anterior incompatible. No se sobrescribe: haz una copia y revisa los parámetros antes de cambiarlo.")
        if not isinstance(antiguo.get("resultados"), dict):
            raise ValueError("Informe anterior inválido")
        return antiguo
    return {**base, "fecha_inicio_utc": datetime.now(timezone.utc).isoformat(), "resultados": {}}


def guardar_reporte(report: dict, ruta: Path = REPORT):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    report["fecha_actualizacion_utc"] = datetime.now(timezone.utc).isoformat()
    # Escritura temporal y sustitución: evita archivos JSON a medio escribir.
    tmp = ruta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ruta)


def resumen(report: dict, casos: list[dict]) -> dict:
    res = report["resultados"]
    simples = [res[c["id"]] for c in casos if c["tipo"] == "simple" and c["id"] in res]
    multiples = [res[c["id"]] for c in casos if c["tipo"] == "multiple" and c["id"] in res]
    return {"evaluados": len(res), "objetivo": len(casos),
            "simples_evaluados": len(simples),
            "hit_at_1": round(sum(x["metricas"]["hit1"] for x in simples)/len(simples),4) if simples else None,
            "hit_at_5": round(sum(x["metricas"]["hit5"] for x in simples)/len(simples),4) if simples else None,
            "mrr_at_5": round(sum(x["metricas"]["rr5"] for x in simples)/len(simples),4) if simples else None,
            "multifuente_evaluados": len(multiples),
            "multifuente_completos": sum(x["metricas"]["completo"] for x in multiples)}


def imprimir_resumen(report: dict, casos: list[dict]):
    m = resumen(report, casos)
    print("CASOS EVALUADOS:", m["evaluados"], "/", m["objetivo"])
    print("PREGUNTAS SIMPLES:", m["simples_evaluados"])
    for etiqueta, clave in (("HIT@1", "hit_at_1"), ("HIT@5", "hit_at_5"), ("MRR@5", "mrr_at_5")):
        print(etiqueta, "=", "pendiente" if m[clave] is None else f"{m[clave]:.4f}")
    print("MULTIFUENTE:", m["multifuente_completos"], "/", m["multifuente_evaluados"], "completos")
    print("INFORME:", REPORT.relative_to(BASE))
    if m["evaluados"] == m["objetivo"]:
        print("RESULTADO: EVALUACIÓN TERMINADA; REQUIERE INTERPRETACIÓN HUMANA DE LOS FALLOS.")
    else:
        print("RESULTADO: EVALUACIÓN PARCIAL; PUEDE REANUDARSE.")


def ejecutar(limite: int, pausa: float):
    e3, col, casos = preflight()
    if not e3.clave_configurada():
        raise ValueError("Configura GEMINI_API_KEY en esta sesión; nunca pegues la clave en el chat.")
    rep = cargar_reporte(casos, e3)
    # Solo se procesan los primeros `limite` casos, conservando los resultados anteriores.
    pendientes = [c for c in casos[:limite] if c["id"] not in rep["resultados"]]
    if not pendientes:
        print("Los casos solicitados ya están guardados; no se repiten solicitudes API.")
    else:
        print(f"Casos seleccionados: {limite}; pendientes API: {len(pendientes)}; completados: {len(rep['resultados'])}")
        client = e3.cliente_gemini()
        for indice, c in enumerate(pendientes, 1):
            v = e3.embedding(client, e3.formato_pregunta(c["pregunta"]))
            cr = col.query(query_embeddings=[v], n_results=TOP_K, include=["metadatas", "distances"])
            resultados = [{"id": i, "metadata": m, "distancia": round(float(d), 6)}
                          for i, m, d in zip(cr["ids"][0], cr["metadatas"][0], cr["distances"][0])]
            metricas = evaluar_resultados(c, resultados)
            rep["resultados"][c["id"]] = {"tipo": c["tipo"], "pregunta": c["pregunta"],
                                            "esperados": c["esperados"], "metricas": metricas,
                                            "resultados": resultados}
            guardar_reporte(rep)
            print(f"[{c['id']}] {'OK' if metricas['completo'] else 'REVISAR'} | posiciones={metricas['posiciones']}")
            for rank, row in enumerate(resultados, 1):
                print(f"  {rank}. {row['id']} | {row['metadata']['seccion'][:75]} | pág {row['metadata']['pagina_inicio']}-{row['metadata']['pagina_fin']} | dist {row['distancia']:.4f}")
            if pausa and indice < len(pendientes):
                time.sleep(pausa)
    imprimir_resumen(rep, casos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sp = parser.add_subparsers(dest="accion", required=True)
    sp.add_parser("check", help="Verifica sin API los 437 IDs y las etiquetas de prueba")
    evalp = sp.add_parser("evaluar", help="Evalúa preguntas mediante Gemini y Chroma, guarda y reanuda")
    evalp.add_argument("--limite", type=int, default=3, help="Número de primeras preguntas a cubrir (máximo 12)")
    evalp.add_argument("--pausa", type=float, default=5.0, help="Espera entre llamadas a Gemini")
    sp.add_parser("resumen", help="Muestra métricas del informe existente, sin API")
    args = parser.parse_args()
    try:
        if args.accion == "check":
            e3, col, casos = preflight()
            print(f"CORPUS: OK {col.count()} embeddings ({e3.MODEL}, {e3.DIM} dimensiones)")
            print(f"BANCO DE PRUEBAS: OK {len(casos)} casos (10 simples, 2 multifuente)")
            print("GEMINI_API_KEY:", "configurada" if e3.clave_configurada() else "pendiente")
            print("RESULTADO: PREFLIGHT RECUPERACIÓN CORRECTO (SIN LLAMADAS API)")
        elif args.accion == "evaluar":
            ncasos = len(leer_casos())
            if not (1 <= args.limite <= ncasos) or args.pausa < 0:
                raise ValueError(f"--limite debe estar entre 1 y {ncasos}; --pausa >= 0")
            ejecutar(args.limite, args.pausa)
        else:
            e3, col, casos = preflight()
            if not REPORT.is_file():
                print("SIN INFORME TODAVÍA: ejecuta primero evaluar --limite 3")
                return
            rep = cargar_reporte(casos, e3)
            imprimir_resumen(rep, casos)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No se han borrado vectores ni documentos. No continúes si el error es de API, cuota o integridad.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
