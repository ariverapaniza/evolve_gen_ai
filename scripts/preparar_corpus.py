"""Etapa 1 Evolve: descarga normativa BOE, valida y extrae páginas sin generar embeddings.

Uso: python scripts/preparar_corpus.py [--descargar]
Este script únicamente procesa los cuatro documentos definidos en DOCUMENTOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF

RAIZ = Path(__file__).resolve().parents[1]
CORPUS = RAIZ / "corpus"
INFORMES = RAIZ / "informes"
DESTINO_TXT = CORPUS / "texto_paginas"

DOCUMENTOS = [
    {
        "id": "DOC-001",
        "filename": "originales/DOC-001_Contrato_Alta_Direccion.pdf",
        "tipo": "contrato_ficticio_alta_direccion",
        "fuente": "Documento aportado por el alumno (Evolve)",
        "origen_url": None,
        "paginas_minimas": 3,
        "caracteres_minimos": 3000,
        "marcadores": ["CONTRATO DE TRABAJO DE ALTA DIRECCIÓN", "Chief Data & AI Officer"],
        "hash_fijado": "5f3bdcbe2db32024c4c7059c2b2c30255202ccf242b592e70b6866ffa09f21fd",
    },
    {
        "id": "DOC-002",
        "filename": "originales/DOC-002_Contrato_Indefinido.pdf",
        "tipo": "contrato_ficticio_indefinido",
        "fuente": "Documento aportado por el alumno (Evolve)",
        "origen_url": None,
        "paginas_minimas": 3,
        "caracteres_minimos": 2500,
        "marcadores": ["CONTRATO DE TRABAJO POR TIEMPO", "Analista de Datos"],
        "hash_fijado": "b1a6df8d44e6237c89f3269e567a8d70ea6310088c5a10ce746896b612aa3515",
    },
    {
        "id": "NOR-001",
        "filename": "normativa/NOR-001_Estatuto_Trabajadores_BOE.pdf",
        "tipo": "normativa_consolidada_informativa",
        "fuente": "Agencia Estatal Boletín Oficial del Estado, RDL 2/2015",
        "origen_url": "https://www.boe.es/buscar/pdf/2015/BOE-A-2015-11430-consolidado.pdf",
        "identificador_boe": "BOE-A-2015-11430",
        "fecha_version_consultada": "2026-10-21",
        "texto_version_esperado": "Última modificación: 21 de octubre de 2026",
        "paginas_minimas": 70,
        "caracteres_minimos": 40000,
        "marcadores": ["BOE-A-2015-11430", "ESTATUTO DE LOS TRABAJADORES"],
        "hash_fijado": None,
    },
    {
        "id": "NOR-002",
        "filename": "normativa/NOR-002_Alta_Direccion_BOE.pdf",
        "tipo": "normativa_consolidada_informativa",
        "fuente": "Agencia Estatal Boletín Oficial del Estado, RD 1382/1985",
        "origen_url": "https://www.boe.es/buscar/pdf/1985/BOE-A-1985-17006-consolidado.pdf",
        "identificador_boe": "BOE-A-1985-17006",
        "fecha_version_consultada": "2012-03-06",
        "texto_version_esperado": "Última modificación: 06 de marzo de 2012",
        "paginas_minimas": 6,
        "caracteres_minimos": 5500,
        "marcadores": ["BOE-A-1985-17006", "alta dirección"],
        "hash_fijado": None,
    },
]


def sha256(path: Path) -> str:
    """Huella SHA-256 del archivo sin alterar sus bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for trozo in iter(lambda: f.read(1024 * 1024), b""):
            h.update(trozo)
    return h.hexdigest()


def descargar_si_falta(doc: dict) -> str | None:
    """Descarga documentos BOE conocidos; nunca sobrescribe una copia existente."""
    url = doc.get("origen_url")
    if not url:
        return None
    ruta = CORPUS / doc["filename"]
    if ruta.exists():
        return None
    if not url.startswith("https://www.boe.es/buscar/pdf/"):
        return "URL de descarga no autorizada"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".pdf.part")
    try:
        peticion = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Evolve corpus academico)"})
        with urllib.request.urlopen(peticion, timeout=35) as respuesta, tmp.open("wb") as salida:
            if respuesta.geturl().split("/")[2].lower() not in {"www.boe.es", "boe.es"}:
                return "Descarga redirigida fuera del BOE; cancelada"
            while True:
                bloque = respuesta.read(1024 * 1024)
                if not bloque:
                    break
                salida.write(bloque)
        if not tmp.read_bytes()[:5] == b"%PDF-":
            return "Descarga no es PDF; conserva pendiente para revisión"
        tmp.replace(ruta)
        return None
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return f"Descarga no realizada: {type(exc).__name__}: {exc}"
    finally:
        tmp.unlink(missing_ok=True)


def validar_archivo(doc: dict, extraer: bool = True) -> dict:
    ruta = CORPUS / doc["filename"]
    resultado = {
        "id": doc["id"], "ruta": str(ruta.relative_to(RAIZ)).replace("\\", "/"),
        "tipo": doc["tipo"], "fuente": doc["fuente"],
        "url_original": doc.get("origen_url"),
        "version_consultada": doc.get("fecha_version_consultada"),
        "estado": "PENDIENTE", "motivos": [],
    }
    if not ruta.exists():
        resultado["motivos"].append("Archivo todavía no descargado" if doc.get("origen_url") else "Archivo original ausente")
        return resultado
    try:
        resultado["bytes"] = ruta.stat().st_size
        resultado["sha256"] = sha256(ruta)
        if doc["hash_fijado"] and resultado["sha256"] != doc["hash_fijado"]:
            raise ValueError("Huella SHA-256 del original diferente de la registrada")
        with fitz.open(ruta) as archivo:
            if archivo.is_encrypted:
                raise ValueError("PDF protegido por contraseña")
            paginas = [pagina.get_text("text", sort=True) for pagina in archivo]
        resultado["paginas"] = len(paginas)
        resultado["caracteres"] = sum(len(p) for p in paginas)
        if len(paginas) < doc["paginas_minimas"]:
            raise ValueError(f"Documento tiene menos de {doc['paginas_minimas']} páginas")
        if resultado["caracteres"] < doc["caracteres_minimos"]:
            raise ValueError("No se obtiene texto suficiente; requiere inspección de PDF")
        texto_completo = "\n".join(paginas).casefold()
        for marcador in doc["marcadores"]:
            if marcador.casefold() not in texto_completo:
                raise ValueError(f"No aparece el marcador esperado: {marcador}")
        if doc.get("texto_version_esperado"):
            normalizado = " ".join(texto_completo.split())
            if doc["texto_version_esperado"].casefold() not in normalizado:
                raise ValueError("La versión del BOE no coincide con la versión consultada: revisar antes de continuar")
        if extraer:
            DESTINO_TXT.mkdir(parents=True, exist_ok=True)
            for idx, contenido in enumerate(paginas, 1):
                destino = DESTINO_TXT / f"{doc['id']}_pagina_{idx:03d}.txt"
                destino.write_text(contenido, encoding="utf-8")
        resultado["estado"] = "VALIDADO"
    except (ValueError, RuntimeError, OSError) as exc:
        resultado["estado"] = "ERROR"
        resultado["motivos"].append(str(exc))
    return resultado


def guardar_reporte(resultados: list[dict], errores_descarga: list[str]) -> Path:
    INFORMES.mkdir(parents=True, exist_ok=True)
    total = sum(r.get("paginas", 0) for r in resultados if r["estado"] == "VALIDADO")
    todos = all(r["estado"] == "VALIDADO" for r in resultados)
    payload = {
        "fecha_validacion": date.today().isoformat(),
        "etapa": "Preparación del corpus (sin embeddings ni indexación)",
        "corpus_completo": todos,
        "cantidad_documentos": len(resultados),
        "documentos_validados": sum(r["estado"] == "VALIDADO" for r in resultados),
        "paginas_validadas": total,
        "avisos_descarga": errores_descarga,
        "documentos": resultados,
        "nota_legal": "Los textos consolidados del BOE son informativos y carecen de valor jurídico oficial.",
    }
    ruta = INFORMES / "manifest_validacion_local.json"
    ruta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ruta


def ejecutar(descargar: bool = False) -> int:
    fallos_descarga = []
    if descargar:
        for doc in DOCUMENTOS:
            error = descargar_si_falta(doc)
            if error:
                fallos_descarga.append(f"{doc['id']}: {error}")
    resultados = [validar_archivo(d) for d in DOCUMENTOS]
    reporte = guardar_reporte(resultados, fallos_descarga)
    for res in resultados:
        extra = f" | {res.get('paginas', 0)} páginas | {res.get('caracteres', 0)} caracteres" if res["estado"] == "VALIDADO" else ""
        print(f"{res['id']}: {res['estado']}{extra}")
        for motivo in res["motivos"]:
            print(f"  Detalle: {motivo}")
    for err in fallos_descarga:
        print(f"AVISO: {err}")
    print("Informe:", reporte.relative_to(RAIZ))
    if any(r["estado"] == "ERROR" for r in resultados):
        print("RESULTADO: ERROR. No avanzar al chunking.")
        return 1
    if any(r["estado"] == "PENDIENTE" for r in resultados):
        print("RESULTADO: PENDIENTE. Faltan fuentes; no avanzar al chunking.")
        return 2
    print("RESULTADO: CORPUS VALIDADO. Revisar informe y fuentes antes del siguiente paso.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descargar", action="store_true", help="Descargar los dos PDF consolidados del BOE si faltan")
    argumentos = parser.parse_args()
    sys.exit(ejecutar(descargar=argumentos.descargar))
