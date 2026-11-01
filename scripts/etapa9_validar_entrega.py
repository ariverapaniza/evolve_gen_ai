"""Validador offline para los artefactos del Proyecto Final Evolve (Etapa 9).

No invoca Gemini, no indexa, no cambia ChromaDB ni los informes.
CLI: python scripts/etapa9_validar_entrega.py check
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Proyecto_Final_Evolve_Gemini_RAG_LangGraph.ipynb"
ARCHIVOS = [
    ROOT / "README.md", ROOT / "docs" / "MEMORIA_TECNICA_PROYECTO_FINAL.md",
    ROOT / "informes" / "etapa8" / "revision_factual_preliminar.md",
    ROOT / "informes" / "etapa8" / "evaluacion_integral.json",
    ROOT / "informes" / "etapa8" / "auditoria_citas_v81.json",
    ROOT / "scripts" / "etapa3_vectores.py",
    ROOT / "scripts" / "etapa5_rag.py",
    ROOT / "scripts" / "etapa6_langgraph.py",
    ROOT / "scripts" / "etapa7_memoria.py",
    NOTEBOOK,
]

def comprobar_notebook(ruta: Path) -> int:
    doc = json.loads(ruta.read_text(encoding="utf-8"))
    if doc.get("nbformat") != 4 or not isinstance(doc.get("cells"), list):
        raise ValueError("Formato de notebook inválido (se espera Jupyter nbformat=4)")
    code_cells = [c for c in doc["cells"] if c.get("cell_type") == "code"]
    if len(code_cells) < 10:
        raise ValueError("Se esperan al menos 10 celdas de código")
    for celda in code_cells:
        src = celda.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        compile(src, str(ruta), "exec")
    return len(code_cells)

def check() -> None:
    faltantes = [str(p.relative_to(ROOT)) for p in ARCHIVOS if not p.is_file()]
    if faltantes:
        raise ValueError("Faltan archivos: " + ", ".join(faltantes))
    n_code = comprobar_notebook(NOTEBOOK)
    sys.path.insert(0, str(ROOT / "scripts"))
    import etapa3_vectores as e3
    import etapa5_rag as rag
    corpus = e3.leer_corpus()
    col = rag.preflight()
    if len(corpus) != 437 or col.count() != 437:
        raise ValueError("Corpus o colección Chroma incompletos")
    informe = json.loads((ROOT / "informes" / "etapa8" / "evaluacion_integral.json").read_text(encoding="utf-8"))
    auditoria = json.loads((ROOT / "informes" / "etapa8" / "auditoria_citas_v81.json").read_text(encoding="utf-8"))
    if len(informe.get("resultados", [])) != 11:
        raise ValueError("La evaluación integral no contiene los 11 casos")
    if not isinstance(auditoria.get("resumen"), dict):
        raise ValueError("Auditoría de citas incompleta")
    print("NOTEBOOK: OK |", n_code, "celdas de código válidas")
    print("DOCUMENTACIÓN: OK | README.md + memoria técnica + revisión factual")
    print("CORPUS: OK |", len(corpus), "fragmentos aprobados |", e3.MODEL, e3.DIM, "dimensiones")
    print("CHROMA: OK |", col.count(), "vectores de la colección completa")
    print("EVALUACIÓN: OK | 11 casos históricos | auditoría presente")
    print("RESULTADO: PREFLIGHT ETAPA 9 CORRECTO (SIN LLAMADAS API)")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=["check"])
    parser.parse_args()
    try:
        check()
    except Exception as error:
        print(f"ERROR DE ENTREGA: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
