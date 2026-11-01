"""Control básico de calidad del chunking legal sin embeddings ni Internet."""
from __future__ import annotations
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CHUNKS = RAIZ / 'corpus' / 'chunks' / 'chunks.jsonl'
REPORTE = RAIZ / 'informes' / 'reporte_chunking.json'
CASOS = [
    ('DOC-001', r'^PRIMERA\.', 2),
    ('DOC-002', r'^TERCERA\.', 2),
    ('NOR-001', r'^Art[íi]culo 15\.', 20),
    ('NOR-002', r'^Art[íi]culo 1\.', 3),
]


def main() -> int:
    if not CHUNKS.is_file() or not REPORTE.is_file():
        print('ERROR: no aparecen chunks/reporte; ejecutar preparar_chunks.py')
        return 1
    chunks = [json.loads(line) for line in CHUNKS.read_text(encoding='utf-8').splitlines() if line.strip()]
    reporte = json.loads(REPORTE.read_text(encoding='utf-8'))
    print(f"CHUNKS: {len(chunks)} | ORIGINALES: {reporte.get('total_paginas_fuente')} páginas | "
          f"INDEXABLES: {reporte.get('total_paginas_con_contenido_indexable')} páginas | "
          f"INDICES EXCLUIDOS: {reporte.get('total_paginas_indice_excluidas')} páginas")
    if 'total_paginas_indice_excluidas' not in reporte:
        print('ERROR: parece persistir la versión anterior de preparar_chunks.py')
        return 1
    errores = []
    for doc, patron, esperada in CASOS:
        lista = [c for c in chunks if c['metadata']['documento_id'] == doc and re.match(patron, c['metadata']['seccion'])]
        if not lista:
            errores.append(f'{doc}: no se encuentra sección {patron}')
            continue
        candidato = lista[0]
        pagina = candidato['metadata']['pagina_inicio']
        contenido = candidato['texto']
        print(f"{doc}: {candidato['metadata']['seccion'][:66]} | página {pagina} | {len(contenido)} caracteres")
        if pagina != esperada:
            errores.append(f'{doc}: página encontrada {pagina}, se esperaba {esperada}; revisar fuente')
        if doc.startswith('NOR-') and (re.search(r'\.{4,}', contenido) or len(contenido) < 150):
            errores.append(f'{doc}: la muestra parece contener sólo un título del índice')
    cortos = [c for c in chunks if len(c['texto']) < 100]
    print(f'CHUNKS CORTOS (<100 caracteres): {len(cortos)}')
    for c in cortos[:15]:
        print(f"  {c['id']} | pág. {c['metadata']['pagina_inicio']} | {len(c['texto'])} car. | {c['metadata']['seccion'][:70]}")
    if errores:
        for error in errores: print('REVISAR:', error)
        print('RESULTADO: REVISIÓN NO SUPERADA.')
        return 1
    print('RESULTADO: CONTROLES AUTOMÁTICOS DE MUESTRAS SUPERADOS. Pendiente inspección humana.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
