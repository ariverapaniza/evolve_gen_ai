r"""Prepara un corpus DERIVADO para embeddings, sin sobrescribir el chunking original.

Ejecutar desde la raíz del proyecto:
  .\.venv\Scripts\python.exe .\scripts\preparar_para_embeddings.py

Solo usa la biblioteca estándar de Python. Mantiene identificadores de origen y páginas.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DOCS_ESPERADOS = {'DOC-001', 'DOC-002', 'NOR-001', 'NOR-002'}
ENCABEZADOS_AISLADOS = {
    'NOR-001-00006', 'NOR-001-00245', 'NOR-001-00252',
    'NOR-001-00296', 'NOR-001-00302'
}
COLA_ID = 'NOR-002-00022'
ANTERIOR_ID = 'NOR-002-00021'
PATRON_MARCA = re.compile(
    r'^[ \t]*(?:P[aá]gina[ \t]+\d+|BOLET[IÍ]N OFICIAL DEL ESTADO|LEGISLACI[OÓ]N CONSOLIDADA)[ \t]*$',
    flags=re.IGNORECASE
)


def hashtexto(texto: str) -> str:
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()


def limpiar_marcas(texto: str) -> tuple[str, int]:
    lineas = texto.splitlines()
    marcas = 0
    res = []
    for linea in lineas:
        if PATRON_MARCA.fullmatch(linea):
            marcas += 1
        else:
            res.append(linea)
    limpio = '\n'.join(res)
    limpio = re.sub(r'\n{3,}', '\n\n', limpio).strip()
    return limpio, marcas


def transformar(chunks: list[dict]) -> tuple[list[dict], dict]:
    if not chunks:
        raise ValueError('Archivo JSONL vacío')
    if {c['metadata']['documento_id'] for c in chunks} != DOCS_ESPERADOS:
        raise ValueError('Los IDs de documentos no coinciden con el corpus validado')
    ids = [c['id'] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('Hay IDs duplicados en el original')
    for c in chunks:
        if hashtexto(c['texto']) != c['metadata']['sha256_texto']:
            raise ValueError(f"Hash de texto incorrecto: {c['id']}")

    por_id = {c['id']: c for c in chunks}
    if not ENCABEZADOS_AISLADOS.issubset(por_id):
        raise ValueError('Faltan encabezados esperados de la auditoría v3; revisar antes de continuar')
    for id_ in ENCABEZADOS_AISLADOS:
        c = por_id[id_]
        if c['metadata']['documento_id'] != 'NOR-001' or not re.fullmatch(
            r'(?:TÍTULO|CAPÍTULO)\s+[IVXLC]+\s*\n.+', c['texto'].strip(), re.DOTALL
        ) or len(c['texto']) >= 100:
            raise ValueError(f'Encabezado no coincide con la auditoría: {id_}')
    anterior, cola = por_id.get(ANTERIOR_ID), por_id.get(COLA_ID)
    if not anterior or not cola or cola['texto'].strip() != 'Título II del Estatuto de los Trabajadores.' or anterior['metadata']['seccion'] != 'Artículo 16. Derechos de representación.' or anterior['metadata']['pagina_fin'] != cola['metadata']['pagina_inicio']:
        raise ValueError('La continuación del artículo 16 no coincide con la auditoría')
    if 'en el' not in anterior['texto'][-35:]:
        raise ValueError('La continuación del artículo 16 no conserva el final esperado')

    res = []
    marcas_total = 0
    eliminados = []
    for original in chunks:
        id_ = original['id']
        if id_ in ENCABEZADOS_AISLADOS or id_ == COLA_ID:
            eliminados.append(id_)
            continue
        c = copy.deepcopy(original)
        texto = c['texto']
        if id_ == ANTERIOR_ID:
            texto = texto.rstrip() + '\n' + cola['texto'].lstrip()
        if c['metadata']['documento_id'].startswith('NOR-'):
            texto, n = limpiar_marcas(texto)
            marcas_total += n
        if not texto:
            raise ValueError(f'Fragmento vacío después de limpieza: {id_}')
        if len(texto) > 1400:
            raise ValueError(f'Fragmento supera 1400 caracteres: {id_}')
        c['texto'] = texto
        c['metadata']['sha256_texto'] = hashtexto(texto)
        # ID, documento, página y sección permanecen sin alterar.
        res.append(c)

    if len(res) != len(chunks) - 6:
        raise AssertionError('Recuento de fragmentos derivado inesperado')
    if any(PATRON_MARCA.fullmatch(line) for c in res for line in c['texto'].splitlines()):
        raise AssertionError('Quedan marcas de encabezado o pie de página del BOE')
    if len({c['id'] for c in res}) != len(res):
        raise AssertionError('IDs duplicados en el derivado')
    if any(hashtexto(c['texto']) != c['metadata']['sha256_texto'] for c in res):
        raise AssertionError('Hashes incorrectos en el derivado')
    for c in res:
        if c['metadata']['pagina_inicio'] < 1 or c['metadata']['pagina_fin'] < c['metadata']['pagina_inicio']:
            raise AssertionError(f'Rango de páginas inválido: {c["id"]}')

    cortos = [{'id': c['id'], 'caracteres': len(c['texto']), 'documento':c['metadata']['documento_id']}
              for c in res if len(c['texto']) < 100]
    report = {
        'estado':'VALIDADO_PARA_PRUEBAS_DE_RECUPERACION',
        'etapa':'2 - Derivado de chunks para embeddings',
        'documentos':dict(Counter(c['metadata']['documento_id'] for c in res)),
        'total_entrada':len(chunks),
        'total_salida':len(res),
        'descartados': sorted(eliminados),
        'fragments_fusionados': {COLA_ID:ANTERIOR_ID},
        'lineas_boilerplate_retiradas': marcas_total,
        'fragmentos_menores_100': cortos,
        'ids_duplicados': 0,
        'hash_texto_verificado': True,
        'nota':'Derivado auditable: se conserva chunks.jsonl sin cambios. No se han probado embeddings ni recuperación semántica.'
    }
    return res, report


def principal() -> None:
    p = argparse.ArgumentParser(description='Limpieza controlada de chunks para embeddings')
    p.add_argument('--origen', default='corpus/chunks/chunks.jsonl')
    p.add_argument('--salida', default='corpus/chunks/chunks_para_embeddings.jsonl')
    p.add_argument('--informe', default='informes/reporte_calidad_chunks.json')
    args = p.parse_args()
    origen = Path(args.origen)
    data = origen.read_bytes()
    chunks = [json.loads(s) for s in data.decode('utf-8-sig').splitlines() if s.strip()]
    salida, informe = transformar(chunks)
    informe['sha256_archivo_origen'] = hashlib.sha256(data).hexdigest()
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    texto = ''.join(json.dumps(c, ensure_ascii=False, separators=(',',':'))+'\n' for c in salida)
    out.write_text(texto, encoding='utf-8', newline='\n')
    informe['sha256_archivo_salida'] = hashlib.sha256(out.read_bytes()).hexdigest()
    informe_path = Path(args.informe)
    informe_path.parent.mkdir(parents=True, exist_ok=True)
    informe_path.write_text(json.dumps(informe, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('DOCUMENTOS:', informe['documentos'])
    print('FRAGMENTOS:', informe['total_entrada'], '->', informe['total_salida'])
    print('MARCAS BOE RETIRADAS:', informe['lineas_boilerplate_retiradas'])
    print('FRAGMENTOS CORTOS:', len(informe['fragmentos_menores_100']))
    print('DATOS:', out)
    print('INFORME:', informe_path)
    print('RESULTADO: CORPUS DERIVADO VALIDADO. Pendiente pruebas de recuperación con embeddings.')

if __name__ == '__main__':
    principal()
