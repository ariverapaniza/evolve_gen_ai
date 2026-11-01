"""Etapa 2 Evolve (v2) - chunking legal sin indices BOE ni llamadas a Internet.

Uso: python scripts/preparar_chunks.py
Precondicion: scripts/preparar_corpus.py ejecutado con cuatro documentos VALIDADOS.
Genera: corpus/chunks/chunks.jsonl + informes/reporte_chunking.json.

Protege originales: solo lectura; verifica hashes contra el manifiesto de Etapa 1.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

RAIZ = Path(__file__).resolve().parents[1]
MANIFEST = RAIZ / 'informes' / 'manifest_validacion_local.json'
DESTINO = RAIZ / 'corpus' / 'chunks' / 'chunks.jsonl'
REPORTE = RAIZ / 'informes' / 'reporte_chunking.json'
MAX_CARACTERES = 1400
SOLAPE = 180
IDS_ESPERADOS = {'DOC-001', 'DOC-002', 'NOR-001', 'NOR-002'}

# Encabezados que suelen delimitar unidades significativas de legislación y contratos.
PATRON_ARTICULO = re.compile(r'^Art[íi]culo\s+\d+(?:\s+(?:bis|ter|qu[aá]ter))?(?=[.\s:]|$)')
PATRON_DISPOSICION = re.compile(r'^Disposici[oó]n\s+(?:adicional|transitoria|derogatoria|final)\b', re.I)
PATRON_CLAUSULA = re.compile(r'^(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|S[EÉ]PTIMA|OCTAVA|NOVENA|D[EÉ]CIMA|UND[EÉ]CIMA|DUOD[EÉ]CIMA|DECIMOTERCERA|DECIMOCUARTA)\s*[.:]', re.I)
PATRON_TITULO = re.compile(r'^(?:T[ÍI]TULO|CAP[ÍI]TULO)\s+(?:[IVXLCDM]+|\d+)\b', re.I)


def sha256_bytes(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def verificar_manifiesto(ruta: Path = MANIFEST) -> list[dict]:
    """Comprobar cuatro estados, rutas y hashes antes de modificar el corpus derivado."""
    if not ruta.is_file():
        raise ValueError(f'No existe el manifiesto: {ruta}')
    data = json.loads(ruta.read_text(encoding='utf-8'))
    docs = data.get('documentos', [])
    if not data.get('corpus_completo') or data.get('documentos_validados') != 4:
        raise ValueError('La Etapa 1 no figura completada: ejecutar primero preparar_corpus.py')
    if {x.get('id') for x in docs} != IDS_ESPERADOS or len(docs) != 4:
        raise ValueError('El manifiesto no contiene exactamente los cuatro documentos esperados')
    for doc in docs:
        if doc.get('estado') != 'VALIDADO' or not doc.get('sha256'):
            raise ValueError(f"Documento no verificado: {doc['id']}")
        archivo = RAIZ / doc['ruta']
        if not archivo.is_file():
            raise ValueError(f'No existe el archivo indicado: {archivo}')
        if sha256_bytes(archivo) != doc['sha256']:
            raise ValueError(f"El archivo {doc['id']} cambió desde la validación inicial")
    return docs


def detectar_encabezado(linea: str, tipo: str) -> str | None:
    linea = ' '.join(linea.strip().split())
    if not linea:
        return None
    patrones = [PATRON_ARTICULO, PATRON_DISPOSICION, PATRON_TITULO]
    if tipo.startswith('contrato_'):
        patrones.insert(0, PATRON_CLAUSULA)
    if any(p.match(linea) for p in patrones):
        return linea[:150]
    return None


# La expresión v2 buscaba "TEXTO CONSOLIDADO" en cualquier lugar de las páginas.
# También coincidía con menciones narrativas en minúscula en el cuerpo de la ley.
# v3 exige encabezado aislado + fecha de modificación dentro de la misma página.
PATRON_INICIO_CONSOLIDADO = re.compile(
    r"(?m)^[ \t]*TEXTO[ \t]+CONSOLIDADO[ \t]*(?=\r?$)", re.I
)
PATRON_FECHA_VERSION = re.compile(r"Última\s+modificaci[oó]n\s*:", re.I)
PATRON_ARTICULO_REAL = re.compile(r"(?m)^Art[íi]culo\s+\d+(?:[.\s]|$)")


def excluir_indice_boe(paginas: list[str], tipo: str) -> tuple[list[str], dict]:
    """Elimina exclusivamente el índice de legislación BOE manteniendo páginas PDF.

    Valida que el encabezado aislado "TEXTO CONSOLIDADO" tenga debajo
    una "Última modificación:" y que exista un índice anterior. No debe
    confundir menciones narrativas de "texto consolidado" con otro inicio.
    """
    salida = list(paginas)
    if tipo != 'normativa_consolidada_informativa':
        return salida, {'pagina_inicio_contenido': 1, 'paginas_indice_excluidas': []}

    candidatos = []
    cabeceras_aisladas = []
    ocurrencias_libres = []
    for i, pagina in enumerate(paginas):
        if re.search(r'TEXTO\s+CONSOLIDADO', pagina, flags=re.I):
            ocurrencias_libres.append(i + 1)
        for m in PATRON_INICIO_CONSOLIDADO.finditer(pagina):
            cabeceras_aisladas.append((i, m.start(), m.end()))
            # El BOE pone la fecha de actualización inmediatamente después
            # del encabezado; se admiten espacios y líneas intermedias.
            cola = pagina[m.end():m.end() + 900]
            if PATRON_FECHA_VERSION.search(cola):
                candidatos.append((i, m.start(), m.end()))

    # Se prioriza la cabecera con fecha. Si no se extrajo la fecha, se acepta
    # únicamente una cabecera aislada, siempre con índice y articulado comprobado.
    if len(candidatos) == 1:
        cabecera = candidatos[0]
    elif not candidatos and len(cabeceras_aisladas) == 1:
        cabecera = cabeceras_aisladas[0]
    else:
        raise ValueError(
            'No se ha identificado un único encabezado TEXTO CONSOLIDADO '
            'del BOE; '
            f'cabeceras={[(i+1) for i, _, _ in cabeceras_aisladas]}; '
            f'cabeceras_con_fecha={[(i+1) for i, _, _ in candidatos]}; '
            f'menciones en páginas={ocurrencias_libres[:25]}. '
            'Ejecutar scripts/diagnosticar_boe_v3.py y revisar la extracción.'
        )

    i, inicio, fin = cabecera
    preliminar = '\n'.join(paginas[:i])
    if i < 1 or not re.search(r'(?mi)^[ \t]*[ÍI]NDICE[ \t]*$', preliminar):
        raise ValueError('No se puede delimitar de forma segura el índice del BOE')

    salida[:i] = [''] * i
    salida[i] = paginas[i][inicio:]
    if not any(PATRON_ARTICULO_REAL.search(pagina) for pagina in salida[i:]):
        raise ValueError('No se ha encontrado articulado normativo tras descartar el índice')
    return salida, {
        'pagina_inicio_contenido': i + 1,
        'paginas_indice_excluidas': list(range(1, i + 1)),
    }


def separar_secciones(paginas: list[str], tipo: str) -> list[dict]:
    """Une continuaciones de un artículo/cláusula entre páginas y conserva sus páginas."""
    secciones = []
    seccion = 'Texto introductorio'
    componentes: list[tuple[int, str]] = []

    def cerrar():
        nonlocal componentes
        if any(texto.strip() for _, texto in componentes):
            secciones.append({'seccion': seccion, 'componentes': componentes})
        componentes = []

    for num_pagina, texto in enumerate(paginas, 1):
        for linea in texto.splitlines():
            encabezado = detectar_encabezado(linea, tipo)
            if encabezado:
                cerrar()
                seccion = encabezado
            componentes.append((num_pagina, linea.rstrip() + '\n'))
    cerrar()
    return secciones


def ventanas(texto: str, max_caracteres: int = MAX_CARACTERES, solape: int = SOLAPE):
    """Ventanas deterministas; prefiere párrafos y saltos antes de cortar caracteres."""
    if not 0 <= solape < max_caracteres // 2:
        raise ValueError('El solape debe ser no negativo e inferior a la mitad del máximo')
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + max_caracteres, len(texto))
        if fin < len(texto):
            minimo = inicio + int(max_caracteres * 0.72)
            opciones = []
            for sep in ('\n\n', '\n', '. ', '; '):
                pos = texto.rfind(sep, minimo, fin)
                if pos >= 0:
                    opciones.append(pos + len(sep))
            if opciones:
                fin = max(opciones)
        contenido = texto[inicio:fin].strip()
        if contenido:
            yield inicio, fin, contenido
        if fin >= len(texto):
            break
        inicio_siguiente = fin - solape
        if inicio_siguiente <= inicio:
            raise RuntimeError('Segmentación sin progreso; revisar MAX_CARACTERES/SOLAPE')
        inicio = inicio_siguiente


def dividir_documento(doc: dict, paginas: list[str], max_caracteres=MAX_CARACTERES, solape=SOLAPE) -> list[dict]:
    chunks = []
    for numero_seccion, seccion in enumerate(separar_secciones(paginas, doc['tipo']), 1):
        partes = seccion['componentes']
        texto = ''.join(part for _, part in partes)
        rangos = []
        offset = 0
        for pagina, parte in partes:
            rangos.append((offset, offset + len(parte), pagina))
            offset += len(parte)
        for inicio, fin, contenido in ventanas(texto, max_caracteres, solape):
            paginas_origen = sorted({pagina for a, b, pagina in rangos if a < fin and b > inicio})
            if not paginas_origen:
                raise RuntimeError('Falta la trazabilidad de una ventana a las páginas originales')
            nro = len(chunks) + 1
            identificador = f"{doc['id']}-{nro:05d}"
            chunks.append({
                'id': identificador,
                'texto': contenido,
                'metadata': {
                    'documento_id': doc['id'],
                    'tipo_documento': doc['tipo'],
                    'fuente': doc['fuente'],
                    'archivo_fuente': doc['ruta'],
                    'url_oficial': doc.get('url_original'),
                    'version_fuente': doc.get('version_consultada'),
                    'sha256_pdf': doc['sha256'],
                    'seccion': seccion['seccion'],
                    'seccion_orden': numero_seccion,
                    'pagina_inicio': min(paginas_origen),
                    'pagina_fin': max(paginas_origen),
                    'numero_chunk': nro,
                    'sha256_texto': hashlib.sha256(contenido.encode('utf-8')).hexdigest(),
                },
            })
    return chunks


def ejecutar() -> int:
    try:
        documentos = verificar_manifiesto()
        chunks = []
        resumen = []
        for doc in documentos:
            pdf = RAIZ / doc['ruta']
            with pymupdf.open(pdf) as original:
                paginas = [p.get_text('text', sort=True) for p in original]
            if len(paginas) != doc['paginas'] or sum(map(len, paginas)) != doc['caracteres']:
                raise ValueError(f"Extracción diferente al manifiesto en {doc['id']}; no continuar")
            paginas_procesadas, auditoria_indice = excluir_indice_boe(paginas, doc['tipo'])
            filas = dividir_documento(doc, paginas_procesadas)
            if not filas:
                raise ValueError(f"No se generaron chunks para {doc['id']}")
            if any(len(f['texto']) > MAX_CARACTERES for f in filas):
                raise ValueError(f"Algún chunk excede el máximo para {doc['id']}")
            if any(not 1 <= f['metadata']['pagina_inicio'] <= f['metadata']['pagina_fin'] <= len(paginas) for f in filas):
                raise ValueError(f"Error de rangos de páginas en {doc['id']}")
            cubiertas = set()
            for f in filas:
                cubiertas.update(range(f['metadata']['pagina_inicio'], f['metadata']['pagina_fin'] + 1))
            paginas_con_contenido = {i for i, pagina in enumerate(paginas_procesadas, 1) if pagina.strip()}
            faltantes = sorted(paginas_con_contenido - cubiertas)
            if faltantes:
                raise ValueError(f"Páginas sin trazabilidad en {doc['id']}: {faltantes[:10]}")
            chunks.extend(filas)
            resumen.append({
                'documento_id': doc['id'], 'paginas': len(paginas), 'chunks': len(filas),
                'secciones': len({f['metadata']['seccion_orden'] for f in filas}),
                'chunks_cortos_menos_100_caracteres': sum(len(f['texto']) < 100 for f in filas),
                'caracteres_extraidos_originales': doc['caracteres'],
                'paginas_con_contenido_indexable': len(paginas_con_contenido),
                'paginas_indice_excluidas': auditoria_indice['paginas_indice_excluidas'],
                'pagina_inicio_contenido': auditoria_indice['pagina_inicio_contenido'],
            })
        ids = [f['id'] for f in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError('IDs de chunks duplicados')
        DESTINO.parent.mkdir(parents=True, exist_ok=True)
        REPORTE.parent.mkdir(parents=True, exist_ok=True)
        contenido_jsonl = ''.join(json.dumps(f, ensure_ascii=False) + '\n' for f in chunks)
        DESTINO.write_text(contenido_jsonl, encoding='utf-8')
        longitudes = [len(f['texto']) for f in chunks]
        reporte = {
            'etapa': '2 - Segmentación legal sin embeddings',
            'estado': 'VALIDADO',
            'max_caracteres': MAX_CARACTERES,
            'solape_caracteres': SOLAPE,
            'total_documentos': len(documentos),
            'total_paginas_fuente': sum(d['paginas'] for d in documentos),
            'total_paginas_con_contenido_indexable': sum(d['paginas_con_contenido_indexable'] for d in resumen),
            'total_paginas_indice_excluidas': sum(len(d['paginas_indice_excluidas']) for d in resumen),
            'total_chunks': len(chunks),
            'min_longitud_chunk': min(longitudes),
            'max_longitud_chunk': max(longitudes),
            'promedio_longitud_chunk': round(sum(longitudes)/len(longitudes), 1),
            'duplicados_ids': len(ids) - len(set(ids)),
            'por_documento': resumen,
            'nota': 'v2: Se excluye el índice de normativa BOE hasta TEXTO CONSOLIDADO. Los tamaños están en caracteres, no tokens. Las páginas se refieren al PDF original.',
        }
        REPORTE.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for d in resumen:
            print(f"{d['documento_id']}: {d['chunks']} chunks | {d['paginas']} páginas | {d['secciones']} secciones")
        print(f'TOTAL: {len(chunks)} chunks | {sum(d["paginas"] for d in documentos)} páginas fuente')
        print(f'PAGINAS CON CONTENIDO INDEXABLE: {reporte["total_paginas_con_contenido_indexable"]}')
        print(f'PAGINAS DE INDICE EXCLUIDAS: {reporte["total_paginas_indice_excluidas"]}')
        print(f'Reporte: {REPORTE.relative_to(RAIZ)}')
        print(f'Datos: {DESTINO.relative_to(RAIZ)}')
        print('RESULTADO: CHUNKING VALIDADO. Pendiente revisión humana de muestras antes de embeddings.')
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        print('RESULTADO: NO VALIDADO. No continuar a embeddings.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(ejecutar())
