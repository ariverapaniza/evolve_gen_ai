"""Etapa 4.1: diagnóstico de recuperación dirigida por fuente para M01 y M02.

No modifica Chroma ni el corpus. Reutiliza embeddings ya indexados.
Solo produce dos embeddings de consulta por caso, al ejecutar `evaluar`.

Uso:
    python scripts/diagnostico_multifuente.py check
    python scripts/diagnostico_multifuente.py evaluar --caso M01
    python scripts/diagnostico_multifuente.py evaluar --caso todos --pausa 5
    python scripts/diagnostico_multifuente.py resumen
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
CASOS = BASE / 'evaluacion' / 'casos_recuperacion.json'
DESTINO = BASE / 'informes' / 'etapa4' / 'diagnostico_multifuente.json'
BASELINE = BASE / 'informes' / 'etapa4' / 'evaluacion_recuperacion.json'
TOP_K_POR_FUENTE = 5

# Las consultas se descomponen según las entidades nombradas por el usuario.
# No incluyen las etiquetas de secciones esperadas: no hacemos trampa con el ground truth.
CONSULTAS = {
    'M01': [
        {'documento_id': 'DOC-001', 'consulta': '¿En qué fecha se inicia la vigencia del contrato de alta dirección del directivo?'},
        {'documento_id': 'DOC-002', 'consulta': '¿En qué fecha se inicia la vigencia del contrato indefinido del analista de datos?'},
    ],
    'M02': [
        {'documento_id': 'DOC-002', 'consulta': '¿Cuántos días de vacaciones anuales establece el contrato indefinido del analista de datos?'},
        {'documento_id': 'NOR-001', 'consulta': '¿Qué dispone el artículo 38 del Estatuto de los Trabajadores sobre vacaciones anuales?'},
    ]
}


def casos_multifuente(ruta: Path = CASOS) -> dict:
    raw = json.loads(ruta.read_text(encoding='utf-8-sig'))
    cases = {x['id']: x for x in raw['casos'] if x['tipo'] == 'multiple'}
    if set(cases) != set(CONSULTAS):
        raise ValueError('Los casos multifuente esperados no coinciden con el banco de pruebas.')
    for cid, caso in cases.items():
        docs = [e['documento_id'] for e in caso['esperados']]
        routed = [e['documento_id'] for e in CONSULTAS[cid]]
        if len(set(docs)) != len(docs) or sorted(docs) != sorted(routed):
            raise ValueError(f'Las fuentes de {cid} no están alineadas con la pregunta.')
        if any(not r['consulta'].strip() for r in CONSULTAS[cid]):
            raise ValueError('Consulta vacía')
    return cases


def coincide(meta: dict, expectativa: dict) -> bool:
    return (meta.get('documento_id') == expectativa['documento_id'] and
            str(meta.get('seccion', '')).casefold().startswith(expectativa['seccion_inicio'].casefold()))


def metrica(caso: dict, por_fuente: dict) -> dict:
    posiciones = {}
    for expectativa in caso['esperados']:
        doc = expectativa['documento_id']
        filas = por_fuente.get(doc, [])
        posicion = next((i for i, fila in enumerate(filas, 1)
                         if coincide(fila['metadata'], expectativa)), None)
        posiciones[doc] = posicion
    return {'posiciones_por_fuente': posiciones,
            'fuentes_cubiertas': sum(p is not None for p in posiciones.values()),
            'fuentes_esperadas': len(posiciones),
            'completo': all(p is not None for p in posiciones.values())}


def preflight():
    import etapa3_vectores as e3
    corpus = e3.leer_corpus()  # huella y conteo comprobados
    casos = casos_multifuente()
    db = e3.cliente_chroma()
    col = e3.coleccion_compatible(db, 'completa', crear=False)
    if col.count() != 437:
        raise ValueError(f'Chroma contiene {col.count()} embeddings; se esperaban 437')
    ids = set(col.get(include=[])['ids'])
    if ids != {r['id'] for r in corpus}:
        raise ValueError('Los IDs de Chroma no coinciden con el corpus derivado.')
    for caso in casos.values():
        for exp in caso['esperados']:
            if not any(coincide(r['metadata'], exp) for r in corpus):
                raise ValueError('Sección esperada ausente del corpus: ' + repr(exp))
    return e3, col, casos


def cargar(casos: dict, modelo: str, huella: str) -> dict:
    config = {'modelo': modelo, 'sha256_corpus': huella,
              'version_consultas': hashlib.sha256(json.dumps(CONSULTAS, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              'top_k_por_fuente': TOP_K_POR_FUENTE}
    if DESTINO.exists():
        rep = json.loads(DESTINO.read_text(encoding='utf-8'))
        for k, v in config.items():
            if rep.get(k) != v:
                raise ValueError('El informe anterior usa otros parámetros. Haz respaldo antes de actualizar.')
        return rep
    return {**config, 'creado_utc': datetime.now(timezone.utc).isoformat(), 'resultados': {}}


def guardar(rep: dict):
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    tmp = DESTINO.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(DESTINO)


def buscar(col, vector: list[float], doc: str, n: int = TOP_K_POR_FUENTE) -> list[dict]:
    # Esencial: el filtro se aplica ANTES de ordenar resultados en Chroma.
    result = col.query(query_embeddings=[vector], n_results=n,
                       where={'documento_id': doc},
                       include=['metadatas', 'distances'])
    return [{'id': id_, 'metadata': m, 'distancia': round(float(d), 6)}
            for id_, m, d in zip(result['ids'][0], result['metadatas'][0], result['distances'][0])]


def evaluar(cid: str, pausa: float):
    e3, col, casos = preflight()
    if not e3.clave_configurada():
        raise ValueError('Falta GEMINI_API_KEY. No pegues la clave en el chat.')
    rep = cargar(casos, e3.MODEL, e3.EXPECTED_SHA256)
    elegidos = sorted(casos) if cid == 'todos' else [cid]
    pendientes = [i for i in elegidos if i not in rep['resultados']]
    print(f'Casos seleccionados={len(elegidos)}, pendientes={len(pendientes)}; top-{TOP_K_POR_FUENTE} POR fuente')
    cliente = e3.cliente_gemini() if pendientes else None
    for i, caso_id in enumerate(pendientes):
        caso = casos[caso_id]
        por_fuente = {}
        for j, ruta in enumerate(CONSULTAS[caso_id]):
            emb = e3.embedding(cliente, e3.formato_pregunta(ruta['consulta']))
            resultados = buscar(col, emb, ruta['documento_id'])
            por_fuente[ruta['documento_id']] = resultados
            print(f'  {caso_id} | {ruta["documento_id"]}:')
            for k, r in enumerate(resultados, 1):
                m = r['metadata']
                print(f'     {k}. {r["id"]} | {m["seccion"][:63]} | pág {m["pagina_inicio"]}-{m["pagina_fin"]} | dist {r["distancia"]:.4f}')
            if pausa and (j < len(CONSULTAS[caso_id])-1 or i < len(pendientes)-1):
                time.sleep(pausa)
        metricas = metrica(caso, por_fuente)
        rep['resultados'][caso_id] = {'pregunta': caso['pregunta'], 'consultas_por_fuente': CONSULTAS[caso_id],
                                      'resultados_por_fuente': por_fuente, 'metricas': metricas}
        guardar(rep)
        print(f'  [{caso_id}] {"OK" if metricas["completo"] else "REVISAR"} | cobertura {metricas["fuentes_cubiertas"]}/{metricas["fuentes_esperadas"]} | posiciones {metricas["posiciones_por_fuente"]}')
    resumen(rep, casos)


def resumen(rep: dict, casos: dict):
    final = rep['resultados']
    completos = sum(bool(r['metricas']['completo']) for r in final.values())
    print(f'DIAGNÓSTICO DIRIGIDO: {completos}/{len(final)} casos multifuente completos ({len(final)}/{len(casos)} evaluados)')
    print('IMPORTANTE: top-5 por fuente (hasta 10 resultados por pregunta), NO equivale al Hit@5 global previo.')
    if BASELINE.is_file():
        before = json.loads(BASELINE.read_text(encoding='utf-8'))
        for cid in sorted(final):
            old = before.get('resultados', {}).get(cid, {}).get('metricas', {})
            print(f'{cid} | posiciones globales previas={old.get("posiciones", "no disponible")} | posiciones con filtros={final[cid]["metricas"]["posiciones_por_fuente"]}')
    print('INFORME:', DESTINO.relative_to(BASE))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='accion', required=True)
    sp.add_parser('check')
    e = sp.add_parser('evaluar')
    e.add_argument('--caso', choices=['M01','M02','todos'], default='M01')
    e.add_argument('--pausa', type=float, default=5)
    sp.add_parser('resumen')
    args = p.parse_args()
    try:
        if args.accion == 'check':
            e3, col, cs = preflight()
            print(f'CHROMA: {col.count()} vectores correctos; corpus inmutable')
            for cid, caso in sorted(cs.items()):
                print(cid, '| fuentes:', ', '.join(r['documento_id'] for r in CONSULTAS[cid]))
            print('RESULTADO: PREFLIGHT MULTIFUENTE CORRECTO (SIN LLAMADAS API)')
        elif args.accion == 'evaluar':
            if args.pausa < 0:
                raise ValueError('--pausa no puede ser negativa')
            evaluar(args.caso, args.pausa)
        else:
            e3, col, cs = preflight()
            if not DESTINO.is_file():
                print('SIN INFORME: ejecuta evaluar --caso M01')
            else:
                resumen(cargar(cs, e3.MODEL, e3.EXPECTED_SHA256), cs)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        print('No se han modificado corpus, embeddings ni evaluación original.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
