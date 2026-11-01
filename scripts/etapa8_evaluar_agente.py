"""Etapa 8 | Evaluación integral (citas trazables, rutas, memoria, revisión humana).

No reindexa ni altera ChromaDB. 11 casos; uno de ellos requiere observar abstención.
CLI: check | evaluar | resumen
Importa el agente de Etapa 7 corregido (etapa7_memoria.py).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'scripts'))
import etapa7_memoria as memoria
import citas_agrupadas

BANCO = RAIZ / 'evaluacion' / 'casos_calidad_agente.json'
INFORME = RAIZ / 'informes' / 'etapa8' / 'evaluacion_integral.json'
REVISION = RAIZ / 'informes' / 'etapa8' / 'revision_humana.md'
VALIDOS = set(memoria.VALIDAS)
CITA = re.compile(r'\[F(\d+)\]')


def normalizar(texto: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', texto.casefold())
                   if unicodedata.category(ch) != 'Mn')


def cargar_casos() -> list[dict]:
    data = json.loads(BANCO.read_text(encoding='utf-8'))
    casos = data['casos']
    if len(casos) != 11 or len({c['id'] for c in casos}) != len(casos):
        raise ValueError('Banco de casos incompleto o con identificadores duplicados')
    for c in casos:
        if not re.fullmatch(r'[A-Z]\d\d', c['id']):
            raise ValueError('Identificador de caso incorrecto')
        if c['ruta'] not in {'directa', 'documental', 'comparacion', 'aclaracion'}:
            raise ValueError('Ruta esperada no válida')
        if not c['pregunta'] or len(c['pregunta']) > memoria.MAX_PREGUNTA:
            raise ValueError('Pregunta inválida')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', c['sesion']):
            raise ValueError('Sesión inválida')
        if len(c['fuentes']) != len(set(c['fuentes'])) or set(c['fuentes']) - VALIDOS:
            raise ValueError('Fuentes explícitas inválidas')
        if any(e['documento_id'] not in VALIDOS or not e['seccion_inicio'] for e in c['esperados']):
            raise ValueError('Secciones esperadas inválidas')
    memoria_ids = [c['id'] for c in casos if c['sesion'] == 'eval_memoria']
    if memoria_ids != ['D06', 'D07', 'D08']:
        raise ValueError('Las tres preguntas de memoria deben ejecutarse consecutivas y en orden')
    return casos


def evaluar_estructura(caso: dict, resultado: dict) -> dict:
    """Comprueba sólo estructura y cobertura; NUNCA valida verdad jurídica."""
    fallos = []
    ruta = resultado.get('ruta')
    if ruta != caso['ruta']:
        fallos.append(f"Ruta distinta: {ruta!r} vs {caso['ruta']!r}")
    memoria_usada = bool(resultado.get('usa_memoria'))
    if memoria_usada != caso['usa_memoria']:
        fallos.append('Uso de memoria diferente del esperado')
    respuesta = resultado.get('respuesta', '')
    if not isinstance(respuesta, str) or not respuesta.strip():
        fallos.append('Respuesta vacía')
        respuesta = ''

    try:
        citas_texto = citas_agrupadas.extraer_referencias(respuesta)
    except ValueError as exc:
        fallos.append(str(exc))
        citas_texto = set()
    refs = resultado.get('referencias', [])
    refs_ids = [m.get('ref') for m in refs]
    if len(refs_ids) != len(set(refs_ids)):
        fallos.append('Referencias repetidas')
    if any(not re.fullmatch(r'F[1-9]\d*', r or '') for r in refs_ids):
        fallos.append('Referencias sin índice original F# (corrección no aplicada)')
    if citas_texto != set(refs_ids):
        fallos.append(f'Las citas del texto {sorted(citas_texto)} no coinciden con las mostradas {sorted(str(x) for x in refs_ids)}')
    if caso['ruta'] in ('documental', 'comparacion') and not caso.get('requiere_abstencion') and not citas_texto:
        fallos.append('La respuesta documental carece de citas')
    if caso['ruta'] in ('directa', 'aclaracion') and (citas_texto or refs):
        fallos.append('La ruta sin RAG presenta citas')

    presentes = []
    for e in caso['esperados']:
        if any(m.get('documento_id') == e['documento_id']
               and str(m.get('seccion','')).casefold().startswith(e['seccion_inicio'].casefold())
               for m in refs):
            presentes.append(e)
        else:
            fallos.append(f"No se citó {e['documento_id']} {e['seccion_inicio']}")

    for frase in caso['texto_requerido']:
        if normalizar(frase) not in normalizar(respuesta):
            fallos.append(f'No se encontró en la respuesta: {frase}')

    # La abstención de D10 exige lectura humana: no se infiere a partir de citas ni palabras.
    return {
        'estado_estructural': 'OK' if not fallos else 'REVISAR',
        'fallos': fallos, 'evidencias_esperadas_citadas': len(presentes),
        'total_evidencias_esperadas': len(caso['esperados']),
        'revision_manual_obligatoria': bool(caso['revision_manual']),
    }


def preflight():
    casos = cargar_casos()
    corpus = memoria.e6.rag.e3.leer_corpus()
    distribucion = {e['id']: e['metadata'] for e in corpus}
    for caso in casos:
        for esperado in caso['esperados']:
            if not any(meta['documento_id'] == esperado['documento_id']
                       and meta['seccion'].casefold().startswith(esperado['seccion_inicio'].casefold())
                       for meta in distribucion.values()):
                raise ValueError(f"Ground truth no presente: {caso['id']}: {esperado}")
    col = memoria.e6.rag.preflight()
    print(f'CHROMA: {col.count()} embeddings (sólo lectura)')
    print(f'BANCO: {len(casos)} casos; memoria en una sesión y aislamiento entre sesiones')
    print('CITAS: se mantienen las etiquetas F# originales de Gemini en la salida y el informe')
    print('RESULTADO: PREFLIGHT DE EVALUACIÓN CORRECTO, SIN LLAMADAS A GEMINI')


def guardar(destino: Path, datos: dict):
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix('.tmp')
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(destino)


def resumen_data(datos: dict) -> dict:
    registros = datos.get('resultados', [])
    completos = [r for r in registros if r.get('estado_estructural')]
    ok = sum(r['estado_estructural'] == 'OK' for r in completos)
    manuales = sum(bool(r.get('revision_manual_obligatoria')) for r in registros)
    return {'casos_ejecutados': len(registros), 'casos_objetivo': len(cargar_casos()),
            'sin_errores_estructurales': ok, 'con_incidencias_o_errores': len(registros)-ok,
            'pendientes_verificacion_humana': manuales}


def plantilla_revision(datos: dict):
    lineas = [
        '# Revisión humana de la Etapa 8 — Proyecto Evolve',
        '', 'Los controles automáticos NO demuestran exactitud jurídica ni fundamentación semántica.',
        'Contrasta cada afirmación (también las adicionales) contra los PDF originales,',
        'incluyendo documento, página, sección y correspondencia de las etiquetas [F#].',
        '', '| Caso | Ruta | Control estructural | Revisión factual (rellenar) | Observaciones |',
        '|---|---|---|---|---|',
    ]
    for r in datos.get('resultados', []):
        lineas.append(f"| {r['id']} | {r.get('ruta_real','error')} | {r.get('estado_estructural','ERROR')} | PENDIENTE | |")
    lineas.extend([
        '', '## Puntos de atención',
        '- **D06-D08:** fecha de inicio de vigencia distinta de la fecha de firma; las dos fechas de inicio deben tener citas a TERCERA.',
        '- **D09:** se citan el contrato DOC-002 y el artículo 38 de NOR-001 con información pertinente.',
        '- **D10:** el asistente debería reconocer que no existe evidencia de satélites en estos documentos; no inventar una cifra.',
        '- **D11:** la sesión independiente debe pedir aclaración y no heredar datos de otra sesión.',
        '- Las referencias [F#] deben coincidir exactamente con la tabla de fuentes impresa.',
        '- Las frases no solicitadas (ej. fechas de firma) también requieren contraste.',
        '', '## Criterio de cierre',
        'Registrar un veredicto humano JUSTIFICADO en cada caso y corregir problemas reproducibles antes de redactar las conclusiones finales.',
    ])
    REVISION.parent.mkdir(parents=True, exist_ok=True)
    REVISION.write_text('\n'.join(lineas)+'\n', encoding='utf-8')


def auditar_guardado():
    """Reevaluar TODAS las citas del JSON existente sin reconsultar Gemini.

    No altera evaluacion_integral.json ni revision_humana.md.
    """
    if not INFORME.exists():
        raise ValueError('Falta informes/etapa8/evaluacion_integral.json; no se recrea automáticamente')
    datos = json.loads(INFORME.read_text(encoding='utf-8'))
    casos = {c['id']: c for c in cargar_casos()}
    salidas = []
    for registro in datos.get('resultados', []):
        nombre = registro['id']
        if nombre not in casos:
            raise ValueError(f'Caso inesperado: {nombre}')
        original = {
            'ruta': registro.get('ruta_real'),
            'usa_memoria': registro.get('usa_memoria_real'),
            'respuesta': registro.get('respuesta',''),
            'referencias': registro.get('referencias',[]),
        }
        control = evaluar_estructura(casos[nombre], original)
        salida = {'id': nombre, 'estado_original': registro.get('estado_estructural'),
                  'estado_revisado': control['estado_estructural'], 'fallos': control['fallos']}
        salidas.append(salida)
        detalle = ' | '.join(control['fallos']) if control['fallos'] else 'sin errores de citas/ruta/formato detectados'
        print(f"{nombre} | {salida['estado_revisado']} | {detalle}")
    resumen = {'casos': len(salidas), 'ok_estructurales_revisados': sum(r['estado_revisado']=='OK' for r in salidas),
               'a_revisar': sum(r['estado_revisado']!='OK' for r in salidas)}
    destino = RAIZ / 'informes' / 'etapa8' / 'auditoria_citas_v81.json'
    guardar(destino, {'origen': str(INFORME.relative_to(RAIZ)), 'resumen': resumen, 'resultados': salidas,
                     'advertencia': 'Sin validación factual. El informe de evaluación original se conserva intacto.'})
    print('RESUMEN CORREGIDO:', json.dumps(resumen, ensure_ascii=False))
    print('INFORME NUEVO:', destino.relative_to(RAIZ))
    print('NOTA: ninguna llamada Gemini; no modifica los registros previos.')


def ejecutar():
    casos = cargar_casos()
    # No se ofrece reanudación automática: el checkpoint InMemorySaver debe vivir
    # durante las tres preguntas D06-D08 en el mismo proceso Python.
    grafo = memoria.crear_grafo()
    datos = {
        'etapa': 8,
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'modelo_llm': memoria.e6.rag.DEFAULT_MODEL,
        'modelo_embeddings': memoria.e6.rag.e3.MODEL,
        'alcance': 'evaluación académica exploratoria, no precisión estadística ni validación jurídica',
        'resultados': [],
    }
    guardar(INFORME, datos)
    print('EVALUACIÓN ETAPA 8 | 11 casos | SIN reindexar corpus')
    print('AVISO: durante D06-D08 mantenemos el MISMO proceso para probar memoria real.')
    for caso in casos:
        print('\n'+'='*72)
        print(f"[{caso['id']}] {caso['pregunta']}")
        try:
            salida = memoria.preguntar(grafo, sesion=caso['sesion'], texto=caso['pregunta'],
                                      fuentes=caso['fuentes'], top_k=5)
        except Exception as exc:
            # Evitar revelar secretos incluso si el SDK incorporara una clave a un error.
            clave = os.getenv('GEMINI_API_KEY', '')
            mensaje = f'{type(exc).__name__}: {str(exc)[:240]}'
            if clave:
                mensaje = mensaje.replace(clave, '[CLAVE_OCULTA]')
            registro = {'id': caso['id'], 'pregunta': caso['pregunta'],
                        'estado_estructural': 'ERROR', 'error': mensaje,
                        'revision_manual_obligatoria': True}
            datos['resultados'].append(registro)
            guardar(INFORME, datos)
            print('ERROR DEL CASO:', mensaje)
            # Una llamada fallida podría invalidar el historial para el siguiente turno.
            print('EVALUACIÓN INTERRUMPIDA. Consulte el informe y el error antes de repetirla.')
            break
        control = evaluar_estructura(caso, salida)
        registro = {
            'id': caso['id'], 'sesion': caso['sesion'], 'pregunta': caso['pregunta'],
            'pregunta_interpretada': salida.get('pregunta_resuelta', ''),
            'ruta_esperada': caso['ruta'], 'ruta_real': salida.get('ruta'),
            'usa_memoria_esperado': caso['usa_memoria'],
            'usa_memoria_real': bool(salida.get('usa_memoria')),
            'fuentes_dirigidas': salida.get('fuentes_resueltas', []),
            'nodos': salida.get('recorrido', []),
            'respuesta': salida.get('respuesta', ''),
            'referencias': salida.get('referencias', []),
            **control,
        }
        datos['resultados'].append(registro)
        guardar(INFORME, datos)
        print('RUTA:', registro['ruta_real'], '| MEMORIA:', 'sí' if registro['usa_memoria_real'] else 'no')
        print('PREGUNTA INTERPRETADA:', registro['pregunta_interpretada'])
        for fuente in registro['referencias']:
            print(f"  [{fuente['ref']}] {fuente['documento_id']} | {fuente['seccion']} | PDF pág. {fuente['pagina_inicio']}-{fuente['pagina_fin']}")
        print('RESPUESTA:\n'+registro['respuesta'])
        print('CONTROL AUTOMÁTICO:', control['estado_estructural'])
        for fallo in control['fallos']:
            print('  REVISAR:', fallo)
        if caso.get('requiere_abstencion'):
            print('AVISO: LA ABSTENCIÓN DE ESTE CASO REQUIERE COMPROBACIÓN HUMANA.')
        if control['revision_manual_obligatoria']:
            print('VERIFICACIÓN DE CONTENIDO CONTRA PDF: PENDIENTE.')
    datos['resumen'] = resumen_data(datos)
    guardar(INFORME, datos)
    plantilla_revision(datos)
    print('\nRESUMEN:', json.dumps(datos['resumen'], ensure_ascii=False))
    print('INFORME:', INFORME.relative_to(RAIZ))
    print('REVISIÓN HUMANA:', REVISION.relative_to(RAIZ))
    if len(datos['resultados']) == len(casos):
        print('RESULTADO: BATERÍA EJECUTADA; PENDIENTE REVISIÓN FACTUAL HUMANA.')
    else:
        print('RESULTADO: BATERÍA INCOMPLETA. REVISAR ERROR ANTES DE CONTINUAR.')


def resumen():
    if not INFORME.exists():
        raise ValueError('No existe el informe; primero ejecuta evaluar')
    datos = json.loads(INFORME.read_text(encoding='utf8'))
    valores = resumen_data(datos)
    print('RESUMEN ETAPA 8:', json.dumps(valores, ensure_ascii=False, indent=2))
    for r in datos.get('resultados', []):
        print(f"{r['id']} | {r.get('estado_estructural','ERROR')} | {r.get('ruta_real', '?')} | "+
              '; '.join(r.get('fallos', [])))
    print('LA REVISIÓN JURÍDICA Y SEMÁNTICA NO ESTÁ AUTOMATIZADA.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion', choices=['check', 'evaluar', 'resumen', 'auditar'])
    args = parser.parse_args()
    try:
        {'check': preflight, 'evaluar': ejecutar, 'resumen': resumen, 'auditar': auditar_guardado}[args.accion]()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
