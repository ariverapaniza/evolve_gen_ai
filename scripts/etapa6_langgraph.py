"""Etapa 6 Evolve — LangGraph condicional sobre RAG validado (sin memoria).

Se reutiliza etapa5_rag.py v5.1, y NO se altera ChromaDB.
CLI: check | plano | preguntar --texto ... [--fuentes DOC-001 DOC-002]
El grafo es deliberadamente sin checkpointer: memoria en Etapa 7.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypedDict
import unicodedata

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
import etapa5_rag as rag


class Estado(TypedDict, total=False):
    pregunta: str
    fuentes_explicitas: list[str]
    top_k: int
    modelo: str
    ruta: str
    fuentes: list[str]
    evidencias: list[dict]
    respuesta: str
    citas: list[int]
    recorrido: list[str]
    motivo_abstencion: str


def sin_acentos(texto: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', texto.casefold())
                   if unicodedata.category(c) != 'Mn').strip()


DIRECTAS = {
    'hola', 'buenos dias', 'buenas tardes', 'buenas noches',
    'gracias', 'muchas gracias', 'adios', 'hasta luego',
    'que puedes hacer', 'que puedes hacer?',
    'ayuda', 'como funciona', 'como funcionas',
}


def ruta_inicial(pregunta: str, fuentes: list[str] | None = None) -> tuple[str, list[str]]:
    """Solo los saludos y la ayuda explícita saltan RAG; el resto consulta documentos."""
    q = sin_acentos(pregunta)
    if not q or len(pregunta) > 800:
        raise ValueError('La pregunta debe tener entre 1 y 800 caracteres')
    autorizadas = rag.elegir_fuentes(pregunta, fuentes or None)
    if not autorizadas and q.strip('¿?¡!., ') in DIRECTAS:
        return 'directa', []
    if len(autorizadas) > 1:
        return 'comparacion', autorizadas
    return 'documental', autorizadas


def crear_grafo(*, modulo_rag=None, coleccion=None, cliente=None):
    """Construye un StateGraph real. Permite dependencias inyectables para pruebas."""
    from langgraph.graph import StateGraph, START, END
    from langchain_core.runnables import RunnableLambda
    r = modulo_rag or rag
    recursos = {'col': coleccion, 'cliente': cliente}

    def servicios():
        if recursos['col'] is None:
            recursos['col'] = r.preflight()  # 437 vectores, en solo lectura
        if recursos['cliente'] is None:
            recursos['cliente'] = r.e3.cliente_gemini()
        return recursos['col'], recursos['cliente']

    def clasificar(estado: Estado) -> dict:
        pregunta = estado.get('pregunta', '')
        ruta, fuentes = ruta_inicial(pregunta, estado.get('fuentes_explicitas'))
        return {'ruta': ruta, 'fuentes': fuentes, 'recorrido': ['clasificar'],
                'evidencias': [], 'respuesta': '', 'citas': [], 'motivo_abstencion': ''}

    def seleccionar(estado: Estado) -> str:
        return estado['ruta']

    def respuesta_directa(estado: Estado) -> dict:
        q = sin_acentos(estado['pregunta'])
        if q.rstrip('!?., ') in {'gracias', 'muchas gracias'}:
            texto = 'De nada. Puedo consultar cláusulas de los contratos ficticios y artículos del BOE.'
        else:
            texto = ('Soy un asistente académico para consultar dos contratos ficticios y '
                     'el Estatuto de los Trabajadores y RD 1382/1985. '
                     'Puedo comparar documentos y citar páginas PDF. '
                     'No sustituyo el asesoramiento jurídico.')
        return {'respuesta': texto, 'recorrido': estado['recorrido'] + ['respuesta_directa']}

    def recuperar(estado: Estado) -> dict:
        col, client = servicios()
        top_k = estado.get('top_k', 5)
        evidencias = r.recuperar(col, client, estado['pregunta'], estado['fuentes'], top_k)
        return {'evidencias': evidencias, 'recorrido': estado['recorrido'] + ['recuperar']}

    def validar(estado: Estado) -> dict:
        evidencias = estado.get('evidencias') or []
        if not evidencias:
            return {'motivo_abstencion': 'No hay fragmentos recuperados.',
                    'recorrido': estado['recorrido'] + ['validar']}
        # Este control también comprueba evidencia suficiente en la comparación de fechas.
        # Si falla, debe interrumpir la ejecución; no es válido simular una respuesta.
        r.comprobar_cobertura(estado['pregunta'], estado['fuentes'], evidencias)
        return {'motivo_abstencion': '', 'recorrido': estado['recorrido'] + ['validar']}

    def hay_contexto(estado: Estado) -> str:
        return 'abstener' if estado.get('motivo_abstencion') else 'generar'

    def abstener(estado: Estado) -> dict:
        return {'respuesta': 'No he encontrado evidencia documental suficiente para responder con rigor.',
                'citas': [], 'recorrido': estado['recorrido'] + ['abstener']}

    def generar(estado: Estado) -> dict:
        _, client = servicios()
        respuesta, citas = r.generar_respuesta(
            client, estado['pregunta'], estado['evidencias'], estado['fuentes'],
            estado.get('modelo', r.DEFAULT_MODEL))
        return {'respuesta': respuesta, 'citas': citas,
                'recorrido': estado['recorrido'] + ['generar']}

    builder = StateGraph(Estado)
    for nombre, funcion in (
        ('clasificar', clasificar), ('respuesta_directa', respuesta_directa),
        ('recuperar', recuperar), ('validar', validar),
        ('abstener', abstener), ('generar', generar),
    ):
        builder.add_node(nombre, RunnableLambda(funcion))
    builder.add_edge(START, 'clasificar')
    builder.add_conditional_edges('clasificar', seleccionar, {
        'directa': 'respuesta_directa',
        'documental': 'recuperar',
        'comparacion': 'recuperar',
    })
    builder.add_edge('recuperar', 'validar')
    builder.add_conditional_edges('validar', hay_contexto, {
        'abstener': 'abstener', 'generar': 'generar',
    })
    for destino in ('respuesta_directa', 'abstener', 'generar'):
        builder.add_edge(destino, END)
    return builder.compile()  # SIN memoria en esta etapa


def imprimir_salida(resultado: dict):
    print('RUTA:', resultado.get('ruta', '?'))
    print('NODOS:', ' -> '.join(resultado.get('recorrido', [])))
    print('FUENTES DIRIGIDAS:', resultado.get('fuentes', []))
    for i, ev in enumerate(resultado.get('evidencias', []), 1):
        m = ev['metadata']
        print(f'[F{i}] {m["documento_id"]} | {m["seccion"]} | '
              f'pág PDF {m["pagina_inicio"]}-{m["pagina_fin"]}')
    print('\nRESPUESTA:\n' + resultado.get('respuesta', ''))
    if resultado.get('citas'):
        print('CITAS:', ', '.join(f'F{i}' for i in sorted(set(resultado['citas']))))
    print('NOTA: las citas numéricas requieren contraste con los PDF; no constituyen verificación jurídica.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='accion', required=True)
    sub.add_parser('check', help='Comprueba LangGraph y 437 vectores SIN llamadas API')
    sub.add_parser('plano', help='Ejemplo de ruta directa SIN Gemini')
    ask = sub.add_parser('preguntar', help='Invoca el grafo condicional para pregunta documental')
    ask.add_argument('--texto', required=True)
    ask.add_argument('--fuentes', nargs='+', choices=rag.VALID_SOURCES)
    ask.add_argument('--top-k', type=int, default=5)
    ask.add_argument('--modelo', default=rag.DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        if args.accion == 'check':
            g = crear_grafo()
            col = rag.preflight()
            print('LANGGRAPH: OK | nodos:', ', '.join(sorted(n for n in g.nodes if not n.startswith('__'))))
            print(f'CHROMA: OK | {col.count()} embeddings | SIN API')
            print('GEMINI_API_KEY:', 'configurada' if rag.e3.clave_configurada() else 'pendiente')
            print('RESULTADO: PREFLIGHT ETAPA 6 CORRECTO')
            return
        if args.accion == 'plano':
            result = crear_grafo().invoke({'pregunta': 'Hola'})
            imprimir_salida(result)
            print('RESULTADO: RUTA DIRECTA EJECUTADA SIN RAG NI API')
            return
        if not 1 <= args.top_k <= 5:
            raise ValueError('--top-k debe estar entre 1 y 5')
        g = crear_grafo()
        resultado = g.invoke({'pregunta': args.texto, 'fuentes_explicitas': args.fuentes or [],
                              'top_k': args.top_k, 'modelo': args.modelo})
        imprimir_salida(resultado)
        print('RESULTADO: GRAFO EJECUTADO (REVISIÓN HUMANA PENDIENTE)')
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('No se ha modificado ChromaDB ni el corpus original.', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
