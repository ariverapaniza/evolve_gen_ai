"""Etapa 7: contexto, sesiones independientes y checkpoints LangGraph."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import etapa7_memoria as mem


def t(tema='fecha_inicio', foco='DOC-001', ruta='documental'):
    return {'pregunta': '¿Cuál es la fecha de inicio del contrato de alta dirección?',
            'pregunta_resuelta': '¿Cuál es la fecha de inicio del contrato de alta dirección?',
            'tema': tema, 'fuente_foco': foco, 'ruta': ruta,
            'fuentes': [foco] if foco else [], 'respuesta': 'Resultado de ejemplo'}


class PruebasReescritura(unittest.TestCase):
    def test_normalizacion(self):
        self.assertEqual(mem.normalizar('Alta Dirección'), 'alta direccion')

    def test_tema_inicio(self):
        self.assertEqual(mem.tema_de('¿Qué fecha de inicio de vigencia tiene?'), 'fecha_inicio')

    def test_tema_vacaciones(self):
        self.assertEqual(mem.tema_de('¿Cuántas vacaciones anuales?'), 'vacaciones')

    def test_identifica_contrato_alta_direccion(self):
        self.assertEqual(mem.documento_mencionado('Contrato de Alta Dirección'), 'DOC-001')

    def test_identifica_contrato_indefinido(self):
        self.assertEqual(mem.documento_mencionado('Contrato indefinido'), 'DOC-002')

    def test_no_hereda_filtro_de_pregunta_nueva(self):
        r = mem.resolver_seguimiento('¿Qué dispone el Estatuto de los Trabajadores?', [t()])
        self.assertEqual(r['fuentes_resueltas'], ['NOR-001'])
        self.assertFalse(r['usa_memoria'])

    def test_mismo_contrato_cambia_tema(self):
        r = mem.resolver_seguimiento('¿Y cuáles son sus vacaciones?', [t()])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-001'])
        self.assertEqual(r['tema'], 'vacaciones')
        self.assertIn('vacaciones', r['pregunta_resuelta'])

    def test_sin_historial_pide_aclaracion(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [])
        self.assertTrue(r['motivo'])
        self.assertEqual(r['fuentes_resueltas'], [])

    def test_otro_contrato_salta_a_doc002(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [t()])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-002'])
        self.assertIn('fecha', r['pregunta_resuelta'])
        self.assertTrue(r['usa_memoria'])

    def test_otro_contrato_salta_a_doc001(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [t(foco='DOC-002')])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-001'])

    def test_referencia_ambigua_no_inventa(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [t(tema=None, foco=None)])
        self.assertTrue(r['motivo'])

    def test_comparar_fechas_y_ambos_documentos(self):
        r = mem.resolver_seguimiento('Compara ambas fechas de inicio', [t()])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-001', 'DOC-002'])
        self.assertEqual(r['pregunta_resuelta'], 'Compara las fechas de inicio de los dos contratos')

    def test_comparar_vacaciones(self):
        r = mem.resolver_seguimiento('Compara ambos contratos', [t(tema='vacaciones')])
        self.assertIn('vacaciones', r['pregunta_resuelta'])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-001', 'DOC-002'])

    def test_explicitas_tienen_prioridad(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [t()], ['NOR-002'])
        self.assertEqual(r['fuentes_resueltas'], ['NOR-002'])
        self.assertFalse(r['usa_memoria'])

    def test_fuentes_invalidas_rechazadas(self):
        with self.assertRaises(ValueError):
            mem.resolver_seguimiento('Hola', [], ['DOC-999'])

    def test_fuentes_duplicadas_rechazadas(self):
        with self.assertRaises(ValueError):
            mem.resolver_seguimiento('Hola', [], ['DOC-001', 'DOC-001'])

    def test_limitador_historial(self):
        prev = [{'i': i} for i in range(8)]
        r = mem.combinar_historial(prev, [{'i': 8}])
        self.assertEqual([x['i'] for x in r], list(range(1,9)))

    def test_longitud_rechazada(self):
        with self.assertRaises(ValueError):
            mem.resolver_seguimiento('Z'*801, [])

    def test_saludo_no_tapa_contexto_anterior(self):
        r = mem.resolver_seguimiento('¿Y qué indica el otro contrato?', [t(), t(ruta='directa', foco=None)])
        self.assertEqual(r['fuentes_resueltas'], ['DOC-002'])


LANGGRAPH = importlib.util.find_spec('langgraph') is not None


@unittest.skipUnless(LANGGRAPH, 'Solo integración real con langgraph instalado')
class PruebasGrafo(unittest.TestCase):
    def setUp(self):
        self.llamadas = []
        def executor(pregunta, fuentes, top_k, modelo):
            self.llamadas.append((pregunta, tuple(fuentes)))
            ruta, docs = mem.e6.ruta_inicial(pregunta, fuentes)
            return {'respuesta': 'FAKE LOCAL: ' + pregunta, 'ruta': ruta,
                    'recorrido': ['clasificar','simular'], 'citas': [], 'evidencias': []}
        self.g = mem.crear_grafo(ejecutor=executor)

    def test_turnos_misma_sesion(self):
        a = mem.preguntar(self.g, sesion='A', texto='¿Cuál es la fecha de inicio del contrato de alta dirección?')
        b = mem.preguntar(self.g, sesion='A', texto='¿Y qué indica el otro contrato?')
        self.assertEqual(len(a['historial']), 1)
        self.assertEqual(len(b['historial']), 2)
        self.assertEqual(b['fuentes_resueltas'], ['DOC-002'])
        self.assertTrue(b['usa_memoria'])

    def test_misma_sesion_compara_tercer_turno(self):
        mem.preguntar(self.g, sesion='A', texto='¿Cuál es la fecha de inicio del contrato de alta dirección?')
        mem.preguntar(self.g, sesion='A', texto='¿Y qué indica el otro contrato?')
        c = mem.preguntar(self.g, sesion='A', texto='Compara ambas fechas de inicio')
        self.assertEqual(c['ruta'], 'comparacion')
        self.assertEqual(c['fuentes_resueltas'], ['DOC-001', 'DOC-002'])
        self.assertEqual(len(c['historial']), 3)

    def test_diferentes_sesiones_sin_fugas(self):
        mem.preguntar(self.g, sesion='A', texto='¿Cuál es la fecha de inicio del contrato de alta dirección?')
        otro = mem.preguntar(self.g, sesion='B', texto='¿Y qué indica el otro contrato?')
        self.assertEqual(otro['ruta'], 'aclaracion')
        self.assertEqual(len(otro['historial']), 1)
        self.assertEqual(len(self.llamadas), 1)  # no llamar a RAG para pregunta ambigua

    def test_get_state_recupera_historial(self):
        mem.preguntar(self.g, sesion='C', texto='Hola')
        snapshot = self.g.get_state({'configurable': {'thread_id': 'C'}})
        self.assertEqual(len(snapshot.values['historial']), 1)

    def test_fuente_explicita_no_fuga_al_siguiente_turno(self):
        mem.preguntar(self.g, sesion='X', texto='Dime la duración', fuentes=['DOC-001'])
        b = mem.preguntar(self.g, sesion='X', texto='¿Qué establece el Estatuto de los Trabajadores sobre vacaciones?')
        self.assertEqual(b['fuentes_resueltas'], ['NOR-001'])

    def test_simulacion_no_genera_citas_imaginarias(self):
        a = mem.preguntar(self.g, sesion='X', texto='Hola')
        self.assertEqual(a['referencias'], [])
        self.assertEqual(a['ruta'], 'directa')


if __name__ == '__main__':
    unittest.main()
