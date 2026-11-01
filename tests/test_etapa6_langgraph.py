"""15 comprobaciones para la Etapa 6. Las tres pruebas de grafo real
se ejecutan cuando langgraph está instalado (en el PC del alumno)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import etapa6_langgraph as e6


class PruebasEnrutamiento(unittest.TestCase):
    def test_saludo_va_directo(self):
        self.assertEqual(e6.ruta_inicial('Hola'), ('directa', []))

    def test_ayuda_va_directo(self):
        self.assertEqual(e6.ruta_inicial('¿Qué puedes hacer?'), ('directa', []))

    def test_gracias_va_directo(self):
        self.assertEqual(e6.ruta_inicial('Muchas gracias'), ('directa', []))

    def test_pregunta_juridica_no_salta_rag(self):
        self.assertEqual(e6.ruta_inicial('¿Qué dispone el artículo 15?')[0], 'documental')

    def test_hola_mas_consulta_no_salta_rag(self):
        self.assertEqual(e6.ruta_inicial('Hola, dime qué indica el contrato.')[0], 'documental')

    def test_comparacion_fecha_ambos_va_multifuente(self):
        ruta, fuentes = e6.ruta_inicial('Compara las fechas de inicio de los dos contratos')
        self.assertEqual(ruta, 'comparacion')
        self.assertEqual(fuentes, ['DOC-001', 'DOC-002'])

    def test_comparacion_vacaciones_estatuto_multifuente(self):
        ruta, fuentes = e6.ruta_inicial('Compara las vacaciones del contrato con el Estatuto')
        self.assertEqual(ruta, 'comparacion')
        self.assertEqual(fuentes, ['DOC-002', 'NOR-001'])

    def test_documento_unico_filtrado(self):
        ruta, fuentes = e6.ruta_inicial('Qué duración tiene?', ['DOC-001'])
        self.assertEqual((ruta, fuentes), ('documental', ['DOC-001']))

    def test_pregunta_general_con_filtro_siempre_consulta(self):
        ruta, fuentes = e6.ruta_inicial('Hola', ['DOC-001'])
        self.assertEqual((ruta, fuentes), ('documental', ['DOC-001']))

    def test_fuente_invalida_no_se_acepta(self):
        with self.assertRaises(ValueError):
            e6.ruta_inicial('Revisa el contrato', ['DOC-999'])

    def test_pregunta_vacia_rechazada(self):
        with self.assertRaises(ValueError):
            e6.ruta_inicial('   ')

    def test_pregunta_muy_larga_rechazada(self):
        with self.assertRaises(ValueError):
            e6.ruta_inicial('A' * 801)


LANGGRAPH = importlib.util.find_spec('langgraph') is not None


@unittest.skipUnless(LANGGRAPH, 'Instalar langgraph antes de ejecutar pruebas de grafo real')
class PruebasGrafoConMocks(unittest.TestCase):
    def setUp(self):
        self.calls = {'recuperar': 0, 'generar': 0, 'comprobar_cobertura': 0}
        self.resultados = [
            {'id': 'DOC-001-00005', 'metadata': {'documento_id': 'DOC-001', 'seccion': 'TERCERA. Duración', 'pagina_inicio': 2, 'pagina_fin': 2}, 'texto': 'vigencia inicio 1 de abril de 2026'},
            {'id': 'DOC-002-00005', 'metadata': {'documento_id': 'DOC-002', 'seccion': 'TERCERA. Duración', 'pagina_inicio': 2, 'pagina_fin': 2}, 'texto': 'vigencia inicio 15 de abril de 2026'},
        ]
        def recuperar(col, cliente, pregunta, fuentes, top_k):
            self.calls['recuperar'] += 1
            return self.resultados
        def verificar(pregunta, fuentes, evidencias):
            self.calls['comprobar_cobertura'] += 1
        def generar(cliente, pregunta, evidencias, fuentes, modelo):
            self.calls['generar'] += 1
            return ('Contrato DOC-001 comienza 1 de abril de 2026 [F1]; DOC-002 el 15 de abril [F2].', [1, 2])
        self.fake = SimpleNamespace(
            DEFAULT_MODEL='falso', recuperar=recuperar, comprobar_cobertura=verificar,
            generar_respuesta=generar)
        self.graph = e6.crear_grafo(modulo_rag=self.fake, coleccion=object(), cliente=object())

    def test_grafo_saludo_evita_retriever_y_llm(self):
        out = self.graph.invoke({'pregunta': 'Hola'})
        self.assertEqual(out['recorrido'], ['clasificar', 'respuesta_directa'])
        self.assertEqual(self.calls['recuperar'], 0)
        self.assertEqual(self.calls['generar'], 0)

    def test_grafo_documental_recupera_y_genera(self):
        out = self.graph.invoke({'pregunta': 'Qué duración tiene el contrato?'})
        self.assertEqual(out['recorrido'], ['clasificar', 'recuperar', 'validar', 'generar'])
        self.assertEqual(self.calls['recuperar'], 1)
        self.assertEqual(self.calls['generar'], 1)

    def test_grafo_sin_evidencias_abstiene(self):
        self.resultados = []
        out = self.graph.invoke({'pregunta': '¿Qué indica el contrato?'})
        self.assertEqual(out['recorrido'], ['clasificar', 'recuperar', 'validar', 'abstener'])
        self.assertEqual(self.calls['generar'], 0)


if __name__ == '__main__':
    unittest.main()
