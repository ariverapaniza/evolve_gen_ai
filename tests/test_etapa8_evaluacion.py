"""Pruebas aisladas del evaluador de calidad; sin API ni ChromaDB."""
from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import importlib.util
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import etapa8_evaluar_agente as ev
import etapa7_memoria as mem


def caso(identificador='D06'):
    return next(c for c in ev.cargar_casos() if c['id'] == identificador)


def r(ref, documento, seccion):
    return {'ref': ref, 'id_chunk': f'{documento}-00001', 'documento_id': documento,
            'seccion': seccion, 'pagina_inicio': 2, 'pagina_fin': 2}


class PruebasBancoYCalidad(unittest.TestCase):
    def test_once_casos(self):
        self.assertEqual(len(ev.cargar_casos()), 11)

    def test_sesiones_misma_memoria(self):
        self.assertEqual([c['id'] for c in ev.cargar_casos() if c['sesion']=='eval_memoria'],
                         ['D06','D07','D08'])

    def test_sesion_aislamiento_independiente(self):
        self.assertNotEqual(caso('D11')['sesion'], caso('D08')['sesion'])

    def test_corpus_de_evaluacion_no_contiene_claves(self):
        texto = ev.BANCO.read_text(encoding='utf8')
        self.assertNotIn('AIza', texto)
        self.assertNotIn('GEMINI_API_KEY=', texto)

    def test_enlace_cita_original_con_documento(self):
        c=caso('D06')
        res={'ruta':'documental','usa_memoria':False,
             'respuesta':'Comienza el 1 de abril de 2026 [F3].',
             'referencias':[r('F3','DOC-001','TERCERA. Duración')]}
        out=ev.evaluar_estructura(c,res)
        self.assertEqual(out['estado_estructural'],'OK')

    def test_rechazar_cita_inventada(self):
        c=caso('D06')
        res={'ruta':'documental','usa_memoria':False,
             'respuesta':'Comienza el 1 de abril de 2026 [F4].',
             'referencias':[r('F3','DOC-001','TERCERA. Duración')]}
        self.assertEqual(ev.evaluar_estructura(c,res)['estado_estructural'],'REVISAR')

    def test_rechazar_seccion_imprecisa(self):
        c=caso('D06')
        res={'ruta':'documental','usa_memoria':False,
             'respuesta':'Comienza el 1 de abril de 2026 [F3].',
             'referencias':[r('F3','DOC-001','DUODÉCIMA. Jurisdicción')]}
        self.assertEqual(ev.evaluar_estructura(c,res)['estado_estructural'],'REVISAR')

    def test_rechazar_comparacion_incompleta(self):
        c=caso('D08')
        res={'ruta':'comparacion','usa_memoria':True,
             'respuesta':'1 de abril de 2026 y 15 de abril de 2026 [F2].',
             'referencias':[r('F2','DOC-001','TERCERA. Duración')]}
        self.assertEqual(ev.evaluar_estructura(c,res)['estado_estructural'],'REVISAR')

    def test_comparacion_ambos_documentos(self):
        c=caso('D08')
        res={'ruta':'comparacion','usa_memoria':True,
             'respuesta':'DOC-001: 1 de abril de 2026 [F2]. DOC-002: 15 de abril de 2026 [F6].',
             'referencias':[r('F2','DOC-001','TERCERA. Duración'),r('F6','DOC-002','TERCERA. Duración')]}
        self.assertEqual(ev.evaluar_estructura(c,res)['estado_estructural'],'OK')

    def test_detectar_memoria_incorrecta(self):
        c=caso('D07')
        res={'ruta':'documental','usa_memoria':False,
             'respuesta':'15 de abril de 2026 [F1].',
             'referencias':[r('F1','DOC-002','TERCERA. Duración')]}
        self.assertEqual(ev.evaluar_estructura(c,res)['estado_estructural'],'REVISAR')

    def test_ruta_directa_sin_citas(self):
        c=caso('D01')
        self.assertEqual(ev.evaluar_estructura(c,{'ruta':'directa','usa_memoria':False,
                         'respuesta':'Hola, puedo ayudarte.', 'referencias':[]})['estado_estructural'],'OK')

    def test_abstencion_no_es_aceptada_automaticamente(self):
        c=caso('D10')
        out=ev.evaluar_estructura(c,{'ruta':'documental','usa_memoria':False,
                            'respuesta':'No encuentro evidencia suficiente.', 'referencias':[]})
        self.assertTrue(out['revision_manual_obligatoria'])

    def test_impresion_usa_etiquetas_originales(self):
        buf=StringIO()
        with redirect_stdout(buf):
            mem.imprimir({'ruta':'documental','pregunta_resuelta':'Demo', 'fuentes_resueltas':[],
                        'usa_memoria':False,'recorrido':['clasificar'], 'respuesta':'Hola [F6].',
                        'referencias':[r('F6','DOC-002','TERCERA. Duración')], 'historial':[]})
        self.assertIn('[F6] DOC-002', buf.getvalue())
        self.assertNotIn('CITA 1:', buf.getvalue())

    def test_historial_original_conserva_modelo(self):
        self.assertIn('eval_memoria', [c['sesion'] for c in ev.cargar_casos()])


if __name__=='__main__':
    unittest.main()
