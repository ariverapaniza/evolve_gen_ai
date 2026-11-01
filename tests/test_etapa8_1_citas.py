"""Pruebas de regresión de referencias agrupadas y reevaluación estructural."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import citas_agrupadas as citas
import etapa5_rag as rag
import etapa8_evaluar_agente as ev


def documento(d):
    return {'metadata': {'documento_id': d}}


class PruebasCitasAgrupadas(unittest.TestCase):
    def test_cita_simple(self):
        self.assertEqual(citas.extraer_numeros('Respuesta [F2].'), [2])

    def test_grupo_con_espacios(self):
        self.assertEqual(citas.extraer_numeros('Una frase [F1, F4].'), [1, 4])

    def test_grupo_sin_espacios(self):
        self.assertEqual(citas.extraer_numeros('Frase [F3,F6,F10].'), [3, 6, 10])

    def test_varios_formatos_y_repeticiones(self):
        self.assertEqual(citas.extraer_numeros('Una [F1] y otra [F1; F3] y [F4 y F5]'), [1, 1, 3, 4, 5])

    def test_rechaza_referencia_agrupada_malformada(self):
        with self.assertRaises(ValueError):
            citas.extraer_numeros('Mala [F1, texto libre]')

    def test_modelo_conserva_todas_las_citas_agrupadas(self):
        self.assertEqual(rag.validar_citas('Comparación [F1, F2].', 2,
                         ['DOC-001','DOC-002'], [documento('DOC-001'),documento('DOC-002')]), [1,2])

    def test_modelo_rechaza_cita_fuera_de_contexto_agrupada(self):
        with self.assertRaises(ValueError):
            rag.validar_citas('Texto [F1, F99].', 2,
                              ['DOC-001','DOC-002'], [documento('DOC-001'), documento('DOC-002')])

    def test_evaluador_detecta_citas_agrupadas_no_impresas(self):
        caso = next(c for c in ev.cargar_casos() if c['id'] == 'D05')
        resultado = {'ruta':'documental', 'usa_memoria':False,
                     'respuesta':'El trabajador disfruta de 22 días [F2]; contrato [F1, F4].',
                     'referencias':[{'ref':'F2','documento_id':'DOC-002','seccion':'SEXTA. Vacaciones'}]}
        self.assertEqual(ev.evaluar_estructura(caso, resultado)['estado_estructural'], 'REVISAR')

    def test_evaluador_acepta_citas_agrupadas_bien_impresas(self):
        caso = next(c for c in ev.cargar_casos() if c['id'] == 'D05')
        resultado = {'ruta':'documental', 'usa_memoria':False,
                     'respuesta':'El trabajador tiene 22 días [F2]; contrato [F1, F4].',
                     'referencias':[
                       {'ref':'F1','documento_id':'DOC-002','seccion':'TERCERA. Duración'},
                       {'ref':'F2','documento_id':'DOC-002','seccion':'SEXTA. Vacaciones'},
                       {'ref':'F4','documento_id':'DOC-002','seccion':'PRIMERA. Puesto de trabajo'},
                     ]}
        self.assertEqual(ev.evaluar_estructura(caso, resultado)['estado_estructural'], 'OK')


if __name__ == '__main__':
    unittest.main()
