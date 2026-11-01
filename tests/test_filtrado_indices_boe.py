"""Pruebas adicionales contra indices confundidos con articulado."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preparar_chunks import excluir_indice_boe, dividir_documento, detectar_encabezado


class PruebasIndicesBOE(unittest.TestCase):
    def setUp(self):
        self.norma = {
            'id': 'NOR-TEST', 'tipo': 'normativa_consolidada_informativa',
            'fuente': 'BOE', 'ruta': 'prueba.pdf', 'url_original': None,
            'version_consultada': 'prueba', 'sha256': 'prueba',
        }
        self.pags = [
            'Referencia: BOE-A-2015-11430\nÍNDICE\nArtículo 15. Duración del contrato. . . . . . 20\n',
            'CAPÍTULO II. Contenido del contrato ........................ 20\n',
            'TEXTO CONSOLIDADO\nÚltima modificación: 21 de octubre de 2026\n'
            'El texto consolidado comienza aquí.\nArtículo 1. Ámbito de aplicación.\n'
            '1. La ley se aplica a los trabajadores.\n',
            'Artículo 15. Duración del contrato de trabajo.\n'
            '1. El contrato de trabajo se presume concertado por tiempo indefinido.\n',
        ]

    def test_excluir_indice_sin_reenumerar_paginas(self):
        limpio, info = excluir_indice_boe(self.pags, self.norma['tipo'])
        self.assertEqual(info['pagina_inicio_contenido'], 3)
        self.assertEqual(info['paginas_indice_excluidas'], [1, 2])
        self.assertEqual(limpio[:2], ['', ''])
        self.assertTrue(limpio[2].startswith('TEXTO CONSOLIDADO'))

    def test_no_cita_indice_como_articulo_real(self):
        limpio, _ = excluir_indice_boe(self.pags, self.norma['tipo'])
        chunks = dividir_documento(self.norma, limpio)
        arts_15 = [x for x in chunks if x['metadata']['seccion'].startswith('Artículo 15.')]
        self.assertEqual(len(arts_15), 1)
        self.assertEqual(arts_15[0]['metadata']['pagina_inicio'], 4)
        self.assertIn('presume concertado', arts_15[0]['texto'])
        self.assertFalse(any(x['metadata']['pagina_inicio'] <= 2 for x in chunks))

    def test_contratos_intactos(self):
        original = ['PRIMERA. Cargo y funciones\nContenido.\n']
        limpio, info = excluir_indice_boe(original, 'contrato_ficticio_alta_direccion')
        self.assertEqual(limpio, original)
        self.assertFalse(info['paginas_indice_excluidas'])

    def test_referencia_minuscula_no_es_encabezado(self):
        self.assertIsNone(detectar_encabezado('artículo 48.4, en el segundo párrafo', self.norma['tipo']))
        self.assertIsNotNone(detectar_encabezado('Artículo 15. Duración del contrato.', self.norma['tipo']))

    def test_falla_cerrado_si_no_existe_marcador(self):
        with self.assertRaisesRegex(ValueError, 'TEXTO CONSOLIDADO'):
            excluir_indice_boe(['ÍNDICE\nArtículo 1. Foo ... 3',
                                'Artículo 1. Foo\nTexto normativo'], self.norma['tipo'])

    def test_falla_cerrado_si_no_aparece_articulado(self):
        with self.assertRaisesRegex(ValueError, 'articulado normativo'):
            excluir_indice_boe(['ÍNDICE\nArtículo 1. Foo ... 3',
                                'TEXTO CONSOLIDADO\nSólo texto introductorio'], self.norma['tipo'])

if __name__ == '__main__':
    unittest.main()
