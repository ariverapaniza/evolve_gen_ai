"""Casos que reproducen el fallo de coincidencias narrativas de v2."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preparar_chunks import excluir_indice_boe

TIPO = 'normativa_consolidada_informativa'


def encabezado():
    return ('BOLETÍN OFICIAL DEL ESTADO\nLEGISLACIÓN CONSOLIDADA\n'
            'TEXTO CONSOLIDADO\nÚltima modificación: 21 de octubre de 2026\n'
            'El decreto se ha elaborado para fines jurídicos.\n'
            'Artículo 1. Ámbito de aplicación.\n'
            '1. El artículo permite relacionar obligaciones.\n')


class PruebasCabeceraBOEV3(unittest.TestCase):
    def test_texto_consolidado_narrativo_no_crea_inicio_adicional(self):
        paginas = [
            'Referencia: BOE-A-2015-11430\nÍNDICE\nArtículo 1. Ambito ..................... 9\n',
            encabezado(),
            'En este párrafo se describe el texto consolidado que se publicará.\n'
            'Artículo 2. Relaciones laborales.\nContiene normativa.\n',
        ]
        salida, info = excluir_indice_boe(paginas, TIPO)
        self.assertEqual(info['paginas_indice_excluidas'], [1])
        self.assertEqual(info['pagina_inicio_contenido'], 2)
        self.assertEqual(salida[0], '')
        self.assertTrue(salida[1].startswith('TEXTO CONSOLIDADO'))
        self.assertIn('texto consolidado', salida[2])

    def test_encabezado_suelto_posterior_sin_fecha_no_cuenta(self):
        paginas = ['ÍNDICE\nArtículo 1. ................ 3\n', encabezado(),
                   'TEXTO CONSOLIDADO\nSe trata de una cita dentro de un anexo.\n']
        _, info = excluir_indice_boe(paginas, TIPO)
        self.assertEqual(info['pagina_inicio_contenido'], 2)

    def test_simula_estatuto_pagina_ocho(self):
        preliminar = ['Referencia: BOE-A-2015-11430\nÍNDICE\nArtículo 15. ... 20\n']
        preliminar.extend('Artículo 15. Duración ....................... 20\n' for _ in range(6))
        preliminar.extend([encabezado(), 'Artículo 2. Relaciones laborales.\nTexto\n'])
        _, info = excluir_indice_boe(preliminar, TIPO)
        self.assertEqual(info['pagina_inicio_contenido'], 8)
        self.assertEqual(info['paginas_indice_excluidas'], list(range(1,8)))

    def test_con_dos_encabezados_y_fechas_falla_cerrado(self):
        paginas = ['ÍNDICE\nArtículo 1. ... 3\n', encabezado(), encabezado()]
        with self.assertRaisesRegex(ValueError, 'único encabezado'):
            excluir_indice_boe(paginas, TIPO)

    def test_sin_cabecera_falla_cerrado(self):
        paginas = ['ÍNDICE\nArtículo 1. ... 3\n', 'Artículo 1. Texto real.\nContenido completo.\n']
        with self.assertRaisesRegex(ValueError, 'TEXTO CONSOLIDADO'):
            excluir_indice_boe(paginas, TIPO)

    def test_una_sola_cabecera_sin_fecha_se_acepta_si_hay_articulado(self):
        paginas = ['ÍNDICE\nArtículo 1. ..... 3\n',
                   'TEXTO CONSOLIDADO\nArtículo 1. Ambito.\nContenido normativo.\n']
        salida, info = excluir_indice_boe(paginas, TIPO)
        self.assertEqual(info['pagina_inicio_contenido'], 2)
        self.assertEqual(salida[0], '')

    def test_contrato_no_se_altera(self):
        original = ['PRIMERA. Cargo y funciones\nTexto contractual.\n']
        salida, _ = excluir_indice_boe(original, 'contrato_ficticio_alta_direccion')
        self.assertEqual(original, salida)


if __name__ == '__main__':
    unittest.main()
