"""Pruebas unitarias de chunking sin llamadas a servicios externos."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preparar_chunks import detectar_encabezado, dividir_documento, verificar_manifiesto, ventanas


class PruebasChunking(unittest.TestCase):
    def setUp(self):
        self.doc = {'id': 'NOR-TEST', 'tipo': 'normativa_consolidada_informativa',
                    'fuente': 'Prueba sintetica', 'ruta': 'prueba.pdf',
                    'url_original': None, 'version_consultada': None, 'sha256': 'prueba'}

    def test_articulo_y_disposicion_se_identifican(self):
        self.assertTrue(detectar_encabezado('Artículo 2. Derechos', self.doc['tipo']))
        self.assertTrue(detectar_encabezado('Disposición adicional primera. Ámbito', self.doc['tipo']))

    def test_clausula_contrato_se_identifica(self):
        self.assertTrue(detectar_encabezado('TERCERA. Duración', 'contrato_ficticio_indefinido'))

    def test_ventanas_respetan_longitud_y_solape(self):
        texto = ' '.join(['obligaciones profesionales, requisitos y responsabilidades'] * 130)
        trozos = list(ventanas(texto, max_caracteres=750, solape=100))
        self.assertGreater(len(trozos), 2)
        self.assertTrue(all(0 < len(t[2]) <= 750 for t in trozos))
        self.assertTrue(all(trozos[i+1][0] < trozos[i][1] for i in range(len(trozos)-1)))

    def test_continuacion_entre_paginas_y_trazabilidad(self):
        pags = ['Artículo 7. Supuesto de prueba\n' + 'Texto explicativo ' * 24,
                'continuación del artículo ' * 28 + '\nArtículo 8. Supuesto siguiente\nOtro texto.']
        filas = dividir_documento(self.doc, pags, max_caracteres=550, solape=60)
        self.assertTrue(any('Artículo 7.' in f['metadata']['seccion'] and
                            f['metadata']['pagina_fin'] == 2 for f in filas))
        self.assertTrue(any('Artículo 8.' in f['metadata']['seccion'] for f in filas))

    def test_ids_deterministas(self):
        pags = ['Artículo 1. Generalidades\n' + 'Texto extenso. ' * 160]
        a = dividir_documento(self.doc, pags)
        b = dividir_documento(self.doc, pags)
        self.assertEqual(a, b)
        self.assertEqual(len({x['id'] for x in a}), len(a))

    def test_rechazar_manifiesto_incompleto(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'manifest.json'
            p.write_text(json.dumps({'corpus_completo': False, 'documentos_validados': 2,
                                     'documentos': []}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Etapa 1'):
                verificar_manifiesto(p)


if __name__ == '__main__':
    unittest.main()
