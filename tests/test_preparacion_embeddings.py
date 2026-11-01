import unittest
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preparar_para_embeddings import limpiar_marcas, PATRON_MARCA, transformar, hashtexto

class PruebasLimpiezaEmbeddings(unittest.TestCase):
    def test_elimina_cabeceras_y_pies_sin_eliminar_texto_legal(self):
        antes = 'Texto legal\n  Página 8\n  BOLETÍN OFICIAL DEL ESTADO\n  LEGISLACIÓN CONSOLIDADA\nOtro texto legal'
        despues, n = limpiar_marcas(antes)
        self.assertEqual(n, 3)
        self.assertIn('Texto legal', despues)
        self.assertIn('Otro texto legal', despues)
        self.assertNotIn('BOLETÍN', despues)
    def test_no_elimina_mencion_narrativa_al_boe(self):
        s = 'Se publicó en el BOLETÍN OFICIAL DEL ESTADO en tal fecha.'
        self.assertEqual(limpiar_marcas(s), (s, 0))
    def test_no_elimina_articulos_que_contienen_titulo(self):
        s = 'El Título II del Estatuto de los Trabajadores contiene disposiciones.'
        self.assertEqual(limpiar_marcas(s), (s, 0))
    def test_detecta_linea_de_pagina(self):
        self.assertIsNotNone(PATRON_MARCA.fullmatch('    Página 12'))
    def test_no_elimina_referencias_a_paginas_en_el_texto(self):
        s = 'Véase la Página 12 del anexo.'
        self.assertEqual(limpiar_marcas(s), (s, 0))
    def test_integracion_corpus_443_y_derivado_437(self):
        fichero = ROOT / 'corpus/chunks/chunks.jsonl'
        if not fichero.exists():
            self.skipTest('Se necesita el corpus local chunks.jsonl')
        originales = [json.loads(s) for s in fichero.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        derivados, r = transformar(originales)
        self.assertEqual(len(originales), 443)
        self.assertEqual(len(derivados), 437)
        self.assertEqual(Counter(c['metadata']['documento_id'] for c in derivados),
            Counter({'DOC-001':14,'DOC-002':13,'NOR-001':387,'NOR-002':23}))
        self.assertEqual(len([c for c in derivados if len(c['texto'])<100]), 2)
        self.assertEqual(r['lineas_boilerplate_retiradas'], 315)
        self.assertTrue(all(hashtexto(c['texto'])==c['metadata']['sha256_texto'] for c in derivados))
        articulo_16 = next(c for c in derivados if c['id']=='NOR-002-00021')
        self.assertIn('Título II del Estatuto de los Trabajadores.', articulo_16['texto'])

if __name__ == '__main__':
    unittest.main()
