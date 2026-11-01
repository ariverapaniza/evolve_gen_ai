"""Pruebas sin acceso a Google ni ChromaDB real."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnostico_multifuente as d

class FakeCollection:
    def __init__(self):
        self.kwargs = None
    def query(self, **kwargs):
        self.kwargs = kwargs
        return {'ids':[['A']], 'metadatas':[[{'documento_id':'DOC-001','seccion':'TERCERA. Duración'}]], 'distances':[[0.25]]}

class TestMultifuente(unittest.TestCase):
    def setUp(self):
        self.m01 = {'id':'M01','tipo':'multiple','esperados':[
          {'documento_id':'DOC-001','seccion_inicio':'TERCERA.'},
          {'documento_id':'DOC-002','seccion_inicio':'TERCERA.'}]}
    def test_dos_rutas_para_m01(self):
        self.assertEqual([r['documento_id'] for r in d.CONSULTAS['M01']], ['DOC-001','DOC-002'])
    def test_no_fuga_de_etiquetas_en_m01(self):
        for r in d.CONSULTAS['M01']:
            self.assertNotIn('TERCERA', r['consulta'])
    def test_filtro_previo_chroma(self):
        c = FakeCollection()
        rows = d.buscar(c,[0.1,0.2],'DOC-001')
        self.assertEqual(c.kwargs['where'], {'documento_id':'DOC-001'})
        self.assertEqual(c.kwargs['n_results'],5)
        self.assertEqual(rows[0]['id'],'A')
    def test_ambos_documentos_cubiertos(self):
        filas = {'DOC-001':[{'metadata':{'documento_id':'DOC-001','seccion':'TERCERA. Duración'}}],
                 'DOC-002':[{'metadata':{'documento_id':'DOC-002','seccion':'TERCERA. Duración'}}]}
        self.assertTrue(d.metrica(self.m01,filas)['completo'])
    def test_un_documento_no_es_suficiente(self):
        filas = {'DOC-001':[], 'DOC-002':[{'metadata':{'documento_id':'DOC-002','seccion':'TERCERA. Duración'}}]}
        m = d.metrica(self.m01,filas)
        self.assertFalse(m['completo'])
        self.assertEqual(m['fuentes_cubiertas'],1)
    def test_seccion_distinta_no_cuenta(self):
        filas = {'DOC-001':[{'metadata':{'documento_id':'DOC-001','seccion':'PRIMERA. Cargo y funciones'}}],
                 'DOC-002':[{'metadata':{'documento_id':'DOC-002','seccion':'TERCERA. Duración'}}]}
        self.assertFalse(d.metrica(self.m01,filas)['completo'])
    def test_mismo_documento_no_cubre_dos(self):
        filas = {'DOC-001':[{'metadata':{'documento_id':'DOC-001','seccion':'TERCERA. Duración'}}],
                 'DOC-002':[{'metadata':{'documento_id':'DOC-001','seccion':'TERCERA. Duración'}}]}
        self.assertFalse(d.metrica(self.m01,filas)['completo'])
    def test_fuentes_de_m02(self):
        self.assertEqual([r['documento_id'] for r in d.CONSULTAS['M02']], ['DOC-002','NOR-001'])

if __name__ == '__main__': unittest.main()
