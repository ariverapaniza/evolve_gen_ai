"""Pruebas locales sin llamadas API de Gemini, independientes de los PDF originales."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import evaluar_recuperacion as e


class PruebasEvaluacionRecuperacion(unittest.TestCase):
    def test_12_casos_y_modalidades(self):
        casos = e.leer_casos()
        self.assertEqual(len(casos), 12)
        self.assertEqual(sum(c['tipo']=='simple' for c in casos), 10)
        self.assertEqual(sum(c['tipo']=='multiple' for c in casos), 2)

    def test_casos_tienen_id_distinto(self):
        casos = e.leer_casos()
        self.assertEqual(len({c['id'] for c in casos}), len(casos))

    def test_coincidencia_exige_documento_y_seccion(self):
        meta = {'documento_id':'DOC-001','seccion':'PRIMERA. Cargo y funciones'}
        self.assertTrue(e.coincide(meta,{'documento_id':'DOC-001','seccion_inicio':'PRIMERA.'}))
        self.assertFalse(e.coincide(meta,{'documento_id':'DOC-002','seccion_inicio':'PRIMERA.'}))
        self.assertFalse(e.coincide(meta,{'documento_id':'DOC-001','seccion_inicio':'SEGUNDA.'}))

    def test_ranking_simple_acierto_primero(self):
        c = e.leer_casos()[0]
        v = [{'metadata':{'documento_id':'DOC-001','seccion':'PRIMERA. Cargo y funciones'}}]
        self.assertEqual(e.evaluar_resultados(c,v)['rr5'],1)
        self.assertTrue(e.evaluar_resultados(c,v)['hit1'])

    def test_ranking_simple_acierto_tercero(self):
        c = e.leer_casos()[0]
        malo = {'metadata':{'documento_id':'NOR-001','seccion':'Artículo 1.'}}
        bueno = {'metadata':{'documento_id':'DOC-001','seccion':'PRIMERA. Cargo y funciones'}}
        x = e.evaluar_resultados(c,[malo,malo,bueno])
        self.assertTrue(x['hit5'])
        self.assertFalse(x['hit1'])
        self.assertEqual(x['posiciones'],[3])
        self.assertAlmostEqual(x['rr5'],1/3,places=5)

    def test_sin_acierto(self):
        c=e.leer_casos()[0]
        x=e.evaluar_resultados(c,[{'metadata':{'documento_id':'NOR-001','seccion':'Artículo 15.'}}])
        self.assertEqual(x['rr5'],0)
        self.assertFalse(x['completo'])

    def test_multifuente_exige_ambos(self):
        c=e.leer_casos()[-2]
        r=[{'metadata':{'documento_id':'DOC-001','seccion':'TERCERA. Duración'}}]
        self.assertFalse(e.evaluar_resultados(c,r)['completo'])
        r.append({'metadata':{'documento_id':'DOC-002','seccion':'TERCERA. Duración'}})
        self.assertTrue(e.evaluar_resultados(c,r)['completo'])

    def test_etiquetas_ground_truth_en_corpus_de_banco(self):
        casos = e.leer_casos()
        corpus = [{'metadata':x} for c in casos for x in (
            {'documento_id':z['documento_id'],'seccion':z['seccion_inicio']+' Ejemplo'} for z in c['esperados'])]
        e.validar_etiquetas_en_corpus(casos,corpus)
        with self.assertRaises(ValueError):
            e.validar_etiquetas_en_corpus(casos,corpus[:1])

    def test_informe_reanuda_sin_reiniciar(self):
        casos=e.leer_casos()
        mod=SimpleNamespace(MODEL='gemini-embedding-2',DIM=768,EXPECTED_SHA256='X',COLLECTIONS={'completa':'coleccion'})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json'
            r=e.cargar_reporte(casos,mod,ruta=p)
            r['resultados']['S01']={'metricas': {'hit1': True,'hit5':True,'rr5':1},'tipo':'simple'}
            e.guardar_reporte(r,p)
            r2=e.cargar_reporte(casos,mod,ruta=p)
            self.assertIn('S01',r2['resultados'])
            with self.assertRaises(ValueError):
                e.cargar_reporte(casos,SimpleNamespace(MODEL='otro',DIM=768,EXPECTED_SHA256='X',COLLECTIONS={'completa':'coleccion'}),ruta=p)

    def test_metricas_en_denominador_evaluado(self):
        casos=e.leer_casos()
        r={'resultados': {'S01': {'metricas': {'hit1': True,'hit5':True,'rr5':1}},
                          'S02': {'metricas': {'hit1': False,'hit5':False,'rr5':0}},
                          'M01': {'metricas': {'completo':True}}}}
        met=e.resumen(r,casos)
        self.assertEqual(met['hit_at_5'],.5)
        self.assertEqual(met['hit_at_1'],.5)
        self.assertEqual(met['mrr_at_5'],.5)
        self.assertEqual(met['multifuente_completos'],1)


if __name__ == '__main__': unittest.main()
