"""Pruebas offline: no requieren API de Gemini ni servicio de Chroma."""
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'etapa3_vectores.py'
spec = importlib.util.spec_from_file_location('evolve_etapa3_vectores', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class PruebasEtapa3(unittest.TestCase):
    def test_modelo_dimension_y_prefijos(self):
        self.assertEqual(mod.MODEL, 'gemini-embedding-2')
        self.assertEqual(mod.DIM, 768)
        self.assertEqual(mod.formato_pregunta('¿Qué dice el artículo 15?'),
                         'task: question answering | query: ¿Qué dice el artículo 15?')

    def test_privacidad_iban_dni(self):
        s = 'DNI: 78123649-K y cuenta ES66 2080 4455 9911 2233 7788'
        t = mod.texto_para_api(s)
        self.assertNotIn('78123649', t)
        self.assertNotIn('2080 4455', t)
        self.assertIn('[IBAN OMITIDO]', t)

    def test_privacidad_domicilio_fecha(self):
        s = 'Domicilio social: Calle Mayor 12\nFecha de nacimiento: 1 de enero\nArtículo 15. Duración'
        t = mod.texto_para_api(s)
        self.assertNotIn('Calle Mayor', t)
        self.assertNotIn('1 de enero', t)
        self.assertIn('Artículo 15.', t)

    def test_doc_formato_y_metadatos(self):
        t = 'Artículo 15. Texto'
        r = {'id':'a', 'texto':t, 'metadata':{'documento_id':'NOR-001','seccion':'Artículo 15.','pagina_inicio':20,
                'pagina_fin':21,'tipo_documento':'norma', 'sha256_texto':hashlib.sha256(t.encode()).hexdigest()}}
        self.assertTrue(mod.formato_documento(r).startswith('title: NOR-001 - Artículo 15. | text:'))
        m = mod.metadatos_chroma(r)
        self.assertIsInstance(m['pagina_inicio'],int)
        self.assertIsInstance(m['url_oficial'],str)

    def test_vector_valido(self):
        self.assertTrue(mod.vector_valido([1.0]*768))
        self.assertFalse(mod.vector_valido([1.0]*100))
        self.assertFalse(mod.vector_valido([float('nan')]*768))

    def test_rechazo_ids_duplicados(self):
        doc='texto'
        m={'documento_id':'DOC-001','seccion':'Primera','pagina_inicio':1,'pagina_fin':1,
           'sha256_texto': hashlib.sha256(doc.encode()).hexdigest()}
        r={'id':'a','texto':doc,'metadata':m}
        with tempfile.TemporaryDirectory() as tmp:
            archivo=Path(tmp)/'chunks.jsonl'
            archivo.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'duplicado'):
                mod.leer_corpus(archivo,verificar_sha=False)

    def test_muestra_contiene_cuatro_documentos_y_cuatro_claves(self):
        datos=[]
        for d,s in [('DOC-001','PRIMERA. Cargo'),('DOC-002','TERCERA. Duración'),
                    ('NOR-001','Artículo 15. Duración'),('NOR-002','Artículo 1. Ambito')]:
            for i in range(4):
                datos.append({'id':f'{d}-{i}', 'texto':'x',
                              'metadata':{'documento_id':d,'seccion':s if i==3 else 'Introduccion'}})
        muestra=mod.seleccionar_muestra(datos)
        self.assertEqual(len(muestra),16)
        self.assertEqual(set(x['metadata']['documento_id'] for x in muestra),
                         {'DOC-001','DOC-002','NOR-001','NOR-002'})

    def test_huella_real_si_archivo_montado(self):
        if mod.CORPUS.exists():
            self.assertEqual(len(mod.leer_corpus()),437)

if __name__ == '__main__':unittest.main()
