"""Regresiones sin Gemini: corrigen el fallo real de la comparación de fechas."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts'))
import etapa5_rag as rag

FUENTES = ['DOC-001', 'DOC-002']
PREGUNTA = 'Compara las fechas de inicio de los dos contratos'


def evidencia(doc, id_, sec, texto):
    return {'id': id_, 'metadata': {'documento_id': doc, 'seccion': sec,
                                   'pagina_inicio': 2, 'pagina_fin': 2},
            'distancia': 0.42, 'texto': texto}


def filas_correctas():
    return [
        evidencia('DOC-001', 'DOC-001-00005', 'TERCERA. Duración',
                  'El contrato se celebra por tiempo indefinido, iniciando su vigencia el 1 de abril de 2026.'),
        evidencia('DOC-002', 'DOC-002-00005', 'TERCERA. Duración',
                  'El contrato se concierta por tiempo indefinido, iniciando su vigencia el 15 de abril de 2026.'),
    ]


class PruebasComparacionesRAG(unittest.TestCase):
    def test_identifica_comparacion_de_fechas(self):
        self.assertEqual(rag.tipo_comparacion(PREGUNTA, FUENTES), 'inicio_contratos')

    def test_pregunta_dirigida_al_contrato_alta_direccion(self):
        consulta = rag.consulta_por_fuente(PREGUNTA, 'DOC-001', FUENTES)
        self.assertIn('vigencia', consulta)
        self.assertIn('alta dirección', consulta)
        self.assertNotIn('1 de abril de 2026', consulta)  # No filtra la respuesta del dataset

    def test_pregunta_dirigida_al_contrato_indefinido(self):
        consulta = rag.consulta_por_fuente(PREGUNTA, 'DOC-002', FUENTES)
        self.assertIn('vigencia', consulta)
        self.assertIn('analista', consulta)
        self.assertNotIn('15 de abril de 2026', consulta)

    def test_no_reescribe_pregunta_ajena_a_comparacion(self):
        q = '¿Cuál es el cargo del directivo?'
        self.assertIsNone(rag.tipo_comparacion(q, FUENTES))
        self.assertEqual(rag.consulta_por_fuente(q, 'DOC-001', FUENTES), rag.pregunta_especifica(q, 'DOC-001'))

    def test_distingue_fecha_de_firma_de_fecha_de_vigencia(self):
        self.assertFalse(rag.es_evidencia_inicio(evidencia('DOC-001', 'x', 'DUODÉCIMA. Jurisdicción',
                                                        'Firmado en Barcelona el 20 de marzo de 2026.')))
        self.assertTrue(rag.es_evidencia_inicio(filas_correctas()[0]))
        self.assertTrue(rag.es_evidencia_inicio(filas_correctas()[1]))

    def test_rechaza_comparacion_sin_fecha_del_primer_contrato(self):
        d = filas_correctas()
        d[0] = evidencia('DOC-001','DOC-001-00012','Firma', 'Se firma el 20 de marzo de 2026.')
        with self.assertRaisesRegex(ValueError, 'FALTAN EVIDENCIAS'):
            rag.comprobar_cobertura(PREGUNTA, FUENTES, d)

    def test_acepta_comparacion_con_ambas_evidencias(self):
        self.assertIsNone(rag.comprobar_cobertura(PREGUNTA, FUENTES, filas_correctas()))

    def test_rechaza_citas_a_fragmentos_irrelevantes_de_un_documento(self):
        d = [evidencia('DOC-001','DOC-001-00012','Firma','Firmado el 20 de marzo de 2026.'),
             *filas_correctas()]
        with self.assertRaisesRegex(ValueError, 'no cita la evidencia'):
            rag.comprobar_citas_pertinentes(PREGUNTA, FUENTES, d, [1, 3])
        self.assertIsNone(rag.comprobar_citas_pertinentes(PREGUNTA, FUENTES, d, [2, 3]))

    def test_comprueba_textos_reales_corpus_si_estan_disponibles(self):
        corpus = BASE / 'corpus' / 'chunks' / 'chunks_para_embeddings.jsonl'
        if not corpus.exists():
            self.skipTest('Corpus real no disponible en este montaje')
        encontrados = {}
        for linea in corpus.read_text(encoding='utf-8').splitlines():
            r = json.loads(linea)
            if r['id'] in ('DOC-001-00005', 'DOC-002-00005'):
                encontrados[r['id']] = r
        self.assertEqual(len(encontrados), 2)
        filas = [dict(encontrados[i], distancia=.42) for i in ('DOC-001-00005', 'DOC-002-00005')]
        self.assertIsNone(rag.comprobar_cobertura(PREGUNTA, FUENTES, filas))

if __name__ == '__main__':
    unittest.main()
