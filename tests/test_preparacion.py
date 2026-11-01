"""Pruebas offline: validan solo originales suministrados (no simulan PDFs del BOE)."""
import hashlib
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from preparar_corpus import DOCUMENTOS, CORPUS, validar_archivo


class PruebasCorpusOriginal(unittest.TestCase):
    def test_hay_exactamente_cuatro_fuentes_definidas(self):
        self.assertEqual(len(DOCUMENTOS), 4)

    def test_originales_inmutables_y_legibles(self):
        for doc in DOCUMENTOS[:2]:
            with self.subTest(doc=doc["id"]):
                resultado = validar_archivo(doc, extraer=False)
                self.assertEqual(resultado["estado"], "VALIDADO", resultado["motivos"])
                self.assertEqual(resultado["sha256"], doc["hash_fijado"])

    def test_un_hash_original_incorrecto_es_detectado(self):
        doc = dict(DOCUMENTOS[0])
        doc["hash_fijado"] = "0" * 64
        resultado = validar_archivo(doc, extraer=False)
        self.assertEqual(resultado["estado"], "ERROR")
        self.assertIn("SHA-256", " ".join(resultado["motivos"]))

    def test_identificadores_boe_estan_especificados(self):
        self.assertEqual([x["identificador_boe"] for x in DOCUMENTOS[2:]],
                         ["BOE-A-2015-11430", "BOE-A-1985-17006"])


if __name__ == "__main__":
    unittest.main()
