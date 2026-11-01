"""12 pruebas offline del notebook y documentación de Etapa 9.

Estas pruebas no invocan la API, no acceden a ChromaDB y no ejecutan celdas.
"""
from __future__ import annotations
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/'notebooks'/'Proyecto_Final_Evolve_Gemini_RAG_LangGraph.ipynb'

def cargar():
    return json.loads(NB.read_text(encoding='utf-8'))

def codigo():
    return '\n'.join(''.join(c.get('source',[])) if isinstance(c.get('source'),list) else c.get('source','') for c in cargar()['cells'] if c['cell_type']=='code')

def markdown():
    return '\n'.join(''.join(c.get('source',[])) if isinstance(c.get('source'),list) else c.get('source','') for c in cargar()['cells'] if c['cell_type']=='markdown')

class PruebasEtapa9(unittest.TestCase):
    def test_notebook_es_version_4(self):
        self.assertEqual(cargar()['nbformat'],4)
    def test_minimo_diez_celdas_codigo(self):
        self.assertGreaterEqual(len([c for c in cargar()['cells'] if c['cell_type']=='code']), 10)
    def test_todas_celdas_codigo_compilan(self):
        for c in cargar()['cells']:
            if c['cell_type']=='code':
                x=c['source']; x=''.join(x) if isinstance(x,list) else x
                compile(x, 'notebook_evolve', 'exec')
    def test_no_contiene_clave_literal(self):
        text=NB.read_text(encoding='utf-8')
        self.assertNotRegex(text,r'AIza[0-9A-Za-z_-]{30,}')
    def test_demos_api_apagadas_por_defecto(self):
        t=codigo()
        for v in ('EJECUTAR_CINCO_EJEMPLOS','EJECUTAR_MEMORIA_REAL','ACTIVAR_CHAT_INTERACTIVO'):
            self.assertIn(v+' = False',t)
    def test_cinco_ejemplos_programaticos(self):
        t=codigo()
        for label in ('E01','E02','E03','E04','E05'):
            self.assertIn('"'+label+'"',t)
    def test_demostracion_memoria_tres_turnos(self):
        t=codigo()
        self.assertIn('¿Y qué indica el otro contrato?',t)
        self.assertIn('Compara ambas fechas de inicio',t)
        self.assertIn('nb_simulacion_unica',t)
    def test_chat_interactivo_con_salir(self):
        t=codigo()
        self.assertIn('input("TÚ> ")',t)
        self.assertIn('"salir"',t)
    def test_no_reindexa_al_ejecutar(self):
        t=codigo()
        self.assertNotIn('ejecutar_indexacion(',t)
        self.assertNotIn('.add(',t)
    def test_readme_documenta_prompt_y_limites(self):
        t=(ROOT/'README.md').read_text(encoding='utf-8')
        for x in ('System prompt','[F#]','InMemorySaver','Hit@5','D05','D09','getpass'):
            self.assertIn(x,t)
    def test_documentacion_academica_existe(self):
        self.assertTrue((ROOT/'docs'/'MEMORIA_TECNICA_PROYECTO_FINAL.md').is_file())
        self.assertIn('5–10 minutos',(ROOT/'docs'/'MEMORIA_TECNICA_PROYECTO_FINAL.md').read_text(encoding='utf-8'))
    def test_validador_check_existe_y_compila(self):
        path=ROOT/'scripts'/'etapa9_validar_entrega.py'
        self.assertTrue(path.is_file())
        compile(path.read_text(encoding='utf-8'),str(path),'exec')

if __name__=='__main__': unittest.main()
