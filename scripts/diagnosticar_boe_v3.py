"""Diagnóstico local (solo lectura) de cabeceras BOE. No modifica PDFs ni corpus."""
from pathlib import Path
import re
import pymupdf

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVOS = {
    'NOR-001': RAIZ/'corpus'/'normativa'/'NOR-001_Estatuto_Trabajadores_BOE.pdf',
    'NOR-002': RAIZ/'corpus'/'normativa'/'NOR-002_Alta_Direccion_BOE.pdf',
}
PATRON_EXACTO = re.compile(r'(?m)^[ \t]*TEXTO[ \t]+CONSOLIDADO[ \t]*(?=\r?$)', re.I)
PATRON_LIBRE = re.compile(r'TEXTO\s+CONSOLIDADO', re.I)
PATRON_FECHA = re.compile(r'Última\s+modificaci[oó]n\s*:', re.I)

for etiqueta, ruta in ARCHIVOS.items():
    print(f'\n== {etiqueta}: {ruta.name} ==')
    if not ruta.exists():
        print('NO EXISTE:', ruta)
        continue
    with pymupdf.open(ruta) as archivo:
        print('Páginas PDF:', len(archivo))
        for numero, pagina in enumerate(archivo, 1):
            texto = pagina.get_text('text', sort=True)
            if not PATRON_LIBRE.search(texto):
                continue
            exactos = list(PATRON_EXACTO.finditer(texto))
            fechas = [bool(PATRON_FECHA.search(texto[m.end():m.end()+900])) for m in exactos]
            print(f'Página {numero}: menciones={len(PATRON_LIBRE.findall(texto))}, '
                  f'cabeceras_exactas={len(exactos)}, cabecera_con_fecha={sum(fechas)}')
            for m in exactos:
                print('  Encabezado:', repr(texto[m.start():m.end()]))
                print('  Texto posterior:', repr(texto[m.end():m.end()+100][:100]))
