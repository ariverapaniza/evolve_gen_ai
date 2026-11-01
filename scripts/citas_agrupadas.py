"""Citas Gemini: identifica [F1], [F1,F4] y [F1, F4, F7] sin perder refs.

El parser falla cerrado para referencias mal formadas dentro de corchetes.
Sin dependencias externas, sin llamadas API.
"""
from __future__ import annotations
import re

BLOQUE = re.compile(r"\[([^\[\]]*)\]")
PATRON = re.compile(r"F[1-9]\d*(?:\s*(?:,|;|\by\b)\s*F[1-9]\d*)*", re.I)
ETIQUETA = re.compile(r"F([1-9]\d*)", re.I)
MENCION = re.compile(r"(?<![A-Za-z0-9])F\d+", re.I)


def extraer_numeros(texto: str) -> list[int]:
    """Devuelve citas en orden, incluyendo las agrupadas; no elimina duplicados."""
    if not isinstance(texto, str):
        raise ValueError("La respuesta debe ser texto")
    numeros = []
    for match in BLOQUE.finditer(texto):
        contenido = match.group(1).strip()
        if MENCION.search(contenido):
            if not PATRON.fullmatch(contenido):
                raise ValueError(f"Cita entre corchetes mal formada: [{contenido[:70]}]")
            numeros.extend(int(n) for n in ETIQUETA.findall(contenido))
    # Citas fuera de corchetes tipo F4 no se consideran referencias válidas.
    return numeros


def extraer_referencias(texto: str) -> set[str]:
    return {f"F{n}" for n in extraer_numeros(texto)}
