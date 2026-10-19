# -*- coding: utf-8 -*-
"""Utilidades comunes de los capítulos: rutas del Entregable 3 y lectura de resultados."""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().resolve()
while not (RAIZ / "estrato3").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))
RES = RAIZ / "resultados"
FIG = RES / "figuras"


def leer_json(nombre):
    """JSON de ``resultados/`` o ``None`` si todavía no existe."""
    p = RES / nombre
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def leer_csv(nombre, **kw):
    p = RES / nombre
    return pd.read_csv(p, **kw) if p.exists() else None


def figura(nombre):
    """Muestra una figura de ``resultados/figuras`` (si existe)."""
    from IPython.display import Image, display
    p = FIG / nombre
    if p.exists():
        display(Image(filename=str(p)))
    else:
        print(f"(figura {nombre} pendiente)")
