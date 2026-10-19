# -*- coding: utf-8 -*-
"""Configuración única del Entregable 3: semilla global, rutas y presupuestos.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Todas las decisiones numéricas del pipeline combinatorio están aquí para que
el experimento sea reproducible y auditable (sección 7.3 de la guía):

- ``SEED``: única semilla global; se propaga a numpy, scikit-learn,
  imbalanced-learn, XGBoost, Optuna y DEAP.
- ``N_SUBMUESTRA``: tamaño de la submuestra estratificada en la que corren las
  combinaciones. Con 30 000 viviendas y 24 evaluaciones, las 128 combinaciones
  con validación anidada costaban unas 630 horas-CPU en un portátil de 4
  núcleos (piloto del 18-oct); con 10 000 viviendas y 12 evaluaciones caben en
  una noche. Los finalistas se reentrenan con todo el entrenamiento y se
  evalúan en el test.
- ``K_EXTERNO`` / ``K_INTERNO``: pliegues de la validación anidada (kNNDM).
- ``PRESUPUESTO``: evaluaciones por búsqueda; es el mismo para Random,
  Bayesiana y Genética, y Grid usa una rejilla de tamaño parecido, de modo que
  la comparación de optimizadores sea por unidad de presupuesto.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42
RAIZ_E3 = Path(__file__).resolve().parents[1]
ML_TRABAJO = RAIZ_E3.parent
# Entregable 1, versión 3: datos procesados, decisiones del EDA, preprocesador y src/vecinos.py (kNNDM).
E1 = Path(os.environ.get("ENTREGABLE1", ML_TRABAJO / "e1_rama"))
RESULTADOS = RAIZ_E3 / "resultados"

N_SUBMUESTRA = 10_000
K_EXTERNO = 5
K_INTERNO = 3
PRESUPUESTO = 12
CLASES = np.arange(1, 7)
K_VECINDAD = 8                      # vecinos del lag espacial físico (versión 3, cap. 6 del E2)


def fijar_semillas(seed=SEED):
    """Fija la semilla de ``random`` y ``numpy`` (las librerías reciben ``random_state=seed`` explícito).

    Parámetros
    ----------
    seed : int
        Semilla global.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def importar_e1():
    """Agrega el Entregable 1 al ``sys.path`` y devuelve sus módulos ``config``, ``modelo``, ``particion`` y ``vecinos``."""
    if str(E1) not in sys.path:
        sys.path.insert(0, str(E1))
    from src import config as C1
    from src import modelo as M1
    from src import particion as P1
    from src import vecinos as V1
    return C1, M1, P1, V1
