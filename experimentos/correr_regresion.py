# -*- coding: utf-8 -*-
"""Pipeline combinatorio de regresión: 7 modelos × 4 optimizadores = 28, con validación anidada.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

El estrato (1-6) se trata como número: es la lectura ordinal del mismo problema. Mismos datos, submuestra y
pliegues (kNNDM) que la clasificación; el balanceo no aplica. El objetivo interno es el RMSE; en el bucle externo
se reportan RMSE, MAE, R² y el F1 macro de la predicción redondeada al estrato más cercano (para compararla con
la clasificación). Cada tarea se guarda en ``resultados/tareas_regresion/``.

Uso (desde e3/):  ../venv_e2/Scripts/python -u experimentos/correr_regresion.py
"""
import argparse
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import experimento as EX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="*", default=None)
    ap.add_argument("--metodos", nargs="*", default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--presupuesto", type=int, default=CF.PRESUPUESTO)
    a = ap.parse_args()
    EX.preparar()
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    carpeta = CF.RESULTADOS / "tareas_regresion"
    costo = {"Random Forest (regresión)": 0, "XGBoost (regresión)": 1, "SVR": 2, "KNN (regresión)": 3}
    modelos = sorted(a.modelos or MD.MODELOS_REG, key=lambda x: costo.get(x, 9))
    metodos = a.metodos or list(OP.METODOS)
    pendientes = [(m, o, f) for m in modelos for f in range(CF.K_EXTERNO) for o in metodos
                  if not (carpeta / f"{EX.nombre_tarea(m, 'no aplica', o, f)[0]}.json").exists()]
    print(f"{len(pendientes)} tareas de regresión pendientes (presupuesto {a.presupuesto})", flush=True)
    t0 = time.time()
    for s, rmse, seg in Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(EX.tarea_reg)(m, o, f, a.presupuesto, str(carpeta)) for m, o, f in pendientes):
        print(f"[{time.time() - t0:7.0f} s] {s}: RMSE externo {rmse:.3f} (búsqueda {seg:.0f} s)", flush=True)
    print("=== FIN", flush=True)


if __name__ == "__main__":
    main()
