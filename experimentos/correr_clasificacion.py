# -*- coding: utf-8 -*-
"""Pipeline combinatorio de clasificación con validación cruzada anidada.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Combinaciones: 7 modelos de la guía (+ el XGBoost geográfico, modelo nuevo del proyecto) × 4 técnicas de balanceo ×
4 optimizadores = 112 (+ 16). Cada combinación se evalúa con validación anidada:

- bucle externo: 5 pliegues kNNDM sobre la submuestra (escenario: la geometría del test del Entregable 1);
- bucle interno: 3 pliegues kNNDM dentro del entrenamiento de cada pliegue externo (escenario: el pliegue externo).

La métrica que se reporta sale del bucle externo. Cada tarea (combinación × pliegue externo) se guarda en
``resultados/tareas/`` (JSON con métricas, hiperparámetros, historial de la búsqueda y tiempos; NPZ con las
probabilidades fuera de muestra), así que el experimento se puede interrumpir y retomar.

Uso (desde e3/):
    ../venv_e2/Scripts/python -u experimentos/correr_clasificacion.py            # todo
    ../venv_e2/Scripts/python -u experimentos/correr_clasificacion.py --piloto   # tiempos con presupuesto 2
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
    ap.add_argument("--piloto", action="store_true", help="presupuesto 2, un pliegue, solo Random (tiempos)")
    ap.add_argument("--modelos", nargs="*", default=None)
    ap.add_argument("--metodos", nargs="*", default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--presupuesto", type=int, default=CF.PRESUPUESTO)
    a = ap.parse_args()
    EX.preparar()
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    modelos = a.modelos or (MD.MODELOS_GUIA + ["XGBoost geográfico"])
    metodos = a.metodos or list(OP.METODOS)
    folds = range(CF.K_EXTERNO)
    presupuesto = a.presupuesto
    carpeta = EX.TAREAS
    if a.piloto:
        metodos, folds, presupuesto = ["Random"], [0], 2
        carpeta = CF.RESULTADOS / "piloto"
    costo = {"Random Forest": 0, "XGBoost geográfico": 1, "XGBoost": 2, "SVM": 3, "Logística L1/L2": 4, "KNN": 5,
             "Árbol de decisión": 6, "Naive Bayes": 7}   # orden del piloto: las tareas caras primero
    pendientes = [(m, b, o, f) for m in sorted(modelos, key=lambda x: costo.get(x, 9)) for f in folds
                  for b in MD.BALANCEOS for o in metodos
                  if not (carpeta / f"{EX.nombre_tarea(m, b, o, f)[0]}.json").exists()]
    print(f"{len(pendientes)} tareas pendientes (presupuesto {presupuesto}, {a.n_jobs} procesos)", flush=True)
    t0 = time.time()
    for s, f1, seg in Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(EX.tarea)(m, b, o, f, presupuesto, str(carpeta)) for m, b, o, f in pendientes):
        print(f"[{time.time() - t0:7.0f} s] {s}: F1 externo {f1:.3f} (búsqueda {seg:.0f} s)", flush=True)
    print("=== FIN", flush=True)


if __name__ == "__main__":
    main()
