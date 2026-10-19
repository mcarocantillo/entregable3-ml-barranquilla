# -*- coding: utf-8 -*-
"""Optimización multi-fidelidad: Successive Halving frente a Random Search (sección 3.3 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Successive Halving (Jamieson y Talwalkar, 2016; ``HalvingRandomSearchCV`` de scikit-learn) prueba muchas
configuraciones con un presupuesto PARCIAL y solo sigue con las más prometedoras:

- **Fidelidad = número de viviendas de entrenamiento** (``resource="n_samples"``). Es la fidelidad natural aquí:
  el costo de todos los modelos crece con n (capítulo 4) y el orden de las configuraciones con pocas viviendas
  se parece al orden con muchas (los hiperparámetros que importan, como la profundidad o el learning rate, no
  dependen mucho de n en este rango).
- **Factor de reducción η = 3**: en cada ronda queda un tercio de las configuraciones y cada una recibe el triple
  de viviendas. Es el valor por defecto y el que recomiendan Li et al. (2018) para Hyperband.

Se compara, en los 5 pliegues externos y con los mismos pliegues internos, contra Random Search con 12
configuraciones a fidelidad completa: F1 del bucle externo, tiempo y número de ajustes equivalentes.

Uso (desde e3/, con el computador libre):  ../venv_e2/Scripts/python -u experimentos/multifidelidad.py
"""
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import f1_score
from sklearn.model_selection import HalvingRandomSearchCV

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import evaluacion as EV  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402
from estrato3 import optimizacion as OP  # noqa: E402

warnings.filterwarnings("ignore")
DISTRIBUCIONES = {
    "XGBoost": {"modelo__n_estimators": randint(50, 300), "modelo__learning_rate": loguniform(0.03, 0.3),
                "modelo__max_depth": randint(3, 9), "modelo__subsample": uniform(0.6, 0.4),
                "modelo__colsample_bytree": uniform(0.5, 0.5), "modelo__min_child_weight": loguniform(1, 20)},
    "Random Forest": {"modelo__n_estimators": randint(50, 250), "modelo__max_depth": randint(6, 26),
                      "modelo__max_features": ["sqrt", 0.3, 0.6], "modelo__min_samples_leaf": randint(1, 51)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="*", default=list(DISTRIBUCIONES))
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--candidatos", type=int, default=27)
    ap.add_argument("--salida", default="multifidelidad.json")
    a = ap.parse_args()
    CF.fijar_semillas()
    _, M1, _, _ = CF.importar_e1()
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    paq["M1"] = M1
    sub = paq["sub"]
    y = sub.estrato_num.to_numpy()
    filas = []
    for nombre in a.modelos:
        dist = DISTRIBUCIONES[nombre]
        for fold, (itr, iva) in enumerate(paq["externos"]):
            if a.folds is not None and fold not in a.folds:
                continue
            Xtr, ytr, Xva, yva = sub.iloc[itr], y[itr], sub.iloc[iva], y[iva]
            base = MD.pipeline(paq, nombre, {}, "ninguno")
            t0 = time.perf_counter()
            sh = HalvingRandomSearchCV(base, dist, n_candidates=a.candidatos, factor=3, resource="n_samples",
                                       min_resources=max(300, len(itr) // a.candidatos), cv=paq["internos"][fold],
                                       scoring="f1_macro", random_state=CF.SEED, n_jobs=4, refit=True).fit(Xtr, ytr)
            t_sh = time.perf_counter() - t0
            f1_sh = EV.metricas(yva, EV.proba_completa(sh.best_estimator_, Xva))["f1_macro"]
            # trabajo en «ajustes equivalentes a fidelidad completa»: Σ (viviendas usadas / viviendas totales)
            r = sh.cv_results_
            equiv = float(np.sum(np.asarray(r["n_resources"]) / len(itr)) * len(paq["internos"][fold]))
            t1 = time.perf_counter()
            obj = OP.Objetivo(paq, Xtr, ytr, paq["internos"][fold], nombre, "ninguno")
            OP.aleatoria(obj, nombre, presupuesto=CF.PRESUPUESTO)
            params, _ = obj.mejor()
            est = MD.pipeline(paq, nombre, params, "ninguno").fit(Xtr, ytr)
            t_rs = time.perf_counter() - t1
            f1_rs = EV.metricas(yva, EV.proba_completa(est, Xva))["f1_macro"]
            filas.append({"modelo": nombre, "fold": fold, "f1_halving": f1_sh, "seg_halving": t_sh,
                          "configuraciones_halving": int(np.sum(np.asarray(r["iter"]) == 0)), "rondas": int(sh.n_iterations_),
                          "ajustes_equivalentes_halving": equiv, "f1_random": f1_rs, "seg_random": t_rs,
                          "ajustes_random": CF.PRESUPUESTO * len(paq["internos"][fold])})
            print(f"{nombre} fold {fold}: halving F1 {f1_sh:.3f} ({t_sh:.0f} s, {int(np.sum(np.asarray(r['iter']) == 0))} config., "
                  f"{equiv:.1f} ajustes equiv.) | random F1 {f1_rs:.3f} ({t_rs:.0f} s)", flush=True)
    (CF.RESULTADOS / a.salida).write_text(json.dumps({"factor": 3, "fidelidad": "n_samples",
                                                                   "resultados": filas}, ensure_ascii=False, indent=1,
                                                                  default=float), encoding="utf-8")
    print("guardado multifidelidad.json", flush=True)


if __name__ == "__main__":
    main()
