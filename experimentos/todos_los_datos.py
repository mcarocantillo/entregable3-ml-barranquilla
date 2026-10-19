# -*- coding: utf-8 -*-
"""Sensibilidad al tamaño de muestra: los finalistas reentrenados con TODO el entrenamiento.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

El pipeline combinatorio se corrió con una submuestra de 10 000 viviendas para que la validación anidada de las
156 combinaciones cupiera en un portátil (capítulo 2). Aquí se mide cuánto cuesta esa decisión: cada finalista
(con los hiperparámetros que eligió su búsqueda, sin buscar de nuevo) y el modelo base del Entregable 1 se
reentrenan una vez con las 257 273 viviendas del entrenamiento y se evalúan en el mismo test (7 bloques de 2 km).

Salidas: ``resultados/todos_los_datos.json`` (métricas de test con IC BCa por bloques, F1 con la submuestra y con
todos los datos, y la diferencia pareada del XGBoost geográfico frente a cada rival) y
``resultados/proba_test_todos_los_datos.npz``.

Uso (desde e3/, después de finalistas.py y con el computador libre):
    ../venv_e2/Scripts/python -u experimentos/todos_los_datos.py
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import estadistica as ES  # noqa: E402
from estrato3 import evaluacion as EV  # noqa: E402
from estrato3 import experimento as EX  # noqa: E402

warnings.filterwarnings("ignore")
CLASES = CF.CLASES
# los más caros primero; Random Forest va por lotes para no llenar la memoria (11.9 GB en el portátil)
ORDEN = ["XGBoost geográfico", "Random Forest", "XGBoost", "Logística L1/L2", "SVM", "KNN", "Árbol de decisión",
         "Naive Bayes", EX.BASE_E1]


def archivo(nombre):
    return nombre.replace(" ", "_").replace("/", "-")


def main():
    from joblib import Parallel, delayed
    from estrato3 import datos as DT
    CF.fijar_semillas()
    d = DT.cargar()
    cols = list(dict.fromkeys(d["num"] + d["cat"] + ["estrato_num", "x_km", "y_km", "npn_edificio", "bloque"]))
    train, test = d["train"][cols].reset_index(drop=True), d["test"][cols].reset_index(drop=True)
    with open(EX.COMPLETO_CACHE, "wb") as fh:
        pickle.dump({"train": train, "test": test,
                     "datos": {k: d[k] for k in ["num", "cat", "num_log", "dec"]}}, fh)
    FN = json.loads((CF.RESULTADOS / "finalistas.json").read_text(encoding="utf-8"))
    fin = FN["finalistas"]
    yte, bloques = test.estrato_num.to_numpy(), test.bloque.to_numpy()
    print(f"entrenamiento completo: {len(train):,} viviendas; test: {len(test):,}", flush=True)

    tareas = [(n, fin[n]["params"] if n in fin else {}, fin[n]["balanceo"] if n in fin else "ninguno")
              for n in ORDEN if n in fin or n == EX.BASE_E1]
    # 3 procesos: con SMOTE el entrenamiento crece a ≈ 460 000 filas y la memoria es el límite, no los núcleos
    res = Parallel(n_jobs=3, backend="loky")(delayed(EX.todo_el_train)(n, p, b) for n, p, b in tareas)
    proba = {n: P for n, P, _ in res}
    segundos = {n: s for n, _, s in res}

    salida = {"n_train": len(train), "n_test": len(test), "n_submuestra": int(FN.get("n_submuestra", 10001)),
              "modelos": {}}
    for n, P in proba.items():
        m = EV.metricas(yte, P)
        pred = CLASES[P.argmax(1)]
        ic = ES.bootstrap_bca_bloques(lambda idx: f1_score(yte[idx], pred[idx], average="macro", zero_division=0),
                                      bloques, n_resamples=999)
        f1_sub = (fin[n]["test"]["f1_macro"] if n in fin else FN["modelo_base"]["test"]["f1_macro"])
        salida["modelos"][n] = {"balanceo": fin[n]["balanceo"] if n in fin else "ninguno",
                                "metodo": fin[n]["metodo"] if n in fin else "fijo (E1: C = 0.01)",
                                "test": m, "ic95_bca_f1_test": list(ic), "f1_test_submuestra": f1_sub,
                                "segundos": segundos[n]}
        print(f"{n}: F1 test {m['f1_macro']:.3f} [{ic[0]:.3f}, {ic[1]:.3f}] (submuestra {f1_sub:.3f}; "
              f"{segundos[n]:.0f} s)", flush=True)

    geo = "XGBoost geográfico"
    pred_geo = CLASES[proba[geo].argmax(1)]
    comp = []
    for rival in [EX.BASE_E1, "Logística L1/L2", "XGBoost", "Random Forest"]:
        pred_r = CLASES[proba[rival].argmax(1)]

        def dif_f1(idx, a=pred_geo, b=pred_r):
            return (f1_score(yte[idx], a[idx], average="macro", zero_division=0)
                    - f1_score(yte[idx], b[idx], average="macro", zero_division=0))
        ic = ES.bootstrap_bca_bloques(dif_f1, bloques, n_resamples=999)
        por_bloque = [dif_f1(np.where(bloques == b)[0]) for b in np.unique(bloques)]
        comp.append({"rival": rival, "test_dif": float(dif_f1(np.arange(len(yte)))), "test_ic95_bca": list(ic),
                     "test_bloques_a_favor": int(np.sum(np.asarray(por_bloque) > 0)),
                     "test_n_bloques": len(por_bloque)})
        print(f"geo vs {rival}: {comp[-1]['test_dif']:+.3f} [{ic[0]:+.3f}, {ic[1]:+.3f}], "
              f"{comp[-1]['test_bloques_a_favor']}/{len(por_bloque)} bloques", flush=True)
    salida["geo_vs_rivales"] = comp
    np.savez_compressed(CF.RESULTADOS / "proba_test_todos_los_datos.npz",
                        **{archivo(k): v.astype(np.float32) for k, v in proba.items()}, y=yte, bloque=bloques)
    (CF.RESULTADOS / "todos_los_datos.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1,
                                                                   default=float), encoding="utf-8")
    EX.COMPLETO_CACHE.unlink(missing_ok=True)
    print("guardado todos_los_datos.json", flush=True)


if __name__ == "__main__":
    main()
