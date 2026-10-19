# -*- coding: utf-8 -*-
"""Optimización computacional (sección 4 de la guía): complejidad teórica frente a tiempo empírico.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada modelo se comparan una versión estándar y una optimizada, variando el número de viviendas n (con las
mismas p variables preprocesadas del pipeline) para estimar el exponente empírico b de tiempo ≈ a·n^b y
contrastarlo con el orden O(·) teórico:

- KNN: búsqueda exhaustiva (O(n·p) por consulta) frente a KD-Tree y Ball Tree (O(log n) solo en dimensión baja), en
  p ≈ 30 (variables del modelo) y en p = 2 (coordenadas).
- Logística y Ridge/Lasso: solver SAGA (costo por época O(n·p)) frente a lbfgs / Cholesky.
- Naive Bayes: ``fit`` frente a ``partial_fit`` por lotes (aprendizaje incremental, memoria O(p·K)).
- XGBoost: ``tree_method="hist"`` (O(n·p) para construir histogramas una vez, luego O(bins·p) por nodo) frente a
  ``"exact"`` (O(n·p·log n) por ordenar), y early stopping con un conjunto de validación independiente.
- SVM: núcleo RBF completo (``SVC``, O(n²)–O(n³)) frente a Nyström + LinearSVC (O(n·m²)) y SGD (O(n·p) por época).

Se mide el tiempo (mediana de 3 repeticiones), la memoria pico (``tracemalloc``), el desempeño (F1 macro en un
conjunto de validación fijo), un perfil ``cProfile`` de una evaluación completa del pipeline y la aceleración de
la validación cruzada con ``n_jobs`` (joblib, backend loky). Salida: ``resultados/complejidad.json`` y figuras.

Uso (desde e3/, con el computador libre):  ../venv_e2/Scripts/python -u experimentos/complejidad.py
"""
import cProfile
import io
import json
import os
import pickle
import pstats
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

warnings.filterwarnings("ignore")
FIG = CF.RESULTADOS / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
TAMANOS = [1000, 2000, 4000, 8000, 16000, 32000]


def medir(fn, rep=3):
    """Mediana del tiempo de ``fn`` y memoria pico (MB) de la última repetición."""
    tiempos = []
    for _ in range(rep):
        tracemalloc.start()
        t = time.perf_counter()
        out = fn()
        tiempos.append(time.perf_counter() - t)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return float(np.median(tiempos)), pico / 1e6, out


def exponente(ns, ts):
    """Pendiente b de log(t) = a + b·log(n) (exponente empírico de escalamiento)."""
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    ok = ts > 0
    return float(np.polyfit(np.log(ns[ok]), np.log(ts[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")


def main():
    CF.fijar_semillas()
    from estrato3 import datos as DT
    d = DT.cargar()
    d["M1"] = d["M1"]
    tr = d["train"].sample(n=max(TAMANOS) + 10000, random_state=CF.SEED).reset_index(drop=True)
    prep = MD.preprocesador(d).fit(tr.iloc[:max(TAMANOS)])
    X = prep.transform(tr)
    y = tr.estrato_num.to_numpy()
    Xv, yv = X[-10000:], y[-10000:]
    XY = tr[["x_km", "y_km"]].to_numpy()
    p = X.shape[1]
    print(f"p = {p} variables tras el preprocesamiento; validación fija de 10 000", flush=True)
    variantes = {
        "KNN | fuerza bruta (p≈30)": lambda n: KNeighborsClassifier(30, algorithm="brute").fit(X[:n], y[:n]).predict(Xv[:2000]),
        "KNN | KD-Tree (p≈30)": lambda n: KNeighborsClassifier(30, algorithm="kd_tree").fit(X[:n], y[:n]).predict(Xv[:2000]),
        "KNN | Ball Tree (p≈30)": lambda n: KNeighborsClassifier(30, algorithm="ball_tree").fit(X[:n], y[:n]).predict(Xv[:2000]),
        "KNN | fuerza bruta (p=2)": lambda n: KNeighborsClassifier(30, algorithm="brute").fit(XY[:n], y[:n]).predict(XY[-2000:]),
        "KNN | KD-Tree (p=2)": lambda n: KNeighborsClassifier(30, algorithm="kd_tree").fit(XY[:n], y[:n]).predict(XY[-2000:]),
        "Logística | lbfgs": lambda n: LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X[:n], y[:n]),
        "Logística | SAGA": lambda n: LogisticRegression(C=0.1, max_iter=2000, solver="saga", tol=1e-3).fit(X[:n], y[:n]),
        "Naive Bayes | fit": lambda n: GaussianNB().fit(X[:n], y[:n]),
        "Naive Bayes | partial_fit (lotes de 1000)": lambda n: _parcial(X[:n], y[:n]),
        "XGBoost | exact": lambda n: XGBClassifier(n_estimators=150, max_depth=6, tree_method="exact", n_jobs=1,
                                                   verbosity=0).fit(X[:n], y[:n] - 1),
        "XGBoost | hist": lambda n: XGBClassifier(n_estimators=150, max_depth=6, tree_method="hist", n_jobs=1,
                                                  verbosity=0).fit(X[:n], y[:n] - 1),
        "XGBoost | hist + early stopping": lambda n: XGBClassifier(n_estimators=600, max_depth=6, tree_method="hist",
                                                                   n_jobs=1, verbosity=0, early_stopping_rounds=20).fit(
            X[:n], y[:n] - 1, eval_set=[(Xv[:3000], yv[:3000] - 1)], verbose=False),
        "SVM | núcleo RBF completo (SVC)": lambda n: SVC(C=1.0, gamma="scale").fit(X[:n], y[:n]) if n <= 16000 else None,
        "SVM | Nyström + LinearSVC": lambda n: MD.SVMNystroem(gamma=0.03, n_components=300, C=1.0).fit(X[:n], y[:n]),
        "SVM | SGD (hinge)": lambda n: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-3,
                                                     random_state=CF.SEED).fit(X[:n], y[:n]),
    }
    filas = []
    for nombre, fn in variantes.items():
        for n in TAMANOS:
            t, mem, est = medir(lambda: fn(n), rep=3 if n <= 8000 else 1)
            if est is None:
                continue
            f1 = np.nan
            if hasattr(est, "predict") and not nombre.startswith("KNN"):
                pr = est.predict(Xv)
                pr = pr + 1 if nombre.startswith("XGBoost") else pr
                f1 = f1_score(yv, pr, average="macro", zero_division=0)
            filas.append({"variante": nombre, "n": n, "segundos": t, "memoria_MB": mem, "f1_validacion": f1,
                          "arboles_usados": int(getattr(est, "best_iteration", -1)) + 1 if "early" in nombre else None})
            print(f"{nombre:45s} n={n:6d}: {t:8.3f} s, {mem:7.1f} MB, F1 {f1:.3f}", flush=True)
    exps = {}
    for nombre in variantes:
        sel = [f for f in filas if f["variante"] == nombre]
        exps[nombre] = exponente([f["n"] for f in sel], [f["segundos"] for f in sel])
    # figura de escalamiento
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for nombre in variantes:
        sel = [f for f in filas if f["variante"] == nombre]
        a = ax[0] if nombre.startswith(("KNN", "SVM")) else ax[1]
        a.loglog([f["n"] for f in sel], [f["segundos"] for f in sel], marker="o", label=f"{nombre} (b = {exps[nombre]:.2f})")
    for a in ax:
        a.set_xlabel("viviendas de entrenamiento (n)")
        a.set_ylabel("segundos (escala log)")
        a.legend(fontsize=7)
    ax[0].set_title("KNN y SVM")
    ax[1].set_title("Logística, Naive Bayes y XGBoost")
    plt.savefig(FIG / "escalamiento.png", dpi=130, bbox_inches="tight")
    plt.close()

    # perfil cProfile de una evaluación completa del pipeline (XGBoost + SMOTE)
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    paq["M1"] = d["M1"]
    sub = paq["sub"]
    pr = cProfile.Profile()
    pr.enable()
    MD.pipeline(paq, "XGBoost", {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1}, "smote").fit(
        sub.iloc[:6000], sub.estrato_num.to_numpy()[:6000])
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    perfil = s.getvalue()
    # rutas relativas, sin la carpeta del usuario: «…/site-packages/xgboost/core.py» → «xgboost/core.py»
    perfil = perfil.replace(str(Path(np.__file__).parents[1]) + os.sep, "").replace(str(AQUI.parent) + os.sep, "")

    # aceleración de la validación cruzada con n_jobs (joblib, loky)
    from joblib import Parallel, delayed

    def un_fold(i):
        idx = np.arange(len(sub)) % 8 != i
        return MD.pipeline(paq, "Random Forest", {"n_estimators": 100, "max_depth": 15}, "ninguno").fit(
            sub.iloc[idx], sub.estrato_num.to_numpy()[idx]).score(sub.iloc[~idx], sub.estrato_num.to_numpy()[~idx])

    acel = {}
    for nj in [1, 2, 4, 8]:
        t = time.perf_counter()
        Parallel(n_jobs=nj, backend="loky")(delayed(un_fold)(i) for i in range(8))
        acel[nj] = time.perf_counter() - t
        print(f"n_jobs={nj}: {acel[nj]:.1f} s", flush=True)

    salida = {"p": p, "tamanos": TAMANOS, "mediciones": filas, "exponentes": exps, "perfil_cprofile": perfil,
              "aceleracion_n_jobs": {str(k): v for k, v in acel.items()},
              "gpu": "No disponible: el portátil (Intel Core i5-8265U) no tiene GPU NVIDIA para device='cuda'."}
    (CF.RESULTADOS / "complejidad.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=float),
                                                     encoding="utf-8")
    print("guardado complejidad.json", flush=True)


def _parcial(X, y):
    m = GaussianNB()
    for i in range(0, len(X), 1000):
        m.partial_fit(X[i:i + 1000], y[i:i + 1000], classes=CF.CLASES)
    return m


if __name__ == "__main__":
    main()
