# -*- coding: utf-8 -*-
"""Tabla comparativa estándar frente a optimizado (sección 4.4 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada familia, la versión estándar y la optimizada con las mismas n = 16 000 viviendas de entrenamiento (el
mayor tamaño en el que el SVM con núcleo completo todavía es viable) y las mismas variables preprocesadas: tiempo
de entrenamiento, tiempo de inferencia sobre 10 000 viviendas, memoria pico de Python (``tracemalloc``; FAISS y
XGBoost reservan memoria en C++ que ``tracemalloc`` no ve, así que para ellos se reporta el tamaño del índice o del
modelo serializado) y F1 macro en esas 10 000 viviendas (validación aleatoria, solo para comparar variantes). Cada
tiempo es la mediana de 3 repeticiones, en un hilo.

Salida: ``resultados/tabla_estandar_optimizado.json``.
Uso (desde e3/, con el computador libre):  ../venv_e2/Scripts/python -u experimentos/tabla_estandar_optimizado.py
"""
import json
import pickle
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

warnings.filterwarnings("ignore")
N = 16000
COMPLEJIDAD = {
    "KNN | fuerza bruta": ("O(1)", "O(n·p) por consulta"),
    "KNN | FAISS HNSW": ("≈ O(n·log n)", "≈ O(log n) por consulta (aproximado)"),
    "Logística | lbfgs": ("O(n·p·K) por iteración", "O(p·K)"),
    "Logística | SAGA": ("O(n·p·K) por época", "O(p·K)"),
    "Naive Bayes | fit": ("O(n·p) (todo en memoria)", "O(p·K)"),
    "Naive Bayes | partial_fit": ("O(n·p) por lotes, memoria O(p·K)", "O(p·K)"),
    "XGBoost | exact": ("O(T·n·p·log n)", "O(T·profundidad)"),
    "XGBoost | hist + early stopping": ("O(n·p) + O(T'·bins·p·nodos), T' ≤ T", "O(T'·profundidad)"),
    "SVM | SVC (núcleo RBF completo)": ("O(n²·p) a O(n³)", "O(n_sv·p)"),
    "SVM | Nyström + LinearSVC": ("O(n·m·p + n·m²)", "O(m·p)"),
    "SVM | SGD (hinge)": ("O(n·p) por época", "O(p)"),
}


class KNNFaiss:
    """KNN con índice HNSW de FAISS (vecinos aproximados) y voto mayoritario."""

    def __init__(self, k=30, M=32, ef=64):
        self.k, self.M, self.ef = k, M, ef

    def fit(self, X, y):
        import faiss
        faiss.omp_set_num_threads(1)
        self.idx_ = faiss.IndexHNSWFlat(X.shape[1], self.M)
        self.idx_.hnsw.efSearch = self.ef
        self.idx_.add(np.ascontiguousarray(X, dtype=np.float32))
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        _, I = self.idx_.search(np.ascontiguousarray(X, dtype=np.float32), self.k)
        cuentas = np.stack([(self.y_[I] == c).sum(1) for c in CF.CLASES], 1)
        return CF.CLASES[cuentas.argmax(1)]


def medir(fn, rep=3):
    ts, picos, out = [], [], None
    for _ in range(rep):
        tracemalloc.start()
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
        picos.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
    return float(np.median(ts)), float(max(picos)), out


def tamano_modelo(est):
    """Tamaño serializado del modelo (MB): índice de FAISS, booster de XGBoost o pickle."""
    try:
        import faiss
        if hasattr(est, "idx_"):
            return len(faiss.serialize_index(est.idx_)) / 1e6
    except ImportError:
        pass
    if isinstance(est, XGBClassifier):
        return len(est.get_booster().save_raw()) / 1e6
    return len(pickle.dumps(est)) / 1e6


def main():
    from estrato3 import datos as DT
    CF.fijar_semillas()
    d = DT.cargar()
    tr = d["train"].sample(n=N + 13000, random_state=CF.SEED).reset_index(drop=True)
    prep = MD.preprocesador(d).fit(tr.iloc[:N])
    X = prep.transform(tr)
    y = tr.estrato_num.to_numpy()
    Xtr, ytr, Xva, yva = X[:N], y[:N], X[-10000:], y[-10000:]
    Xes, yes = X[N:N + 3000], y[N:N + 3000]       # conjunto independiente para el early stopping
    variantes = {
        "KNN | fuerza bruta": lambda: KNeighborsClassifier(30, algorithm="brute", n_jobs=1),
        "KNN | FAISS HNSW": lambda: KNNFaiss(30),
        "Logística | lbfgs": lambda: LogisticRegression(C=0.47, max_iter=2000, solver="lbfgs"),
        "Logística | SAGA": lambda: LogisticRegression(C=0.47, max_iter=2000, solver="saga", tol=1e-3),
        "Naive Bayes | fit": lambda: GaussianNB(),
        "Naive Bayes | partial_fit": None,
        "XGBoost | exact": lambda: XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                 tree_method="exact", n_jobs=1, verbosity=0),
        "XGBoost | hist + early stopping": lambda: XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                                                 tree_method="hist", n_jobs=1, verbosity=0,
                                                                 early_stopping_rounds=20),
        "SVM | SVC (núcleo RBF completo)": lambda: SVC(C=10.0, gamma="scale"),
        "SVM | Nyström + LinearSVC": lambda: MD.SVMNystroem(gamma=0.03, n_components=300, C=10.0),
        "SVM | SGD (hinge)": lambda: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-3,
                                                   random_state=CF.SEED),
    }
    filas = []
    for nombre, nuevo in variantes.items():
        xgb = nombre.startswith("XGBoost")
        yt = ytr - 1 if xgb else ytr

        def ajustar():
            if nombre == "Naive Bayes | partial_fit":
                m = GaussianNB()
                for i in range(0, len(Xtr), 1000):
                    m.partial_fit(Xtr[i:i + 1000], ytr[i:i + 1000], classes=CF.CLASES)
                return m
            m = nuevo()
            if "early" in nombre:
                return m.fit(Xtr, yt, eval_set=[(Xes, yes - 1)], verbose=False)
            return m.fit(Xtr, yt)
        t_fit, mem, est = medir(ajustar)
        t_pred, _, pred = medir(lambda: est.predict(Xva))
        pred = pred + 1 if xgb else pred
        f1 = f1_score(yva, pred, average="macro", zero_division=0)
        filas.append({"variante": nombre, "familia": nombre.split(" | ")[0],
                      "complejidad_entrenamiento": COMPLEJIDAD[nombre][0],
                      "complejidad_inferencia": COMPLEJIDAD[nombre][1], "n": N, "seg_entrenamiento": t_fit,
                      "seg_inferencia_10000": t_pred, "memoria_pico_MB": mem, "modelo_MB": tamano_modelo(est),
                      "f1_macro": f1,
                      "arboles": int(getattr(est, "best_iteration", -1)) + 1 if "early" in nombre else None})
        print(f"{nombre:36s} entrenar {t_fit:8.3f} s | inferir {t_pred:7.3f} s | {mem:7.1f} MB | "
              f"modelo {filas[-1]['modelo_MB']:7.2f} MB | F1 {f1:.3f}", flush=True)
    (CF.RESULTADOS / "tabla_estandar_optimizado.json").write_text(
        json.dumps({"n": N, "filas": filas}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("guardado tabla_estandar_optimizado.json", flush=True)


if __name__ == "__main__":
    main()
