# -*- coding: utf-8 -*-
"""Modelos propios del proyecto (copiados del Entregable 2, ``avanzados.py``) y envoltorios.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

- ``XGBEstrato``: XGBoost multiclase con pesos de clase opcionales.
- ``GeoXGBEstrato``: XGBoost geográfico adaptado a CLASIFICACIÓN (Grekousis, 2025, propuso la versión de
  regresión): un XGBoost global más XGBoost locales alrededor de anclas con kernel espacial adaptativo; la
  predicción mezcla p = alpha·p_global + (1 − alpha)·p_local. Es el modelo nuevo del proyecto.
- ``GeoXGBSk``: el mismo modelo con la interfaz estándar ``fit(X, y)``: las dos últimas columnas de X son las
  coordenadas sin escalar (x_km, y_km), para poder usarlo dentro de un Pipeline y de las búsquedas.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

CLASES = np.arange(1, 7)


def _proba_completa(est, X, clases_est):
    """Probabilidades en las 6 columnas de CLASES aunque el modelo no haya visto alguna clase."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(CLASES)))
    for j, k in enumerate(clases_est):
        out[:, int(k) - 1] = p[:, j]
    return out


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
class XGBEstrato(ClassifierMixin, BaseEstimator):
    """XGBoost multiclase con pesos de clase opcionales.

    Parámetros principales: ``max_depth``, ``n_estimators``, ``learning_rate``
    y ``balanceado`` (si True, cada vivienda pesa 1/frecuencia de su clase).
    """

    def __init__(self, max_depth=6, n_estimators=300, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, balanceado=False, n_jobs=2, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.balanceado = balanceado
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _nuevo(self, n_clases):
        return XGBClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate, subsample=self.subsample,
                             colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                             tree_method="hist", n_jobs=self.n_jobs, random_state=self.random_state,
                             objective="multi:softprob" if n_clases > 2 else "binary:logistic", verbosity=0)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        mapa = {k: i for i, k in enumerate(self.classes_)}
        w = compute_sample_weight("balanced", y) if self.balanceado else np.ones(len(y))
        if sample_weight is not None:
            w = w * sample_weight
        self.modelo_ = self._nuevo(len(self.classes_)).fit(X, np.vectorize(mapa.get)(y), sample_weight=w)
        return self

    def predict_proba(self, X):
        p = self.modelo_.predict_proba(X)
        if p.ndim == 1 or p.shape[1] == 1:
            p = np.c_[1 - p.ravel(), p.ravel()]
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _Constante:
    """Modelo local trivial para zonas donde el entrenamiento tiene un solo estrato."""

    def __init__(self, clases):
        self.classes_ = np.asarray(clases)

    def predict_proba(self, X):
        return np.ones((len(X), 1))


# ---------------------------------------------------------------------------
# XGBoost geográfico para clasificación
# ---------------------------------------------------------------------------
class GeoXGBEstrato(ClassifierMixin, BaseEstimator):
    """XGBoost geográfico adaptado a clasificación multiclase.

    Idea (Grekousis, 2025, para regresión): además de un modelo global, se
    entrenan modelos LOCALES en los que cada observación pesa según su
    distancia al punto local mediante un kernel espacial; la predicción final
    combina ambos. Adaptación propuesta aquí:

    1. Modelo global: ``XGBEstrato`` sobre todo el entrenamiento.
    2. Anclas: centroides de las celdas de ``tam_ancla_km`` con al menos
       ``min_ancla`` viviendas de entrenamiento.
    3. Ancho de banda ADAPTATIVO: para cada ancla, h = distancia a su vecino
       de entrenamiento número k, con k = ``frac_vecinos`` × n (o ``k_vecinos`` fijo); en zonas densas la
       ventana es pequeña.
    4. Kernel bicuadrado: w = (1 - (d/h)^2)^2 si d < h, 0 si no. Cada ancla
       entrena un XGBoost local (más pequeño) con esos pesos.
    5. Predicción de una vivienda: promedio de las probabilidades de las anclas
       que la cubren, ponderado por el mismo kernel; si ninguna la cubre se usa
       solo el global. Mezcla final: p = alpha·p_global + (1 − alpha)·p_local.

    Requiere que las columnas ``col_x`` y ``col_y`` (km) de X sean las
    coordenadas SIN escalar; por eso recibe aparte ``xy`` en fit y predict.
    """

    def __init__(self, params_global=None, params_local=None, k_vecinos=8000, frac_vecinos=None,
                 tam_ancla_km=2.0, min_ancla=300, alpha=0.5, random_state=42):
        self.params_global = params_global
        self.params_local = params_local
        self.k_vecinos = k_vecinos
        self.frac_vecinos = frac_vecinos
        self.tam_ancla_km = tam_ancla_km
        self.min_ancla = min_ancla
        self.alpha = alpha
        self.random_state = random_state

    @staticmethod
    def _kernel(d, h):
        u = d / h
        return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)

    def fit(self, X, y, xy):
        y = np.asarray(y)
        xy = np.asarray(xy, float)
        self.classes_ = CLASES
        self.global_ = XGBEstrato(**(self.params_global or {})).fit(X, y)
        # Anclas: centroide de las viviendas de cada celda con suficiente entrenamiento.
        celda = np.floor(xy / self.tam_ancla_km).astype(int)
        _, inv, cuenta = np.unique(celda, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        anclas = np.array([xy[inv == g].mean(0) for g in np.where(cuenta >= self.min_ancla)[0]])
        arbol = cKDTree(xy)
        # Con frac_vecinos la ventana es una FRACCIÓN del entrenamiento: así se traslada igual de la CV
        # (folds más pequeños) al modelo final (todo el entrenamiento).
        k = int(self.frac_vecinos * len(xy)) if self.frac_vecinos else self.k_vecinos
        k = min(k, len(xy))
        self.k_usado_ = k
        d_k, _ = arbol.query(anclas, k=[k])
        self.anclas_, self.h_ = anclas, d_k.ravel()
        self.locales_ = []
        pl = dict(max_depth=4, n_estimators=150, learning_rate=0.1)
        pl.update(self.params_local or {})
        for a, h in zip(self.anclas_, self.h_):
            idx = np.array(arbol.query_ball_point(a, r=h))
            w = self._kernel(np.linalg.norm(xy[idx] - a, axis=1), h)
            sel = w > 0
            if len(np.unique(y[idx][sel])) < 2:
                # Zona con un solo estrato: el modelo local es esa clase con probabilidad 1.
                self.locales_.append(_Constante(np.unique(y[idx][sel])))
                continue
            m = XGBEstrato(**pl, random_state=self.random_state).fit(X[idx][sel], y[idx][sel],
                                                                    sample_weight=w[sel])
            self.locales_.append(m)
        return self

    def proba_partes(self, X, xy):
        """Devuelve (p_global, p_local, cubierto): útil para afinar alpha sin reentrenar."""
        xy = np.asarray(xy, float)
        pg = _proba_completa(self.global_, X, self.global_.classes_)
        num = np.zeros_like(pg)
        den = np.zeros(len(X))
        for a, h, m in zip(self.anclas_, self.h_, self.locales_):
            w = self._kernel(np.linalg.norm(xy - a, axis=1), h)
            sel = w > 0
            if sel.any():
                num[sel] += w[sel, None] * _proba_completa(m, X[sel], m.classes_)
                den[sel] += w[sel]
        cubierto = den > 0
        pl = pg.copy()
        pl[cubierto] = num[cubierto] / den[cubierto, None]
        return pg, pl, cubierto

    def predict_proba(self, X, xy):
        pg, pl, _ = self.proba_partes(X, xy)
        return self.alpha * pg + (1 - self.alpha) * pl

    def predict(self, X, xy):
        return CLASES[np.argmax(self.predict_proba(X, xy), axis=1)]


class GeoXGBSk(ClassifierMixin, BaseEstimator):
    """``GeoXGBEstrato`` con interfaz estándar: las dos últimas columnas de X son x_km, y_km sin escalar.

    Parámetros
    ----------
    max_depth, n_estimators, learning_rate : hiperparámetros del XGBoost global.
    frac_vecinos : float
        Ancho de banda adaptativo como fracción del entrenamiento (k = frac·n vecinos por ancla).
    alpha : float
        Peso del modelo global en la mezcla (1 = XGBoost global puro).
    tam_ancla_km, min_ancla : rejilla de anclas y mínimo de viviendas por ancla.
    balanceado : bool
        Pesos 1/frecuencia de clase en el global y en los locales (equivale a class_weight='balanced').
    """

    def __init__(self, max_depth=6, n_estimators=300, learning_rate=0.1, frac_vecinos=0.1, alpha=0.5,
                 tam_ancla_km=3.0, min_ancla=100, balanceado=False, random_state=42):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.frac_vecinos = frac_vecinos
        self.alpha = alpha
        self.tam_ancla_km = tam_ancla_km
        self.min_ancla = min_ancla
        self.balanceado = balanceado
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        pg = dict(max_depth=self.max_depth, n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                  balanceado=self.balanceado, n_jobs=1, random_state=self.random_state)
        self.modelo_ = GeoXGBEstrato(params_global=pg, params_local=dict(balanceado=self.balanceado, n_jobs=1, n_estimators=80, max_depth=4),
                                     frac_vecinos=self.frac_vecinos, tam_ancla_km=self.tam_ancla_km,
                                     min_ancla=self.min_ancla, alpha=self.alpha,
                                     random_state=self.random_state).fit(X[:, :-2], y, X[:, -2:])
        self.classes_ = CLASES
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return self.modelo_.predict_proba(X[:, :-2], X[:, -2:])

    def predict(self, X):
        return CLASES[np.argmax(self.predict_proba(X), axis=1)]
