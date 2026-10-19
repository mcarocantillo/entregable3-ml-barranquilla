# -*- coding: utf-8 -*-
"""Modelos de la revisión bibliográfica (Tabla III) que no se habían probado en el estrato.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Todas las funciones entrenan y predicen en un solo paso (``*_proba(Xtr, ytr, Xev, ...)``) y descartan los árboles
a medida que predicen: con 207 000 viviendas por pliegue, guardar cientos de árboles completos no cabe en la memoria
del portátil.

- ``rf_proba``: Random Forest entrenado por lotes de árboles (equivale a un bosque del tamaño total con semillas
  distintas por lote).
- ``rotation_forest_proba``: Rotation Forest (Rodríguez, Kuncheva y Alonso, 2006; candidato 5 de la revisión,
  Sartono et al., 2018). Para cada árbol: las variables se reparten al azar en grupos de M; en cada grupo se toma un
  subconjunto aleatorio de clases y el 75 % de sus filas, se hace PCA y sus ejes forman un bloque de la matriz de
  rotación; el árbol se entrena con todas las filas rotadas. La rotación ataca la colinealidad entre área, baños,
  habitaciones y sus promedios de vecindad.
- ``rf_svm_proba``: ensamble en dos etapas (candidato 4, Zhai et al., 2026): un Random Forest ordena las variables
  por importancia y una SVM (Nyström + LinearSVC) clasifica con las k más importantes.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from estrato3.modelos import SVMNystroem
from estrato3.modelos_propios import _proba_completa


def rf_proba(Xtr, ytr, Xev, n_estimators=150, lote=25, seed=42, n_jobs=2, devolver_importancia=False, **params):
    """Random Forest por lotes: promedio de las probabilidades de ``n_estimators`` árboles.

    Devuelve las probabilidades (n_ev, 6) y, si se pide, la importancia media por impureza de cada variable.
    """
    suma = np.zeros((len(Xev), 6))
    imp = np.zeros(Xtr.shape[1])
    hechos = 0
    i = 0
    while hechos < n_estimators:
        n = min(lote, n_estimators - hechos)
        m = RandomForestClassifier(n_estimators=n, random_state=seed + i, n_jobs=n_jobs, **params).fit(Xtr, ytr)
        suma += n * _proba_completa(m, Xev, m.classes_)
        imp += n * m.feature_importances_
        hechos += n
        i += 1
        del m
    p = suma / n_estimators
    return (p, imp / n_estimators) if devolver_importancia else p


def _rotacion(Xtr, ytr, M, rng, max_filas=20000):
    """Matriz de rotación de un árbol de Rotation Forest (bloques de PCA sobre grupos de M variables)."""
    p = Xtr.shape[1]
    R = np.zeros((p, p))
    perm = rng.permutation(p)
    clases = np.unique(ytr)
    for g in range(0, p, M):
        cols = perm[g:g + M]
        sub_cl = clases[rng.random(len(clases)) < 0.5]
        if len(sub_cl) == 0:
            sub_cl = clases[[rng.integers(len(clases))]]
        filas = np.where(np.isin(ytr, sub_cl))[0]
        n = min(max(int(0.75 * len(filas)), len(cols) + 1), max_filas)
        filas = rng.choice(filas, size=min(n, len(filas)), replace=False)
        A = Xtr[np.ix_(filas, cols)].astype(float)
        A = A - A.mean(0)
        # ejes principales (todas las componentes): si una variable es constante en la muestra, sigue siendo base
        _, _, vt = np.linalg.svd(A, full_matrices=False)   # vt es M × M (hay más filas que columnas)
        R[np.ix_(cols, cols)] = vt.T
    return R


def rotation_forest_proba(Xtr, ytr, Xev, n_arboles=50, M=3, min_samples_leaf=5, class_weight=None, seed=42):
    """Rotation Forest: promedio de las probabilidades de ``n_arboles`` árboles, cada uno con su rotación."""
    rng = np.random.default_rng(seed)
    suma = np.zeros((len(Xev), 6))
    for t in range(n_arboles):
        R = _rotacion(Xtr, ytr, M, rng)
        arbol = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                       random_state=seed + t).fit(Xtr @ R, ytr)
        suma += _proba_completa(arbol, Xev @ R, arbol.classes_)
    return suma / n_arboles


def rf_svm_proba(Xtr, ytr, Xev, orden_importancia, k=20, gamma=0.03, n_components=300, C=10.0,
                 class_weight="balanced", seed=42):
    """Dos etapas: las ``k`` variables más importantes según el Random Forest y una SVM Nyström sobre ellas.

    ``predict_proba`` de la SVM es el softmax de su función de decisión (puntajes, no probabilidades calibradas).
    """
    cols = np.asarray(orden_importancia)[:k]
    m = SVMNystroem(gamma=gamma, n_components=n_components, C=C, class_weight=class_weight,
                    random_state=seed).fit(Xtr[:, cols], ytr)
    return _proba_completa(m, Xev[:, cols], m.classes_)
