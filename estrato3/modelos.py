# -*- coding: utf-8 -*-
"""Catálogo de modelos, espacios de búsqueda y construcción del pipeline con balanceo.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Cada modelo tiene:

- ``espacio``: hiperparámetros con su tipo y rango, usado por Random, Bayesiana y
  Genética. Tipos: ``("log", a, b)`` real en escala logarítmica, ``("float", a, b)``,
  ``("int", a, b)``, ``("intlog", a, b)`` entero en escala logarítmica y
  ``("cat", [opciones])``. Los rangos se justifican en el capítulo de
  optimización del libro (escala log para C, λ, learning rate y número de vecinos).
- ``rejilla``: la rejilla de Grid Search, de tamaño parecido al presupuesto.

El pipeline es preprocesamiento del Entregable 1 (winsorización → imputación +
indicador → log1p → estandarización; one-hot con categorías infrecuentes) →
[SMOTE | ADASYN] → modelo, todo dentro de ``imblearn.pipeline.Pipeline``: el
remuestreo y todos los pasos que aprenden de los datos se ajustan solo con el
entrenamiento de cada pliegue, y el remuestreo no se aplica al predecir.
"""
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import configuracion as CF
from .modelos_propios import GeoXGBSk, XGBEstrato

BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]


# ---------------------------------------------------------------------------
# Envoltorios
# ---------------------------------------------------------------------------
class SVMNystroem(ClassifierMixin, BaseEstimator):
    """SVM con kernel RBF aproximado (Nyström) y LinearSVC.

    El SVM con núcleo completo cuesta O(n²)–O(n³) en entrenamiento; con
    Nyström (m componentes) el costo es O(n·m²) y el problema lineal resultante
    se resuelve con LinearSVC (sección 4.2 de la guía). ``predict_proba``
    devuelve el softmax de la función de decisión: son puntajes, no
    probabilidades calibradas (la calibración se evalúa y corrige aparte).

    Parámetros
    ----------
    gamma : float
        Ancho del kernel RBF, k(x, z) = exp(−γ‖x − z‖²). Valores grandes hacen el kernel más local.
    n_components : int
        Número m de puntos de referencia de Nyström (calidad de la aproximación frente a costo).
    C : float
        Inverso de la regularización del LinearSVC.
    class_weight : {None, "balanced"}
        Pesos de clase del LinearSVC.
    random_state : int
        Semilla de la selección de puntos de Nyström y del LinearSVC.

    Supuestos
    ---------
    Las variables llegan estandarizadas (las escala el preprocesador del pipeline), porque el kernel RBF depende
    de las distancias.
    """

    def __init__(self, gamma=0.1, n_components=500, C=1.0, class_weight=None, random_state=CF.SEED):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        """Ajusta el mapa de Nyström y el LinearSVC sobre ``X`` (n × p) y ``y``; devuelve ``self``."""
        self.mapa_ = Nystroem(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
        Z = self.mapa_.fit_transform(X)
        self.svm_ = LinearSVC(C=self.C, class_weight=self.class_weight, random_state=self.random_state,
                              max_iter=3000).fit(Z, y)
        self.classes_ = self.svm_.classes_
        return self

    def decision_function(self, X):
        """Puntaje de cada clase (n × K): distancia con signo al hiperplano en el espacio de Nyström."""
        return self.svm_.decision_function(self.mapa_.transform(X))

    def predict_proba(self, X):
        """Softmax de ``decision_function`` (n × K): suma 1 por fila, pero no está calibrado."""
        d = self.decision_function(X)
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        """Clase con el mayor puntaje de decisión."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class CorreccionPriors(ClassifierMixin, BaseEstimator):
    """Equivalente de ``class_weight='balanced'`` para modelos que no lo admiten (KNN).

    Divide las probabilidades por la frecuencia de cada clase en el
    entrenamiento y renormaliza: p̃_k ∝ p_k / π_k. Es la corrección bayesiana que
    convierte un clasificador con priors empíricos en uno con priors uniformes,
    el mismo efecto que pesar cada clase por 1/π_k.

    Parámetros
    ----------
    estimador : clasificador de scikit-learn con ``predict_proba``
        Modelo base (aquí, ``KNeighborsClassifier``); se clona al ajustar.

    Supuestos
    ---------
    Las probabilidades del modelo base estiman p(k | x) con los priors del entrenamiento π_k (frecuencias de
    clase), y todas las clases aparecen en el entrenamiento (π_k > 0).
    """

    def __init__(self, estimador=None):
        self.estimador = estimador

    def fit(self, X, y):
        """Ajusta el modelo base y guarda los priors empíricos ``priors_`` de ``y``; devuelve ``self``."""
        self.estimador_ = clone(self.estimador).fit(X, y)
        self.classes_ = self.estimador_.classes_
        self.priors_ = np.array([np.mean(np.asarray(y) == k) for k in self.classes_])
        return self

    def predict_proba(self, X):
        """Probabilidades corregidas p̃_k ∝ p_k / π_k, renormalizadas para sumar 1 (n × K)."""
        p = self.estimador_.predict_proba(X) / self.priors_
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        """Clase con la mayor probabilidad corregida."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ADASYNPorClase(BaseEstimator):
    """ADASYN aplicado estrato por estrato (cada minoritario hasta el tamaño del mayoritario).

    ADASYN reparte las muestras sintéticas según la proporción de vecinos de OTRA clase alrededor de cada
    ejemplo minoritario. Si un estrato no tiene vecinos de otra clase (todos sus vecinos son de su mismo
    estrato, algo frecuente con un estrato tan concentrado), esa densidad es cero e imbalanced-learn falla
    ("No samples will be generated"). Aquí ese estrato se deja sin muestras sintéticas y se registra en
    ``omitidos_``; los demás se sobremuestrean normalmente.

    Parámetros
    ----------
    n_neighbors : int
        Vecinos que usa ADASYN para medir la dificultad de cada ejemplo y para interpolar.
    random_state : int
        Semilla del muestreo sintético.

    Supuestos
    ---------
    Va dentro de un ``imblearn.pipeline.Pipeline``, después del preprocesamiento: se aplica solo al entrenamiento
    de cada pliegue y nunca al predecir.
    """

    def __init__(self, n_neighbors=5, random_state=CF.SEED):
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Devuelve ``(X, y)`` con los ejemplos sintéticos de cada estrato minoritario agregados al final.

        Parámetros
        ----------
        X : array (n, p)
            Variables ya preprocesadas.
        y : array (n,)
            Estratos.

        Devuelve
        --------
        X_res : array (n', p)
        y_res : array (n',)
            n' ≥ n; cada estrato llega al tamaño del mayoritario salvo los de ``omitidos_``.
        """
        X, y = np.asarray(X), np.asarray(y)
        clases, cuenta = np.unique(y, return_counts=True)
        objetivo = cuenta.max()
        Xs, ys = [X], [y]
        self.omitidos_ = []
        for k, n in zip(clases, cuenta):
            if n >= objetivo:
                continue
            try:
                Xr, yr = ADASYN(sampling_strategy={k: objetivo}, n_neighbors=self.n_neighbors,
                                random_state=self.random_state).fit_resample(X, y)
                Xs.append(Xr[len(X):])
                ys.append(yr[len(y):])
            except (ValueError, RuntimeError):
                self.omitidos_.append(int(k))
        return np.vstack(Xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------
CATALOGO = {
    "KNN": {
        "espacio": {"n_neighbors": ("intlog", 5, 200), "weights": ("cat", ["uniform", "distance"]),
                    "p": ("cat", [1, 2])},
        "rejilla": {"n_neighbors": [10, 30, 100], "weights": ["uniform", "distance"], "p": [1, 2]},
        "familia": "vecinos"},
    "Naive Bayes": {
        "espacio": {"var_smoothing": ("log", 1e-11, 1e-1)},
        "rejilla": {"var_smoothing": list(np.logspace(-11, -1, 12))},
        "familia": "probabilistico"},
    "Logística L1/L2": {
        "espacio": {"penalty": ("cat", ["l1", "l2"]), "C": ("log", 1e-3, 10.0)},
        "rejilla": {"penalty": ["l1", "l2"], "C": list(np.logspace(-3, 1, 6))},
        "familia": "lineal"},
    "Árbol de decisión": {
        "espacio": {"max_depth": ("int", 3, 30), "min_samples_leaf": ("intlog", 1, 300),
                    "criterion": ("cat", ["gini", "entropy"])},
        "rejilla": {"max_depth": [5, 10, 20], "min_samples_leaf": [5, 50], "criterion": ["gini", "entropy"]},
        "familia": "arbol"},
    "Random Forest": {
        "espacio": {"n_estimators": ("intlog", 50, 250), "max_depth": ("int", 6, 25),
                    "max_features": ("cat", ["sqrt", 0.3, 0.6]), "min_samples_leaf": ("intlog", 1, 50)},
        "rejilla": {"n_estimators": [150], "max_depth": [10, 20], "max_features": ["sqrt", 0.3, 0.6],
                    "min_samples_leaf": [2, 10]},
        "familia": "ensamble"},
    "XGBoost": {
        "espacio": {"n_estimators": ("intlog", 50, 300), "learning_rate": ("log", 0.03, 0.3),
                    "max_depth": ("int", 3, 8), "subsample": ("float", 0.6, 1.0),
                    "colsample_bytree": ("float", 0.5, 1.0), "min_child_weight": ("log", 1.0, 20.0)},
        "rejilla": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.15], "max_depth": [3, 5, 8]},
        "familia": "ensamble"},
    "SVM": {
        "espacio": {"gamma": ("log", 1e-3, 1.0), "n_components": ("intlog", 100, 500), "C": ("log", 1e-3, 10.0)},
        "rejilla": {"gamma": [0.003, 0.03, 0.3], "n_components": [300], "C": [0.01, 0.1, 1.0, 10.0]},
        "familia": "kernel"},
    "XGBoost geográfico": {
        "espacio": {"n_estimators": ("intlog", 50, 200), "learning_rate": ("log", 0.03, 0.3),
                    "max_depth": ("int", 3, 8), "frac_vecinos": ("log", 0.02, 0.3), "alpha": ("float", 0.0, 1.0)},
        "rejilla": {"n_estimators": [150], "max_depth": [4, 7], "frac_vecinos": [0.05, 0.15],
                    "alpha": [0.0, 0.33, 0.67]},
        "familia": "propio"},
}
MODELOS_GUIA = ["KNN", "Naive Bayes", "Logística L1/L2", "Árbol de decisión", "Random Forest", "XGBoost", "SVM"]


def estimador(nombre, params, balanceo, seed=CF.SEED):
    """Modelo final (último paso del pipeline) con los hiperparámetros y el balanceo pedidos.

    ``balanceo == "class_weight"`` usa ``class_weight='balanced'`` donde existe;
    XGBoost y el geográfico usan pesos de muestra 1/frecuencia; Naive Bayes usa
    priors uniformes y KNN la corrección de priors (``CorreccionPriors``).

    Parámetros
    ----------
    nombre : str
        Clave de ``CATALOGO`` (uno de los 8 modelos de clasificación).
    params : dict
        Hiperparámetros del modelo (los que propone el optimizador).
    balanceo : str
        Una de ``BALANCEOS``. SMOTE y ADASYN no cambian el modelo: se agregan como paso del pipeline.
    seed : int
        Semilla del modelo (los que tienen aleatoriedad).

    Devuelve
    --------
    estimador de scikit-learn sin ajustar.

    Supuestos
    ---------
    Cada modelo usa un solo hilo (``n_jobs=1``): el paralelismo va entre tareas, no dentro del modelo.
    """
    cw = balanceo == "class_weight"
    p = dict(params)
    if nombre == "KNN":
        m = KNeighborsClassifier(n_jobs=1, **p)
        return CorreccionPriors(m) if cw else m
    if nombre == "Naive Bayes":
        return GaussianNB(priors=np.full(len(CF.CLASES), 1 / len(CF.CLASES)) if cw else None, **p)
    if nombre == "Logística L1/L2":
        return LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, class_weight="balanced" if cw else None,
                                  random_state=seed, **p)
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(class_weight="balanced" if cw else None, random_state=seed, **p)
    if nombre == "Random Forest":
        return RandomForestClassifier(class_weight="balanced" if cw else None, n_jobs=1, random_state=seed, **p)
    if nombre == "XGBoost":
        return XGBEstrato(balanceado=cw, n_jobs=1, random_state=seed, **p)
    if nombre == "SVM":
        return SVMNystroem(class_weight="balanced" if cw else None, random_state=seed, **p)
    if nombre == "XGBoost geográfico":
        return GeoXGBSk(balanceado=cw, random_state=seed, **p)
    raise ValueError(nombre)


def preprocesador(datos, con_xy=False):
    """Preprocesador del Entregable 1 (y, para el modelo geográfico, x_km e y_km sin escalar al final).

    Parámetros
    ----------
    datos : dict
        Paquete de ``datos.cargar()`` o de ``resultados/datos_clasificacion.pkl``: listas ``num``, ``cat`` y
        ``num_log``, decisiones ``dec`` del Entregable 1 y el módulo ``M1`` (``src/modelo.py`` del E1).
    con_xy : bool
        Si es verdadero, agrega al final las coordenadas x_km e y_km sin transformar, que el XGBoost geográfico
        usa para ubicar las anclas.

    Devuelve
    --------
    ColumnTransformer sin ajustar: winsorización (cuantiles del E1) → imputación por mediana con indicador de
    faltante → log1p (solo ``num_log``) → estandarización en las numéricas; imputación y one-hot con categorías
    infrecuentes en las categóricas.

    Supuestos
    ---------
    Se ajusta dentro de cada pliegue (forma parte del pipeline), así que no hay fuga de información.
    """
    num, cat, num_log = datos["num"], datos["cat"], datos["num_log"]
    q_inf, q_sup = datos["dec"]["winsorizacion_cuantiles"]["valor"]
    prep = datos["M1"].construir_preprocesador([c for c in num if c in num_log], [c for c in num if c not in num_log],
                                               cat, q_inf=q_inf, q_sup=q_sup,
                                               min_frecuencia=datos["dec"]["onehot_min_frecuencia"]["valor"])
    if not con_xy:
        return prep
    return ColumnTransformer([("base", prep, num + cat), ("xy", "passthrough", ["x_km", "y_km"])])


def pipeline(datos, nombre, params, balanceo, seed=CF.SEED):
    """Pipeline completo: preprocesamiento → [SMOTE | ADASYN] → modelo.

    Parámetros
    ----------
    datos : dict
        Salida de ``datos.cargar()``.
    nombre : str
        Clave de ``CATALOGO``.
    params : dict
        Hiperparámetros del modelo.
    balanceo : str
        Una de ``BALANCEOS``.
    seed : int
        Semilla del muestreo sintético y del modelo.

    Devuelve
    --------
    imblearn.pipeline.Pipeline
    """
    pasos = [("prep", preprocesador(datos, con_xy=nombre == "XGBoost geográfico"))]
    if balanceo == "smote":
        pasos.append(("muestreo", SMOTE(k_neighbors=5, random_state=seed)))
    elif balanceo == "adasyn":
        pasos.append(("muestreo", ADASYNPorClase(n_neighbors=5, random_state=seed)))
    pasos.append(("modelo", estimador(nombre, params, balanceo, seed)))
    return Pipeline(pasos)


def columnas(datos):
    """Columnas de entrada del pipeline (el geográfico usa además x_km, y_km, que ya están en ``num``).

    Parámetros
    ----------
    datos : dict
        Paquete de datos con las listas ``num`` y ``cat``.

    Devuelve
    --------
    list of str
        ``num + cat``.
    """
    return datos["num"] + datos["cat"]


# ---------------------------------------------------------------------------
# Regresión: el estrato (1-6) tratado como número
# ---------------------------------------------------------------------------
class SVRNystroem(BaseEstimator):
    """SVR con kernel RBF aproximado (Nyström) y LinearSVR: costo O(n·m²) en vez de O(n²)–O(n³).

    Parámetros
    ----------
    gamma : float
        Ancho del kernel RBF.
    n_components : int
        Puntos de referencia de Nyström.
    C : float
        Inverso de la regularización.
    epsilon : float
        Ancho del tubo insensible de la pérdida ε-insensible (en estratos).
    random_state : int
        Semilla de Nyström y del LinearSVR.

    Supuestos
    ---------
    Variables estandarizadas (las escala el preprocesador del pipeline).
    """

    def __init__(self, gamma=0.1, n_components=300, C=1.0, epsilon=0.1, random_state=CF.SEED):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X, y):
        """Ajusta el mapa de Nyström y el LinearSVR sobre ``X`` y el estrato numérico ``y``; devuelve ``self``."""
        from sklearn.svm import LinearSVR
        self.mapa_ = Nystroem(gamma=self.gamma, n_components=self.n_components, random_state=self.random_state)
        self.svr_ = LinearSVR(C=self.C, epsilon=self.epsilon, random_state=self.random_state, max_iter=5000,
                              dual="auto").fit(self.mapa_.fit_transform(X), y)
        return self

    def predict(self, X):
        """Estrato predicho como número real (n,)."""
        return self.svr_.predict(self.mapa_.transform(X))


CATALOGO_REG = {
    "KNN (regresión)": {
        "espacio": {"n_neighbors": ("intlog", 5, 200), "weights": ("cat", ["uniform", "distance"]), "p": ("cat", [1, 2])},
        "rejilla": {"n_neighbors": [10, 30, 100], "weights": ["uniform", "distance"], "p": [1, 2]}},
    "Ridge": {
        "espacio": {"alpha": ("log", 1e-3, 1e3)},
        "rejilla": {"alpha": list(np.logspace(-3, 3, 12))}},
    "Lasso": {
        "espacio": {"alpha": ("log", 1e-5, 1.0)},
        "rejilla": {"alpha": list(np.logspace(-5, 0, 12))}},
    "Árbol (regresión)": {
        "espacio": {"max_depth": ("int", 3, 30), "min_samples_leaf": ("intlog", 1, 300)},
        "rejilla": {"max_depth": [5, 10, 15, 20], "min_samples_leaf": [5, 30, 150]}},
    "Random Forest (regresión)": {
        "espacio": {"n_estimators": ("intlog", 50, 250), "max_depth": ("int", 6, 25),
                    "max_features": ("cat", ["sqrt", 0.3, 0.6]), "min_samples_leaf": ("intlog", 1, 50)},
        "rejilla": {"n_estimators": [150], "max_depth": [10, 20], "max_features": ["sqrt", 0.3, 0.6],
                    "min_samples_leaf": [2, 10]}},
    "XGBoost (regresión)": {
        "espacio": {"n_estimators": ("intlog", 50, 300), "learning_rate": ("log", 0.03, 0.3),
                    "max_depth": ("int", 3, 8), "subsample": ("float", 0.6, 1.0),
                    "colsample_bytree": ("float", 0.5, 1.0), "min_child_weight": ("log", 1.0, 20.0)},
        "rejilla": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.15], "max_depth": [3, 5, 8]}},
    "SVR": {
        "espacio": {"gamma": ("log", 1e-3, 1.0), "n_components": ("intlog", 100, 500), "C": ("log", 1e-3, 10.0),
                    "epsilon": ("float", 0.0, 0.5)},
        "rejilla": {"gamma": [0.003, 0.03, 0.3], "n_components": [300], "C": [0.01, 0.1, 1.0, 10.0], "epsilon": [0.1]}},
}
MODELOS_REG = list(CATALOGO_REG)


def catalogo(nombre):
    """Entrada del catálogo (clasificación o regresión) de un modelo.

    Parámetros
    ----------
    nombre : str
        Clave de ``CATALOGO`` o de ``CATALOGO_REG``.

    Devuelve
    --------
    dict
        Con ``espacio`` (rangos para Random, Bayesiana y Genética) y ``rejilla`` (Grid Search).
    """
    return CATALOGO[nombre] if nombre in CATALOGO else CATALOGO_REG[nombre]


def estimador_reg(nombre, params, seed=CF.SEED):
    """Regresor final con sus hiperparámetros.

    Parámetros
    ----------
    nombre : str
        Clave de ``CATALOGO_REG`` (uno de los 7 modelos de regresión).
    params : dict
        Hiperparámetros del modelo.
    seed : int
        Semilla de los modelos con aleatoriedad.

    Devuelve
    --------
    regresor de scikit-learn (o compatible) sin ajustar, con un solo hilo.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    p = dict(params)
    if nombre == "KNN (regresión)":
        return KNeighborsRegressor(n_jobs=1, **p)
    if nombre == "Ridge":
        return Ridge(solver="saga", max_iter=5000, tol=1e-4, random_state=seed, **p)
    if nombre == "Lasso":
        return Lasso(selection="random", max_iter=5000, tol=1e-4, random_state=seed, **p)
    if nombre == "Árbol (regresión)":
        return DecisionTreeRegressor(random_state=seed, **p)
    if nombre == "Random Forest (regresión)":
        return RandomForestRegressor(n_jobs=1, random_state=seed, **p)
    if nombre == "XGBoost (regresión)":
        return XGBRegressor(tree_method="hist", n_jobs=1, random_state=seed, verbosity=0, **p)
    if nombre == "SVR":
        return SVRNystroem(random_state=seed, **p)
    raise ValueError(nombre)


def pipeline_reg(datos, nombre, params, seed=CF.SEED):
    """Pipeline de regresión: preprocesamiento del Entregable 1 → regresor (sin balanceo: no aplica).

    Parámetros
    ----------
    datos : dict
        Paquete de datos (ver ``preprocesador``).
    nombre : str
        Clave de ``CATALOGO_REG``.
    params : dict
        Hiperparámetros del regresor.
    seed : int
        Semilla del regresor.

    Devuelve
    --------
    imblearn.pipeline.Pipeline
        Pasos ``prep`` y ``modelo``; el objetivo es el estrato (1–6) como número.
    """
    return Pipeline([("prep", preprocesador(datos)), ("modelo", estimador_reg(nombre, params, seed))])
