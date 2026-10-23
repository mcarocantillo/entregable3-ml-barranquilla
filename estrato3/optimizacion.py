# -*- coding: utf-8 -*-
"""Optimización de hiperparámetros: Grid, Random, Bayesiana (Optuna/TPE) y Genética (DEAP).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Las cuatro búsquedas comparten la misma función objetivo (``Objetivo``): el F1
macro medio en los pliegues INTERNOS de la validación anidada, ajustando el
pipeline completo (preprocesamiento + remuestreo + modelo) dentro de cada
pliegue, de modo que no hay fuga de información. Todas registran cada
evaluación (configuración, puntaje, tiempo) para las curvas de desempeño en
cualquier momento (*anytime performance*) y la comparación por presupuesto.

- **Grid Search**: recorre la rejilla del catálogo (de tamaño parecido al presupuesto).
- **Random Search**: ``PRESUPUESTO`` configuraciones al azar del espacio (Bergstra y Bengio, 2012).
- **Bayesiana (Optuna)**: muestreador TPE (*Tree-structured Parzen Estimator*). El modelo sustituto no es un
  proceso gaussiano: TPE modela por separado la densidad de las configuraciones buenas, l(x), y la de las
  malas, g(x) (corte en el cuantil γ del puntaje), y propone la configuración que maximiza l(x)/g(x), que es
  equivalente a maximizar la mejora esperada (*Expected Improvement*; Bergstra et al., 2011). Las primeras
  ``n_startup`` (4) evaluaciones son aleatorias (exploración) y luego domina la explotación.
- **Genética (DEAP)**: cada individuo es un vector en [0, 1]^d que se traduce al espacio de búsqueda. Selección
  por torneo (tamaño 3), cruce *blend* (α = 0.5, probabilidad 0.7), mutación gaussiana (σ = 0.15, probabilidad
  por gen 1/d, probabilidad por individuo 0.3), elitismo de 1 individuo y población de 4 (con 12 evaluaciones
  alcanza para unas 3 generaciones). Se registra la
  diversidad genética (distancia media entre individuos) por generación como indicador de convergencia
  prematura.
"""
import itertools
import math
import time
import warnings

import numpy as np
import optuna
from deap import base, creator, tools

from . import configuracion as CF
from . import modelos as MD

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Espacios
# ---------------------------------------------------------------------------
def de_unitario(espacio, u):
    """Traduce un vector u ∈ [0, 1]^d a una configuración del espacio (usado por Random y Genética).

    Parámetros
    ----------
    espacio : dict
        ``{hiperparámetro: (tipo, ...)}`` con tipo ``log``, ``float``, ``int``, ``intlog`` o ``cat``.
    u : sequence of float
        Un valor por hiperparámetro, en el mismo orden del espacio; se recorta a [0, 1].

    Devuelve
    --------
    dict
        Configuración: interpolación lineal (``float``, ``int``) o logarítmica (``log``, ``intlog``) entre los
        extremos, y para ``cat`` la opción del intervalo en que cae u.
    """
    out = {}
    for (nombre, spec), v in zip(espacio.items(), u):
        v = float(np.clip(v, 0, 1))
        tipo = spec[0]
        if tipo == "log":
            out[nombre] = float(math.exp(math.log(spec[1]) + v * (math.log(spec[2]) - math.log(spec[1]))))
        elif tipo == "float":
            out[nombre] = float(spec[1] + v * (spec[2] - spec[1]))
        elif tipo == "int":
            out[nombre] = int(round(spec[1] + v * (spec[2] - spec[1])))
        elif tipo == "intlog":
            out[nombre] = int(round(math.exp(math.log(spec[1]) + v * (math.log(spec[2]) - math.log(spec[1])))))
        elif tipo == "cat":
            out[nombre] = spec[1][min(int(v * len(spec[1])), len(spec[1]) - 1)]
    return out


def sugerir(trial, espacio):
    """Traduce el espacio a las llamadas ``trial.suggest_*`` de Optuna.

    Parámetros
    ----------
    trial : optuna.trial.Trial
        Prueba en curso del estudio.
    espacio : dict
        Espacio de búsqueda (mismo formato que en ``de_unitario``).

    Devuelve
    --------
    dict
        Configuración propuesta por el muestreador TPE (escala log donde el espacio lo indica).
    """
    out = {}
    for nombre, spec in espacio.items():
        tipo = spec[0]
        if tipo == "log":
            out[nombre] = trial.suggest_float(nombre, spec[1], spec[2], log=True)
        elif tipo == "float":
            out[nombre] = trial.suggest_float(nombre, spec[1], spec[2])
        elif tipo == "int":
            out[nombre] = trial.suggest_int(nombre, spec[1], spec[2])
        elif tipo == "intlog":
            out[nombre] = trial.suggest_int(nombre, spec[1], spec[2], log=True)
        elif tipo == "cat":
            out[nombre] = trial.suggest_categorical(nombre, spec[1])
    return out


def clave(params):
    """Clave hashable de una configuración (para no evaluar dos veces la misma).

    Parámetros
    ----------
    params : dict
        Configuración de hiperparámetros.

    Devuelve
    --------
    tuple
        Pares ``(nombre, valor)`` ordenados, con los reales redondeados a 10 decimales.
    """
    return tuple(sorted((k, round(v, 10) if isinstance(v, float) else v) for k, v in params.items()))


# ---------------------------------------------------------------------------
# Objetivo
# ---------------------------------------------------------------------------
class Objetivo:
    """F1 macro medio en los pliegues internos, con el pipeline completo dentro de cada pliegue.

    Parámetros
    ----------
    datos : dict
        Salida de ``datos.cargar()`` (se usan las columnas y el preprocesador).
    X, y : DataFrame y array del entrenamiento del pliegue EXTERNO.
    folds : list
        Pliegues internos (índices posicionales sobre X).
    nombre, balanceo : str
        Modelo y técnica de balanceo (``"no aplica"`` en regresión).
    seed : int
        Semilla del pipeline en cada evaluación.
    tarea : {"clasificacion", "regresion"}
        En regresión el puntaje es −RMSE (más alto es mejor).

    Devuelve
    --------
    Al llamarse con una configuración, el puntaje medio en los pliegues internos (float). Una configuración que
    falla recibe el peor puntaje (0 o −10) y su error queda en el historial; una configuración repetida se
    devuelve de la memoria sin volver a entrenar.

    Supuestos
    ---------
    Los pliegues internos son índices posicionales sobre ``X`` y se construyeron solo con el entrenamiento del
    pliegue externo (kNNDM), así que la búsqueda nunca ve la validación externa.
    """

    def __init__(self, datos, X, y, folds, nombre, balanceo, seed=CF.SEED, tarea="clasificacion"):
        self.datos, self.X, self.y, self.folds = datos, X, y, folds
        self.nombre, self.balanceo, self.seed, self.tarea = nombre, balanceo, seed, tarea
        self.historial = []
        self._cache = {}
        self._t0 = time.perf_counter()

    def __call__(self, params):
        k = clave(params)
        if k in self._cache:
            return self._cache[k]
        t = time.perf_counter()
        puntajes = []
        error = None
        for itr, iva in self.folds:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.tarea == "regresion":
                        est = MD.pipeline_reg(self.datos, self.nombre, params, self.seed)
                    else:
                        est = MD.pipeline(self.datos, self.nombre, params, self.balanceo, self.seed)
                    est.fit(self.X.iloc[itr], self.y[itr])
                    pred = est.predict(self.X.iloc[iva])
                if self.tarea == "regresion":
                    puntajes.append(-float(np.sqrt(np.mean((self.y[iva] - pred) ** 2))))   # −RMSE (más alto es mejor)
                else:
                    from sklearn.metrics import f1_score
                    puntajes.append(f1_score(self.y[iva], pred, average="macro", zero_division=0))
            except Exception as e:  # una configuración que falla se registra y recibe el peor puntaje
                error = f"{type(e).__name__}: {e}"[:200]
                puntajes.append(-10.0 if self.tarea == "regresion" else 0.0)
        s = float(np.mean(puntajes))
        self.historial.append({"evaluacion": len(self.historial) + 1, "params": dict(params), "f1_interno": s,
                               "f1_interno_sd": float(np.std(puntajes)), "segundos": time.perf_counter() - t,
                               "segundos_acum": time.perf_counter() - self._t0, "error": error})
        self._cache[k] = s
        return s

    def mejor(self):
        """Devuelve ``(params, puntaje)`` de la mejor evaluación del historial."""
        h = max(self.historial, key=lambda r: r["f1_interno"])
        return h["params"], h["f1_interno"]


# ---------------------------------------------------------------------------
# Búsquedas
# ---------------------------------------------------------------------------
def grid(obj, nombre, presupuesto=CF.PRESUPUESTO, seed=CF.SEED):
    """Grid Search sobre la rejilla del catálogo (si excede el presupuesto, se toma una muestra fija).

    Parámetros
    ----------
    obj : Objetivo
        Función objetivo del pliegue externo; registra cada evaluación en ``obj.historial``.
    nombre : str
        Modelo (clave de ``modelos.CATALOGO`` o ``CATALOGO_REG``), de donde salen el espacio o la rejilla.
    presupuesto : int
        Número de evaluaciones distintas (12 en el experimento); las repetidas no cuentan.
    seed : int
        Semilla del método.

    Devuelve
    --------
    dict
        Información extra del método (vacía).
    """
    rej = MD.catalogo(nombre)["rejilla"]
    combos = [dict(zip(rej, v)) for v in itertools.product(*rej.values())]
    if len(combos) > presupuesto:
        combos = [combos[i] for i in sorted(np.random.default_rng(seed).choice(len(combos), presupuesto, replace=False))]
    for c in combos:
        obj(c)
    return {}


def aleatoria(obj, nombre, presupuesto=CF.PRESUPUESTO, seed=CF.SEED):
    """Random Search: configuraciones uniformes en el espacio (log-uniformes donde corresponde).

    Parámetros
    ----------
    obj : Objetivo
        Función objetivo del pliegue externo; registra cada evaluación en ``obj.historial``.
    nombre : str
        Modelo (clave de ``modelos.CATALOGO`` o ``CATALOGO_REG``), de donde salen el espacio o la rejilla.
    presupuesto : int
        Número de evaluaciones distintas (12 en el experimento); las repetidas no cuentan.
    seed : int
        Semilla del método.

    Devuelve
    --------
    dict
        Información extra del método (vacía).
    """
    esp = MD.catalogo(nombre)["espacio"]
    rng = np.random.default_rng(seed)
    intentos = 0
    while len(obj.historial) < presupuesto and intentos < 20 * presupuesto:
        obj(de_unitario(esp, rng.random(len(esp))))
        intentos += 1
    return {}


def bayesiana(obj, nombre, presupuesto=CF.PRESUPUESTO, seed=CF.SEED, n_startup=4):
    """Optimización bayesiana con Optuna (TPE; adquisición equivalente a Expected Improvement).

    Parámetros
    ----------
    obj : Objetivo
        Función objetivo del pliegue externo; registra cada evaluación en ``obj.historial``.
    nombre : str
        Modelo (clave de ``modelos.CATALOGO`` o ``CATALOGO_REG``), de donde salen el espacio o la rejilla.
    presupuesto : int
        Número de evaluaciones distintas (12 en el experimento); las repetidas no cuentan.
    seed : int
        Semilla del método.
    n_startup : int
        Evaluaciones aleatorias iniciales antes de que TPE use su modelo sustituto.

    Devuelve
    --------
    dict
        ``{"n_startup": n_startup}``.
    """
    esp = MD.catalogo(nombre)["espacio"]
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup, multivariate=True)
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    intentos = 0
    while len(obj.historial) < presupuesto and intentos < 20 * presupuesto:
        trial = estudio.ask()
        estudio.tell(trial, obj(sugerir(trial, esp)))
        intentos += 1
    return {"n_startup": n_startup}


def genetica(obj, nombre, presupuesto=CF.PRESUPUESTO, seed=CF.SEED, poblacion=4, p_cruce=0.7, p_mutacion=0.3,
             sigma=0.15, torneo=3, elite=1):
    """Algoritmo genético (DEAP) sobre vectores en [0, 1]^d; devuelve la diversidad por generación.

    Parámetros
    ----------
    obj : Objetivo
        Función objetivo del pliegue externo; registra cada evaluación en ``obj.historial``.
    nombre : str
        Modelo (clave de ``modelos.CATALOGO`` o ``CATALOGO_REG``), de donde salen el espacio o la rejilla.
    presupuesto : int
        Número de evaluaciones distintas (12 en el experimento); las repetidas no cuentan.
    seed : int
        Semilla del método.
    poblacion : int
        Individuos por generación.
    p_cruce : float
        Probabilidad de cruce *blend* (α = 0.5) de cada pareja.
    p_mutacion : float
        Probabilidad de mutación gaussiana de cada individuo (cada gen con probabilidad 1/d).
    sigma : float
        Desviación de la mutación gaussiana, en la escala del genoma [0, 1].
    torneo : int
        Tamaño del torneo de selección.
    elite : int
        Individuos que pasan intactos a la siguiente generación.

    Devuelve
    --------
    dict
        ``diversidad`` (por generación: distancia media entre individuos, mejor puntaje y evaluaciones
        acumuladas) y los parámetros del algoritmo.
    """
    esp = MD.catalogo(nombre)["espacio"]
    d = len(esp)
    rng = np.random.default_rng(seed)
    import random as _random
    _random.seed(seed)
    if not hasattr(creator, "AptitudMax"):
        creator.create("AptitudMax", base.Fitness, weights=(1.0,))
        creator.create("Individuo", list, fitness=creator.AptitudMax)
    tb = base.Toolbox()
    tb.register("individuo", lambda: creator.Individuo(rng.random(d).tolist()))
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=sigma, indpb=1.0 / d)
    tb.register("select", tools.selTournament, tournsize=torneo)

    def evaluar(ind):
        """Recorta el individuo a [0, 1], lo traduce a una configuración y devuelve su aptitud (tupla)."""
        ind[:] = list(np.clip(ind, 0, 1))
        return (obj(de_unitario(esp, ind)),)

    pob = [tb.individuo() for _ in range(poblacion)]
    for ind in pob:
        ind.fitness.values = evaluar(ind)
    diversidad = []
    gen = 0
    while len(obj.historial) < presupuesto and gen < 50:
        G = np.array(pob)
        diversidad.append({"generacion": gen, "diversidad": float(np.mean(
            [np.linalg.norm(a - b) for a, b in itertools.combinations(G, 2)])) if len(G) > 1 else 0.0,
            "mejor": max(i.fitness.values[0] for i in pob), "evaluaciones": len(obj.historial)})
        elites = [creator.Individuo(list(i)) for i in tools.selBest(pob, elite)]
        for e, o in zip(elites, tools.selBest(pob, elite)):
            e.fitness.values = o.fitness.values
        hijos = [creator.Individuo(list(i)) for i in tb.select(pob, poblacion - elite)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if rng.random() < p_cruce:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for h in hijos:
            if rng.random() < p_mutacion:
                tb.mutate(h)
                if h.fitness.valid:
                    del h.fitness.values
        for h in hijos:
            if not h.fitness.valid:
                if len(obj.historial) >= presupuesto:
                    h.fitness.values = (-1.0,)
                else:
                    h.fitness.values = evaluar(h)
        pob = elites + hijos
        gen += 1
    return {"diversidad": diversidad, "poblacion": poblacion, "p_cruce": p_cruce, "p_mutacion": p_mutacion,
            "sigma": sigma, "torneo": torneo, "elite": elite}


METODOS = {"Grid": grid, "Random": aleatoria, "Bayesiana": bayesiana, "Genética": genetica}
