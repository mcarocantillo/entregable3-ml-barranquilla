# -*- coding: utf-8 -*-
"""Tareas del pipeline combinatorio (una combinación × un pliegue externo de la validación anidada).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Vive en un módulo (y no en el script) para que cada proceso trabajador cargue los datos una sola vez.
"""
import hashlib
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

from . import configuracion as CF

CACHE = CF.RESULTADOS / "datos_clasificacion.pkl"
TAREAS = CF.RESULTADOS / "tareas"
_DATOS = None


def preparar():
    """Submuestra, folds externos e internos (una sola vez; se guardan en ``CACHE``).

    Crea la submuestra estratificada de ``CF.N_SUBMUESTRA`` viviendas, los ``CF.K_EXTERNO`` pliegues externos
    kNNDM (escenario = geometría del test) y, para cada uno, ``CF.K_INTERNO`` pliegues internos kNNDM (escenario =
    ese pliegue externo). No hace nada si ``CACHE`` ya existe, así que todas las combinaciones usan los mismos
    pliegues.

    Devuelve
    --------
    None
        Escribe ``resultados/datos_clasificacion.pkl``.
    """
    if CACHE.exists():
        return
    from estrato3 import datos as DT
    CF.fijar_semillas()
    t0 = time.time()
    d = DT.cargar()
    sub = DT.submuestra(d["train"])
    ext = DT.folds_externos(sub, d["test"], d["V"])
    internos = [DT.folds_internos(sub, itr, iva, d["V"]) for itr, iva in ext["folds"]]
    cols = d["num"] + d["cat"] + ["estrato_num", "x_km", "y_km", "npn_edificio", "bloque"]
    cols = list(dict.fromkeys(cols))
    paquete = {"sub": sub[cols], "num": d["num"], "cat": d["cat"], "num_log": d["num_log"], "dec": d["dec"],
               "externos": ext["folds"], "internos": internos,
               "knndm": {"q": ext["q"], "W_km": ext["W"], "W_aleatoria_km": ext["W_loo"]}}
    CF.RESULTADOS.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(paquete, f)
    print(f"preparado: submuestra {len(sub):,}, q externo = {ext['q']} ({time.time() - t0:.0f} s)", flush=True)


def _datos():
    """Paquete de datos en el proceso trabajador (con el módulo ``modelo`` del E1 importado aquí)."""
    global _DATOS
    if _DATOS is None:
        with open(CACHE, "rb") as f:
            _DATOS = pickle.load(f)
        _, M1, _, _ = CF.importar_e1()
        _DATOS["M1"] = M1
    return _DATOS


def nombre_tarea(modelo, balanceo, metodo, fold):
    """Identificador estable de una tarea.

    Parámetros
    ----------
    modelo, balanceo, metodo : str
        Combinación (``"no aplica"`` como balanceo en regresión).
    fold : int
        Pliegue externo.

    Devuelve
    --------
    (str, str)
        Hash MD5 de 12 caracteres (nombre del archivo en ``resultados/tareas``) y la descripción legible.
    """
    s = f"{modelo}|{balanceo}|{metodo}|{fold}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12], s


def tarea(modelo, balanceo, metodo, fold, presupuesto, carpeta=None):
    """Una combinación en un pliegue externo: búsqueda en el bucle interno, reajuste y evaluación externa.

    Parámetros
    ----------
    modelo : str
        Clave de ``modelos.CATALOGO``.
    balanceo : str
        Una de ``modelos.BALANCEOS``.
    metodo : str
        Clave de ``optimizacion.METODOS`` (Grid, Random, Bayesiana o Genética).
    fold : int
        Pliegue externo (0 a 4).
    presupuesto : int
        Evaluaciones de la búsqueda.
    carpeta : str o Path, opcional
        Dónde guardar el resultado (por defecto ``resultados/tareas``).

    Devuelve
    --------
    (str, float, float)
        Descripción de la tarea, F1 macro en el pliegue externo y segundos de búsqueda. Además escribe
        ``<hash>.json`` (semilla, hiperparámetros, métricas externas, tiempos e historial completo) y ``<hash>.npz``
        (probabilidades fuera de muestra).

    Supuestos
    ---------
    La métrica que se reporta sale del pliegue externo, nunca del óptimo interno (validación anidada).
    """
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    h, s = nombre_tarea(modelo, balanceo, metodo, fold)
    CF.fijar_semillas()
    d = _datos()
    sub = d["sub"]
    itr, iva = d["externos"][fold]
    Xtr, ytr = sub.iloc[itr], sub.estrato_num.to_numpy()[itr]
    Xva, yva = sub.iloc[iva], sub.estrato_num.to_numpy()[iva]
    obj = OP.Objetivo(d, Xtr, ytr, d["internos"][fold], modelo, balanceo)
    t0 = time.perf_counter()
    extra = OP.METODOS[metodo](obj, modelo, presupuesto=presupuesto)
    t_busqueda = time.perf_counter() - t0
    params, f1_int = obj.mejor()
    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MD.pipeline(d, modelo, params, balanceo).fit(Xtr, ytr)
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    proba = EV.proba_completa(est, Xva)
    t_inferencia = time.perf_counter() - t2
    m = EV.metricas(yva, proba)
    salida = {"tarea": s, "modelo": modelo, "balanceo": balanceo, "metodo": metodo, "fold": fold,
              "presupuesto": presupuesto, "semilla": CF.SEED, "params": params, "f1_interno_mejor": f1_int,
              "metricas": m, "n_evaluaciones": len(obj.historial), "seg_busqueda": t_busqueda,
              "seg_ajuste_final": t_ajuste, "seg_inferencia": t_inferencia, "n_train": len(itr), "n_val": len(iva),
              "historial": obj.historial, "extra": extra}
    carpeta = Path(carpeta) if carpeta else TAREAS
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{h}.json").write_text(json.dumps(salida, ensure_ascii=False, default=float), encoding="utf-8")
    np.savez_compressed(carpeta / f"{h}.npz", proba=proba.astype(np.float32), iva=iva)
    return s, m["f1_macro"], t_busqueda


TEST_CACHE = CF.RESULTADOS / "test_clasificacion.pkl"
_TEST = None


def _test():
    """Test del Entregable 1 en el proceso trabajador (lo guarda ``finalistas.py`` en ``TEST_CACHE``)."""
    global _TEST
    if _TEST is None:
        with open(TEST_CACHE, "rb") as f:
            _TEST = pickle.load(f)
    return _TEST


def finalista(modelo, balanceo, metodo, presupuesto=CF.PRESUPUESTO):
    """Búsqueda final en toda la submuestra (pliegues externos como validación), ajuste y probabilidades en test.

    Parámetros
    ----------
    modelo, balanceo, metodo : str
        La mejor combinación del modelo según el bucle externo.
    presupuesto : int
        Evaluaciones de la búsqueda final.

    Devuelve
    --------
    dict
        ``params``, ``f1_busqueda_final``, ``proba_test`` (65 145 × 6) y ``segundos``.
    """
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    CF.fijar_semillas()
    d, test = _datos(), _test()
    sub = d["sub"]
    y = sub.estrato_num.to_numpy()
    t0 = time.perf_counter()
    obj = OP.Objetivo(d, sub, y, d["externos"], modelo, balanceo)
    OP.METODOS[metodo](obj, modelo, presupuesto=presupuesto)
    params, f1_cv = obj.mejor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MD.pipeline(d, modelo, params, balanceo).fit(sub, y)
    return {"modelo": modelo, "balanceo": balanceo, "metodo": metodo, "params": params, "f1_busqueda_final": f1_cv,
            "proba_test": EV.proba_completa(est, test), "segundos": time.perf_counter() - t0}


def calibrado(modelo, params, balanceo, metodo_cal):
    """Recalibración (``sigmoid`` = Platt o ``isotonic``) con los pliegues externos; probabilidades en test.

    Parámetros
    ----------
    modelo : str
        Finalista.
    params : dict
        Sus hiperparámetros finales.
    balanceo : str
        Su técnica de balanceo.
    metodo_cal : {"sigmoid", "isotonic"}
        Platt o regresión isotónica (``CalibratedClassifierCV``).

    Devuelve
    --------
    (str, str, numpy.ndarray)
        Modelo, método y probabilidades recalibradas en el test.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    CF.fijar_semillas()
    d, test = _datos(), _test()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = CalibratedClassifierCV(MD.pipeline(d, modelo, params, balanceo), method=metodo_cal,
                                   cv=d["externos"]).fit(d["sub"], d["sub"].estrato_num.to_numpy())
    return modelo, metodo_cal, EV.proba_completa(c, test)


def con_semilla(modelo, params, balanceo, semilla):
    """El finalista reentrenado con otra semilla (robustez); probabilidades en test.

    Parámetros
    ----------
    modelo, params, balanceo
        El finalista y su configuración.
    semilla : int
        Semilla del modelo y del muestreo sintético.

    Devuelve
    --------
    (str, int, numpy.ndarray)
        Modelo, semilla y probabilidades en el test.
    """
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    d, test = _datos(), _test()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = MD.pipeline(d, modelo, params, balanceo, seed=semilla).fit(d["sub"], d["sub"].estrato_num.to_numpy())
    return modelo, semilla, EV.proba_completa(e, test)


COMPLETO_CACHE = CF.RESULTADOS / "train_test_completo.pkl"
BASE_E1 = "Logística B (modelo base del E1)"


def todo_el_train(modelo, params, balanceo, lote_rf=15):
    """Un finalista reentrenado con TODO el entrenamiento del E1; devuelve sus probabilidades en el test.

    Random Forest se entrena por lotes de ``lote_rf`` árboles (semillas distintas) y se promedian las
    probabilidades: es el mismo bosque en distribución, sin guardar en memoria los 150 árboles a la vez.
    ``modelo == BASE_E1`` es la logística B del E1 (C = 0.01, sin pesos, sin vecindad física).

    Parámetros
    ----------
    modelo : str
        Finalista o ``BASE_E1``.
    params : dict
        Hiperparámetros elegidos en la submuestra (no se vuelven a buscar).
    balanceo : str
        Técnica de balanceo del finalista.
    lote_rf : int
        Árboles por lote en Random Forest.

    Devuelve
    --------
    (str, numpy.ndarray, float)
        Modelo, probabilidades en el test y segundos.

    Supuestos
    ---------
    ``COMPLETO_CACHE`` existe (lo escribe ``experimentos/todos_los_datos.py``).
    """
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    CF.fijar_semillas()
    with open(COMPLETO_CACHE, "rb") as f:
        c = pickle.load(f)
    _, M1, _, _ = CF.importar_e1()
    datos = dict(c["datos"], M1=M1)
    train, test = c["train"], c["test"]
    y = train.estrato_num.to_numpy()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if modelo == BASE_E1:
            datos_b = dict(datos, num=[col for col in datos["num"] if not col.startswith("knn")])
            est = Pipeline([("prep", MD.preprocesador(datos_b)),
                            ("modelo", LogisticRegression(C=0.01, max_iter=1000))]).fit(train, y)
            P = EV.proba_completa(est, test)
        elif modelo == "Random Forest":
            total = int(params["n_estimators"])
            P = 0.0
            for i, ini in enumerate(range(0, total, lote_rf)):
                n = min(lote_rf, total - ini)
                est = MD.pipeline(datos, modelo, dict(params, n_estimators=n), balanceo, seed=CF.SEED + i).fit(train, y)
                P = P + EV.proba_completa(est, test) * n
                del est
            P = P / total
        else:
            est = MD.pipeline(datos, modelo, params, balanceo).fit(train, y)
            P = EV.proba_completa(est, test)
    return modelo, P, time.perf_counter() - t0


def tarea_reg(modelo, metodo, fold, presupuesto, carpeta):
    """Regresión del estrato como número: búsqueda interna (−RMSE), reajuste y evaluación externa.

    Parámetros
    ----------
    modelo : str
        Clave de ``modelos.CATALOGO_REG``.
    metodo : str
        Optimizador.
    fold : int
        Pliegue externo.
    presupuesto : int
        Evaluaciones de la búsqueda.
    carpeta : str o Path
        Dónde guardar el resultado (``resultados/tareas_regresion``).

    Devuelve
    --------
    (str, float, float)
        Descripción, RMSE externo y segundos de búsqueda; escribe el JSON y el ``.npz`` con las predicciones.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    from estrato3 import evaluacion as EV
    from estrato3 import modelos as MD
    from estrato3 import optimizacion as OP
    h, s = nombre_tarea(modelo, "no aplica", metodo, fold)
    CF.fijar_semillas()
    d = _datos()
    sub = d["sub"]
    itr, iva = d["externos"][fold]
    y = sub.estrato_num.to_numpy().astype(float)
    Xtr, ytr, Xva, yva = sub.iloc[itr], y[itr], sub.iloc[iva], y[iva]
    obj = OP.Objetivo(d, Xtr, ytr, d["internos"][fold], modelo, "no aplica", tarea="regresion")
    t0 = time.perf_counter()
    extra = OP.METODOS[metodo](obj, modelo, presupuesto=presupuesto)
    t_busqueda = time.perf_counter() - t0
    params, s_int = obj.mejor()
    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = MD.pipeline_reg(d, modelo, params).fit(Xtr, ytr)
    t_ajuste = time.perf_counter() - t1
    t2 = time.perf_counter()
    pred = est.predict(Xva)
    t_inferencia = time.perf_counter() - t2
    clase = np.clip(np.rint(pred), 1, 6).astype(int)
    m = {"rmse": float(np.sqrt(np.mean((yva - pred) ** 2))), "mae": float(mean_absolute_error(yva, pred)),
         "r2": float(r2_score(yva, pred)), "f1_macro_redondeado": EV.metricas_clase(yva.astype(int), clase)}
    salida = {"tarea": s, "modelo": modelo, "balanceo": "no aplica", "metodo": metodo, "fold": fold,
              "presupuesto": presupuesto, "semilla": CF.SEED, "params": params, "rmse_interno_mejor": -s_int,
              "metricas": m, "n_evaluaciones": len(obj.historial), "seg_busqueda": t_busqueda,
              "seg_ajuste_final": t_ajuste, "seg_inferencia": t_inferencia, "n_train": len(itr), "n_val": len(iva),
              "historial": obj.historial, "extra": extra}
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{h}.json").write_text(json.dumps(salida, ensure_ascii=False, default=float), encoding="utf-8")
    np.savez_compressed(carpeta / f"{h}.npz", pred=pred.astype(np.float32), iva=iva)
    return s, m["rmse"], t_busqueda
