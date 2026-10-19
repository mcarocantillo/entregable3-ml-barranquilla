# -*- coding: utf-8 -*-
"""Utilidades comunes de las mejoras posteriores al Entregable 3.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Protocolo de todas las mejoras (el de la fase 2 de la versión 3 del Entregable 2, con todos los datos):

- 257 273 viviendas de entrenamiento y las variables del Entregable 3 (modelo B + vecindad física KNN, k = 8);
- los MISMOS 5 pliegues kNNDM de zonas nuevas del Entregable 2 v3 (q = 30, semilla 42), copiados en
  ``datos/knndm_e2v3.pkl``; los hiperparámetros se eligen con la media del F1 macro en esos pliegues;
- el test de los 7 bloques de 2 km se mira UNA sola vez, con los modelos ya elegidos (``03_test.py``);
- criterio de reemplazo del proyecto: diferencia pareada de F1 por pliegue > 1 SE (ddof = 1) y el test lo confirma.

Las probabilidades fuera de muestra de cada pliegue se guardan en ``chk/`` para reanudar y para el stacking.
``RAPIDO=1`` es una prueba de humo: pocas filas por pliegue, carpetas ``chk_rapido/`` y ``resultados_rapido/``.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.stdout.reconfigure(encoding="utf-8")

from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import datos as D  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

RAPIDO = os.environ.get("RAPIDO") == "1"
SUF = "_rapido" if RAPIDO else ""
CHK = AQUI / f"chk{SUF}"
RES = AQUI / f"resultados{SUF}"
KNNDM = AQUI / "datos" / "knndm_e2v3.pkl"
N_TRABAJOS = int(os.environ.get("N_TRABAJOS", 4))
HILOS = int(os.environ.get("HILOS", 2))
CLASES = CF.CLASES
T0 = time.time()


def log(msg):
    """Imprime con el tiempo transcurrido."""
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def f1_macro(y, p):
    """F1 macro por tabla de confusión (igual a ``f1_score(average='macro', zero_division=0)``)."""
    cm = np.bincount(np.asarray(y, int) * 7 + np.asarray(p, int), minlength=49).reshape(7, 7)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    pres = den > 0
    return float(np.mean(np.where(pres, 2 * tp / np.maximum(den, 1), 0.0)[pres]))


def pred(proba):
    """Estrato con mayor probabilidad (columnas = estratos 1..6)."""
    return CLASES[np.argmax(proba, axis=1)]


def cargar():
    """Datos del Entregable 3 con todo el entrenamiento y los pliegues kNNDM del Entregable 2 v3.

    Devuelve
    --------
    dict
        Lo de ``estrato3.datos.cargar()`` más ``folds`` [(itr, iva)], ``y_tr``, ``y_te``, ``xy_tr``, ``xy_te`` y
        ``num_B`` (variables del modelo base, sin vecindad).
    """
    CF.fijar_semillas()
    d = D.cargar()
    with open(KNNDM, "rb") as fh:
        kn = pickle.load(fh)["zonas nuevas"]
    assert len(kn["fold_fila"]) == len(d["train"]), "los pliegues no corresponden a este entrenamiento"
    folds = [(np.asarray(a), np.asarray(b)) for a, b in kn["folds"]]
    if RAPIDO:
        rng = np.random.default_rng(CF.SEED)
        folds = [(rng.choice(a, 12000, replace=False), rng.choice(b, 4000, replace=False)) for a, b in folds[:2]]
    d["folds"] = folds
    d["q"] = int(kn["q"])
    d["y_tr"] = d["train"].estrato_num.to_numpy().astype(int)
    d["y_te"] = d["test"].estrato_num.to_numpy().astype(int)
    d["xy_tr"] = d["train"][["x_km", "y_km"]].to_numpy(float)
    d["xy_te"] = d["test"][["x_km", "y_km"]].to_numpy(float)
    d["num_B"] = [c for c in d["num"] if not c.startswith("knn")]
    return d


def matrices(d, itr, iev, df_ev=None, base=False):
    """Preprocesador del Entregable 1 ajustado solo con las filas ``itr`` de train.

    Parámetros
    ----------
    d : dict
        Salida de ``cargar()``.
    itr : array
        Filas de entrenamiento.
    iev : array or None
        Filas de train que se evalúan (validación); se ignora si se pasa ``df_ev``.
    df_ev : pandas.DataFrame, optional
        Conjunto de evaluación externo (el test).
    base : bool
        Si True, solo las variables del modelo base (B, sin vecindad).

    Devuelve
    --------
    (ndarray, ndarray)
        Matrices float32 de entrenamiento y evaluación.
    """
    dd = dict(d)
    if base:
        dd["num"] = d["num_B"]
    cols = dd["num"] + dd["cat"]
    tr = d["train"].iloc[itr][cols]
    ev = (df_ev if df_ev is not None else d["train"].iloc[iev])[cols]
    prep = MD.preprocesador(dd).fit(tr)
    return prep.transform(tr).astype(np.float32), prep.transform(ev).astype(np.float32)


def cache_matrices(d, f, base=False):
    """Matrices del pliegue ``f`` (o del test si ``f == 'test'``), guardadas en ``chk/`` para los procesos."""
    CHK.mkdir(parents=True, exist_ok=True)
    ruta = CHK / f"X_{'B' if base else 'BV'}_{f}.npz"
    if ruta.exists():
        z = np.load(ruta)
        return z["tr"], z["ev"]
    if f == "test":
        Xtr, Xev = matrices(d, np.arange(len(d["train"])), None, df_ev=d["test"], base=base)
    else:
        itr, iva = d["folds"][f]
        Xtr, Xev = matrices(d, itr, iva, base=base)
    np.savez(ruta, tr=Xtr, ev=Xev)
    return Xtr, Xev


def guardar(nombre, obj):
    CHK.mkdir(parents=True, exist_ok=True)
    with open(CHK / f"{nombre}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def leer(nombre):
    ruta = CHK / f"{nombre}.pkl"
    if ruta.exists():
        with open(ruta, "rb") as fh:
            return pickle.load(fh)
    return None


def pareada(a, b):
    """Diferencia pareada por pliegue a − b: media, SE (ddof = 1), pliegues a favor y si supera 1 SE."""
    dif = np.asarray(a) - np.asarray(b)
    se = float(dif.std(ddof=1) / np.sqrt(len(dif))) if len(dif) > 1 else float("nan")
    return {"dif_por_pliegue": dif.round(4).tolist(), "media": float(dif.mean()), "SE": se,
            "pliegues_a_favor": int((dif > 0).sum()), "supera_1SE": bool(dif.mean() > se)}


def escribir_json(nombre, obj):
    RES.mkdir(parents=True, exist_ok=True)

    def limpio(o):
        if isinstance(o, dict):
            return {str(k): limpio(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [limpio(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return limpio(o.tolist())
        return o

    (RES / nombre).write_text(json.dumps(limpio(obj), ensure_ascii=False, indent=1), encoding="utf-8")
