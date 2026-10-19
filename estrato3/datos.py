# -*- coding: utf-8 -*-
"""Carga de datos, variables, submuestra y folds de la validación anidada.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Reutiliza sin cambios la partición del Entregable 1 (los mismos 7 bloques de
test) y la validación por emparejamiento de distancias al vecino más cercano
(kNNDM) de su versión 3. Las variables son las del modelo B (físicas y de
ubicación) más la vecindad física con grafo KNN (versión 3, capítulo 6 del
Entregable 2), que no usa el estrato y está disponible al predecir.
"""
import numpy as np
import pandas as pd

from . import configuracion as CF

ATRIBUTOS_VECINDAD = ["log_area", "banios", "habitaciones", "antiguedad", "piso", "prop_PH"]


def cargar():
    """Datos del Entregable 1 v3 con la vecindad física KNN.

    Devuelve
    --------
    dict
        ``train`` y ``test`` (DataFrames con las variables del modelo y la
        vecindad), ``num`` y ``cat`` (listas de columnas numéricas y
        categóricas), ``num_log`` (numéricas que reciben log1p), ``dec`` (las
        decisiones del EDA) y ``V`` (el módulo ``vecinos`` del E1).
    """
    C1, M1, P1, V1 = CF.importar_e1()
    dec = C1.leer_decisiones()
    vm = dec["variables_modelo"]["valor"]
    datos = P1.cargar_particion()
    train = datos[datos.particion == "train"].reset_index(drop=True)
    test = datos[datos.particion == "test"].reset_index(drop=True)
    vec = vecindad_knn(pd.concat([train, test], ignore_index=True), V1, CF.K_VECINDAD)
    train = pd.concat([train, vec.iloc[:len(train)].reset_index(drop=True)], axis=1)
    test = pd.concat([test, vec.iloc[len(train):].reset_index(drop=True)], axis=1)
    num = vm["numericas"] + vm["espaciales"] + list(vec.columns)
    return {"train": train, "test": test, "num": num, "cat": vm["categoricas"],
            "num_log": dec["num_log"]["valor"], "dec": dec, "V": V1, "M1": M1}


def vecindad_knn(todo, V, k):
    """Promedio de atributos físicos de los k edificios más cercanos (sin el propio edificio).

    Parámetros
    ----------
    todo : pandas.DataFrame
        Todas las viviendas con coordenadas (train y test): los atributos
        físicos del catastro existen también para las viviendas que se predicen.
    V : module
        ``src.vecinos`` del Entregable 1 v3.
    k : int
        Número de edificios vecinos.

    Devuelve
    --------
    pandas.DataFrame
        Una columna por atributo (``knn{k}_*``) y la distancia media a los vecinos.
    """
    attrs = pd.DataFrame({
        "log_area": np.log1p(todo.area_construida), "banios": todo.total_banios,
        "habitaciones": todo.total_habitaciones, "antiguedad": todo.antiguedad,
        "piso": todo.planta_ubicacion, "prop_PH": (todo.condicion_predio == "PH_Unidad_Predial").astype(float)})
    attrs = attrs.fillna(attrs.median())
    xy_g, _, cod = V.puntos_unicos(todo[["x_km", "y_km"]].to_numpy(), todo.npn_edificio.to_numpy())
    A_g = attrs.groupby(cod).mean().to_numpy()
    g = np.arange(len(xy_g))
    lag, dist = V.lag_knn(xy_g, xy_g, A_g, k, grupo_destino=g, grupo_fuente=g)
    out = pd.DataFrame(lag[cod], columns=[f"knn{k}_{c}" for c in ATRIBUTOS_VECINDAD])
    out[f"knn{k}_dist_km"] = dist[cod]
    return out


def submuestra(train, n=CF.N_SUBMUESTRA, seed=CF.SEED):
    """Submuestra estratificada por estrato (mismas proporciones de clase que train).

    Parámetros
    ----------
    train : pandas.DataFrame
        Entrenamiento completo del Entregable 1.
    n : int
        Tamaño de la submuestra.
    seed : int
        Semilla.

    Devuelve
    --------
    pandas.DataFrame
        La submuestra con índice nuevo (0..n-1).
    """
    rng = np.random.default_rng(seed)
    frac = n / len(train)
    partes = [g.iloc[rng.permutation(len(g))[:int(round(len(g) * frac))]] for _, g in train.groupby("estrato_num")]
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def folds_externos(sub, test, V, k=CF.K_EXTERNO, seed=CF.SEED):
    """Bucle externo: kNNDM sobre la submuestra, con la geometría del test como escenario.

    Devuelve el resultado completo de ``V.knndm`` (folds, distancias y q elegido).
    """
    xy_te, n_te, _ = V.puntos_unicos(test[["x_km", "y_km"]].to_numpy(), test.npn_edificio.to_numpy())
    return V.knndm(sub[["x_km", "y_km"]].to_numpy(), sub.npn_edificio.to_numpy(), sub.estrato_num.to_numpy(),
                   xy_te, n_te, k=k, seed=seed, verbose=False)


def folds_internos(sub, itr, iva, V, k=CF.K_INTERNO, seed=CF.SEED):
    """Bucle interno: kNNDM dentro del entrenamiento de un pliegue externo.

    El escenario del bucle interno es el propio pliegue externo de validación:
    los folds internos imitan la distancia a la que el modelo tendrá que
    predecir ese pliegue. Devuelve folds con índices posicionales relativos a
    ``itr`` (listos para indexar ``sub.iloc[itr]``).
    """
    tr, va = sub.iloc[itr], sub.iloc[iva]
    xy_va, n_va, _ = V.puntos_unicos(va[["x_km", "y_km"]].to_numpy(), va.npn_edificio.to_numpy())
    r = V.knndm(tr[["x_km", "y_km"]].to_numpy(), tr.npn_edificio.to_numpy(), tr.estrato_num.to_numpy(),
                xy_va, n_va, k=k, seed=seed, verbose=False,
                q_rejilla=[k, 5, 8, 12, 16, 20, 30, 50, 80, 120, 200, 400])
    return r["folds"]
