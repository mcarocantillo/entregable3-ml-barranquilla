# -*- coding: utf-8 -*-
"""Mejora 1: búsqueda de hiperparámetros del XGBoost geográfico con TODOS los datos.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Motivo. En el Entregable 2 los hiperparámetros del modelo nuevo quedaron fijos (α = 0.75, 15 % de vecinos por
ancla, profundidad 4: nunca se buscaron con kNNDM) y en el Entregable 3 se buscaron con solo 10 001 viviendas, donde
cada ancla tiene pocos vecinos y la búsqueda casi apagó los modelos locales (α = 0.97). Aquí se buscan con las
257 273 viviendas y los pliegues kNNDM del Entregable 2 v3.

Búsqueda factorizada. La predicción es p = α·p_global + (1 − α)·p_local y las dos partes se entrenan por separado,
así que no hace falta reentrenar cada combinación:
  - 6 configuraciones del XGBoost GLOBAL (profundidad, árboles, tasa de aprendizaje, pesos de clase);
  - 9 configuraciones de los modelos LOCALES (fracción de vecinos por ancla, tamaño de la rejilla de anclas,
    profundidad y árboles de cada modelo local, pesos de clase);
  - α en una rejilla de 0 a 1 (paso 0.05), aplicado sobre las probabilidades ya guardadas.
Son 6 × 9 × 21 = 1 134 combinaciones con solo (6 + 9) × 5 = 75 ajustes. Con α = 1 el modelo es el XGBoost global:
la misma búsqueda da el XGBoost afinado (la ablación sin la parte geográfica).

Uso (desde e3/):  ../venv_e2/Scripts/python -u mejoras/01_geo_busqueda.py      (RAPIDO=1: prueba de humo)
"""
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

import comun as K
from estrato3.modelos_propios import GeoXGBEstrato, XGBEstrato, _Constante, _proba_completa

GLOBALES = {
    "G1_e2": dict(max_depth=4, n_estimators=300, learning_rate=0.1, balanceado=True),
    "G2": dict(max_depth=6, n_estimators=300, learning_rate=0.1, balanceado=True),
    "G3": dict(max_depth=8, n_estimators=300, learning_rate=0.1, balanceado=True),
    "G4": dict(max_depth=6, n_estimators=600, learning_rate=0.05, balanceado=True),
    "G5": dict(max_depth=6, n_estimators=300, learning_rate=0.1, balanceado=False),
    "G6": dict(max_depth=8, n_estimators=600, learning_rate=0.05, balanceado=False, min_child_weight=10),
}
_L = dict(frac=0.15, tam=2.0, min_ancla=300, max_depth=4, n_estimators=150, learning_rate=0.1, balanceado=True)
LOCALES = {
    "L1_e2": dict(_L),
    "L2": {**_L, "frac": 0.05},
    "L3": {**_L, "frac": 0.02},
    "L4": {**_L, "frac": 0.30},
    "L5": {**_L, "frac": 0.05, "tam": 1.0},
    "L6": {**_L, "frac": 0.05, "max_depth": 6},
    "L7": {**_L, "max_depth": 6, "n_estimators": 300, "learning_rate": 0.05},
    "L8": {**_L, "frac": 0.02, "tam": 1.0, "min_ancla": 150, "max_depth": 3, "n_estimators": 100},
    "L9": {**_L, "frac": 0.08, "balanceado": False},
}
ALFAS = np.round(np.arange(0, 1.0001, 0.05), 2)
REF_E2 = ("G1_e2", "L1_e2", 0.75)


def ajustar_locales(X, y, xy, cfg, hilos, seed=42):
    """Modelos locales del XGBoost geográfico (misma construcción que ``GeoXGBEstrato.fit``, sin el global).

    Devuelve (anclas, h, modelos, k).
    """
    celda = np.floor(xy / cfg["tam"]).astype(int)
    _, inv, cuenta = np.unique(celda, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    anclas = np.array([xy[inv == g].mean(0) for g in np.where(cuenta >= cfg["min_ancla"])[0]]).reshape(-1, 2)
    arbol = cKDTree(xy)
    k = min(int(cfg["frac"] * len(xy)), len(xy))
    if len(anclas) == 0:
        return anclas, np.zeros(0), [], k
    d_k, _ = arbol.query(anclas, k=[k])
    h = d_k.ravel()
    pl = dict(max_depth=cfg["max_depth"], n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
              balanceado=cfg["balanceado"], n_jobs=hilos, random_state=seed)
    modelos = []
    for a, hh in zip(anclas, h):
        idx = np.array(arbol.query_ball_point(a, r=hh))
        w = GeoXGBEstrato._kernel(np.linalg.norm(xy[idx] - a, axis=1), hh)
        sel = w > 0
        if len(np.unique(y[idx][sel])) < 2:
            modelos.append(_Constante(np.unique(y[idx][sel])))
            continue
        modelos.append(XGBEstrato(**pl).fit(X[idx][sel], y[idx][sel], sample_weight=w[sel]))
    return anclas, h, modelos, k


def proba_local(anclas, h, modelos, X, xy, n_clases=6):
    """Promedio de las probabilidades de las anclas que cubren cada vivienda (kernel), y la máscara de cobertura."""
    num = np.zeros((len(X), n_clases))
    den = np.zeros(len(X))
    for a, hh, m in zip(anclas, h, modelos):
        w = GeoXGBEstrato._kernel(np.linalg.norm(xy - a, axis=1), hh)
        sel = w > 0
        if sel.any():
            num[sel] += w[sel, None] * _proba_completa(m, X[sel], m.classes_)
            den[sel] += w[sel]
    cub = den > 0
    out = np.full((len(X), n_clases), np.nan)
    out[cub] = num[cub] / den[cub, None]
    return out, cub


def tarea(tipo, nombre, f, ruta_X, ytr, xytr, xyev, hilos):
    """Ajusta una parte (global o local) en el pliegue f y guarda sus probabilidades de validación."""
    clave = f"geo_{tipo}_{nombre}_f{f}"
    hecho = K.leer(clave)
    if hecho is not None:
        return clave, hecho["seg"]
    z = np.load(ruta_X)
    Xtr, Xev = z["tr"], z["ev"]
    t0 = time.time()
    if tipo == "global":
        m = XGBEstrato(**GLOBALES[nombre], n_jobs=hilos, random_state=42).fit(Xtr, ytr)
        out = {"p": _proba_completa(m, Xev, m.classes_).astype(np.float32)}
    else:
        anclas, h, modelos, k = ajustar_locales(Xtr, ytr, xytr, LOCALES[nombre], hilos)
        p, cub = proba_local(anclas, h, modelos, Xev, xyev)
        out = {"p": p.astype(np.float32), "cub": cub, "anclas": len(anclas), "k": k}
    out["seg"] = time.time() - t0
    K.guardar(clave, out)
    K.log(f"  {clave}: {out['seg']:.0f} s" + (f", {out['anclas']} anclas, k = {out['k']}, cobertura "
                                              f"{out['cub'].mean():.2f}" if tipo == "local" else ""))
    return clave, out["seg"]


def evaluar(d):
    """F1 por pliegue de todas las combinaciones (global, local, α) a partir de las probabilidades guardadas."""
    nf = len(d["folds"])
    yv = [d["y_tr"][iva] for _, iva in d["folds"]]
    PG = {g: [K.leer(f"geo_global_{g}_f{f}")["p"] for f in range(nf)] for g in GLOBALES}
    PL = {loc: [K.leer(f"geo_local_{loc}_f{f}") for f in range(nf)] for loc in LOCALES}
    filas = []
    for g in GLOBALES:
        for loc in LOCALES:
            for a in ALFAS:
                f1 = []
                for f in range(nf):
                    pg, ol = PG[g][f], PL[loc][f]
                    plc = np.where(ol["cub"][:, None], ol["p"], pg)
                    f1.append(K.f1_macro(yv[f], K.pred(a * pg + (1 - a) * plc)))
                filas.append({"global": g, "local": loc, "alpha": float(a), "f1_pliegues": f1,
                              "f1": float(np.mean(f1))})
    return filas


def main():
    d = K.cargar()
    nf = len(d["folds"])
    K.log(f"train {len(d['train']):,} | {nf} pliegues kNNDM (q = {d['q']}) | {len(GLOBALES)} globales × "
          f"{len(LOCALES)} locales × {len(ALFAS)} α")
    for f in range(nf):
        K.cache_matrices(d, f)
    K.log("matrices de los pliegues listas")
    tareas = []
    for f, (itr, iva) in enumerate(d["folds"]):
        ruta = str(K.CHK / f"X_BV_{f}.npz")
        args = (ruta, d["y_tr"][itr], d["xy_tr"][itr], d["xy_tr"][iva], K.HILOS)
        # primero las más pesadas (locales con muchos vecinos y globales profundos)
        tareas += [("local", loc, f) + args for loc in LOCALES]
        tareas += [("global", g, f) + args for g in GLOBALES]
    orden = {"L4": 0, "L7": 1, "G6": 2, "G4": 3, "L1_e2": 4}
    tareas.sort(key=lambda t: orden.get(t[1], 9))
    Parallel(n_jobs=K.N_TRABAJOS, pre_dispatch="n_jobs")(delayed(tarea)(*t) for t in tareas)
    K.log("ajustes terminados; evaluando combinaciones")

    filas = evaluar(d)
    filas.sort(key=lambda r: -r["f1"])
    mejor = filas[0]
    ref = next(r for r in filas if (r["global"], r["local"], r["alpha"]) == REF_E2)
    xgb = sorted([r for r in filas if r["alpha"] == 1.0], key=lambda r: -r["f1"])[0]
    por_alfa = {}
    for r in filas:
        if (r["global"], r["local"]) == (mejor["global"], mejor["local"]):
            por_alfa[r["alpha"]] = r["f1"]
    seg = {k: [K.leer(f"geo_{t}_{k}_f{f}")["seg"] for f in range(nf)] for t, dd in
           (("global", GLOBALES), ("local", LOCALES)) for k in dd}
    loc_info = {loc: {"anclas": [K.leer(f"geo_local_{loc}_f{f}")["anclas"] for f in range(nf)],
                      "cobertura": [float(K.leer(f"geo_local_{loc}_f{f}")["cub"].mean()) for f in range(nf)]}
                for loc in LOCALES}
    K.escribir_json("geo_busqueda.json", {
        "protocolo": "todos los datos; pliegues kNNDM de zonas nuevas del E2 v3 (q = 30); F1 macro medio por pliegue",
        "globales": GLOBALES, "locales": LOCALES, "alfas": ALFAS,
        "mejor": mejor, "referencia_E2": ref, "xgboost_afinado_alpha1": xgb,
        "curva_alpha_mejor": por_alfa, "top20": filas[:20], "segundos": seg, "locales_info": loc_info,
        "n_combinaciones": len(filas)})
    K.log(f"MEJOR: {mejor['global']} + {mejor['local']}, α = {mejor['alpha']}: F1 {mejor['f1']:.4f} "
          f"{np.round(mejor['f1_pliegues'], 3).tolist()}")
    K.log(f"Referencia E2 (G1_e2 + L1_e2, α = 0.75): F1 {ref['f1']:.4f} (E2 v3 publicó 0.572)")
    K.log(f"XGBoost afinado (α = 1): {xgb['global']}: F1 {xgb['f1']:.4f}")


if __name__ == "__main__":
    main()
