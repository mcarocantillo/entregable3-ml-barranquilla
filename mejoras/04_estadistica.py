# -*- coding: utf-8 -*-
"""Mejora 4: ¿son estadísticamente significativas las diferencias? Secuencia jerárquica del capítulo 7.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Sin reentrenar nada (usa las probabilidades guardadas en ``chk/`` por 01–03):

1. **Validación, bloques espaciales.** F1 macro de las predicciones fuera de muestra en cada bloque de 2 km del
   entrenamiento con al menos ``MIN_BLOQUE`` viviendas → Friedman → Nemenyi (diferencia crítica) → comparaciones
   dirigidas con Wilcoxon de rangos con signo, ajustadas por Holm (y Benjamini-Hochberg), con delta de Cliff.
2. **Validación, pliegues.** t de Student pareada corregida de Nadeau y Bengio (2003) sobre los 5 pliegues
   (la varianza se infla por 1/k + n_val/n_entrenamiento porque los pliegues comparten entrenamiento), con Holm.
3. **Test.** Prueba exacta de permutación (cambio de signo) sobre las diferencias de F1 en los 7 bloques de test,
   con Holm; se acompaña del IC 95 % BCa de ``mejoras.json``.

Familias de comparaciones (Holm dentro de cada una): cada modelo frente al modelo base (10) y frente a la logística
con vecindad (9), más dos dirigidas (geográfico afinado − XGBoost afinado; Rotation Forest − geográfico del E2).

Uso (desde e3/):  ../venv_e2/Scripts/python -u mejoras/04_estadistica.py
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import comun as K  # noqa: E402
from estrato3 import estadistica as ES  # noqa: E402
from estrato3.modelos_propios import _proba_completa  # noqa: E402

T3 = __import__("03_test")
MIN_BLOQUE = 200
BASE, LV = "Modelo base (logística B, sin vecindad)", "Logística con vecindad"
DIRIGIDAS = [("XGBoost geográfico afinado", "XGBoost afinado"),
             ("Rotation Forest", "XGBoost geográfico (configuración del E2)")]


def stacking_oof(bases_oof, yv, meta_params, f):
    """Probabilidades del metamodelo en el pliegue f, entrenado con los otros pliegues."""
    nf = len(yv)
    Z = [T3.rasgos([b[g] for b in bases_oof]) for g in range(nf)]
    otros = [g for g in range(nf) if g != f]
    meta = LogisticRegression(max_iter=5000, **meta_params).fit(np.vstack([Z[g] for g in otros]),
                                                               np.concatenate([yv[g] for g in otros]))
    return _proba_completa(meta, Z[f], meta.classes_)


def wilcoxon_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def t_corregida(a, b, n_val, n_tr):
    """t pareada corregida de Nadeau y Bengio (2003): var × (1/k + n_val/n_tr)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"t": float("nan"), "p": 1.0}
    t = d.mean() / np.sqrt((1 / k + n_val / n_tr) * var)
    return {"t": float(t), "p": float(2 * stats.t.sf(abs(t), k - 1))}


def permutacion_signo(d):
    """p exacto (dos colas) del test de permutación por cambio de signo de las diferencias d (pocas: 2^n casos)."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    sig = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float(np.mean(np.abs((sig * d).mean(1)) >= obs - 1e-12))


def main():
    d = K.cargar()
    nf = len(d["folds"])
    yv = [d["y_tr"][iva] for _, iva in d["folds"]]
    geo, rev, mj = T3.leer_json("geo_busqueda.json"), T3.leer_json("revision_cv.json"), T3.leer_json("mejoras.json")
    elec = T3.elegir(geo, rev)
    OOF = {n: T3.oof(e, nf) for n, e in elec.items()}
    for nombre, info in mj["stacking"].items():
        hecho = K.leer(f"stack_oof_{info['meta']}_{len(info['bases'])}")
        if hecho is None:
            K.log(f"stacking fuera de muestra: {nombre}")
            hecho = Parallel(n_jobs=min(K.N_TRABAJOS, nf))(
                delayed(stacking_oof)([OOF[b] for b in info["bases"]], yv, T3.META[info["meta"]], f) for f in range(nf))
            K.guardar(f"stack_oof_{info['meta']}_{len(info['bases'])}", hecho)
        OOF[nombre] = hecho
    modelos = list(OOF)

    # --- predicciones fuera de muestra por vivienda y F1 por bloque espacial --------------------------------------
    n = len(d["train"])
    pred = {}
    for m in modelos:
        p = np.zeros(n, int)
        for f, (_, iva) in enumerate(d["folds"]):
            p[iva] = K.pred(OOF[m][f])
        pred[m] = p
    bloques = d["train"].bloque.to_numpy()
    ub, cuenta = np.unique(bloques, return_counts=True)
    usados = ub[cuenta >= MIN_BLOQUE]
    y = d["y_tr"]
    tabla = pd.DataFrame({m: [K.f1_macro(y[bloques == b], pred[m][bloques == b]) for b in usados] for m in modelos},
                         index=usados)
    K.log(f"{len(usados)} bloques de 2 km con ≥ {MIN_BLOQUE} viviendas (de {len(ub)})")

    fr = ES.friedman(tabla)
    cd = ES.diferencia_critica(len(modelos), len(tabla))
    nem = ES.nemenyi(tabla)
    K.log(f"Friedman χ² = {fr['chi2']:.1f}, gl = {fr['gl']}, p = {fr['p']:.2e}; CD = {cd:.2f}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ES.diagrama_cd(fr["rangos"], cd, titulo=f"Nemenyi sobre {len(tabla)} bloques de 2 km (todos los datos)", ax=ax)
    (K.RES / "figuras").mkdir(parents=True, exist_ok=True)
    fig.savefig(K.RES / "figuras" / "cd_mejoras.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- comparaciones dirigidas ----------------------------------------------------------------------------------
    n_val = np.mean([len(iva) for _, iva in d["folds"]])
    n_tr = np.mean([len(itr) for itr, _ in d["folds"]])
    te = mj["test"]
    y_te = d["y_te"]
    PTE = {}
    for m in modelos:
        r = K.leer(f"test_{m}")
        if r is not None:
            PTE[m] = K.pred(r["p"].astype(float))
    # stacking en el test: se reconstruye como en 03_test.py
    for nombre, info in mj["stacking"].items():
        Ztr = np.vstack([T3.rasgos([OOF[b][f] for b in info["bases"]]) for f in range(nf)])
        meta = LogisticRegression(max_iter=5000, **T3.META[info["meta"]]).fit(Ztr, np.concatenate(yv))
        Zte = T3.rasgos([K.leer(f"test_{b}")["p"].astype(float) for b in info["bases"]])
        PTE[nombre] = K.pred(_proba_completa(meta, Zte, meta.classes_))
    bte = d["test"].bloque.to_numpy()
    ubt = np.unique(bte)
    f1_te_bloque = {m: np.array([K.f1_macro(y_te[bte == b], PTE[m][bte == b]) for b in ubt]) for m in PTE}
    for m in PTE:   # control: el F1 del test reconstruido debe ser el de 03_test.py
        assert abs(K.f1_macro(y_te, PTE[m]) - te[m]["f1"]) < 1e-9, m

    def comparar(a, b):
        fa, fb = mj["cv"][a]["f1_pliegues"], mj["cv"][b]["f1_pliegues"]
        delta, mag = ES.delta_cliff(tabla[a], tabla[b])
        dif_te = f1_te_bloque[a] - f1_te_bloque[b]
        return {"a": a, "b": b,
                "bloques_dif_mediana": float(np.median(tabla[a] - tabla[b])),
                "bloques_a_favor": int((tabla[a] > tabla[b]).sum()), "n_bloques": len(tabla),
                "wilcoxon_p": wilcoxon_p(tabla[a], tabla[b]),
                "nemenyi_p": float(nem.loc[a, b]),
                "delta_cliff": delta, "magnitud": mag,
                "cv_dif": float(np.mean(fa) - np.mean(fb)), **{f"cv_{k}": v for k, v in
                                                               t_corregida(fa, fb, n_val, n_tr).items()},
                "test_dif": float(K.f1_macro(y_te, PTE[a]) - K.f1_macro(y_te, PTE[b])),
                "test_bloques_a_favor": int((dif_te > 0).sum()), "test_perm_p": permutacion_signo(dif_te)}

    familias = {"vs modelo base": [comparar(m, BASE) for m in modelos if m != BASE],
                "vs logística con vecindad": [comparar(m, LV) for m in modelos if m not in (BASE, LV)],
                "dirigidas": [comparar(a, b) for a, b in DIRIGIDAS]}
    for fam, filas in familias.items():
        for clave in ("wilcoxon_p", "cv_p", "test_perm_p"):
            aj = ES.holm([r[clave] for r in filas])
            bh = ES.benjamini_hochberg([r[clave] for r in filas])
            for r, h, b in zip(filas, aj, bh):
                r[f"{clave}_holm"] = float(h)
                r[f"{clave}_bh"] = float(b)
        K.log(f"--- {fam}")
        for r in sorted(filas, key=lambda r: -r["cv_dif"]):
            K.log(f"{r['a'][:42]:42s} bloques {r['bloques_a_favor']:3d}/{r['n_bloques']} Wilcoxon p_Holm "
                  f"{r['wilcoxon_p_holm']:.2g} δ {r['delta_cliff']:+.2f} ({r['magnitud']}) | CV {r['cv_dif']:+.3f} "
                  f"p_Holm {r['cv_p_holm']:.2g} | test {r['test_dif']:+.3f} ({r['test_bloques_a_favor']}/7) "
                  f"p_Holm {r['test_perm_p_holm']:.2g}")
    K.escribir_json("estadistica.json", {
        "bloques": {"min_viviendas": MIN_BLOQUE, "n_usados": len(usados), "n_total": len(ub)},
        "friedman": {"chi2": fr["chi2"], "gl": fr["gl"], "p": fr["p"], "cd": cd,
                     "rangos": fr["rangos"].round(3).to_dict()},
        "nemenyi_p": nem.round(4).to_dict(), "comparaciones": familias,
        "nota": ("Wilcoxon y Nemenyi sobre el F1 por bloque de 2 km (predicciones fuera de muestra); t corregida de "
                 "Nadeau-Bengio sobre los 5 pliegues; permutación exacta por cambio de signo sobre los 7 bloques de "
                 "test (p mínimo posible 2/128 = 0.016); Holm dentro de cada familia.")})


if __name__ == "__main__":
    main()
