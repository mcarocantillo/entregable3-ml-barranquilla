# -*- coding: utf-8 -*-
"""Análisis de la rama de regresión (secciones 5.2 y 6.3 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Secuencia jerárquica: MCS sobre las 28 combinaciones → Giacomini-White entre finalistas no anidados y Clark-West
para el par anidado (Ridge con y sin vecindad física) → Diebold-Mariano (HLN) entre el mejor y el segundo, con
bootstrap estacionario → tamaño del efecto (d de Cohen sobre la diferencia de pérdidas). Además, supuestos de los
residuos del mejor modelo (White, BDS, Ljung-Box/ACF, histograma y normalidad, I de Moran) y la figura de
entrenamiento / validación / prueba con sus predicciones.

Uso (desde e3/):  ../venv_e2/Scripts/python -u experimentos/analizar_regresion.py
"""
import glob
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import estadistica_regresion as ER  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

warnings.filterwarnings("ignore")
FIG = CF.RESULTADOS / "figuras"
TAREAS = CF.RESULTADOS / "tareas_regresion"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    from estrato3 import datos as DT
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    d = DT.cargar()
    paq["M1"] = d["M1"]
    sub, test = paq["sub"], d["test"]
    y = sub.estrato_num.to_numpy().astype(float)
    filas, oof = [], {}
    for f in glob.glob(str(TAREAS / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        fila = {k: r[k] for k in ["modelo", "metodo", "fold", "n_evaluaciones", "seg_busqueda", "seg_ajuste_final",
                                  "seg_inferencia", "rmse_interno_mejor"]}
        fila.update({k: v for k, v in r["metricas"].items()})
        fila["params"] = json.dumps(r["params"], ensure_ascii=False, default=float)
        filas.append(fila)
        z = np.load(f.replace(".json", ".npz"))
        oof.setdefault((r["modelo"], r["metodo"]), np.full(len(sub), np.nan))[z["iva"]] = z["pred"]
    t = pd.DataFrame(filas)
    t.to_csv(CF.RESULTADOS / "tabla_maestra_regresion.csv", index=False, encoding="utf-8")
    res = t.groupby(["modelo", "metodo"])[["rmse", "mae", "r2", "f1_macro_redondeado", "seg_busqueda"]].agg(
        ["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.sort_values("rmse_mean")
    res.to_csv(CF.RESULTADOS / "resumen_regresion.csv", encoding="utf-8")
    salida = {"n_tareas": len(t)}

    # pérdidas fuera de muestra en orden espacial (sur → norte)
    orden = np.argsort(sub.y_km.to_numpy())
    completos = {k: v for k, v in oof.items() if not np.isnan(v).any()}
    L = pd.DataFrame({" | ".join(k): (y - v)[orden] ** 2 for k, v in completos.items()})
    mcs = ER.mcs(L, alpha=0.10)
    salida["mcs"] = mcs.round(4).reset_index().to_dict(orient="records")
    fin = list(mcs[mcs.en_el_conjunto].index)[:4] or list(mcs.index[:2])
    E = {k: (y - completos[tuple(k.split(" | "))])[orden] for k in mcs.index}
    gw = []
    for i in range(len(fin)):
        for j in range(i + 1, len(fin)):
            gw.append({"A": fin[i], "B": fin[j], **ER.giacomini_white(E[fin[i]], E[fin[j]])})
    salida["giacomini_white"] = gw
    mejor, segundo = mcs.index[0], mcs.index[1]
    dm = ER.dm_hln(E[mejor], E[segundo])
    dif = E[mejor] ** 2 - E[segundo] ** 2
    dm["ic95_bootstrap_estacionario"] = ER.bootstrap_estacionario_ic(dif)
    dm["d_cohen"] = float(dif.mean() / dif.std(ddof=1))
    salida["diebold_mariano_hln"] = {"mejor": mejor, "segundo": segundo, **dm}

    # Clark-West: Ridge con vecindad (general) frente a Ridge sin vecindad (restringido, anidado)
    ridge = [k for k in completos if k[0] == "Ridge"]
    if ridge:
        r = t[(t.modelo == "Ridge") & (t.metodo == ridge[0][1])]
        alpha = float(np.median([json.loads(p)["alpha"] for p in r.params]))
        sin_vec = dict(paq)
        sin_vec["num"] = [c for c in paq["num"] if not c.startswith(f"knn{CF.K_VECINDAD}_")]
        pred_r = np.full(len(sub), np.nan)
        for itr, iva in paq["externos"]:
            est = MD.pipeline_reg(sin_vec, "Ridge", {"alpha": alpha}).fit(sub.iloc[itr], y[itr])
            pred_r[iva] = est.predict(sub.iloc[iva])
        cw = ER.clark_west(y[orden], pred_r[orden], completos[ridge[0]][orden])
        salida["clark_west_ridge_vecindad"] = {"alpha": alpha, **cw}

    # residuos del mejor modelo
    k_mejor = tuple(mejor.split(" | "))
    res_mejor = (y - completos[k_mejor])[orden]
    Xw = np.c_[completos[k_mejor][orden], sub.y_km.to_numpy()[orden], sub.x_km.to_numpy()[orden]]
    sup = ER.supuestos_residuos(res_mejor, X_white=Xw)
    C1, _, _, V1 = CF.importar_e1()
    xy_g, _, cod = V1.puntos_unicos(sub[["x_km", "y_km"]].to_numpy(), sub.npn_edificio.to_numpy())
    r_g = pd.Series(y - completos[k_mejor]).groupby(cod).mean().to_numpy()
    sup["moran_residuos_k8"] = float(V1.moran_por_k(xy_g, r_g, [8]).moran_I.iloc[0])
    salida["supuestos_mejor"] = sup
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].bar(range(len(sup["acf"])), sup["acf"], color="#2a78d6")
    ax[0].set_title("ACF de los residuos (orden sur → norte)")
    ax[1].hist(res_mejor, bins=60, color="#86b6ef")
    ax[1].set_title("Histograma de los residuos")
    ax[2].scatter(completos[k_mejor][orden], res_mejor, s=2, alpha=0.3)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("predicción")
    ax[2].set_ylabel("residuo")
    ax[2].set_title("Residuos frente a la predicción")
    plt.suptitle(f"Diagnóstico de residuos — {mejor}")
    plt.savefig(FIG / "residuos_regresion.png", dpi=120, bbox_inches="tight")
    plt.close()

    # entrenamiento / validación / prueba con las predicciones del mejor modelo
    r = t[(t.modelo == k_mejor[0]) & (t.metodo == k_mejor[1])]
    params = json.loads(r.sort_values("rmse").params.iloc[0])
    est = MD.pipeline_reg(paq, k_mejor[0], params).fit(sub, y)
    p_tr = est.predict(sub)
    p_te = est.predict(test)
    yte = test.estrato_num.to_numpy().astype(float)
    salida["test_mejor"] = {"modelo": mejor, "params": params, "rmse": float(np.sqrt(np.mean((yte - p_te) ** 2))),
                            "mae": float(np.mean(np.abs(yte - p_te))),
                            "f1_macro_redondeado": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                                yte, np.clip(np.rint(p_te), 1, 6), average="macro", zero_division=0))}
    rng = np.random.default_rng(CF.SEED)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for a, (nom, yy, pp) in zip(ax, [("entrenamiento (ajuste)", y, p_tr), ("validación (fuera de muestra)", y,
                                     completos[k_mejor]), ("prueba (test)", yte, p_te)]):
        jit = rng.uniform(-0.25, 0.25, len(yy))
        a.scatter(yy + jit, pp, s=2, alpha=0.25)
        a.plot([1, 6], [1, 6], "r--")
        rmse = float(np.sqrt(np.mean((yy - pp) ** 2)))
        a.set_title(f"{nom}: RMSE {rmse:.3f}")
        a.set_xlabel("estrato real (con ruido para ver la densidad)")
    ax[0].set_ylabel("estrato predicho (número)")
    plt.savefig(FIG / "regresion_train_val_test.png", dpi=120, bbox_inches="tight")
    plt.close()
    (CF.RESULTADOS / "analisis_regresion.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1,
                                                                      default=float), encoding="utf-8")
    print("guardado analisis_regresion.json", flush=True)


if __name__ == "__main__":
    main()
