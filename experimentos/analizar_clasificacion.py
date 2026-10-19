# -*- coding: utf-8 -*-
"""Análisis de la corrida combinatoria de clasificación (secciones 3, 5.4 y 6 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Lee ``resultados/tareas/`` (una tarea = combinación × pliegue externo) y produce:

- ``tabla_maestra_clasificacion.csv``: una fila por tarea (modelo, balanceo, optimizador, pliegue,
  hiperparámetros finales, métricas del bucle externo, tiempos, evaluaciones y semilla).
- ``resumen_combinaciones.csv``: media ± desviación estándar entre pliegues externos por combinación.
- ``analisis_clasificacion.json``: Friedman y Nemenyi (todas las combinaciones y la mejor por modelo, con
  bloques = bloques espaciales de 2 km del Entregable 1 sobre las predicciones fuera de muestra), comparación de
  optimizadores (calidad por presupuesto, tiempo, Friedman entre métodos), comparación de balanceos, DeLong por
  estrato con Holm entre finalistas, delta de Cliff e intervalos BCa por bloques.
- Figuras en ``resultados/figuras/``: diagramas de diferencia crítica, curvas de desempeño en cualquier momento
  y diversidad del algoritmo genético.

Uso (desde e3/):  ../venv_e2/Scripts/python experimentos/analizar_clasificacion.py
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import estadistica as ES  # noqa: E402

TAREAS = CF.RESULTADOS / "tareas"
FIG = CF.RESULTADOS / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
CLASES = CF.CLASES


def cargar_tareas():
    filas, proba = [], {}
    for f in glob.glob(str(TAREAS / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        fila = {k: r[k] for k in ["modelo", "balanceo", "metodo", "fold", "presupuesto", "semilla", "n_evaluaciones",
                                  "seg_busqueda", "seg_ajuste_final", "seg_inferencia", "n_train", "n_val",
                                  "f1_interno_mejor"]}
        fila.update(r["metricas"])
        fila["params"] = json.dumps(r["params"], ensure_ascii=False, default=float)
        fila["seg_por_evaluacion"] = r["seg_busqueda"] / max(1, r["n_evaluaciones"])
        fila["errores_busqueda"] = sum(1 for h in r["historial"] if h["error"])
        filas.append(fila)
        z = np.load(f.replace(".json", ".npz"))
        proba[(r["modelo"], r["balanceo"], r["metodo"], r["fold"])] = (z["iva"], z["proba"])
    return pd.DataFrame(filas), proba


def oof(proba, combo, n):
    """Probabilidades fuera de muestra de una combinación (las 5 partes del bucle externo)."""
    P = np.full((n, len(CLASES)), np.nan)
    for f in range(CF.K_EXTERNO):
        clave = combo + (f,)
        if clave in proba:
            iva, p = proba[clave]
            P[iva] = p
    return P


def main():
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    sub = paq["sub"]
    y = sub.estrato_num.to_numpy()
    bloques = sub.bloque.to_numpy()
    t, proba = cargar_tareas()
    t.to_csv(CF.RESULTADOS / "tabla_maestra_clasificacion.csv", index=False, encoding="utf-8")
    completas = t.groupby(["modelo", "balanceo", "metodo"]).fold.nunique()
    completas = completas[completas == CF.K_EXTERNO].index
    tc = t.set_index(["modelo", "balanceo", "metodo"]).loc[completas].reset_index()
    print(f"tareas: {len(t)} | combinaciones completas: {len(completas)}", flush=True)

    met = ["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "auc_ovr", "log_loss",
           "brier", "ece", "mae_ordinal", "acc_pm1", "kappa_cuadratico"]
    res = tc.groupby(["modelo", "balanceo", "metodo"])[met + ["seg_busqueda", "seg_por_evaluacion"]].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res = res.sort_values("f1_macro_mean", ascending=False)
    res.to_csv(CF.RESULTADOS / "resumen_combinaciones.csv", encoding="utf-8")
    salida = {"n_tareas": len(t), "n_combinaciones_completas": len(completas),
              "submuestra": len(sub), "knndm_externo": paq["knndm"]}

    # --- F1 por bloque espacial con las predicciones fuera de muestra (bloques de Friedman) -------------------
    combos = list(completas)
    ub = np.unique(bloques)
    f1_bloque = {}
    for c in combos:
        P = oof(proba, c, len(sub))
        ok = ~np.isnan(P).any(1)
        pred = CLASES[np.argmax(np.nan_to_num(P, nan=-1), axis=1)]
        f1_bloque[" | ".join(c)] = [f1_score(y[(bloques == b) & ok], pred[(bloques == b) & ok], average="macro",
                                             zero_division=0) if ((bloques == b) & ok).any() else np.nan
                                    for b in ub]
    tb = pd.DataFrame(f1_bloque, index=ub)
    tb = tb.dropna(axis=0)          # bloques sin predicciones (solo con resultados parciales)
    tf = tc.pivot_table(index="fold", columns=["modelo", "balanceo", "metodo"], values="f1_macro")
    tf.columns = [" | ".join(c) for c in tf.columns]

    def bloque_friedman(tabla, nombre, max_cd=30):
        fr = ES.friedman(tabla)
        cd = ES.diferencia_critica(tabla.shape[1], tabla.shape[0])
        ES.diagrama_cd(fr["rangos"], cd, titulo=f"Diferencia crítica (Nemenyi) — {nombre}", max_modelos=max_cd)
        plt.savefig(FIG / f"cd_{nombre.replace(' ', '_')}.png", dpi=130, bbox_inches="tight")
        plt.close()
        return {"chi2": fr["chi2"], "gl": fr["gl"], "p": fr["p"], "n_bloques": tabla.shape[0],
                "cd": cd, "rangos": fr["rangos"].round(3).to_dict()}

    if tb.shape[1] >= 3:
        salida["friedman_todas_bloques"] = bloque_friedman(tb, "todas las combinaciones (bloques espaciales)")
        salida["friedman_todas_pliegues"] = bloque_friedman(tf, "todas las combinaciones (pliegues externos)")
        # mejor combinación de cada modelo (por F1 medio externo)
        mejor = res.reset_index().groupby("modelo").head(1)
        cols = [" | ".join(r) for r in mejor[["modelo", "balanceo", "metodo"]].itertuples(index=False)]
        tbm = tb[[c for c in cols if c in tb]]
        tbm.columns = [c.split(" | ")[0] for c in tbm.columns]
        if tbm.shape[1] >= 3:
            salida["friedman_mejor_por_modelo"] = bloque_friedman(tbm, "mejor combinación por modelo")
            salida["nemenyi_mejor_por_modelo"] = ES.nemenyi(tbm).round(4).to_dict()
        salida["mejor_por_modelo"] = mejor[["modelo", "balanceo", "metodo", "f1_macro_mean", "f1_macro_std",
                                            "auc_ovr_mean", "mae_ordinal_mean"]].to_dict(orient="records")

    # --- Optimizadores: calidad por presupuesto, tiempo, Friedman entre métodos ------------------------------
    op = tc.groupby("metodo")[["f1_macro", "f1_interno_mejor", "seg_busqueda", "seg_por_evaluacion",
                               "n_evaluaciones"]].agg(["mean", "std"])
    op.columns = [f"{a}_{b}" for a, b in op.columns]
    salida["optimizadores"] = op.round(4).to_dict(orient="index")
    tm = tc.pivot_table(index=["modelo", "balanceo", "fold"], columns="metodo", values="f1_macro").dropna()
    if tm.shape[1] >= 3 and len(tm) > 3:
        fr = ES.friedman(tm)
        salida["friedman_optimizadores"] = {"chi2": fr["chi2"], "p": fr["p"], "n_bloques": len(tm),
                                            "rangos": fr["rangos"].round(3).to_dict(),
                                            "cd": ES.diferencia_critica(tm.shape[1], len(tm))}
    # curvas de desempeño en cualquier momento: mejor F1 interno hasta la evaluación t, normalizado por tarea
    curvas = {}
    for f in glob.glob(str(TAREAS / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        s = np.array([h["f1_interno"] for h in r["historial"]])
        if len(s) == 0:
            continue
        mejor_t = np.maximum.accumulate(s)
        curvas.setdefault(r["metodo"], []).append(mejor_t / max(mejor_t[-1], 1e-9))
    fig, ax = plt.subplots(figsize=(8, 5))
    resumen_curvas = {}
    for mtd, cs in curvas.items():
        L = max(len(c) for c in cs)
        M = np.array([np.pad(c, (0, L - len(c)), mode="edge") for c in cs])
        media = M.mean(0)
        resumen_curvas[mtd] = {"media": media.round(4).tolist(), "area_bajo_curva": float(media.mean()),
                               "evals_para_99pct": int(np.argmax(media >= 0.99) + 1) if (media >= 0.99).any() else None}
        ax.plot(np.arange(1, L + 1), media, marker="o", label=mtd)
    ax.set_xlabel("evaluación t")
    ax.set_ylabel("mejor F1 interno hasta t / mejor final")
    ax.set_title("Desempeño en cualquier momento (promedio de todas las tareas)")
    ax.legend()
    plt.savefig(FIG / "anytime_optimizadores.png", dpi=130, bbox_inches="tight")
    plt.close()
    salida["anytime"] = resumen_curvas
    # diversidad del genético
    div = []
    for f in glob.glob(str(TAREAS / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if r["metodo"] == "Genética" and r["extra"].get("diversidad"):
            for g in r["extra"]["diversidad"]:
                div.append({"modelo": r["modelo"], **g})
    if div:
        dv = pd.DataFrame(div).groupby("generacion").diversidad.agg(["mean", "std", "count"])
        salida["diversidad_genetica"] = dv.round(4).to_dict(orient="index")
        dv["mean"].plot(marker="o", title="Diversidad genética media por generación")
        plt.ylabel("distancia media entre individuos (genoma en [0, 1]^d)")
        plt.savefig(FIG / "diversidad_genetica.png", dpi=130, bbox_inches="tight")
        plt.close()

    # --- Balanceo ---------------------------------------------------------------------------------------
    bl = tc.groupby("balanceo")[["f1_macro", "recall_macro", "auc_ovr", "ece", "seg_busqueda"]].agg(["mean", "std"])
    bl.columns = [f"{a}_{b}" for a, b in bl.columns]
    salida["balanceo"] = bl.round(4).to_dict(orient="index")
    tbal = tc.pivot_table(index=["modelo", "metodo", "fold"], columns="balanceo", values="f1_macro").dropna()
    if tbal.shape[1] >= 3 and len(tbal) > 3:
        fr = ES.friedman(tbal)
        salida["friedman_balanceo"] = {"chi2": fr["chi2"], "p": fr["p"], "n_bloques": len(tbal),
                                       "rangos": fr["rangos"].round(3).to_dict()}

    # --- Finalistas: DeLong por estrato (Holm), Cliff, BCa por bloques ------------------------------------
    if "mejor_por_modelo" in salida:
        top = sorted(salida["mejor_por_modelo"], key=lambda r: -r["f1_macro_mean"])[:3]
        base = res.reset_index()
        base = base[base.modelo == "Logística L1/L2"].head(1)
        fin = [(r["modelo"], r["balanceo"], r["metodo"]) for r in top]
        if len(base) and tuple(base.iloc[0][["modelo", "balanceo", "metodo"]]) not in fin:
            fin.append(tuple(base.iloc[0][["modelo", "balanceo", "metodo"]]))
        P = {c: oof(proba, c, len(sub)) for c in fin}
        pruebas = []
        for i in range(len(fin)):
            for j in range(i + 1, len(fin)):
                for k_i, k in enumerate(CLASES):
                    d = ES.delong(y == k, P[fin[i]][:, k_i], P[fin[j]][:, k_i])
                    pruebas.append({"A": " | ".join(fin[i]), "B": " | ".join(fin[j]), "estrato": int(k), **d})
        if pruebas:
            dl = pd.DataFrame(pruebas)
            dl["p_holm"] = ES.holm(dl.p)
            dl["p_bh"] = ES.benjamini_hochberg(dl.p)
            salida["delong_finalistas"] = dl.round(5).to_dict(orient="records")
        cliff, ic = [], {}
        for c in fin:
            nom = " | ".join(c)
            pred = CLASES[np.argmax(P[c], axis=1)]
            ic[nom] = ES.bootstrap_bca_bloques(lambda idx: f1_score(y[idx], pred[idx], average="macro",
                                                                    zero_division=0), bloques, n_resamples=999)
        for i in range(len(fin)):
            for j in range(i + 1, len(fin)):
                a, b = " | ".join(fin[i]), " | ".join(fin[j])
                d, etq = ES.delta_cliff(tb[a], tb[b]) if a in tb and b in tb else (np.nan, "")
                cliff.append({"A": a, "B": b, "delta_cliff_bloques": d, "magnitud": etq})
        salida["cliff_finalistas"] = cliff
        salida["ic_bca_f1_finalistas"] = {k: list(v) for k, v in ic.items()}

    (CF.RESULTADOS / "analisis_clasificacion.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("guardado analisis_clasificacion.json", flush=True)


if __name__ == "__main__":
    main()
