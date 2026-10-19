# -*- coding: utf-8 -*-
"""Finalistas: evaluación en el test, calibración, robustez a semillas e interpretabilidad (secciones 5.1, 5.3 y 5.4).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para cada modelo se toma su mejor combinación (balanceo y optimizador con el mayor F1 macro medio del bucle
externo). Sus hiperparámetros finales salen de repetir esa búsqueda en toda la submuestra (con los pliegues
externos como validación interna), y el modelo se ajusta en la submuestra y se evalúa UNA vez en el test del
Entregable 1 (los mismos 7 bloques de siempre).

Salidas (``resultados/finalistas.json`` y figuras):
- tabla de métricas en test (accuracy, precision, recall, F1, AUC, MAE ordinal) con IC BCa por bloques;
- matrices de confusión y curvas ROC (uno contra el resto);
- calibración: diagrama de confiabilidad, Brier y ECE, antes y después de recalibrar (Platt e isotónica,
  ajustadas con validación cruzada dentro de la submuestra);
- sensibilidad a 5 semillas (modelos con aleatoriedad);
- el modelo base del Entregable 1 (logística B, sin vecindad) como referencia, y la comparación del XGBoost
  geográfico con cada rival (criterio de reemplazo en los pliegues externos y diferencia pareada en test);
- SHAP (resumen global y explicaciones locales de una vivienda bien clasificada y una con error grande) y LIME
  para XGBoost, el XGBoost geográfico y el mejor modelo en test.

Las búsquedas, recalibraciones y semillas corren en paralelo (joblib, 8 procesos; ``estrato3/experimento.py``).

Uso (desde e3/, después de la corrida y de analizar_clasificacion.py):
    ../venv_e2/Scripts/python -u experimentos/finalistas.py
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_curve  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import estadistica as ES  # noqa: E402
from estrato3 import evaluacion as EV  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402
from estrato3 import optimizacion as OP  # noqa: E402

warnings.filterwarnings("ignore")
FIG = CF.RESULTADOS / "figuras"
CLASES = CF.CLASES


BASE = "Logística B (modelo base del E1)"


def archivo(nombre):
    """Nombre de archivo seguro para un modelo («Logística L1/L2» → «Logística_L1-L2»)."""
    return nombre.replace(" ", "_").replace("/", "-")


def guardar(salida):
    (CF.RESULTADOS / "finalistas.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=float),
                                                    encoding="utf-8")


def modelo_base(paq, test, yte):
    """Modelo base del Entregable 1 (logística B v3: multinomial, C = 0.01, sin pesos, SIN vecindad física).

    Se ajusta con la misma submuestra, en los 5 pliegues externos (para la comparación pareada) y en el test.
    """
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import Pipeline
    datos_b = dict(paq, num=[c for c in paq["num"] if not c.startswith("knn")])
    sub = paq["sub"]
    y = sub.estrato_num.to_numpy()

    def nuevo():
        return Pipeline([("prep", MD.preprocesador(datos_b)), ("modelo", LogisticRegression(C=0.01, max_iter=1000))])

    f1_pliegues = []
    for itr, iva in paq["externos"]:
        e = nuevo().fit(sub.iloc[itr], y[itr])
        f1_pliegues.append(f1_score(y[iva], e.predict(sub.iloc[iva]), average="macro", zero_division=0))
    return nuevo().fit(sub, y), f1_pliegues


def main():
    from joblib import Parallel, delayed
    from estrato3 import experimento as EX
    FIG.mkdir(parents=True, exist_ok=True)
    CF.fijar_semillas()
    from estrato3 import datos as DT
    d = DT.cargar()
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    paq["M1"] = d["M1"]
    sub, test = paq["sub"], d["test"][paq["sub"].columns]
    with open(EX.TEST_CACHE, "wb") as fh:
        pickle.dump(test, fh)
    ysub, yte = sub.estrato_num.to_numpy(), test.estrato_num.to_numpy()
    bloques_te = test.bloque.to_numpy()
    resumen = pd.read_csv(CF.RESULTADOS / "resumen_combinaciones.csv")
    mejores = resumen.sort_values("f1_macro_mean", ascending=False).groupby("modelo").head(1)
    salida = {"finalistas": {}}
    proba_te = {}

    def resumen_test(P):
        m = EV.metricas(yte, P)
        pred = CLASES[P.argmax(1)]
        ic = ES.bootstrap_bca_bloques(lambda idx: f1_score(yte[idx], pred[idx], average="macro", zero_division=0),
                                      bloques_te, n_resamples=999)
        return m, ic, confusion_matrix(yte, pred, labels=CLASES).tolist()

    # 1) búsqueda final de cada finalista, en paralelo (una tarea por modelo; las más caras primero)
    res = Parallel(n_jobs=8, backend="loky")(delayed(EX.finalista)(r.modelo, r.balanceo, r.metodo)
                                             for _, r in mejores.iterrows())
    for r in res:
        nombre = r["modelo"]
        proba_te[nombre] = r["proba_test"]
        m, ic, conf = resumen_test(r["proba_test"])
        salida["finalistas"][nombre] = {"balanceo": r["balanceo"], "metodo": r["metodo"], "params": r["params"],
                                        "f1_busqueda_final": r["f1_busqueda_final"], "test": m,
                                        "ic95_bca_f1_test": ic, "segundos": r["segundos"], "confusion": conf}
        print(f"{nombre} ({r['balanceo']}, {r['metodo']}): F1 test {m['f1_macro']:.3f} [{ic[0]:.3f}, {ic[1]:.3f}]",
              flush=True)

    # 2) modelo base del E1 (sin vecindad) como referencia directa
    est_b, f1_b = modelo_base(paq, test, yte)
    proba_te[BASE] = EV.proba_completa(est_b, test)
    m, ic, conf = resumen_test(proba_te[BASE])
    salida["modelo_base"] = {"nombre": BASE, "params": {"C": 0.01, "class_weight": None},
                             "variables": "modelo B sin vecindad física", "f1_pliegues_externos": f1_b,
                             "test": m, "ic95_bca_f1_test": ic, "confusion": conf}
    print(f"{BASE}: F1 externo {np.mean(f1_b):.3f}; F1 test {m['f1_macro']:.3f} [{ic[0]:.3f}, {ic[1]:.3f}]", flush=True)
    np.savez_compressed(CF.RESULTADOS / "proba_test_finalistas.npz",
                        **{archivo(k): v.astype(np.float32) for k, v in proba_te.items()}, y=yte, bloque=bloques_te)
    guardar(salida)

    # 3) XGBoost geográfico frente a cada rival: criterio de reemplazo en los pliegues externos y diferencia en test
    tm = pd.read_csv(CF.RESULTADOS / "tabla_maestra_clasificacion.csv")
    geo = "XGBoost geográfico"

    def f1_pliegues(nombre):
        if nombre == BASE:
            return np.asarray(f1_b)
        fi = salida["finalistas"][nombre]
        t = tm[(tm.modelo == nombre) & (tm.balanceo == fi["balanceo"]) & (tm.metodo == fi["metodo"])]
        return t.sort_values("fold").f1_macro.to_numpy()

    pred_geo = CLASES[proba_te[geo].argmax(1)]
    comp = []
    for rival in [BASE, "Logística L1/L2", "XGBoost", "Random Forest"]:
        dif = f1_pliegues(geo) - f1_pliegues(rival)
        se = dif.std(ddof=1) / np.sqrt(len(dif))
        pred_r = CLASES[proba_te[rival].argmax(1)]

        def dif_f1(idx, a=pred_geo, b=pred_r):
            return (f1_score(yte[idx], a[idx], average="macro", zero_division=0)
                    - f1_score(yte[idx], b[idx], average="macro", zero_division=0))
        todos = np.arange(len(yte))
        ic = ES.bootstrap_bca_bloques(dif_f1, bloques_te, n_resamples=999)
        por_bloque = [dif_f1(np.where(bloques_te == b)[0]) for b in np.unique(bloques_te)]
        comp.append({"rival": rival, "cv_dif_media": float(dif.mean()), "cv_se": float(se),
                     "cv_pliegues_a_favor": int((dif > 0).sum()), "cv_supera_1se": bool(dif.mean() > se),
                     "test_dif": float(dif_f1(todos)), "test_ic95_bca": list(ic),
                     "test_bloques_a_favor": int(np.sum(np.asarray(por_bloque) > 0)), "test_n_bloques": len(por_bloque)})
        print(f"geo vs {rival}: CV {dif.mean():+.3f} (SE {se:.3f}, {(dif > 0).sum()}/5) | test {dif_f1(todos):+.3f} "
              f"[{ic[0]:+.3f}, {ic[1]:+.3f}], {np.sum(np.asarray(por_bloque) > 0)}/{len(por_bloque)} bloques", flush=True)
    salida["geo_vs_rivales"] = comp
    guardar(salida)

    # 4) matrices de confusión y ROC
    n = len(proba_te)
    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    for a, (nombre, P) in zip(ax.ravel(), proba_te.items()):
        ConfusionMatrixDisplay(confusion_matrix(yte, CLASES[P.argmax(1)], labels=CLASES, normalize="true"),
                               display_labels=CLASES).plot(ax=a, colorbar=False, values_format=".2f")
        a.set_title(nombre, fontsize=10)
    for a in ax.ravel()[n:]:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_finalistas.png", dpi=110)
    plt.close()
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for k_i, (a, k) in enumerate(zip(ax.ravel(), CLASES)):
        for nombre, P in proba_te.items():
            fpr, tpr, _ = roc_curve(yte == k, P[:, k_i])
            a.plot(fpr, tpr, lw=1.3, label=nombre)
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        a.set_title(f"ROC estrato {k} contra el resto")
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "roc_finalistas.png", dpi=120)
    plt.close()

    # 5) calibración y recalibración (Platt / isotónica) con los pliegues externos, en paralelo
    fin = salida["finalistas"]
    cals = Parallel(n_jobs=8, backend="loky")(
        delayed(EX.calibrado)(nombre, fin[nombre]["params"], fin[nombre]["balanceo"], met)
        for nombre in fin for met in ["sigmoid", "isotonic"])
    cal = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre in fin:
        P = proba_te[nombre]
        acierto = (CLASES[P.argmax(1)] == yte).astype(int)
        fr, mp = calibration_curve(acierto, P.max(1), n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=nombre)
        cal[nombre] = {"original": {"brier": EV.metricas(yte, P)["brier"], "ece": EV.ece(yte, P)}}
    for nombre, met, Pc in cals:
        cal[nombre]["platt" if met == "sigmoid" else "isotonica"] = {
            "brier": EV.metricas(yte, Pc)["brier"], "ece": EV.ece(yte, Pc), "f1": EV.metricas(yte, Pc)["f1_macro"]}
    for nombre in fin:
        print(f"calibración {nombre}: {cal[nombre]}", flush=True)
    ax.plot([0, 1], [0, 1], "k:")
    ax.set_xlabel("confianza predicha (probabilidad del estrato elegido)")
    ax.set_ylabel("proporción de aciertos")
    ax.set_title("Diagrama de confiabilidad en test")
    ax.legend(fontsize=7)
    plt.savefig(FIG / "confiabilidad_finalistas.png", dpi=120, bbox_inches="tight")
    plt.close()
    salida["calibracion"] = cal
    guardar(salida)

    # 6) sensibilidad a semillas (modelos con aleatoriedad), en paralelo
    aleatorios = [n for n in ["Random Forest", "XGBoost", "XGBoost geográfico", "SVM"] if n in fin]
    sems = Parallel(n_jobs=8, backend="loky")(
        delayed(EX.con_semilla)(nombre, fin[nombre]["params"], fin[nombre]["balanceo"], s)
        for nombre in aleatorios for s in [1, 7, 42, 123, 2026])
    sem = {}
    for nombre in aleatorios:
        f1s = [EV.metricas(yte, P)["f1_macro"] for n_, s, P in sems if n_ == nombre]
        sem[nombre] = {"f1": f1s, "media": float(np.mean(f1s)), "sd": float(np.std(f1s, ddof=1))}
        print(f"semillas {nombre}: {np.mean(f1s):.3f} ± {np.std(f1s, ddof=1):.3f}", flush=True)
    salida["semillas"] = sem
    guardar(salida)

    # 7) SHAP y LIME: XGBoost (lo pide la guía), el XGBoost geográfico (el modelo nuevo) y el mejor en test
    mejor = max(fin, key=lambda k: fin[k]["test"]["f1_macro"])
    modelos_finales = {}
    for nombre in dict.fromkeys(["XGBoost", geo, mejor]):
        modelos_finales[nombre] = MD.pipeline(paq, nombre, fin[nombre]["params"], fin[nombre]["balanceo"]).fit(sub, ysub)
    salida["interpretabilidad"] = interpretar(salida, modelos_finales, paq, sub, test, yte, proba_te)
    guardar(salida)
    print("guardado finalistas.json", flush=True)


def interpretar(salida, modelos_finales, paq, sub, test, yte, proba_te):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    out = {}
    for nombre in modelos_finales:
        est = modelos_finales[nombre]
        prep = est.named_steps["prep"]
        modelo = est.named_steps["modelo"]
        cols = list(prep.get_feature_names_out())
        Xs = prep.transform(sub.sample(2000, random_state=CF.SEED))
        Xt = prep.transform(test)
        rng = np.random.default_rng(CF.SEED)
        idx_t = rng.choice(len(test), 600, replace=False)
        f = (lambda Z: modelo.predict_proba(Z))
        if nombre == "XGBoost":
            expl = shap.TreeExplainer(modelo.modelo_)
            sv = expl.shap_values(Xt[idx_t])
        else:
            fondo = shap.sample(Xs, 100, random_state=CF.SEED)
            expl = shap.PermutationExplainer(f, fondo)
            sv = expl(Xt[idx_t[:200]]).values
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[0] != len(CLASES):
            sv = np.moveaxis(sv, -1, 0)          # (clases, n, p)
        importancia = np.abs(sv).mean(axis=(0, 1)) if sv.ndim == 3 else np.abs(sv).mean(0)
        orden = np.argsort(importancia)[::-1][:15]
        out[nombre] = {"shap_importancia_global": {cols[i]: float(importancia[i]) for i in orden}}
        plt.figure(figsize=(8, 6))
        plt.barh([cols[i].split("__")[-1] for i in orden][::-1], importancia[orden][::-1], color="#2a78d6")
        plt.title(f"SHAP: importancia global media |φ| — {nombre}")
        plt.savefig(FIG / f"shap_global_{archivo(nombre)}.png", dpi=120, bbox_inches="tight")
        plt.close()
        # una vivienda bien clasificada y una con error grande (≥ 2 estratos)
        pred = CLASES[proba_te[nombre].argmax(1)]
        bien = int(np.where(pred == yte)[0][0])
        mal_c = np.where(np.abs(pred - yte) >= 2)[0]
        mal = int(mal_c[0]) if len(mal_c) else int(np.where(pred != yte)[0][0])
        lime_exp = LimeTabularExplainer(Xs, feature_names=cols, class_names=[str(k) for k in CLASES],
                                        discretize_continuous=True, random_state=CF.SEED)
        locales = {}
        for etq, i in [("bien_clasificada", bien), ("error_grande", mal)]:
            k_pred = int(np.argmax(proba_te[nombre][i]))
            if nombre == "XGBoost":
                phi = np.asarray(expl.shap_values(Xt[i:i + 1]))
                phi = phi[k_pred, 0] if phi.ndim == 3 and phi.shape[0] == len(CLASES) else phi[0, :, k_pred]
            else:
                phi = np.asarray(expl(Xt[i:i + 1]).values)[0][:, k_pred]
            top = np.argsort(np.abs(phi))[::-1][:8]
            le = lime_exp.explain_instance(Xt[i], f, labels=(k_pred,), num_features=8)
            locales[etq] = {"real": int(yte[i]), "predicho": int(CLASES[k_pred]),
                            "shap_top": {cols[j]: float(phi[j]) for j in top},
                            "lime_top": dict(le.as_list(label=k_pred))}
        out[nombre]["locales"] = locales
        print(f"interpretabilidad {nombre}: listo", flush=True)
    return out


if __name__ == "__main__":
    main()
