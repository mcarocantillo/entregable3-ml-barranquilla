# -*- coding: utf-8 -*-
"""Interpretabilidad global y local con SHAP, y comparación con LIME (sección 5.3 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Para el mejor modelo de cada tarea y para XGBoost (que la guía pide explícitamente), además del modelo nuevo:

- Clasificación: XGBoost (TreeSHAP exacto), XGBoost geográfico y el mejor finalista en test (SHAP por
  permutaciones, que no depende de la forma del modelo).
- Regresión: el mejor modelo según el test (Random Forest, TreeSHAP exacto).

Figuras: importancia global (|φ| medio), *summary plot* (beeswarm: cada punto es una vivienda, el color es el
valor de la variable y la posición su efecto) para los estratos 1 y 6 y, en regresión, para el estrato como
número; *waterfall* de una vivienda bien clasificada y de una con error grande (≥ 2 estratos); y la explicación
LIME de esas mismas viviendas. Las viviendas se toman del test con una semilla fija.

Salida: ``resultados/interpretabilidad.json`` y ``resultados/figuras/shap_*.png``, ``lime_*.png``.

Uso (desde e3/, después de finalistas.py y analizar_regresion.py):
    ../venv_e2/Scripts/python -u experimentos/interpretabilidad.py
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from estrato3 import configuracion as CF  # noqa: E402
from estrato3 import experimento as EX  # noqa: E402
from estrato3 import modelos as MD  # noqa: E402

warnings.filterwarnings("ignore")
FIG = CF.RESULTADOS / "figuras"
CLASES = CF.CLASES
N_GLOBAL = {"arbol": 800, "permutacion": 200}


def archivo(nombre):
    return nombre.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")


def nombres_cortos(cols):
    """«base__num__log_area» → «log_area»; sin prefijos del ColumnTransformer.

    En el XGBoost geográfico, x_km e y_km aparecen dos veces: estandarizadas entre las variables del modelo y en
    km (sin escalar) al final, para ubicar las anclas. Las segundas se llaman «x_km (ancla)» y «y_km (ancla)».
    """
    return [c.split("__")[-1] + (" (ancla)" if c.startswith("xy__") else "") for c in cols]


def guardar(nombre_fig):
    plt.savefig(FIG / nombre_fig, dpi=120, bbox_inches="tight")
    plt.close("all")


def explicacion_clase(shap, ex, k_i):
    """Corte de una explicación multiclase (n, p, K) para la clase ``k_i``."""
    return shap.Explanation(values=ex.values[..., k_i], base_values=np.asarray(ex.base_values)[..., k_i],
                            data=ex.data, feature_names=ex.feature_names)


def clasificacion(shap, LimeTabularExplainer, nombre, est, sub, test, yte, P):
    prep, modelo = est.named_steps["prep"], est.named_steps["modelo"]
    cols = nombres_cortos(prep.get_feature_names_out())
    Xs = np.asarray(prep.transform(sub.sample(2000, random_state=CF.SEED)), dtype=float)
    Xt = np.asarray(prep.transform(test), dtype=float)
    rng = np.random.default_rng(CF.SEED)
    f = (lambda Z: modelo.predict_proba(Z))
    if nombre == "XGBoost":
        idx = rng.choice(len(test), N_GLOBAL["arbol"], replace=False)
        expl = shap.TreeExplainer(modelo.modelo_)
        explicar = (lambda Z: expl(Z))
    else:
        idx = rng.choice(len(test), N_GLOBAL["permutacion"], replace=False)
        # fondo de 50 viviendas de la submuestra; 500 evaluaciones por vivienda (≈ 5 permutaciones antitéticas)
        expl = shap.PermutationExplainer(f, shap.sample(Xs, 50, random_state=CF.SEED), seed=CF.SEED)
        explicar = (lambda Z: expl(Z))
    ex = explicar(Xt[idx])
    ex.feature_names = cols
    vals = np.asarray(ex.values)                     # (n, p, K)
    importancia = np.abs(vals).mean(axis=(0, 2))
    orden = np.argsort(importancia)[::-1]
    out = {"n_viviendas_global": len(idx), "importancia_global": {cols[i]: float(importancia[i]) for i in orden[:20]},
           "importancia_por_estrato": {int(k): {cols[i]: float(np.abs(vals[:, i, k_i]).mean()) for i in orden[:10]}
                                       for k_i, k in enumerate(CLASES)}}
    plt.figure(figsize=(8, 6))
    top = orden[:15]
    plt.barh([cols[i] for i in top][::-1], importancia[top][::-1], color="#2a78d6")
    plt.xlabel("|φ| medio (promedio sobre viviendas y estratos)")
    plt.title(f"SHAP: importancia global — {nombre}")
    guardar(f"shap_global_{archivo(nombre)}.png")
    for k in [1, 6]:
        k_i = int(np.where(CLASES == k)[0][0])
        shap.plots.beeswarm(explicacion_clase(shap, ex, k_i), max_display=12, show=False)
        plt.title(f"SHAP summary plot — {nombre}, estrato {k}")
        guardar(f"shap_beeswarm_{archivo(nombre)}_estrato{k}.png")

    pred = CLASES[P.argmax(1)]
    bien = int(rng.choice(np.where(pred == yte)[0]))
    mal = int(rng.choice(np.where(np.abs(pred - yte) >= 2)[0]))
    lime_exp = LimeTabularExplainer(Xs, feature_names=cols, class_names=[str(k) for k in CLASES],
                                    discretize_continuous=True, random_state=CF.SEED)
    locales = {}
    for etq, i in [("bien_clasificada", bien), ("error_grande", mal)]:
        k_i = int(P[i].argmax())
        e1 = explicar(Xt[i:i + 1])
        e1.feature_names = cols
        e_k = explicacion_clase(shap, e1, k_i)[0]
        shap.plots.waterfall(e_k, max_display=10, show=False)
        plt.title(f"{nombre} — vivienda {etq.replace('_', ' ')}: real {yte[i]}, predicho {CLASES[k_i]}", fontsize=10)
        guardar(f"shap_waterfall_{archivo(nombre)}_{etq}.png")
        le = lime_exp.explain_instance(Xt[i], f, labels=(k_i,), num_features=8, num_samples=3000)
        fig = le.as_pyplot_figure(label=k_i)
        fig.set_size_inches(8, 4.5)
        plt.title(f"LIME — {nombre}, vivienda {etq.replace('_', ' ')} (estrato predicho {CLASES[k_i]})", fontsize=10)
        guardar(f"lime_{archivo(nombre)}_{etq}.png")
        phi = np.asarray(e_k.values)
        top = np.argsort(np.abs(phi))[::-1][:8]
        lime_top = dict(le.as_list(label=k_i))
        shap_top = {cols[j]: float(phi[j]) for j in top}
        # ¿coinciden las 8 variables más importantes de cada método? (las de LIME vienen discretizadas: «x <= 0.3»)
        lime_vars = {c for c in cols if any(c in regla for regla in lime_top)}
        locales[etq] = {"fila_test": i, "real": int(yte[i]), "predicho": int(CLASES[k_i]),
                        "probabilidad": float(P[i, k_i]), "base_shap": float(e_k.base_values),
                        "shap_top": shap_top, "lime_top": lime_top, "lime_r2_local": float(le.score),
                        "variables_en_comun_top8": sorted(set(shap_top) & lime_vars)}
    out["locales"] = locales
    print(f"interpretabilidad {nombre}: listo", flush=True)
    return out


def regresion(shap, paq, sub, test):
    AR = json.loads((CF.RESULTADOS / "analisis_regresion.json").read_text(encoding="utf-8"))
    mejor = AR["test_mejor"]
    nombre = mejor["modelo"].split(" | ")[0]
    est = MD.pipeline_reg(paq, nombre, mejor["params"]).fit(sub, sub.estrato_num.to_numpy().astype(float))
    prep, modelo = est.named_steps["prep"], est.named_steps["modelo"]
    cols = nombres_cortos(prep.get_feature_names_out())
    Xt = np.asarray(prep.transform(test), dtype=float)
    yte = test.estrato_num.to_numpy().astype(float)
    pred = est.predict(test)
    rng = np.random.default_rng(CF.SEED)
    idx = rng.choice(len(test), 400, replace=False)
    expl = shap.TreeExplainer(modelo)
    ex = expl(Xt[idx])
    ex.feature_names = cols
    imp = np.abs(ex.values).mean(0)
    orden = np.argsort(imp)[::-1]
    shap.plots.beeswarm(ex, max_display=12, show=False)
    plt.title(f"SHAP summary plot — {nombre} (estrato como número)")
    guardar(f"shap_beeswarm_{archivo(nombre)}.png")
    err = np.abs(pred - yte)
    bien = int(rng.choice(np.where(err < 0.15)[0]))
    mal = int(rng.choice(np.where(err >= 2)[0]))
    locales = {}
    for etq, i in [("bien_predicha", bien), ("error_grande", mal)]:
        e1 = expl(Xt[i:i + 1])
        e1.feature_names = cols
        shap.plots.waterfall(e1[0], max_display=10, show=False)
        plt.title(f"{nombre} — vivienda {etq.replace('_', ' ')}: real {yte[i]:.0f}, predicho {pred[i]:.2f}",
                  fontsize=10)
        guardar(f"shap_waterfall_{archivo(nombre)}_{etq}.png")
        phi = np.asarray(e1.values[0])
        top = np.argsort(np.abs(phi))[::-1][:8]
        locales[etq] = {"fila_test": i, "real": float(yte[i]), "predicho": float(pred[i]),
                        "base_shap": float(np.asarray(e1.base_values).ravel()[0]),
                        "shap_top": {cols[j]: float(phi[j]) for j in top}}
    print(f"interpretabilidad {nombre}: listo", flush=True)
    return {"modelo": mejor["modelo"], "params": mejor["params"], "n_viviendas_global": len(idx),
            "importancia_global": {cols[i]: float(imp[i]) for i in orden[:20]}, "locales": locales}


def main():
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    FIG.mkdir(parents=True, exist_ok=True)
    CF.fijar_semillas()
    _, M1, _, _ = CF.importar_e1()
    with open(CF.RESULTADOS / "datos_clasificacion.pkl", "rb") as fh:
        paq = pickle.load(fh)
    paq["M1"] = M1
    with open(EX.TEST_CACHE, "rb") as fh:
        test = pickle.load(fh)
    sub = paq["sub"]
    ysub, yte = sub.estrato_num.to_numpy(), test.estrato_num.to_numpy()
    FN = json.loads((CF.RESULTADOS / "finalistas.json").read_text(encoding="utf-8"))
    fin = FN["finalistas"]
    proba = np.load(CF.RESULTADOS / "proba_test_finalistas.npz")
    mejor = max(fin, key=lambda k: fin[k]["test"]["f1_macro"])
    salida = {"clasificacion": {}, "mejor_clasificacion_test": mejor}
    for nombre in dict.fromkeys(["XGBoost", "XGBoost geográfico", mejor]):
        est = MD.pipeline(paq, nombre, fin[nombre]["params"], fin[nombre]["balanceo"]).fit(sub, ysub)
        P = proba[nombre.replace(" ", "_").replace("/", "-")]      # claves como las guarda finalistas.py
        salida["clasificacion"][nombre] = clasificacion(shap, LimeTabularExplainer, nombre, est, sub, test, yte, P)
        (CF.RESULTADOS / "interpretabilidad.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1,
                                                                         default=float), encoding="utf-8")
    salida["regresion"] = regresion(shap, paq, sub, test)
    (CF.RESULTADOS / "interpretabilidad.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1,
                                                                     default=float), encoding="utf-8")
    print("guardado interpretabilidad.json", flush=True)


if __name__ == "__main__":
    main()
