# -*- coding: utf-8 -*-
"""Mejora 3: stacking (candidato 8 de la revisión), elección final por CV y el test, mirado UNA sola vez.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

1. Toma de ``01_geo_busqueda.py`` y ``02_revision.py`` la mejor configuración de cada familia según la media del F1
   macro en los pliegues kNNDM (nunca según el test).
2. Stacking: un metamodelo (logística multinomial) aprende a combinar las probabilidades fuera de muestra de la
   logística con vecindad, el XGBoost afinado y el Random Forest (S1), y además del XGBoost geográfico afinado (S2).
   En la CV, el metamodelo de cada pliegue se entrena con las predicciones fuera de muestra de los OTROS pliegues.
   (Leve optimismo conocido: esas predicciones vienen de modelos que vieron el pliegue evaluado; el test no lo tiene.)
3. Comparaciones pareadas por pliegue (criterio del proyecto: > 1 SE) frente al modelo base y frente a la logística
   con las mismas variables; el XGBoost geográfico afinado además frente al XGBoost afinado (ablación).
4. Reentrena cada modelo elegido con las 257 273 viviendas, predice el test una vez y calcula IC 95 % BCa
   remuestreando los 7 bloques.

Uso (desde e3/):  ../venv_e2/Scripts/python -u mejoras/03_test.py      (RAPIDO=1: prueba de humo)
"""
import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

import comun as K
import modelos_revision as MR
from estrato3.estadistica import bootstrap_bca_bloques
from estrato3.modelos_propios import XGBEstrato, _proba_completa

GEO = __import__("01_geo_busqueda")
REV = __import__("02_revision")
META = {f"C{c}_{'bal' if cw else 'nobal'}": dict(C=c, class_weight=cw) for c in (0.1, 1.0) for cw in (None, "balanced")}


def leer_json(nombre):
    return json.loads((K.RES / nombre).read_text(encoding="utf-8"))


def mezcla(pg, ol, a):
    plc = np.where(ol["cub"][:, None], ol["p"], pg)
    return a * pg + (1 - a) * plc


def elegir(geo, rev):
    """Configuración elegida por CV de cada modelo que pasa al test."""
    fam = rev["familias"]
    mejor = lambda nombre_fam: max(fam[nombre_fam], key=lambda n: fam[nombre_fam][n]["f1"])  # noqa: E731
    g = geo["mejor"]
    return {
        "Modelo base (logística B, sin vecindad)": ("logistica_base", "base"),
        "Logística con vecindad": ("logistica_vecindad", mejor("logistica_vecindad")),
        "XGBoost afinado": ("xgb", geo["xgboost_afinado_alpha1"]["global"]),
        "XGBoost geográfico (configuración del E2)": ("geo", ("G1_e2", "L1_e2", 0.75)),
        "XGBoost geográfico afinado": ("geo", (g["global"], g["local"], g["alpha"])),
        "Random Forest": ("random_forest", mejor("random_forest")),
        "Rotation Forest": ("rotation_forest", mejor("rotation_forest")),
        "RF + SVM (dos etapas)": ("rf_svm", mejor("rf_svm")),
        "SVM sola (referencia del RF + SVM)": ("svm_sola", mejor("svm_sola")),
    }


def oof(eleccion, nf):
    """Probabilidades fuera de muestra por pliegue de un modelo elegido."""
    tipo, cfg = eleccion
    if tipo == "xgb":
        return [K.leer(f"geo_global_{cfg}_f{f}")["p"].astype(float) for f in range(nf)]
    if tipo == "geo":
        g, loc, a = cfg
        return [mezcla(K.leer(f"geo_global_{g}_f{f}")["p"].astype(float), K.leer(f"geo_local_{loc}_f{f}"), a)
                for f in range(nf)]
    return [K.leer(f"rev_{tipo}_{cfg}_f{f}")["p"].astype(float) for f in range(nf)]


def rasgos(probas):
    """Variables del metamodelo: log de las probabilidades de cada modelo base, una al lado de la otra."""
    return np.hstack([np.log(np.clip(p, 1e-4, 1)) for p in probas])


def stacking_cv(bases_oof, yv, meta_params):
    """F1 por pliegue del metamodelo entrenado con los OTROS pliegues."""
    nf = len(yv)
    Z = [rasgos([b[f] for b in bases_oof]) for f in range(nf)]
    f1 = []
    for f in range(nf):
        otros = [g for g in range(nf) if g != f]
        meta = LogisticRegression(max_iter=5000, **meta_params).fit(np.vstack([Z[g] for g in otros]),
                                                                   np.concatenate([yv[g] for g in otros]))
        f1.append(K.f1_macro(yv[f], K.pred(_proba_completa(meta, Z[f], meta.classes_))))
    return f1, Z


def ajustar_test(nombre, eleccion, rutas, ytr, xytr, xyte, hilos):
    """Reentrena con todo el entrenamiento y devuelve las probabilidades del test (se guardan)."""
    clave = f"test_{nombre}"
    hecho = K.leer(clave)
    if hecho is not None:
        return nombre, hecho
    tipo, cfg = eleccion
    zb, zv = np.load(rutas["B"]), np.load(rutas["BV"])
    t0 = time.time()
    out = {}
    if tipo == "logistica_base":
        p = REV.ajustar("logistica_base", REV.LOGISTICA_BASE, zb["tr"], ytr, zb["ev"])
    elif tipo in ("logistica_vecindad", "random_forest", "rotation_forest", "svm_sola"):
        p = REV.ajustar(tipo, REV.familias()[tipo][cfg], zv["tr"], ytr, zv["ev"], hilos=hilos)
    elif tipo == "rf_svm":
        _, imp = MR.rf_proba(zv["tr"], ytr, zv["ev"][:10], n_jobs=hilos, devolver_importancia=True,
                             **REV.RF_IMPORTANCIA)
        p = REV.ajustar(tipo, REV.RFSVM[cfg], zv["tr"], ytr, zv["ev"], np.argsort(-imp), hilos)
    elif tipo == "xgb":
        m = XGBEstrato(**GEO.GLOBALES[cfg], n_jobs=hilos, random_state=42).fit(zv["tr"], ytr)
        p = _proba_completa(m, zv["ev"], m.classes_)
    elif tipo == "geo":
        g, loc, a = cfg
        m = XGBEstrato(**GEO.GLOBALES[g], n_jobs=hilos, random_state=42).fit(zv["tr"], ytr)
        pg = _proba_completa(m, zv["ev"], m.classes_)
        anclas, h, modelos, k = GEO.ajustar_locales(zv["tr"], ytr, xytr, GEO.LOCALES[loc], hilos)
        pl, cub = GEO.proba_local(anclas, h, modelos, zv["ev"], xyte)
        p = mezcla(pg, {"p": pl, "cub": cub}, a)
        out.update(pg=pg.astype(np.float32), anclas=len(anclas), k=k, cobertura=float(cub.mean()))
    else:
        raise ValueError(tipo)
    out.update(p=p.astype(np.float32), seg=time.time() - t0)
    K.guardar(clave, out)
    K.log(f"  test {nombre}: {out['seg']:.0f} s")
    return nombre, out


def main():
    d = K.cargar()
    nf = len(d["folds"])
    yv = [d["y_tr"][iva] for _, iva in d["folds"]]
    geo, rev = leer_json("geo_busqueda.json"), leer_json("revision_cv.json")
    elec = elegir(geo, rev)
    OOF = {n: oof(e, nf) for n, e in elec.items()}
    F1CV = {n: [K.f1_macro(yv[f], K.pred(OOF[n][f])) for f in range(nf)] for n in elec}

    # --- stacking (candidato 8) ----------------------------------------------------------------------------------
    bases = {"Stacking S1 (logística, XGBoost, RF)": ["Logística con vecindad", "XGBoost afinado", "Random Forest"],
             "Stacking S2 (S1 + XGBoost geográfico)": ["Logística con vecindad", "XGBoost afinado", "Random Forest",
                                                       "XGBoost geográfico afinado"]}
    stack_info = {}
    for nombre, bs in bases.items():
        res = {m: stacking_cv([OOF[b] for b in bs], yv, p) for m, p in META.items()}
        m = max(res, key=lambda c: np.mean(res[c][0]))
        F1CV[nombre] = res[m][0]
        stack_info[nombre] = {"bases": bs, "meta": m, "f1_por_meta": {c: float(np.mean(r[0])) for c, r in res.items()}}
        K.log(f"{nombre}: metamodelo {m}, F1 CV {np.mean(F1CV[nombre]):.4f}")

    # --- comparaciones en la CV -------------------------------------------------------------------------------------
    base, lv = "Modelo base (logística B, sin vecindad)", "Logística con vecindad"
    cv = {n: {"f1_pliegues": f, "f1": float(np.mean(f)), "sd": float(np.std(f, ddof=1)) if nf > 1 else 0.0,
              "vs_base": K.pareada(f, F1CV[base]), "vs_logistica_vecindad": K.pareada(f, F1CV[lv])}
          for n, f in F1CV.items()}
    cv["XGBoost geográfico afinado"]["vs_xgboost_afinado"] = K.pareada(F1CV["XGBoost geográfico afinado"],
                                                                      F1CV["XGBoost afinado"])
    cv["XGBoost geográfico afinado"]["vs_geo_E2"] = K.pareada(F1CV["XGBoost geográfico afinado"],
                                                             F1CV["XGBoost geográfico (configuración del E2)"])
    for n in sorted(cv, key=lambda n: -cv[n]["f1"]):
        K.log(f"CV {n:45s} {cv[n]['f1']:.4f}  vs base {cv[n]['vs_base']['media']:+.3f} (SE {cv[n]['vs_base']['SE']:.3f})"
              f"  vs logística vecindad {cv[n]['vs_logistica_vecindad']['media']:+.3f}")

    # --- test: se mira una sola vez ----------------------------------------------------------------------------------
    test = d["test"]
    itr_all = np.arange(len(d["train"]))
    if K.RAPIDO:
        rng = np.random.default_rng(42)
        itr_all = rng.choice(itr_all, 15000, replace=False)
        test = test.iloc[rng.choice(len(test), 6000, replace=False)].reset_index(drop=True)
    y_te = test.estrato_num.to_numpy().astype(int)
    rutas = {}
    for b in ("B", "BV"):
        ruta = K.CHK / f"X_{b}_test.npz"
        if not ruta.exists():
            Xtr, Xte = K.matrices(d, itr_all, None, df_ev=test, base=b == "B")
            np.savez(ruta, tr=Xtr, ev=Xte)
        rutas[b] = str(ruta)
    K.log("matrices de test listas; reentrenando con todo el entrenamiento")
    xy_te = test[["x_km", "y_km"]].to_numpy(float)
    trabajos = sorted(elec.items(), key=lambda kv: 0 if kv[1][0] in ("geo", "rotation_forest") else 1)
    res = dict(Parallel(n_jobs=K.N_TRABAJOS, pre_dispatch="n_jobs")(
        delayed(ajustar_test)(n, e, rutas, d["y_tr"][itr_all], d["xy_tr"][itr_all], xy_te, K.HILOS)
        for n, e in trabajos))
    PTE = {n: r["p"].astype(float) for n, r in res.items()}
    # stacking en el test: metamodelo con TODAS las predicciones fuera de muestra
    for nombre, info in stack_info.items():
        Ztr = np.vstack([rasgos([OOF[b][f] for b in info["bases"]]) for f in range(nf)])
        meta = LogisticRegression(max_iter=5000, **META[info["meta"]]).fit(Ztr, np.concatenate(yv))
        PTE[nombre] = _proba_completa(meta, rasgos([PTE[b] for b in info["bases"]]), meta.classes_)

    bloques = test.bloque.to_numpy()
    ub = np.unique(bloques)
    PRED = {n: K.pred(p) for n, p in PTE.items()}

    def ic_modelo(n):
        return bootstrap_bca_bloques(lambda i: K.f1_macro(y_te[i], PRED[n][i]), bloques)

    def comparar(a, b):
        dif = K.f1_macro(y_te, PRED[a]) - K.f1_macro(y_te, PRED[b])
        ic = bootstrap_bca_bloques(lambda i: K.f1_macro(y_te[i], PRED[a][i]) - K.f1_macro(y_te[i], PRED[b][i]),
                                   bloques)
        favor = sum(K.f1_macro(y_te[bloques == u], PRED[a][bloques == u]) >
                    K.f1_macro(y_te[bloques == u], PRED[b][bloques == u]) for u in ub)
        return {"dif": dif, "ic95_bca": ic, "bloques_a_favor": int(favor), "n_bloques": len(ub)}

    tst = {}
    for n in PTE:
        tst[n] = {"f1": K.f1_macro(y_te, PRED[n]), "ic95_bca": ic_modelo(n),
                  "vs_base": comparar(n, base) if n != base else None,
                  "vs_logistica_vecindad": comparar(n, lv) if n != lv else None}
    tst["XGBoost geográfico afinado"]["vs_xgboost_afinado"] = comparar("XGBoost geográfico afinado", "XGBoost afinado")
    for n in sorted(tst, key=lambda n: -tst[n]["f1"]):
        vb = tst[n]["vs_base"]
        K.log(f"TEST {n:45s} {tst[n]['f1']:.4f} {np.round(tst[n]['ic95_bca'], 3).tolist()}"
              + (f"  vs base {vb['dif']:+.3f} {np.round(vb['ic95_bca'], 3).tolist()} ({vb['bloques_a_favor']}/7)"
                 if vb else ""))
    info_geo = {k: res[n][k] for n in ("XGBoost geográfico afinado",) for k in ("anclas", "k", "cobertura")}
    K.escribir_json("mejoras.json", {
        "protocolo": ("todos los datos (257 273); pliegues kNNDM de zonas nuevas del E2 v3 (q = 30); elección por F1 "
                      "macro medio en la CV; test de 7 bloques mirado una sola vez; IC 95 % BCa por bloques"),
        "elegidos": {n: e for n, e in elec.items()}, "stacking": stack_info, "cv": cv, "test": tst,
        "geo_afinado_test": info_geo,
        "segundos_test": {n: r["seg"] for n, r in res.items()}})


if __name__ == "__main__":
    main()
