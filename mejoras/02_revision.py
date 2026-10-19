# -*- coding: utf-8 -*-
"""Mejora 2: modelos de la revisión bibliográfica y rivales, en los mismos pliegues kNNDM con todos los datos.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Guarda en ``chk/`` las probabilidades fuera de muestra de cada configuración y pliegue:

- **Modelo base** (logística B v3 del Entregable 1: C = 0.01, sin pesos, SIN vecindad). Verifica los pliegues:
  su F1 medio debe ser el del Entregable 2 v3 (0.460).
- **Logística con vecindad** (rival con las mismas variables): C ∈ {0.01, 0.1, 1} × pesos {no, balanceados}.
- **Random Forest** (rival y base del stacking): 3 configuraciones, entrenado por lotes.
- **Rotation Forest** (candidato 5 de la revisión): 4 configuraciones.
- **RF + SVM en dos etapas** (candidato 4): k ∈ {10, 20} variables × 2 configuraciones de la SVM; y la misma SVM
  con todas las variables como referencia (¿ayuda la selección del Random Forest?).

El stacking (candidato 8) va en ``03_test.py`` porque usa el mejor XGBoost geográfico de ``01_geo_busqueda.py``.

Uso (desde e3/):  ../venv_e2/Scripts/python -u mejoras/02_revision.py      (RAPIDO=1: prueba de humo)
"""
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

import comun as K
import modelos_revision as MR
from estrato3.modelos_propios import _proba_completa

LOGISTICA_BASE = dict(C=0.01, class_weight=None)
LOGISTICAS = {f"LV_C{c}_{'bal' if cw else 'nobal'}": dict(C=c, class_weight=cw)
              for c in (0.01, 0.1, 1.0) for cw in (None, "balanced")}
RFS = {
    "RF1": dict(n_estimators=150, max_features="sqrt", min_samples_leaf=5),
    "RF2": dict(n_estimators=150, max_features=0.3, min_samples_leaf=2, max_depth=21),
    "RF3": dict(n_estimators=150, max_features="sqrt", min_samples_leaf=20, class_weight="balanced_subsample"),
}
ROTF = {
    "RoF1": dict(n_arboles=40, M=3, min_samples_leaf=5),
    "RoF2": dict(n_arboles=40, M=6, min_samples_leaf=5),
    "RoF3": dict(n_arboles=40, M=3, min_samples_leaf=20),
}
# S1 = hiperparámetros de la SVM finalista del Entregable 3. Una segunda configuración (γ = 0.1, 400 componentes,
# C = 1) se retiró: con ~200 000 filas cada ajuste tardaba 15–20 min y la búsqueda completa no cabía en el día.
SVMS = {"S1": dict(gamma=0.03, n_components=300, C=10.0)}
MEMORIA_ALTA = ("rf_svm", "svm_sola")   # Nyström con ~200 000 filas: como mucho 2 a la vez
RFSVM = {f"RFSVM_k{k}_{s}": dict(k=k, **p) for k in (10, 20) for s, p in SVMS.items()}
SVM_SOLA = {f"SVM_todas_{s}": dict(p) for s, p in SVMS.items()}
RF_IMPORTANCIA = dict(n_estimators=100, max_features="sqrt", min_samples_leaf=10)


def familias():
    """Nombre de familia -> {config: parámetros} (el orden de las tareas va de la más pesada a la más liviana)."""
    return {"rotation_forest": ROTF, "random_forest": RFS, "rf_svm": RFSVM, "svm_sola": SVM_SOLA,
            "logistica_vecindad": LOGISTICAS, "logistica_base": {"base": LOGISTICA_BASE}}


def ajustar(familia, params, Xtr, ytr, Xev, imp_orden=None, hilos=2):
    """Entrena una configuración y devuelve las probabilidades de evaluación (n_ev, 6)."""
    if familia in ("logistica_base", "logistica_vecindad"):
        m = LogisticRegression(max_iter=1000, **params).fit(Xtr, ytr)
        return _proba_completa(m, Xev, m.classes_)
    if familia == "random_forest":
        return MR.rf_proba(Xtr, ytr, Xev, n_jobs=hilos, **params)
    if familia == "rotation_forest":
        return MR.rotation_forest_proba(Xtr, ytr, Xev, **params)
    if familia == "rf_svm":
        return MR.rf_svm_proba(Xtr, ytr, Xev, imp_orden, **params)
    if familia == "svm_sola":
        return MR.rf_svm_proba(Xtr, ytr, Xev, np.arange(Xtr.shape[1]), k=Xtr.shape[1], **params)
    raise ValueError(familia)


def orden_importancia(f, ruta, ytr, hilos):
    """Variables ordenadas por importancia (impureza) de un Random Forest entrenado en el pliegue (se guarda)."""
    clave = f"rfimp_{f}"
    hecho = K.leer(clave)
    if hecho is None:
        z = np.load(ruta)
        _, imp = MR.rf_proba(z["tr"], ytr, z["ev"][:10], n_jobs=hilos, devolver_importancia=True, **RF_IMPORTANCIA)
        hecho = {"orden": np.argsort(-imp), "imp": imp}
        K.guardar(clave, hecho)
    return hecho["orden"]


def tarea(familia, nombre, params, f, ruta, ytr, yev, hilos):
    """Una configuración en un pliegue: guarda probabilidades y F1."""
    clave = f"rev_{familia}_{nombre}_f{f}"
    hecho = K.leer(clave)
    if hecho is not None:
        return clave, hecho["f1"]
    z = np.load(ruta)
    t0 = time.time()
    imp = orden_importancia(f, ruta.replace("X_B_", "X_BV_"), ytr, hilos) if familia == "rf_svm" else None
    p = ajustar(familia, params, z["tr"], ytr, z["ev"], imp, hilos)
    out = {"p": p.astype(np.float32), "f1": K.f1_macro(yev, K.pred(p)), "seg": time.time() - t0}
    K.guardar(clave, out)
    K.log(f"  {clave}: F1 {out['f1']:.4f} ({out['seg']:.0f} s)")
    return clave, out["f1"]


def main():
    d = K.cargar()
    nf = len(d["folds"])
    K.log(f"train {len(d['train']):,} | {nf} pliegues kNNDM")
    for f in range(nf):
        K.cache_matrices(d, f)
        K.cache_matrices(d, f, base=True)
    K.log("matrices listas")
    tareas = []
    for fam, confs in familias().items():
        for nombre, params in confs.items():
            for f, (itr, iva) in enumerate(d["folds"]):
                ruta = str(K.CHK / f"X_{'B' if fam == 'logistica_base' else 'BV'}_{f}.npz")
                tareas.append((fam, nombre, params, f, ruta, d["y_tr"][itr], d["y_tr"][iva], K.HILOS))
    # la importancia del Random Forest se calcula antes, una vez por pliegue, para que las 4 configuraciones RF+SVM
    # no la recalculen en paralelo
    for f, (itr, _) in enumerate(d["folds"]):
        orden_importancia(f, str(K.CHK / f"X_BV_{f}.npz"), d["y_tr"][itr], K.HILOS)
    K.log("importancias del Random Forest listas")
    Parallel(n_jobs=K.N_TRABAJOS, pre_dispatch="n_jobs")(delayed(tarea)(*t) for t in tareas
                                                         if t[0] not in MEMORIA_ALTA)
    Parallel(n_jobs=min(2, K.N_TRABAJOS), pre_dispatch="n_jobs")(delayed(tarea)(*t) for t in tareas
                                                                 if t[0] in MEMORIA_ALTA)

    resumen = {}
    for fam, confs in familias().items():
        resumen[fam] = {}
        for nombre in confs:
            f1 = [K.leer(f"rev_{fam}_{nombre}_f{f}")["f1"] for f in range(nf)]
            seg = [K.leer(f"rev_{fam}_{nombre}_f{f}")["seg"] for f in range(nf)]
            resumen[fam][nombre] = {"params": confs[nombre], "f1_pliegues": f1, "f1": float(np.mean(f1)),
                                    "segundos": seg}
        mejor = max(resumen[fam], key=lambda n: resumen[fam][n]["f1"])
        K.log(f"{fam}: mejor {mejor} F1 {resumen[fam][mejor]['f1']:.4f}")
    K.escribir_json("revision_cv.json", {
        "protocolo": "todos los datos; pliegues kNNDM de zonas nuevas del E2 v3 (q = 30); F1 macro medio por pliegue",
        "familias": resumen,
        "verificacion_pliegues": {"logistica_base_f1": resumen["logistica_base"]["base"]["f1"],
                                  "E2_v3_publicado": 0.460054}})


if __name__ == "__main__":
    main()
