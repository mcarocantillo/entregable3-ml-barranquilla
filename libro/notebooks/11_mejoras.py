# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 11. Después del entregable: todos los datos y otros modelos de la revisión
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# **Pregunta.** ¿Mejora el XGBoost geográfico si sus hiperparámetros se buscan con **todos los datos**? ¿Funcionan
# mejor otros **modelos originales de la revisión bibliográfica** (Rotation Forest, stacking y RF + SVM)? ¿Son
# **estadísticamente significativas** las diferencias?
#
# **Respuesta corta.** Afinar el XGBoost geográfico con las 257 273 viviendas sube su F1 de validación de 0.572 a
# 0.581, pero en el test no mejora (0.566 frente a 0.585 con la configuración del Entregable 2), y su parte
# geográfica no se distingue del XGBoost global. De los modelos originales de la revisión, el **Rotation Forest** es
# el único que supera al modelo base en la validación y en el test (+0.099 de F1, IC 95 % [+0.024, +0.170]).
# Estadísticamente, **la mejora sobre el modelo base es significativa** para todos los modelos con vecindad física,
# originales o no (30 bloques espaciales: Friedman p = 2.6·10⁻⁵; Wilcoxon con Holm p < 0.01; efecto mediano a grande),
# pero **ninguna diferencia entre ellos lo es**: ningún modelo original supera significativamente a los convencionales
# que reciben la misma información.
# ```
#
# Este capítulo recoge dos de los próximos pasos de la sección 10.6. Se hizo después de cerrar el pipeline
# combinatorio, con todos los datos y el protocolo de la fase 2 del Entregable 2; el código está en `mejoras/`.
#
# ## 11.1 Por qué y cómo
#
# **Dos motivos.**
#
# 1. **Los hiperparámetros del modelo nuevo nunca se buscaron con todos los datos.** En el Entregable 2 quedaron fijos
#    (α = 0.75, 15 % de vecinos por ancla, profundidad 4) y en este entregable se buscaron con 10 001 viviendas, donde
#    cada ancla tiene pocos vecinos y la búsqueda casi apagó los modelos locales (α = 0.97, sección 9.4).
# 2. **La revisión bibliográfica dejó otros candidatos.** Su Tabla III lista nueve modelos con evidencia de reducción
#    de error que nunca se han aplicado a la clasificación del estrato con el catastro. El Entregable 2 eligió el
#    primero, el XGBoost geográfico; aquí se prueban tres más: **Rotation Forest** (candidato 5), **RF + SVM en dos
#    etapas** (candidato 4) y **stacking** (candidato 8). Bagged TAO Trees (3) no tiene una implementación abierta
#    confiable en Python; GWO-XGBoost (2) y TPE-SVM (7) son métodos de optimización (el capítulo 3 mostró que los
#    optimizadores empatan, y la SVM con optimización bayesiana TPE ya está en el pipeline); EFSHB (6) y NSGA-II (9)
#    seleccionan variables, y aquí hay solo 35.
#
# **Protocolo** (el de la fase 2 de la versión 3 del Entregable 2, sección 9.1):
#
# - las 257 273 viviendas de entrenamiento y las variables de este entregable (modelo B + vecindad física, k = 8);
# - los **mismos 5 pliegues kNNDM** de zonas nuevas del Entregable 2 v3 (q = 30). Es verificable: el modelo base
#   vuelve a dar F1 0.460 y el XGBoost geográfico con la configuración del Entregable 2, 0.572, las cifras publicadas;
# - los hiperparámetros se eligen con la media del F1 macro en esos pliegues, nunca con el test;
# - el test de 7 bloques se mira **una sola vez**, al final, con IC 95 % BCa remuestreando bloques;
# - el criterio de reemplazo del proyecto: diferencia pareada por pliegue mayor que un error estándar, y el test lo
#   confirma.

# %% tags=["hide-input"]
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _comun import RAIZ

RM = RAIZ / "mejoras" / "resultados"
leer = lambda n: json.loads((RM / n).read_text(encoding="utf-8")) if (RM / n).exists() else None  # noqa: E731
GB, RV, MJ = leer("geo_busqueda.json"), leer("revision_cv.json"), leer("mejoras.json")
print("Resultados de mejoras/:", "completos" if GB and RV and MJ else "pendientes")

# %% [markdown]
# ## 11.2 Hiperparámetros del XGBoost geográfico con todos los datos
#
# **Una búsqueda factorizada.** La predicción del modelo es $p = \alpha\,p_{\text{global}} + (1-\alpha)\,p_{\text{local}}$
# y sus dos partes se entrenan por separado, así que no hace falta reentrenar cada combinación. Se ajustaron 6
# configuraciones del XGBoost global (profundidad 4 a 8, 300 o 600 árboles, con o sin pesos de clase) y 9 de los
# modelos locales (fracción de vecinos por ancla de 2 % a 30 %, rejilla de anclas de 1 o 2 km, profundidad y árboles
# de cada modelo local, con o sin pesos), y α se recorrió de 0 a 1 en pasos de 0.05 sobre las probabilidades ya
# guardadas: 1 134 combinaciones con solo 75 ajustes. Con α = 1 el modelo es el XGBoost global, así que la misma
# búsqueda da también el **XGBoost afinado**, la ablación sin la parte geográfica.

# %% tags=["hide-input"]
if GB:
    top = pd.DataFrame(GB["top20"][:8])
    top["F1 por pliegue"] = top.f1_pliegues.map(lambda v: ", ".join(f"{x:.3f}" for x in v))
    display(top[["global", "local", "alpha", "f1", "F1 por pliegue"]]
            .rename(columns={"alpha": "α", "f1": "F1 macro (validación)"}).round(4))
    curva = sorted((float(a), f) for a, f in GB["curva_alpha_mejor"].items())
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot([a for a, _ in curva], [f for _, f in curva], marker="o", color="#2a78d6")
    ax.axhline(GB["referencia_E2"]["f1"], ls="--", color="#5F6B76")
    ax.text(0.01, GB["referencia_E2"]["f1"] + 0.0006, "configuración del Entregable 2", color="#5F6B76", fontsize=9)
    ax.set_xlabel("α (peso del XGBoost global)")
    ax.set_ylabel("F1 macro en validación")
    ax.set_title(f"Mejor combinación ({GB['mejor']['global']} + {GB['mejor']['local']}) según α")
    ax.grid(alpha=0.3)
    plt.show()
else:
    print("(búsqueda pendiente)")

# %% [markdown]
# **Lectura.** La mejor combinación usa un XGBoost global de profundidad 6 sin pesos de clase (G5) y modelos locales
# en una rejilla de **1 km** (76 a 78 anclas) con el **5 % de los vecinos** por ancla, unas 10 000 viviendas (L5),
# mezclados con **α = 0.5**: con todos los datos, los modelos locales sí reciben peso, a diferencia de la submuestra
# (α = 0.97). Su F1 de validación es 0.581.
#
# Dos señales de cautela. La superficie es plana: las ocho mejores combinaciones están entre 0.579 y 0.581. Y la
# curva de α muestra que los modelos locales solos son peores (0.556 con α = 0), la mezcla llega a 0.581 y el XGBoost
# global solo da 0.575: la parte geográfica aporta, en el mejor caso, medio punto de F1.

# %% tags=["hide-input"]
NOMBRES_GEO = ["XGBoost geográfico (configuración del E2)", "XGBoost geográfico afinado", "XGBoost afinado"]
if MJ:
    cv, te = MJ["cv"], MJ["test"]
    filas = {n: {"F1 validación": cv[n]["f1"], "F1 test": te[n]["f1"],
                 "IC 95 % test": f"[{te[n]['ic95_bca'][0]:.3f}, {te[n]['ic95_bca'][1]:.3f}]"} for n in NOMBRES_GEO}
    display(pd.DataFrame(filas).T.round(3))
    comp = {
        "afinado − configuración del E2 (validación)": cv["XGBoost geográfico afinado"]["vs_geo_E2"],
        "afinado − XGBoost afinado (validación)": cv["XGBoost geográfico afinado"]["vs_xgboost_afinado"],
    }
    display(pd.DataFrame({k: {"Δ F1": v["media"], "SE": v["SE"], "pliegues a favor": f"{v['pliegues_a_favor']}/5",
                              "¿> 1 SE?": "sí" if v["supera_1SE"] else "no"} for k, v in comp.items()}).T.round(3))
    t = te["XGBoost geográfico afinado"]["vs_xgboost_afinado"]
    print(f"Test, afinado − XGBoost afinado: {t['dif']:+.3f}, IC 95 % BCa [{t['ic95_bca'][0]:+.3f}, "
          f"{t['ic95_bca'][1]:+.3f}], {t['bloques_a_favor']} de {t['n_bloques']} bloques")

# %% [markdown]
# **Lectura.** En la validación, la búsqueda mejora a la configuración del Entregable 2 por +0.009 (SE 0.007, 4 de 5
# pliegues), algo más de un error estándar; frente al XGBoost con el mismo modelo global, la parte geográfica aporta
# +0.005 (SE 0.006), menos de un error estándar. **En el test la mejora no aparece:** el modelo afinado da 0.566,
# por debajo de la configuración del Entregable 2 (0.585), y queda igual que el XGBoost afinado (+0.001, IC
# [−0.027, +0.068], 4 de 7 bloques).
#
# La lectura es la misma de los capítulos 9 y 10, ahora con más datos: la ganancia de validación es del tamaño del
# ruido de elegir la mejor entre 1 134 combinaciones, y en zonas nuevas la parte geográfica no agrega F1 medible
# sobre un XGBoost global con las mismas variables. La configuración del Entregable 2 sigue siendo la mejor versión
# del modelo nuevo en el test.
#
# ## 11.3 Otros modelos originales de la revisión
#
# - **Rotation Forest** (Rodríguez, Kuncheva y Alonso, 2006; candidato 5, Sartono et al., 2018). Cada árbol se entrena
#   con las variables **rotadas**: se reparten al azar en grupos de M, en cada grupo se toma un subconjunto aleatorio
#   de clases y el 75 % de sus filas, y los ejes del PCA de ese grupo forman un bloque de la matriz de rotación. La
#   revisión lo propone por la colinealidad entre área, baños, habitaciones y sus promedios de vecindad. Implementación
#   propia (`mejoras/modelos_revision.py`): 40 árboles, M ∈ {3, 6}, hojas mínimas ∈ {5, 20}.
# - **RF + SVM en dos etapas** (candidato 4, Zhai et al., 2026). Un Random Forest ordena las variables por importancia
#   y una SVM (Nyström + LinearSVC, con los hiperparámetros de la SVM finalista del capítulo 5) clasifica con las
#   k ∈ {10, 20} más importantes. La misma SVM con todas las variables sirve de referencia: ¿ayuda la selección?
# - **Stacking** (candidato 8, Sharma y Tomar, 2026). Un metamodelo (logística multinomial) combina las probabilidades
#   fuera de muestra de la logística con vecindad, el XGBoost afinado y el Random Forest (S1), y además del XGBoost
#   geográfico afinado (S2). En la validación, el metamodelo de cada pliegue se entrena con las predicciones de los
#   **otros** pliegues; esas predicciones vienen de modelos que sí vieron el pliegue evaluado, lo que introduce un
#   leve optimismo que el test no tiene.
#
# La tabla reúne todos los modelos con la configuración que eligió su validación, frente al modelo base y frente a
# la logística con las mismas variables de vecindad.

# %% tags=["hide-input"]
if MJ:
    base, lv = "Modelo base (logística B, sin vecindad)", "Logística con vecindad"
    ic = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"  # noqa: E731
    filas = {}
    for n in sorted(te, key=lambda n: -te[n]["f1"]):
        c, t = cv[n], te[n]
        fila = {"F1 validación": round(c["f1"], 3), "F1 test": round(t["f1"], 3)}
        if n != base:
            vb, tb = c["vs_base"], t["vs_base"]
            fila.update({"vs base, validación (SE)": f"{vb['media']:+.3f} ({vb['SE']:.3f})",
                         "vs base, test [IC 95 %]": f"{tb['dif']:+.3f} {ic(tb['ic95_bca'])}",
                         "bloques": f"{tb['bloques_a_favor']}/7"})
        if n not in (base, lv):
            tl = t["vs_logistica_vecindad"]
            fila["vs logística con vecindad, test [IC 95 %]"] = f"{tl['dif']:+.3f} {ic(tl['ic95_bca'])}"
        filas[n] = fila
    display(pd.DataFrame(filas).T.fillna("—"))

# %% tags=["hide-input"]
if MJ:
    nombres = [n for n in sorted(te, key=lambda n: te[n]["vs_base"]["dif"] if te[n]["vs_base"] else -1) if n != base]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, n in enumerate(nombres):
        tb = te[n]["vs_base"]
        nuevo = n in ("Rotation Forest", "Stacking S1 (logística, XGBoost, RF)", "Stacking S2 (S1 + XGBoost geográfico)",
                      "RF + SVM (dos etapas)") or n.startswith("XGBoost geográfico")
        color = "#E06A2C" if nuevo else "#7891A6"
        ax.plot(tb["ic95_bca"], [i, i], color=color, lw=2.5)
        ax.plot(tb["dif"], i, "o", color=color, ms=8)
    ax.axvline(0, color="#3E4A56", lw=1)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres)
    ax.set_xlabel("Δ F1 macro en el test frente al modelo base (IC 95 % BCa por bloques)")
    ax.set_title("Naranja: modelos originales (revisión); gris: referencias")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Lectura.**
#
# - **Rotation Forest:** es el único modelo original que cumple el criterio del proyecto frente al modelo base en las
#   dos miradas: +0.138 en la validación (SE 0.029, 5 de 5 pliegues) y **+0.099 en el test, IC [+0.024, +0.170]**,
#   ganando en 6 de los 7 bloques. Empata con el mejor en validación (0.598) y su test es 0.593.
# - **Stacking:** S1 tiene la misma validación que el Rotation Forest (0.598), pero en el test su ventaja sobre el
#   base (+0.102) tiene un intervalo que toca el 0 ([−0.006, +0.173]). Agregarle el XGBoost geográfico (S2) lo
#   **empeora** en las dos miradas (0.583 y 0.568): sus probabilidades no traen información nueva para el metamodelo.
# - **RF + SVM:** supera al base en la validación (+0.093), pero no en el test (+0.024, IC [−0.060, +0.088]). La
#   selección de variables del Random Forest sí ayuda a la SVM (test 0.517 frente a 0.493 con todas las variables),
#   pero la SVM queda por debajo de los ensambles de árboles, como en el capítulo 5.
# - **Frente a la logística con las mismas variables de vecindad, ningún modelo se confirma en el test.** El Rotation
#   Forest y el stacking S1 la superan en la validación por apenas un error estándar (+0.023, SE 0.023 y 0.022), y en
#   el test sus intervalos incluyen el 0 (+0.018 [−0.019, +0.064] y +0.021 [−0.032, +0.080]). El Random Forest de la
#   guía, que no es original, tiene el mejor F1 de test (0.601).
#
# ## 11.4 ¿Son estadísticamente significativas las diferencias?
#
# Con 11 modelos hay 55 pares posibles, así que se sigue la secuencia jerárquica del capítulo 7: una prueba global,
# después comparaciones dirigidas con corrección por comparaciones múltiples (Holm dentro de cada familia) y siempre
# el tamaño del efecto. Se miran tres unidades de réplica, de la más a la menos potente:
#
# 1. **30 bloques espaciales de 2 km** del entrenamiento (los que tienen al menos 200 viviendas, de 34): el F1 macro
#    de las predicciones fuera de muestra en cada bloque. Friedman → Nemenyi → Wilcoxon de rangos con signo, delta de
#    Cliff. Es la misma unidad que usó el capítulo 7.
# 2. **5 pliegues kNNDM:** t de Student pareada **corregida de Nadeau y Bengio (2003)**, que infla la varianza porque
#    los pliegues comparten entrenamiento.
# 3. **7 bloques de test:** prueba exacta de permutación por cambio de signo de las diferencias por bloque. Con 7
#    bloques el p-valor más pequeño posible es 2/128 = 0.016, y con 10 comparaciones Holm exige 0.005 para la primera:
#    **el test no puede declarar significancia después de la corrección**; sirve para confirmar la dirección y el
#    intervalo.

# %% tags=["hide-input"]
ESt = leer("estadistica.json")
if ESt:
    fr = ESt["friedman"]
    print(f"Friedman sobre {ESt['bloques']['n_usados']} bloques de 2 km: χ² = {fr['chi2']:.1f}, gl = {fr['gl']}, "
          f"p = {fr['p']:.1e}; diferencia crítica de Nemenyi = {fr['cd']:.2f} rangos")
    rangos = pd.Series(fr["rangos"]).rename("rango medio (1 = mejor)")
    nem = pd.DataFrame(ESt["nemenyi_p"])
    display(pd.DataFrame({"rango medio": rangos,
                          "Nemenyi p vs base": nem["Modelo base (logística B, sin vecindad)"],
                          "Nemenyi p vs logística con vecindad": nem["Logística con vecindad"]}).round(4))
    from IPython.display import Image
    ruta_cd = RM / "figuras" / "cd_mejoras.png"
    if ruta_cd.exists():
        display(Image(filename=str(ruta_cd)))

    def tabla_comp(fam):
        filas = {}
        for r in ESt["comparaciones"][fam]:
            filas[r["a"] if fam != "dirigidas" else f"{r['a']} − {r['b']}"] = {
                "bloques a favor": f"{r['bloques_a_favor']}/{r['n_bloques']}",
                "Wilcoxon p (Holm)": f"{r['wilcoxon_p']:.4f} ({r['wilcoxon_p_holm']:.4f})",
                "δ de Cliff": f"{r['delta_cliff']:+.2f} ({r['magnitud']})",
                "pliegues: t corregida p (Holm)": f"{r['cv_p']:.3f} ({r['cv_p_holm']:.3f})",
                "test: permutación p (Holm)": f"{r['test_perm_p']:.3f} ({r['test_perm_p_holm']:.3f})"}
        return pd.DataFrame(filas).T

    for fam in ("vs modelo base", "vs logística con vecindad", "dirigidas"):
        print(fam.upper())
        display(tabla_comp(fam))
else:
    print("(pruebas de significancia pendientes)")

# %% [markdown]
# **Lectura.**
#
# - **Hay diferencias entre los modelos** (Friedman χ² = 39.0, gl = 10, p = 2.6·10⁻⁵).
# - **Frente al modelo base, la mejora es significativa para todos los modelos con vecindad física**, originales y
#   de referencia: XGBoost geográfico (afinado y del E2), Rotation Forest, los dos stacking, XGBoost, Random Forest y
#   la logística con vecindad (Wilcoxon con Holm, p entre 0.004 y 0.009). El efecto es **mediano a grande**: δ de Cliff
#   entre 0.35 (logística con vecindad) y 0.51 (XGBoost geográfico afinado, el de mejor rango medio, 4.6). Nemenyi, que
#   compara los 55 pares a la vez y es más conservador, confirma la diferencia para el XGBoost geográfico (p = 0.0005
#   el afinado, 0.008 el del E2), XGBoost, Random Forest y Rotation Forest (p = 0.045). El RF + SVM y la SVM sola no
#   superan significativamente al base (p de Holm 0.10).
# - **Entre los modelos que reciben la misma vecindad, ninguna diferencia es significativa.** Frente a la logística
#   con vecindad, todos los p de Holm son ≥ 0.54 y los efectos, despreciables o pequeños (|δ| ≤ 0.24). La parte
#   geográfica del modelo nuevo frente al XGBoost afinado: δ = +0.02, p = 1. Rotation Forest frente al XGBoost
#   geográfico del E2: δ = −0.03, p = 1.
# - **Con las unidades menos potentes nada sobrevive a la corrección.** Con la t corregida sobre los 5 pliegues, los
#   p crudos frente al base van de 0.008 a 0.12, pero con Holm ninguno baja de 0.05 (el menor, 0.078). En el test, los
#   p crudos de la permutación son 0.031 para el Rotation Forest, el Random Forest y la logística con vecindad, y 0.31
#   después de Holm, como se anticipó.
#
# ## 11.5 Conclusión
#
# ```{admonition} Qué se puede afirmar con rigor estadístico
# :class: important
# 1. **Los modelos con vecindad física superan al modelo base en zonas nuevas, con significancia estadística y un
#    efecto mediano a grande**, en la prueba con más potencia (30 bloques espaciales, Wilcoxon con Holm, p < 0.01). Eso
#    incluye al **modelo nuevo del proyecto, el XGBoost geográfico** (δ = 0.51 afinado, 0.43 con la configuración del
#    Entregable 2) y a otro modelo original de la revisión, el **Rotation Forest** (δ = 0.45), que además es el único
#    original cuyo intervalo de test frente al base excluye el 0 (+0.099, IC [+0.024, +0.170]).
# 2. **Ningún modelo original supera significativamente a los modelos convencionales que reciben la misma
#    información**, en ninguna de las tres miradas. El aporte que sí es significativo es el de la **vecindad física**,
#    no el de un algoritmo en particular.
# 3. **Afinar el XGBoost geográfico con todos los datos no mejora su test**, y su parte geográfica no agrega F1 medible
#    sobre el XGBoost global en zonas nuevas.
# 4. **Límite del diseño:** con 7 bloques de test ninguna diferencia puede ser significativa después de corregir por
#    comparaciones múltiples; las conclusiones firmes salen de los bloques espaciales de la validación y del acuerdo
#    de dirección con el test.
# ```
#
# **Sobre la respuesta del capítulo 9.** El XGBoost geográfico sigue superando al modelo base en zonas nuevas con
# la submuestra (+0.094, IC BCa [+0.042, +0.147], sección 9.3). Con todos los datos, la configuración del Entregable 2
# da +0.092 y la afinada +0.073, pero con el intervalo BCa de este entregable ninguna de las dos excluye el 0
# ([−0.024, +0.172] y [−0.119, +0.149]).
#
# **Sobre el intervalo del Entregable 2.** El Entregable 2 publicó para la configuración del XGBoost geográfico
# +0.092 con IC [+0.034, +0.199], calculado con bootstrap de percentiles sobre los 7 bloques. La cifra puntual se
# reproduce exactamente; con el método BCa que usa este entregable, el intervalo es [−0.024, +0.172]. Con solo 7
# bloques de test, los dos métodos son aproximaciones gruesas y la confirmación de esa diferencia **depende del método
# del intervalo**.
#
# **Sobre las comparaciones múltiples.** El test se miró una sola vez, pero se comparan 10 modelos con el base. El
# intervalo del Rotation Forest excluye el 0 con margen pequeño (+0.024 en su extremo inferior), así que conviene
# leerlo como evidencia favorable, no como prueba concluyente.
#
# ## 11.6 Reproducir
#
# Desde la carpeta del Entregable 3, con `PYTHONPATH` apuntando a ella y a `mejoras/`:
#
# ```
# ..\venv_e2\Scripts\python -u mejoras\01_geo_busqueda.py   # búsqueda del XGBoost geográfico (≈ 3 h)
# ..\venv_e2\Scripts\python -u mejoras\02_revision.py       # modelos de la revisión y rivales (≈ 2 h)
# ..\venv_e2\Scripts\python -u mejoras\03_test.py           # stacking, comparaciones y test (≈ 2 h)
# ..\venv_e2\Scripts\python -u mejoras\04_estadistica.py    # Friedman, Nemenyi, Wilcoxon, t corregida, Holm (≈ 15 min)
# ```
#
# Los tiempos son de un portátil de 4 núcleos con 3 procesos; las probabilidades de cada pliegue quedan en
# `mejoras/chk/` y los scripts se reanudan solos. Los resultados están en `mejoras/resultados/` y el detalle del
# protocolo en `mejoras/README.md`.
