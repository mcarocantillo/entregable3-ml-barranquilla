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
# # 9. El modelo nuevo frente al modelo base
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# **Pregunta.** ¿Supera el XGBoost geográfico al modelo base? Es la pregunta del proyecto.
#
# **Respuesta corta.** Sí en zonas nuevas, con la configuración del Entregable 2 y con la submuestra de este
# entregable (+0.09 de F1 en el test en ambos casos), pero no con todos los datos y los hiperparámetros elegidos
# en la submuestra. Y frente a modelos que reciben la misma vecindad física no gana en zonas nuevas: la mejora
# sobre el modelo base la explica sobre todo esa vecindad. La respuesta completa está en la sección 9.5.
# ```
#
# ```{admonition} El objetivo del proyecto
# :class: tip
# Usar un modelo que no se había aplicado a este problema y comprobar si **supera al modelo base**. El modelo
# nuevo es el **XGBoost geográfico adaptado a clasificación**: un XGBoost global más XGBoost locales entrenados
# alrededor de anclas con un kernel espacial adaptativo; la predicción mezcla
# $p = \alpha\,p_{\text{global}} + (1-\alpha)\,p_{\text{local}}$. Grekousis (2025) lo propuso para regresión; la
# versión multiclase con mezcla de probabilidades es la adaptación de este proyecto. El modelo base es la
# regresión logística multinomial del Entregable 1.
#
# Los demás modelos del pipeline son **referencias**, no la propuesta: sirven para que la comparación sea justa y para
# saber de dónde viene la mejora (introducción, recuadro «¿Por qué aparecen modelos conocidos…?»).
# ```
#
# **Criterio de reemplazo** (el mismo desde el Entregable 1): un modelo reemplaza al base si su diferencia
# pareada de F1 macro por pliegue supera **un error estándar** en la validación cruzada; el test (los mismos 7
# bloques de siempre, con IC por bootstrap de bloques) es la confirmación.
#
# La comparación se hace en dos niveles:
#
# 1. **Con todos los datos** (257 273 viviendas, fase 2 de la versión 3 del Entregable 2): validación kNNDM de 5
#    pliegues y test, en los dos escenarios de predicción.
# 2. **En el pipeline combinatorio de este entregable** (submuestra de 10 000, validación anidada): el XGBoost
#    geográfico con sus 16 combinaciones de balanceo y optimizador frente a los otros 7 modelos.
#
# ## 9.1 Con todos los datos: zonas nuevas y zonas conocidas

# %% tags=["hide-input"]
import pandas as pd

from _comun import leer_csv, leer_json

F2 = leer_json("e2_fase2_arboles_knndm.json")
filas = []
for esc, e in F2["xgb_geografico_vs_logistica"].items():
    for var, v in e["variantes"].items():
        for ref, c in v.items():
            if not ref.startswith("vs "):
                continue
            t = c.get("test_F1") or {}
            filas.append({"escenario": esc, "XGBoost geográfico con": var.split(" | ")[1],
                          "frente a": ref[3:], "Δ F1 en CV": c["cv_F1"]["media"], "SE": c["cv_F1"]["SE"],
                          "pliegues a favor": f"{c['cv_F1']['folds_a_favor']}/5",
                          "¿reemplaza? (> 1 SE)": "sí" if c["cv_F1"]["supera_por_1SE"] else "no",
                          "Δ F1 en test": t.get("diferencia"),
                          "IC 95 % test": (f"[{t['IC95_inf']:+.3f}, {t['IC95_sup']:+.3f}]" if t else "—")})
tab = pd.DataFrame(filas)
base = tab["frente a"].str.startswith("Logística B v3")
display(tab[base].round(3))

# %% [markdown]
# **Frente al modelo base, el modelo nuevo gana.** En **zonas nuevas**, el XGBoost geográfico con vecindad física
# supera a la logística B por más de un error estándar en validación (+0.11 a +0.13 de F1, ganando en los 5
# pliegues), y en el test la ventaja se confirma con la vecindad del grafo KNN: +0.09 (IC 95 % [+0.03, +0.20]), y
# +0.07 ([+0.03, +0.18]) si además usa el estrato de los vecinos conocidos. Con la vecindad por radios la ventaja en
# test es la misma (+0.09) pero su intervalo incluye el 0 ([−0.02, +0.21]). Sin vecindad, el XGBoost geográfico
# gana en validación (+0.07, SE 0.04, 4 de 5 pliegues) pero no en el test (−0.01). En **zonas conocidas** la ventaja es mucho mayor: +0.19 a +0.23
# en validación, en los 5 pliegues.
#
# ¿Es mérito del modelo o de las variables? La tabla siguiente lo compara con una logística que recibe **las
# mismas** variables de vecindad:

# %% tags=["hide-input"]
display(tab[~base & (tab["XGBoost geográfico con"] != "B")].round(3))

# %% [markdown]
# Con las mismas variables, en **zonas nuevas** el modelo nuevo **empata** con la logística (diferencias de ±0.02
# en validación, todas dentro de un error estándar, e intervalos de test que incluyen el 0): donde no hay datos
# cerca, lo que mejora la predicción es describir el entorno, y una logística lo aprovecha igual de bien. En
# **zonas conocidas** el XGBoost geográfico con vecindad y estrato de los vecinos **sí supera** a la mejor
# logística (F1 0.778 frente a 0.756; +0.022, SE 0.010, 4 de 5 pliegues; MAE 0.240 frente a 0.276): donde hay
# datos cerca, sus modelos locales aprenden el mapa.
#
# ## 9.2 En el pipeline combinatorio

# %% tags=["hide-input"]
R = leer_csv("resumen_combinaciones.csv")
A = leer_json("analisis_clasificacion.json") or {}
if R is not None:
    mejor = R.sort_values("f1_macro_mean", ascending=False).groupby("modelo").head(1)
    display(mejor[["modelo", "balanceo", "metodo", "f1_macro_mean", "f1_macro_std", "auc_ovr_mean",
                   "mae_ordinal_mean"]].round(4))
if A.get("friedman_mejor_por_modelo"):
    print("Rangos medios (1 = mejor) del mejor de cada modelo:", A["friedman_mejor_por_modelo"]["rangos"])

# %% [markdown]
# **Lectura.** En el pipeline combinatorio, con las mismas variables para todos (modelo B + vecindad física de 8
# vecinos), el XGBoost geográfico tiene el mejor rango medio por bloques espaciales (3.03, capítulo 7) y su
# mejor combinación (`class_weight` + optimización bayesiana) alcanza F1 macro 0.656 ± 0.063 en el bucle
# externo, igual que XGBoost (0.656) y Random Forest (0.658).
#
# Con el criterio de reemplazo (diferencia pareada por pliegue externo frente a la mejor combinación de cada
# rival):
#
# | XGBoost geográfico frente a | Δ F1 medio | SE | pliegues a favor | ¿> 1 SE? |
# |---|---|---|---|---|
# | Logística L1/L2 (misma vecindad) | +0.048 | 0.015 | 5/5 | sí |
# | XGBoost global | −0.000 | 0.008 | 2/5 | no (empate) |
# | Random Forest | −0.002 | 0.006 | 3/5 | no (empate) |
#
# Frente a la logística la ventaja es de más de tres errores estándar y aparece en todos los pliegues; el MAE
# ordinal baja de 0.403 a 0.320 estratos. Frente a los otros dos ensambles, empate.
#
# ¿Los modelos locales aportan algo? La comparación con XGBoost responde esa pregunta, porque XGBoost es el mismo
# algoritmo que la parte global del modelo nuevo, sin los modelos locales: es un experimento de ablación. El peso del XGBoost global, α, lo
# eligieron los optimizadores en las 80
# búsquedas (16 combinaciones × 5 pliegues): mediana 0.65 (rango intercuartílico 0.53 a 0.74), solo 1 de 80
# eligió α ≥ 0.95 (el XGBoost global puro) y 18 dieron más peso a la parte local (α ≤ 0.5). La búsqueda prefiere
# siempre una mezcla, aunque el F1 final quede igual que el del XGBoost global: la parte local reparte el
# acierto de otra forma (mejores AUC en los estratos 1, 4 y 6, peores en el 3 y el 5; capítulo 7) más que
# aumentarlo.
#
# Este resultado no contradice la sección 9.1, pero tampoco es idéntico: allí, con todos los datos, el XGBoost
# geográfico **empataba** en zonas nuevas con una logística que tenía la misma vecindad. Las condiciones son
# distintas: aquí cada modelo se entrena con unas 8 000 viviendas dispersas por la ciudad (no con todo el
# entrenamiento), con 12 evaluaciones de búsqueda, y la geometría de validación es la del test respecto a esa
# submuestra. Una explicación posible es que, con menos datos y vecinos más lejanos, las interacciones que
# capturan los árboles pesan más; el test de la sección 9.3, con los mismos 7 bloques de siempre, sirve de
# árbitro.
#
# ## 9.3 La prueba de fuego: el test
#
# Los finalistas del capítulo 5 se evaluaron una sola vez en el test (los mismos 7 bloques de 2 km de todos los
# entregables). La tabla resume el XGBoost geográfico frente a cada rival en las dos miradas: el criterio de
# reemplazo en los 5 pliegues externos (diferencia pareada de F1 macro y su error estándar) y la diferencia en el
# test, con IC 95 % BCa remuestreando bloques y el número de bloques de test en los que gana.

# %% tags=["hide-input"]
FN = leer_json("finalistas.json") or {}
if FN.get("geo_vs_rivales"):
    g = pd.DataFrame(FN["geo_vs_rivales"])
    g["pliegues a favor"] = g.cv_pliegues_a_favor.astype(str) + "/5"
    g["IC 95 % test"] = g.test_ic95_bca.map(lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]")
    g["bloques de test a favor"] = g.test_bloques_a_favor.astype(str) + "/" + g.test_n_bloques.astype(str)
    g["¿> 1 SE en CV?"] = g.cv_supera_1se.map({True: "sí", False: "no"})
    display(g[["rival", "cv_dif_media", "cv_se", "pliegues a favor", "¿> 1 SE en CV?", "test_dif", "IC 95 % test",
               "bloques de test a favor"]].rename(columns={"cv_dif_media": "Δ F1 en CV", "cv_se": "SE",
                                                         "test_dif": "Δ F1 en test"}).round(3))
else:
    print("(finalistas pendientes)")

# %% [markdown]
# **Lectura.** Las dos miradas coinciden frente al modelo base y discrepan frente a los rivales que usan la
# misma vecindad.
#
# - **Frente al modelo base del E1:** +0.197 en validación (SE 0.034, 5 de 5 pliegues) y **+0.094 en el test (IC 95 %
#   [+0.042, +0.147])**, ganando en 6 de los 7 bloques. Cumple el criterio de reemplazo y el intervalo excluye el 0.
#   Es la misma magnitud que con todos los datos en el Entregable 2 (+0.09).
# - **Frente a la logística con la misma vecindad:** +0.048 en validación (5 de 5), pero **−0.050 en el test**
#   (IC [−0.107, +0.011]; gana en 3 de 7 bloques). La ventaja de validación no se confirma: el intervalo incluye el 0,
#   pero apunta en contra.
# - **Frente a XGBoost y Random Forest:** empate en validación y −0.044 y −0.035 en el test, con intervalos que
#   incluyen el 0 (gana en 3 de 7 bloques).
#
# ¿Por qué cae en el test? Su búsqueda final, con toda la submuestra, eligió α = 0.97: el modelo quedó casi en
# el XGBoost global (profundidad 8), con una parte local mínima, porque con 10 000 viviendas dispersas cada
# ancla tiene pocos vecinos para aprender un modelo local. En el test pierde además los estratos 3 y 6 (recall
# 0.25 y 0.20, frente a 0.70 y 0.57 de la logística), justo los que más pesan en el F1 macro de esos 7 bloques.
#
# ## 9.4 ¿Y con todos los datos?
#
# Los finalistas anteriores se entrenaron con la submuestra de 10 001 viviendas. Para medir cuánto pesa esa
# decisión, cada finalista (con los hiperparámetros que eligió su búsqueda, sin buscar de nuevo) y el modelo base se
# reentrenaron **una vez con las 257 273 viviendas** del entrenamiento y se evaluaron en el mismo test. Los
# hiperparámetros se eligieron para 10 000 viviendas, así que con 25 veces más datos no tienen por qué ser los
# óptimos (sobre todo los que cuentan viviendas, como las hojas mínimas); el ejercicio mide la sensibilidad al
# tamaño de muestra, no el mejor desempeño posible de cada modelo.

# %% tags=["hide-input"]
TD = leer_json("todos_los_datos.json") or {}
if TD.get("modelos"):
    td = pd.DataFrame({n: {"F1 test (10 001)": v["f1_test_submuestra"], "F1 test (257 273)": v["test"]["f1_macro"],
                           "IC 95 % BCa (257 273)": f"[{v['ic95_bca_f1_test'][0]:.3f}, {v['ic95_bca_f1_test'][1]:.3f}]",
                           "MAE ordinal (257 273)": v["test"]["mae_ordinal"], "minutos": v["segundos"] / 60}
                       for n, v in TD["modelos"].items()}).T
    td["cambio"] = td["F1 test (257 273)"].astype(float) - td["F1 test (10 001)"].astype(float)
    num = ["F1 test (10 001)", "F1 test (257 273)", "cambio", "MAE ordinal (257 273)", "minutos"]
    td[num] = td[num].astype(float).round(3)
    display(td.sort_values("F1 test (257 273)", ascending=False))
    g = pd.DataFrame(TD["geo_vs_rivales"])
    g["IC 95 % test"] = g.test_ic95_bca.map(lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]")
    g["bloques a favor"] = g.test_bloques_a_favor.astype(str) + "/" + g.test_n_bloques.astype(str)
    display(g[["rival", "test_dif", "IC 95 % test", "bloques a favor"]]
            .rename(columns={"test_dif": "Δ F1 del XGBoost geográfico en test"}).round(3))
else:
    print("(reentrenamiento con todos los datos pendiente)")

# %% [markdown]
# **Lectura.** Con 25 veces más datos, la mayoría de los modelos mejora poco: Random Forest +0.025 (F1 0.582),
# XGBoost +0.010 (0.576), la logística con vecindad +0.006 (0.578), SVM +0.010 y Naive Bayes +0.018. El que más
# gana es el modelo base del E1 (+0.065, de 0.428 a 0.493). Tres empeoran: el árbol (−0.075), KNN (−0.018) y el
# XGBoost geográfico (−0.008, 0.514). Los tres tienen hiperparámetros elegidos para 10 000 viviendas que no se
# trasladan a 257 273: el árbol, profundidad 10 con hojas de 5 viviendas; KNN, 54 vecinos; el geográfico,
# profundidad 8 con casi todo el peso en el modelo global (α = 0.97).
#
# Para el modelo nuevo el resultado es desfavorable. Con todos los datos y los hiperparámetros de la submuestra
# ya **no se distingue del modelo base** (+0.021, IC [−0.079, +0.082], 5 de 7 bloques) y queda **por debajo de
# XGBoost** (−0.062, IC [−0.072, −0.045]) **y de Random Forest** (−0.068, IC [−0.096, −0.019]), con intervalos
# que excluyen el 0; frente a la logística con vecindad, −0.064 (IC [−0.109, +0.005]).
#
# Dos conclusiones:
#
# 1. **La submuestra se justifica a posteriori.** Para los modelos cuyos hiperparámetros no dependen de n, pasar de
#    10 001 a 257 273 viviendas mueve el F1 de test entre +0.006 y +0.025, y los tres mejores (logística con
#    vecindad, XGBoost y Random Forest) siguen empatados. La guía pide reducir el costo manteniendo el desempeño: la
#    submuestra redujo la validación anidada de unas 630 a unas 38 horas a cambio de, como mucho, 0.025 de F1.
# 2. **El XGBoost geográfico es el modelo más sensible a sus hiperparámetros.** En el Entregable 2, con
#    hiperparámetros elegidos con todos los datos (α = 0.75, es decir, un 25 % de peso para los modelos locales, y
#    profundidad 4), superaba al modelo base por +0.09 en el test. Con los elegidos en la submuestra (α = 0.97,
#    profundidad 8), su ventaja desaparece.
#
# ## 9.5 Respuesta
#
# ```{admonition} ¿El XGBoost geográfico supera al modelo base?
# :class: important
# **Sí, en zonas nuevas, pero la ventaja depende de la configuración y viene sobre todo de la vecindad física.**
#
# | Escenario | XGBoost geográfico − modelo base (F1 macro en test) | ¿Lo reemplaza? |
# |---|---|---|
# | Todos los datos, hiperparámetros del E2 (sección 9.1) | +0.09, IC [+0.03, +0.20] | sí |
# | Submuestra de 10 001 viviendas (sección 9.3) | +0.094, IC [+0.042, +0.147]; 6 de 7 bloques | sí |
# | Todos los datos, hiperparámetros de la submuestra (sección 9.4) | +0.021, IC [−0.079, +0.082] | no se distingue |
#
# **Frente a modelos con la misma vecindad no gana en zonas nuevas.** En el test empata o pierde con la logística
# con vecindad, XGBoost y Random Forest, y con todos los datos pierde de forma clara frente a los dos últimos. Su
# única ventaja demostrada sobre la mejor logística con las mismas variables está en **zonas conocidas** (Entregable
# 2: F1 0.778 frente a 0.756, +0.022, 4 de 5 pliegues), donde sus modelos locales aprenden el mapa.
# ```
#
# La lectura honesta para el proyecto: el modelo nuevo **sí mejora al modelo base**, como se buscaba, pero la mayor
# parte de esa mejora la explica la **vecindad física de los 8 vecinos más cercanos**, que también hace subir a la
# logística, a XGBoost y a Random Forest. Los modelos locales del XGBoost geográfico solo aportan cuando hay datos
# cerca (zonas conocidas) y cuando se les da peso (α ≈ 0.75). Con pocos datos por ancla su aporte es inestable: en
# los pliegues externos la búsqueda les dio un peso mediano del 35 % (α = 0.65), pero la búsqueda final con toda la
# submuestra casi los apagó (α = 0.97) y el modelo quedó reducido a un XGBoost global más profundo.
