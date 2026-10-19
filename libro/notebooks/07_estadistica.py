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
# # 7. Comparación estadística
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# **Pregunta.** De todas las diferencias observadas, ¿cuáles son reales y cuáles son ruido?
#
# **Respuesta corta.** Friedman confirma que los modelos difieren. Con la mejor combinación de cada uno, el
# XGBoost geográfico tiene el mejor rango medio, pero Nemenyi solo separa a Naive Bayes del grupo de cabeza.
# Frente a la logística, el geográfico ordena mejor los estratos 1 a 4 y peor el 5 y el 6 (DeLong con Holm), con
# un efecto pequeño (δ de Cliff = 0.28). En regresión, el MCS deja 8 de 28 combinaciones, y la diferencia entre
# las dos mejores es significativa pero sin relevancia práctica.
# ```
#
# Con 140 modelos hay 9 730 pares posibles; probarlos uno a uno con α = 0.05 daría unos 487 falsos positivos por
# azar. Por eso la comparación es **jerárquica**: primero una prueba global, después comparaciones dirigidas entre
# pocos finalistas con corrección por comparaciones múltiples, y siempre con tamaño del efecto.
#
# ## 7.1 Clasificación: Friedman → Nemenyi
#
# **Bloques.** Con 5 pliegues externos, la prueba de Friedman tiene muy poca potencia (la diferencia crítica de
# Nemenyi para 8 modelos sería de casi 5 rangos). Se usan además, como bloques, los **bloques espaciales de 2 km**
# del Entregable 1: el F1 macro de las predicciones fuera de muestra de cada combinación, calculado en cada bloque.
# Ambas versiones se reportan.

# %% tags=["hide-input"]
import pandas as pd

from _comun import figura, leer_json

A = leer_json("analisis_clasificacion.json") or {}
for clave, titulo in [("friedman_todas_pliegues", "Todas las combinaciones, bloques = pliegues externos"),
                      ("friedman_todas_bloques", "Todas las combinaciones, bloques = bloques espaciales"),
                      ("friedman_mejor_por_modelo", "Mejor combinación de cada modelo, bloques espaciales")]:
    if A.get(clave):
        f = A[clave]
        print(f"{titulo}: χ²_F = {f['chi2']:.2f}, gl = {f['gl']}, p = {f['p']:.3g}, n = {f['n_bloques']} bloques, "
              f"diferencia crítica (Nemenyi, α = 0.05) = {f['cd']:.2f} rangos")
figura("cd_mejor_combinación_por_modelo.png")
figura("cd_todas_las_combinaciones_(bloques_espaciales).png")

# %% tags=["hide-input"]
if A.get("nemenyi_mejor_por_modelo"):
    display(pd.DataFrame(A["nemenyi_mejor_por_modelo"]).round(3))

# %% [markdown]
# **Lectura.** Las tres versiones de Friedman rechazan que todas las combinaciones sean equivalentes (p entre
# 10⁻⁴⁶ y 10⁻¹³¹). Con 128 combinaciones, sin embargo, la diferencia crítica es enorme (103 rangos con 5
# pliegues; 41 con 32 bloques espaciales) y Nemenyi solo separa los extremos.
#
# La comparación útil es la de la **mejor combinación de cada modelo** con 32 bloques espaciales (χ² = 47.6, p =
# 4·10⁻⁸, CD = 1.86 rangos). **El XGBoost geográfico tiene el mejor rango medio (3.03)**, seguido de XGBoost
# (3.41) y Random Forest (4.05); KNN (4.63), la logística (4.69), el árbol (4.83) y SVM (4.86) forman un grupo
# intermedio, y Naive Bayes es el último (6.52). En Nemenyi, lo único claramente separado es Naive Bayes, peor
# que los tres ensambles y que KNN (p < 0.05). El XGBoost geográfico aventaja a la logística en 1.66 rangos,
# menos que la CD (p = 0.12), y a SVM en 1.83 (p = 0.057). Nemenyi es conservador con 8 modelos; por eso las
# comparaciones dirigidas de 7.2 y 7.3 se hacen solo entre finalistas.
#
# ## 7.2 Comparaciones dirigidas: DeLong con Holm
#
# La prueba de DeLong compara dos AUC calculadas sobre las mismas observaciones teniendo en cuenta su correlación.
# Es una prueba para AUC binarias, así que se aplica a cada estrato contra el resto (6 pruebas por par de
# modelos), solo entre los tres mejores modelos y el modelo base, y los p-valores se ajustan con **Holm-Bonferroni**
# (control del error familiar) y **Benjamini-Hochberg** (tasa de falsos descubrimientos).

# %% tags=["hide-input"]
if A.get("delong_finalistas"):
    dl = pd.DataFrame(A["delong_finalistas"])
    display(dl[["A", "B", "estrato", "auc_1", "auc_2", "diferencia", "z", "p", "p_holm", "p_bh"]].round(4))

# %% [markdown]
# **Lectura.** Las AUC se calculan con las predicciones fuera de muestra del bucle externo (10 001 viviendas),
# así que diferencias pequeñas resultan significativas; lo que importa es el signo y el tamaño.
#
# - **XGBoost geográfico frente a la logística** (la mejor combinación de la familia del modelo base): AUC mayor en
#   los estratos 1 a 4 (+0.016 a +0.051, p de Holm < 10⁻⁴) y **menor en los estratos 5 y 6** (−0.019 y −0.011,
#   también significativas).
# - XGBoost y Random Forest muestran el mismo patrón frente a la logística: ganan en los estratos 1 a 4, empatan en
#   el 5 y pierden en el 6. La logística ordena mejor los estratos altos; posiblemente porque son pocos, están
#   concentrados y las variables físicas los separan casi linealmente, mientras que los estratos bajos y medios se
#   mezclan en el espacio y ahí ayudan las interacciones de los árboles.
# - Entre ensambles las diferencias son de milésimas y cambian de signo según el estrato. Random Forest ordena algo
#   mejor que los dos XGBoost en la mayoría de estratos; el XGBoost geográfico supera al global en los estratos 1 y
#   6 (y en el 4 sin significancia tras Holm) y pierde en el 3 y el 5.
#
# Una AUC por estrato mide la capacidad de **ordenar**; el F1 macro mide la decisión final. Por eso Random
# Forest puede tener AUC algo mayores y aun así quedar detrás del XGBoost geográfico en el rango de F1 por
# bloques.
#
# ## 7.3 Tamaño del efecto e intervalos de confianza
#
# **Delta de Cliff** (sobre el F1 por bloque espacial): probabilidad de que un modelo supere al otro en un bloque
# menos la probabilidad contraria; |δ| < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano, mayor grande.
# **Intervalos BCa** (*bias-corrected and accelerated*) del F1 macro, remuestreando bloques espaciales completos.

# %% tags=["hide-input"]
if A.get("cliff_finalistas"):
    display(pd.DataFrame(A["cliff_finalistas"]).round(3))
if A.get("ic_bca_f1_finalistas"):
    display(pd.DataFrame(A["ic_bca_f1_finalistas"], index=["IC 95 % inf", "IC 95 % sup"]).T.round(3))

# %% [markdown]
# **Lectura.** Los intervalos BCa del F1 macro (remuestreando bloques espaciales) se solapan entre los tres
# ensambles: Random Forest [0.640, 0.728], XGBoost [0.654, 0.729] y XGBoost geográfico [0.648, 0.732]. El de la
# logística, [0.584, 0.691], está desplazado hacia abajo, aunque se solapa en parte con los demás.
#
# El delta de Cliff confirma la lectura. Entre ensambles es despreciable (|δ| ≤ 0.13). Frente a la logística es
# **pequeño y a favor de los ensambles**, mayor para el XGBoost geográfico (δ = 0.28: sin empates, gana en ≈ 64
# % de los pares de bloques y pierde en ≈ 36 %) que para XGBoost (0.24) y Random Forest (0.17).
#
# **Conclusión de la comparación en zonas nuevas:** el XGBoost geográfico está en el grupo de los mejores, sin
# diferencia demostrable con XGBoost y Random Forest, y mejora a la logística con un efecto pequeño pero
# consistente en las tres miradas (rango, AUC de los estratos 1–4 y delta de Cliff).
#
# ## 7.4 Regresión: MCS → Giacomini-White / Clark-West → Diebold-Mariano
#
# - **Model Confidence Set** (Hansen, Lunde y Nason, 2011): parte de las 28 combinaciones y elimina, con bootstrap
#   por bloques, las significativamente peores; quedan las indistinguibles de la mejor (α = 0.10).
# - **Giacomini-White** (2006) entre finalistas no anidados: ¿un modelo es mejor de forma condicional (dada la
#   diferencia de pérdida de la vivienda vecina en el orden sur → norte)?
# - **Clark-West** (2007) para un par **anidado**: Ridge con vecindad física frente a Ridge sin ella. Corrige el
#   sesgo del Diebold-Mariano a favor del modelo grande.
# - **Diebold-Mariano** con la corrección de Harvey, Leybourne y Newbold entre el mejor y el segundo, con intervalo
#   por **bootstrap estacionario** (Politis y Romano, 1994) y d de Cohen.

# %% tags=["hide-input"]
AR = leer_json("analisis_regresion.json") or {}
if AR.get("mcs"):
    display(pd.DataFrame(AR["mcs"]).round(4))
if AR.get("giacomini_white"):
    display(pd.DataFrame(AR["giacomini_white"]).round(4))
for k in ["clark_west_ridge_vecindad", "diebold_mariano_hln"]:
    if AR.get(k):
        print(k, AR[k])

# %% [markdown]
# **Lectura.** El MCS (α = 0.10) conserva 8 de las 28 combinaciones: Random Forest con Grid, Random y Genética,
# XGBoost con Random y Bayesiana, y SVR con Random, Grid y Bayesiana; elimina KNN, los lineales y el árbol.
#
# Giacomini-White distingue a los finalistas de forma condicional (p < 10⁻¹⁸ en los 6 pares, siempre a favor del
# de menor pérdida media): conociendo la diferencia de pérdida de la vivienda vecina se puede anticipar cuál
# modelo se equivocará menos, otra vez porque los errores están correlacionados en el espacio. Con 10 001
# viviendas, además, la prueba detecta diferencias muy pequeñas.
#
# Diebold-Mariano (HLN) entre los dos primeros (Random Forest con Grid y con Random) encuentra una diferencia
# estadísticamente significativa (−0.0096 en pérdida cuadrática, p = 1.5·10⁻⁸; IC 95 % por bootstrap
# estacionario [−0.019, −0.001]) pero **sin relevancia práctica**: d de Cohen = −0.06.
#
# Clark-West confirma que la vecindad física aporta información: Ridge con las 7 variables de vecindad predice
# mejor que el mismo Ridge sin ellas (estadístico 47.4, p ≈ 0).
