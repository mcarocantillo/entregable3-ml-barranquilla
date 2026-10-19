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
# # 6. Regresión: el estrato como número
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# **Pregunta.** ¿Ayuda tratar el estrato como un número (1 a 6) en vez de como una categoría?
#
# **Respuesta corta.** No para clasificar. El mejor regresor (Random Forest, RMSE 0.57 estratos) comprime los
# extremos y, al redondear, su F1 macro queda por debajo del de los clasificadores. Sus residuos están
# correlacionados en el espacio: la ubicación deja información sin explicar.
# ```
#
# La guía pide también 7 modelos de regresión. El problema del proyecto es de clasificación, así que la rama de
# regresión se plantea como **la lectura ordinal del mismo problema**: predecir el estrato (1–6) como número y
# medir el error en estratos. Así se puede comparar con la clasificación (redondeando la predicción al estrato
# más cercano) y responder si tratar el estrato como continuo ayuda. Mismos datos, submuestra y pliegues; el
# balanceo no aplica.
#
# ## 6.1 Métricas en el bucle externo

# %% tags=["hide-input"]
import pandas as pd

from _comun import figura, leer_csv, leer_json

RR = leer_csv("resumen_regresion.csv")
AR = leer_json("analisis_regresion.json") or {}
if RR is not None:
    t = RR.copy()
    t.columns = [c.replace("_mean", " (media)").replace("_std", " (desv.)") for c in t.columns]
    display(t.round(4))

# %% [markdown]
# **Lectura.** Los ensambles de árboles vuelven a ganar: Random Forest (RMSE 0.571 ± 0.100 estratos, R² 0.83) y
# XGBoost (0.581) van por delante de SVR (0.596), KNN (0.633), los lineales Ridge y Lasso (≈ 0.67,
# indistinguibles entre sí y entre optimizadores) y el árbol individual (0.76). Un RMSE de 0.57 quiere decir que
# el error típico es de algo más de medio estrato; el MAE, de 0.43 estratos.
#
# Al redondear la predicción al estrato más cercano, el F1 macro de los mejores (0.57 a 0.60) queda por debajo
# del de los clasificadores (≈ 0.66, capítulo 5). La causa es que la regresión **comprime los extremos hacia el
# centro**: en las predicciones fuera de muestra de Random Forest, las viviendas de estrato 6 reciben en
# promedio 5.03 y solo el 19 % se redondea a 6; las de estrato 1 reciben 1.48 (65 % de acierto). La pérdida
# cuadrática empuja hacia la media en las clases raras de los extremos, que son justo las que más pesan en el F1
# macro. Para este problema, entonces, tratar el estrato como número no mejora la clasificación; sirve como
# lectura ordinal del error.
#
# Los tiempos de búsqueda de esta tabla están inflados: por un error de ejecución corrieron dos copias
# simultáneas de la regresión, así que solo se pueden comparar entre sí, no con los de clasificación. Aun con
# esa salvedad, Ridge con SAGA es el más lento (770 a 910 s por búsqueda) por la convergencia del solver con
# regularización débil (capítulo 4).
#
# ## 6.2 Entrenamiento, validación y prueba

# %% tags=["hide-input"]
figura("regresion_train_val_test.png")
if AR.get("test_mejor"):
    print(AR["test_mejor"])

# %% [markdown]
# ## 6.3 Análisis de residuos del mejor modelo
#
# Las pruebas de independencia de la guía (BDS, Ljung-Box, ACF) se diseñaron para series de tiempo. El catastro no
# tiene tiempo, así que los residuos se **ordenan en el espacio** (de sur a norte, la dirección del gradiente del
# estrato): la «autocorrelación serial» se lee como parecido entre viviendas cercanas en esa dirección. Se
# complementa con el I de Moran de los residuos (8 vecinos), la prueba natural de dependencia espacial.
#
# - **White**: H0 = varianza constante de los errores (regresores: predicción y coordenadas).
# - **BDS**: H0 = residuos independientes e idénticamente distribuidos (detecta dependencia no lineal).
# - **Ljung-Box** (20 rezagos): H0 = sin autocorrelación.

# %% tags=["hide-input"]
figura("residuos_regresion.png")
if AR.get("supuestos_mejor"):
    s = AR["supuestos_mejor"]
    display(pd.DataFrame({"estadístico": [s.get("white", {}).get("lm"), s["bds"]["estadistico_dim2"],
                                          s["ljung_box"]["q"], s["normalidad"]["asimetria"], s["moran_residuos_k8"]],
                          "p-valor": [s.get("white", {}).get("p"), s["bds"]["p_dim2"], s["ljung_box"]["p"],
                                      s["normalidad"]["p_dagostino"], None]},
                         index=["White", "BDS (dim. 2)", "Ljung-Box (20)", "asimetría / D'Agostino", "I de Moran"]))

# %% [markdown]
# **Lectura.** Los residuos del mejor modelo (Random Forest con Grid) no cumplen ningún supuesto clásico. White
# rechaza la varianza constante (LM = 721, p ≈ 10⁻¹⁴⁹); BDS y Ljung-Box rechazan la independencia (p ≈ 0); la
# autocorrelación en el orden sur → norte es 0.46 en el primer rezago y decae despacio (0.30 en el cuarto, 0.07
# en el vigésimo). El I de Moran de los residuos con 8 vecinos es 0.35: viviendas vecinas comparten el error.
# Las colas son más pesadas que las de una normal (curtosis en exceso 2.3), aunque la distribución es simétrica
# (asimetría −0.02).
#
# Esto importa para el proyecto por dos razones. Primero, aun con la vecindad física de 8 vecinos como
# variables, queda estructura espacial sin explicar en el error: es justo lo que el XGBoost geográfico intenta
# capturar con modelos locales. Segundo, como los errores no son independientes, las pruebas e intervalos deben
# remuestrear bloques espaciales (capítulo 7), no viviendas sueltas.
