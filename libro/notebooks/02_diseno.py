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
# # 2. Diseño experimental
#
# *Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*
#
# ```{admonition} Qué responde este capítulo
# :class: tip
# **Pregunta.** ¿Cómo se entrenan y evalúan 156 combinaciones de forma justa, sin fuga de información y en un
# portátil?
#
# **Respuesta corta.** Todas pasan por la misma validación anidada (5 pliegues externos × 3 internos, armados con
# kNNDM para imitar zonas nuevas), con el preprocesamiento y el balanceo dentro de cada pliegue, 12 evaluaciones
# de búsqueda y una submuestra estratificada de 10 001 viviendas. Cada una de las 640 tareas queda registrada en
# una tabla maestra.
# ```
#
# ## 2.1 Un pipeline combinatorio
#
# La guía pide entrenar el **producto** de alternativas, no su suma: cada modelo con cada técnica de balanceo y
# cada optimizador. A los 7 modelos de clasificación de la guía se agrega el **XGBoost geográfico**, el modelo
# nuevo del proyecto, para medirlo con exactamente el mismo protocolo.
#
# ```{note}
# **Propuesta y referencias.** Los 7 modelos de la guía no son la propuesta del proyecto: son los modelos de referencia
# (*benchmarks*) contra los que se pone a prueba el XGBoost geográfico, el único que no se había usado antes para
# clasificar el estrato (revisión bibliográfica del Entregable 2). La introducción explica el papel de cada uno.
# ```
#
# | Tarea | Modelos | Balanceo | Optimización | Total |
# |---|---|---|---|---|
# | Clasificación | 7 + 1 (XGBoost geográfico) | 4 | 4 | 7 × 4 × 4 = **112** (+ 16) |
# | Regresión | 7 | no aplica | 4 | 7 × 4 = **28** |
# | **Total** | | | | **140** (+ 16) |
#
# El pipeline está programado con funciones parametrizadas (modelo → balanceo → optimizador) y cada corrida se
# registra en una tabla maestra (sección 2.5).
#
# **Equivalentes de `class_weight='balanced'`.** Naive Bayes no tiene pesos de clase: se usan priors uniformes,
# que es el mismo efecto. KNN tampoco: se divide cada probabilidad por la frecuencia de la clase en el
# entrenamiento y se renormaliza ($\tilde p_k \propto p_k/\pi_k$), la corrección bayesiana que convierte priors
# empíricos en uniformes. XGBoost y el geográfico usan pesos de muestra $1/\text{frecuencia}$.
#
# **SMOTE y ADASYN** van dentro de `imblearn.pipeline.Pipeline`, después del preprocesamiento: se aplican solo al
# entrenamiento de cada pliegue y nunca al predecir. ADASYN reparte las muestras sintéticas según cuántos vecinos
# de otra clase tiene cada ejemplo; cuando un estrato no tiene ninguno (algo frecuente con un estrato tan
# concentrado), su densidad es cero y la librería falla. En ese caso se aplica estrato por estrato y el que falla
# se deja sin muestras sintéticas.
#
# ## 2.2 Validación cruzada anidada con kNNDM
#
# Elegir hiperparámetros mirando el mismo esquema con que se reporta el desempeño da una cifra optimista. Por
# eso:
#
# - **Bucle externo (5 pliegues):** estima el desempeño de generalización. Sus pliegues se arman con **kNNDM**
#   (*k-fold nearest neighbour distance matching*; Linnenbrink et al., 2024): grupos compactos de K-Means
#   repartidos en pliegues de modo que la distancia de cada vivienda de validación a la vivienda de entrenamiento
#   más cercana se parezca a la del escenario real (la geometría del test del Entregable 1). Las unidades de un
#   mismo edificio nunca se separan (ICC 0.99).
# - **Bucle interno (3 pliegues):** dentro del entrenamiento de cada pliegue externo, otro kNNDM cuyo escenario
#   es el propio pliegue externo: la búsqueda de hiperparámetros se valida a la misma distancia a la que luego se
#   evalúa.
#
# **Por qué 5 × 3.** Con 5 pliegues externos hay una media y una desviación entre pliegues (sección 5.4) y bloques
# para la prueba de Friedman. Con 3 internos cada evaluación cuesta 3 ajustes; 5 internos habrían multiplicado el
# costo por 5/3 sin cambiar la elección del óptimo en el piloto.
#
# ## 2.3 Una submuestra para la búsqueda
#
# Con las 257 273 viviendas de entrenamiento, las 156 combinaciones con validación anidada costaban unas 630
# horas-CPU en el portátil del proyecto (Intel Core i5-8265U, 4 núcleos; piloto del 18-oct). Se usa una
# **submuestra estratificada de 10 000 viviendas** (mismas proporciones de estrato), con 12 evaluaciones por
# búsqueda: 38 horas de cómputo en tareas, que con 8 procesos cupieron en 4.8 horas. Los finalistas se reentrenan
# y se evalúan en el test del Entregable 1, y además se reentrenan con las 257 273 viviendas para medir cuánto
# cuesta la submuestra (sección 9.4).

# %% tags=["hide-input"]
import pickle

import numpy as np
import pandas as pd

from _comun import RES, leer_csv

with open(RES / "datos_clasificacion.pkl", "rb") as fh:
    paq = pickle.load(fh)
sub = paq["sub"]
print(f"Submuestra: {len(sub):,} viviendas | kNNDM externo: q = {paq['knndm']['q']} grupos, "
      f"Wasserstein {paq['knndm']['W_km']:.3f} km (validación aleatoria: {paq['knndm']['W_aleatoria_km']:.3f} km)")
pd.DataFrame({"pliegue externo": range(5),
              "entrenamiento": [len(a) for a, _ in paq["externos"]],
              "validación": [len(b) for _, b in paq["externos"]],
              "estratos en validación": [len(np.unique(sub.estrato_num.to_numpy()[b])) for _, b in paq["externos"]],
              "pliegues internos (tamaños de validación)": [", ".join(str(len(v)) for _, v in i) for i in paq["internos"]]})

# %% tags=["hide-input"]
(sub.estrato_num.value_counts(normalize=True).sort_index().rename("proporción en la submuestra").to_frame()
 .style.format("{:.1%}"))

# %% [markdown]
# ## 2.4 Semilla única
#
# Una sola semilla global (`SEED = 42`, en `estrato3/configuracion.py`) se propaga a numpy, scikit-learn,
# imbalanced-learn (SMOTE/ADASYN), XGBoost, Optuna (TPE) y DEAP. La sensibilidad a otras semillas se mide en el
# capítulo 5.
#
# ## 2.5 Tabla maestra de experimentos
#
# Cada tarea (combinación × pliegue externo) guarda su semilla, sus hiperparámetros finales, las métricas del
# bucle externo, el tiempo de búsqueda, de ajuste y de inferencia, el número de evaluaciones y el historial
# completo de la búsqueda. La tabla maestra las reúne:

# %% tags=["hide-input"]
tm = leer_csv("tabla_maestra_clasificacion.csv")
if tm is not None:
    print(f"{len(tm)} tareas registradas ({tm.groupby(['modelo', 'balanceo', 'metodo']).ngroups} combinaciones)")
    display(tm[["modelo", "balanceo", "metodo", "fold", "f1_macro", "auc_ovr", "n_evaluaciones", "seg_busqueda",
                "params"]].head(12))
