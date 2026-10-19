# Estrato socioeconómico en Barranquilla — Entregable 3

**Optimización de hiperparámetros, balanceo, validación anidada y comparación estadística de modelos**

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Machine Learning,
Maestría, Universidad del Norte · Profesor Lihki Rubio

Este repositorio continúa los Entregables 1 y 2 (catastro abierto de Barranquilla, estrato 1–6). Construye el
pipeline combinatorio que pide la guía del entregable y pone a prueba el modelo nuevo del proyecto, el
**XGBoost geográfico adaptado a clasificación**, frente al modelo base (regresión logística).

**Libro (Jupyter Book):** https://mcarocantillo.github.io/entregable3-ml-barranquilla/ ·
Entregable 1: https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/ ·
Entregable 2: https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/

## Resultado principal

El XGBoost geográfico **supera al modelo base en zonas nuevas** (test: +0.094 de F1 macro, IC 95 % [+0.042, +0.147],
6 de 7 bloques), pero la mejora la explica sobre todo la **vecindad física** (cómo son los 8 edificios más cercanos):
frente a la logística, XGBoost y Random Forest con esas mismas variables no gana en zonas nuevas, y con todos los datos
y los hiperparámetros de la submuestra ya no se distingue del modelo base. Su ventaja aparece en zonas con datos cerca
(Entregable 2). Además: el modelo pesa más que el balanceo o el optimizador; los cuatro optimizadores empatan con el
mismo presupuesto (la bayesiana llega antes); `class_weight` es el mejor balanceo y ADASYN el peor. Detalle en los
capítulos 9 y 10 del libro.

## Qué se entrena

| Tarea | Modelos | Balanceo | Optimizadores | Combinaciones |
|---|---|---|---|---|
| Clasificación (estrato 1–6) | KNN, Naive Bayes, Logística L1/L2, Árbol, Random Forest, XGBoost, SVM + **XGBoost geográfico** | ninguno, SMOTE, ADASYN, `class_weight` | Grid, Random, Bayesiana (Optuna/TPE), Genética (DEAP) | 7 × 4 × 4 = 112 (+16 del modelo nuevo) |
| Regresión (estrato como número) | KNN, Ridge, Lasso, Árbol, Random Forest, XGBoost, SVR | no aplica | los mismos 4 | 7 × 4 = 28 |

Cada combinación se evalúa con **validación cruzada anidada**: 5 pliegues externos y 3 internos, ambos con
**kNNDM** (validación por emparejamiento de distancias al vecino más cercano, versión 3 del Entregable 1).
Las métricas que se reportan salen siempre del bucle externo.

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `estrato3/configuracion.py` | Semilla global (42), rutas, tamaño de la submuestra, pliegues y presupuesto |
| `estrato3/datos.py` | Carga del Entregable 1 v3, vecindad física KNN, submuestra y folds anidados (kNNDM) |
| `estrato3/modelos.py` | Catálogo de modelos y espacios de búsqueda; pipeline preprocesamiento → remuestreo → modelo |
| `estrato3/modelos_propios.py` | XGBoost multiclase y XGBoost geográfico (del Entregable 2) |
| `estrato3/optimizacion.py` | Objetivo (validación interna) y los cuatro optimizadores |
| `estrato3/evaluacion.py` | Métricas de clasificación, ordinales y de calibración (ECE, Brier) |
| `estrato3/estadistica.py` | Friedman, Nemenyi y diagrama CD, DeLong, Holm/BH, delta de Cliff, bootstrap BCa por bloques |
| `estrato3/estadistica_regresion.py` | MCS, Giacomini-White, Clark-West, Diebold-Mariano (HLN), bootstrap estacionario, supuestos de residuos |
| `estrato3/experimento.py` | Una tarea = combinación × pliegue externo (se guarda en disco: el experimento se puede retomar) |
| `experimentos/` | Ejecutores (`correr_clasificacion.py`, `correr_regresion.py`) y análisis (`analizar_*.py`, `finalistas.py`, `multifidelidad.py`, `complejidad*.py`, `tabla_estandar_optimizado.py`, `todos_los_datos.py`, `interpretabilidad.py`, `verificar_reproducibilidad.py`) |
| `resultados/` | Tabla maestra, resúmenes, JSON de análisis, probabilidades fuera de muestra (`tareas/`) y figuras. Los `.pkl` con la submuestra y el test no se suben: los regeneran `correr_clasificacion.py` y `finalistas.py` |
| `libro/` | Jupyter Book del entregable: cuadernos en formato jupytext (`.py`) y ejecutados (`.ipynb`); `construir_libro.py` los ejecuta y construye el HTML |
| `requirements-e3.txt` | Versiones exactas (Python 3.14.6) |

## Cómo reproducir

```bash
python -m venv venv && venv/Scripts/pip install -r requirements-e3.txt
# el Entregable 1 v3 (rama v3) debe estar en ../e1_rama o en la variable de entorno ENTREGABLE1
python -u experimentos/correr_clasificacion.py      # 640 tareas (~4.8 h con 8 procesos)
python -u experimentos/correr_regresion.py          # 140 tareas
python experimentos/analizar_clasificacion.py
python experimentos/analizar_regresion.py
python -u experimentos/finalistas.py                # test, calibración, semillas (paralelo)
python -u experimentos/multifidelidad.py
python -u experimentos/complejidad.py               # con el computador libre (mide tiempos)
python -u experimentos/complejidad_faiss.py         # con el computador libre
python -u experimentos/tabla_estandar_optimizado.py # con el computador libre
python -u experimentos/todos_los_datos.py           # finalistas con las 257 273 viviendas
python -u experimentos/interpretabilidad.py         # SHAP y LIME (clasificación y regresión)
python -u experimentos/verificar_reproducibilidad.py
cd libro && python construir_libro.py
```

Cada tarea se guarda en `resultados/tareas/` con su semilla, hiperparámetros finales, métricas del bucle externo,
historial completo de la búsqueda y tiempos; si la corrida se interrumpe, al relanzarla continúa donde iba.
