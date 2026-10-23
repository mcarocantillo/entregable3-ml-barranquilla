# A. Reproducibilidad e ingeniería del pipeline

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

```{admonition} Qué responde este capítulo
:class: tip
**Pregunta.** ¿Cómo se reproduce todo y qué tan reproducible es?

**Respuesta corta.** Con el entorno fijado, la semilla única, las tareas registradas y el orden de ejecución de
la sección A.5. Al repetir las búsquedas en procesos nuevos, XGBoost elige exactamente lo mismo; el genético de
Random Forest puede cambiar por diferencias de redondeo numérico, sin cambiar ninguna conclusión (sección A.6).
```

## A.1 Entorno fijado

Python 3.14.6 en Windows 11 (Intel Core i5-8265U, 4 núcleos y 8 hilos, sin GPU NVIDIA). Versiones exactas en
`requirements.txt`:

```{literalinclude} ../../requirements.txt
```

## A.2 Semilla única

`SEED = 42` en `estrato3/configuracion.py`; `fijar_semillas()` fija `random`, `numpy` y `PYTHONHASHSEED`, y todos
los componentes reciben `random_state=SEED` explícito: submuestra, K-Means de kNNDM, SMOTE/ADASYN, modelos,
Random Search, TPE de Optuna y el genético de DEAP. Los pliegues (externos e internos) se calculan una sola vez y
se guardan en `resultados/datos_clasificacion.pkl`, así que todas las combinaciones usan exactamente los mismos.

## A.3 Registro estructurado de experimentos

Cada tarea (modelo, balanceo, optimizador, pliegue externo) se guarda en `resultados/tareas/<hash>.json` con:
semilla, presupuesto, hiperparámetros finales, F1 interno del óptimo, métricas del bucle externo, número de
evaluaciones, tiempos (búsqueda, ajuste final e inferencia) y el **historial completo** de la búsqueda
(configuración, puntaje y tiempo de cada evaluación); las probabilidades fuera de muestra van en un `.npz`. La
tabla maestra (`tabla_maestra_clasificacion.csv` y `tabla_maestra_regresion.csv`) las reúne. Como cada tarea se
guarda al terminar, el experimento se puede interrumpir y retomar sin repetir trabajo.

## A.4 Modularidad

| Módulo | Responsabilidad |
|---|---|
| `estrato3/datos.py` | Carga, variables, submuestra y pliegues |
| `estrato3/modelos.py`, `estrato3/modelos_propios.py` | Modelos, espacios de búsqueda y pipelines |
| `estrato3/optimizacion.py` | Función objetivo y los cuatro optimizadores |
| `estrato3/evaluacion.py` | Métricas |
| `estrato3/estadistica.py`, `estrato3/estadistica_regresion.py` | Pruebas estadísticas |
| `estrato3/experimento.py` | Una tarea del pipeline combinatorio |
| `experimentos/*.py` | Ejecutores y análisis |

Todas las funciones reutilizadas tienen docstrings estilo NumPy (parámetros, retorno y supuestos).

## A.5 Orden de ejecución

```bash
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

## A.6 Verificación de reproducibilidad

La semilla única se comprobó repitiendo, en procesos nuevos, la búsqueda final de los dos finalistas con
aleatoriedad en el modelo, el balanceo y la búsqueda (`experimentos/verificar_reproducibilidad.py`,
`resultados/reproducibilidad.json`):

| Finalista | Optimizador | ¿Mismos hiperparámetros? | F1 de test (corrida del libro → repetición) |
|---|---|---|---|
| XGBoost (SMOTE) | Random | sí | 0.5665 → 0.5670 |
| Random Forest (SMOTE) | Genética | no (profundidad 21 y hojas de 2 → profundidad 15 y hojas de 3) | 0.557 → 0.565 |

La semilla fija todo lo que es aleatorio, pero no las diferencias de redondeo de las bibliotecas numéricas. La
causa más probable es BLAS, que reparte las operaciones de matrices entre varios hilos según la carga del
computador y así cambia el orden de las sumas en la última cifra decimal. En SMOTE, eso basta para que cambie un
vecino empatado, y entonces cambian algunas viviendas sintéticas. En una búsqueda que compara valores de F1 muy
parecidos, como el algoritmo genético de Random Forest, eso puede cambiar la configuración elegida. La diferencia
(0.008) es menor que la variación de Random Forest entre semillas (±0.013, sección 5.6), así que no cambia ninguna
conclusión. Para obtener resultados idénticos bit a bit hay que fijar además `OMP_NUM_THREADS=1` y
`OPENBLAS_NUM_THREADS=1` antes de correr.
