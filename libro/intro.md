# Estrato socioeconómico y catastro en Barranquilla — Entregable 3

**Optimización de hiperparámetros, balanceo de clases, validación anidada y comparación estadística**

**Machine Learning — Maestría, Universidad del Norte · Profesor Lihki Rubio**

**Autores:** María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329) · Octubre de 2026

```{admonition} En una página
:class: tip
**Pregunta del proyecto.** ¿Se puede predecir el estrato socioeconómico (1 a 6) de una vivienda de Barranquilla con
sus datos catastrales abiertos y su ubicación?

**Objetivo del proyecto.** Usar un modelo que no se había aplicado a este problema, el **XGBoost geográfico adaptado
a clasificación** (la versión original de Grekousis, 2025, es de regresión), y comprobar si supera al **modelo
base**, la regresión logística multinomial del Entregable 1.

**Qué hace este entregable.** Somete ese modelo y los 7 modelos de la guía al mismo protocolo: **140 combinaciones**
de modelo, balanceo de clases y optimizador de hiperparámetros (más 16 del modelo nuevo), cada una con validación
cruzada anidada, y las compara con pruebas estadísticas jerárquicas, análisis de costo computacional,
calibración e interpretabilidad.

**Respuesta.** El XGBoost geográfico **supera al modelo base en zonas nuevas** (en el test, +0.094 de F1 macro, IC 95 %
[+0.042, +0.147]), pero esa mejora la explica sobre todo la **vecindad física** (cómo son los 8 edificios más
cercanos), que también hace mejorar a la logística, a XGBoost y a Random Forest. Frente a modelos con la misma
información, el modelo nuevo no gana en zonas nuevas; su ventaja aparece en zonas con datos cerca (capítulos 9 y 10).

**Después del entregable (capítulo 11).** Con todos los datos y pruebas corregidas por comparaciones múltiples, la
mejora sobre el modelo base es **estadísticamente significativa** (efecto mediano a grande) para todos los modelos con
vecindad física, incluidos el XGBoost geográfico y el **Rotation Forest**, otro modelo original de la revisión
bibliográfica. Ningún modelo original supera significativamente a los convencionales que reciben la misma información.

**Otros hallazgos.** (1) El modelo pesa mucho más que el balanceo o el optimizador. (2) Con el mismo presupuesto, los
cuatro optimizadores llegan al mismo resultado; la bayesiana llega antes. (3) `class_weight` es el mejor balanceo y
ADASYN el peor. (4) Lo que decide el estrato es la ubicación y el entorno de la vivienda, no la vivienda misma
(SHAP).
```

```{admonition} ¿Por qué aparecen modelos conocidos si el proyecto busca un modelo no usado antes?
:class: important
**La regla del proyecto.** La revisión bibliográfica del Entregable 2
([capítulo 1](https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/notebooks/01_revision.html)) buscó en
Scopus quién había clasificado el nivel socioeconómico con aprendizaje supervisado y datos catastrales o
geoespaciales. **No encontró ningún estudio que clasifique el estrato de una vivienda con sus datos catastrales.** Los
trabajos más cercanos hacen otra cosa: predicen el *precio* de la vivienda (con XGBoost, Random Forest o SVR) o usan
una regresión logística con datos agregados por sector censal (1998). Por eso el proyecto propone un modelo que no se
había usado para esto: el **XGBoost geográfico**, que solo existía para regresión y que adaptamos a clasificación.

**El papel de cada modelo en este entregable.** Los modelos conocidos no son la propuesta: son la **competencia** contra
la que se pone a prueba.

| Modelo | Papel en el proyecto | ¿Se había usado para clasificar el estrato de una vivienda con el catastro? |
|---|---|---|
| **XGBoost geográfico adaptado a clasificación** | **La propuesta** (el modelo nuevo) | No. La adaptación a clasificación es nuestra y no hay aplicaciones al estrato. |
| Logística B del Entregable 1 | **El modelo base**, la línea que la propuesta debe superar | No con datos de viviendas; solo hay un antecedente de 1998 con datos agregados. |
| KNN, Naive Bayes, logística L1/L2, árbol de decisión, Random Forest, XGBoost y SVM | **Modelos de referencia (*benchmarks*)** que exige la guía de este entregable (sección 7.2) | No para este problema; algunos sí en problemas vecinos, como el precio de la vivienda. |

**Por qué hacen falta los modelos de referencia.**

1. **Para que la comparación sea justa.** Decir que un modelo nuevo aporta algo exige medirlo contra los métodos
   conocidos con el mismo protocolo: las mismas variables, los mismos pliegues, el mismo presupuesto de búsqueda y las
   mismas pruebas estadísticas. Eso es lo que hace el pipeline combinatorio.
2. **Para saber de dónde viene la mejora.** El XGBoost de la guía es el mismo algoritmo que la parte global del modelo
   nuevo, pero sin los modelos locales, así que compararlos funciona como un **experimento de ablación**: muestra cuánto aporta la parte geográfica (los
   modelos locales alrededor de las anclas). Y la logística con las mismas variables de vecindad muestra cuánto aporta
   esa información por sí sola.

El aporte del proyecto sigue siendo el mismo del Entregable 2: modelos no usados antes en esta aplicación, puestos a
prueba con más rigor. Lo que se encontró sobre el XGBoost geográfico, incluidas sus limitaciones, está en el capítulo 9;
los demás candidatos originales de la revisión (Rotation Forest, stacking y RF + SVM) están en el capítulo 11 de este
libro y en el capítulo 9 del Entregable 2 (versión 3).
```

## Cómo está organizado

El libro sigue el orden de un experimento: cómo se diseñó, cómo se optimizó, qué resultó y qué responde.

| Parte | Capítulo | Qué responde | Sección de la guía |
|---|---|---|---|
| **I. Datos y diseño** | 1. Datos | ¿De dónde salen los datos y qué se corrigió desde la primera entrega? | 7.1 |
| | 2. Diseño experimental | ¿Cómo se entrenan y evalúan 156 combinaciones sin hacer trampa y en un portátil? | 2, 3.2, 7.3 |
| **II. Optimización** | 3. Optimización de hiperparámetros | ¿Qué optimizador encuentra mejores hiperparámetros con el mismo presupuesto? | 3 |
| | 4. Optimización computacional | ¿Cuánto cuesta cada modelo, coincide con la teoría y qué versión conviene? | 4 |
| **III. Resultados** | 5. Clasificación | ¿Qué combinación clasifica mejor, en validación y en el test? ¿Están calibradas sus probabilidades? | 5.1, 5.4 |
| | 6. Regresión | ¿Ayuda tratar el estrato como un número? | 5.2 |
| | 7. Comparación estadística | ¿Cuáles diferencias son reales y cuáles son ruido? | 6 |
| | 8. Interpretabilidad | ¿Qué variables deciden el estrato, en general y en una vivienda concreta? | 5.3 |
| | **9. El modelo nuevo frente al modelo base** | **La respuesta a la pregunta del proyecto** | Objetivo del proyecto |
| **Cierre** | 10. Conclusiones | Respuesta, lecciones, limitaciones y próximos pasos | 8, 9 |
| **Después del entregable** | 11. Todos los datos y otros modelos de la revisión | ¿Mejora el modelo nuevo con todos los datos? ¿Funcionan otros modelos originales de la revisión? ¿Son significativas las diferencias? | Próximos pasos (10.6) |
| **Apéndice** | A. Reproducibilidad | Entorno, semilla, registro de experimentos, módulos, orden de ejecución y verificación | 7.3 |

## Cómo leer este libro

- **Cada capítulo empieza con su pregunta y su respuesta corta** en un recuadro, y cada tabla o figura va seguida de
  una **lectura** que dice qué se ve y por qué importa.
- **El código está oculto** para no interrumpir la lectura; se despliega con el botón de cada celda. Las celdas solo
  leen resultados ya calculados (`resultados/`); los experimentos se corren con los scripts de `experimentos/`
  (apéndice A).
- **Relación con los entregables anteriores.** La limpieza, el EDA completo, la partición en bloques y el modelo base
  están en el Entregable 1 (https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/); la revisión
  bibliográfica, el modelo nuevo y la vecindad física, en el Entregable 2
  (https://mcarocantillo.github.io/entregable2-ml-barranquilla/v3/). Este entregable usa la misma partición y la
  misma semilla.

```{admonition} Cinco términos que se usan en todo el libro
:class: note
- **Zonas nuevas / zonas conocidas.** Predecir el estrato en un barrio sin ninguna vivienda de entrenamiento cerca
  (como el test, 7 bloques de 2 km), o completar el mapa donde sí hay datos cerca.
- **Validación anidada.** Un bucle interno elige los hiperparámetros; un bucle externo, que nunca los vio, mide el
  desempeño. Así la cifra reportada no está inflada por la búsqueda.
- **kNNDM.** La forma de armar los pliegues de validación para que la distancia entre validación y entrenamiento sea
  la misma que entre el test y el entrenamiento (zonas nuevas).
- **Vecindad física.** Siete variables que describen a los 8 edificios más cercanos: área, baños, habitaciones,
  antigüedad, piso, proporción en propiedad horizontal y distancia media. No usan el estrato de los vecinos.
- **F1 macro.** El promedio del F1 de los seis estratos, sin ponderar por su tamaño: el estrato 6 (1 % del test)
  pesa lo mismo que el estrato 1 (35 %). Es la métrica principal del proyecto (sección 5.1).
```
