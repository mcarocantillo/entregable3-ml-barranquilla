# 10. Conclusiones

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

```{admonition} Qué responde este capítulo
:class: tip
**Pregunta.** ¿Qué responde el proyecto, qué se aprendió del pipeline combinatorio y qué queda pendiente?

**Respuesta corta.** El modelo nuevo supera al modelo base en zonas nuevas, pero su ventaja frente a modelos con
la misma información solo aparece en zonas con datos cerca. Del pipeline: el modelo pesa más que el balanceo y
el optimizador; la validación y el test no siempre coinciden; y la complejidad teórica guía bien el costo.
```

## 10.1 ¿El modelo nuevo supera al modelo base?

El objetivo del proyecto era usar un modelo que no se había aplicado a este problema, el **XGBoost geográfico
adaptado a clasificación**, y comprobar si supera al **modelo base** (la logística B del Entregable 1).

**Respuesta: sí lo supera en zonas nuevas, pero no siempre, y la mejora viene sobre todo de la vecindad física.**

| Evidencia | XGBoost geográfico − modelo base | Lectura |
|---|---|---|
| E2, todos los datos, validación kNNDM | +0.11 de F1 (SE 0.03, 5 de 5 pliegues) | lo reemplaza |
| E2, todos los datos, test | +0.09, IC [+0.03, +0.20] | se confirma |
| E3, submuestra, validación anidada | +0.20 (SE 0.03, 5 de 5 pliegues) | lo reemplaza |
| E3, submuestra, test | +0.094, IC [+0.042, +0.147], 6 de 7 bloques | se confirma |
| E3, todos los datos con hiperparámetros de la submuestra, test | +0.021, IC [−0.079, +0.082] | no se distingue |

Frente a modelos que reciben **las mismas variables de vecindad**, el modelo nuevo no gana en zonas nuevas: en el
test empata o pierde con la logística con vecindad, XGBoost y Random Forest, y con todos los datos pierde
claramente frente a los dos últimos. Su ventaja sobre la mejor logística con las mismas variables solo se demostró
en **zonas conocidas** (E2: +0.022, 4 de 5 pliegues), donde los modelos locales tienen datos cerca para aprender.

**¿Sigue siendo un aporte nuevo?** Sí. La revisión bibliográfica del Entregable 2 no encontró ningún estudio que
clasifique el estrato de una vivienda con datos catastrales, y el XGBoost geográfico no se había usado en
clasificación ni para el estrato. Los modelos conocidos que aparecen en este entregable son los modelos de
referencia que exige la guía: no reemplazan a la propuesta, la ponen a prueba. Gracias a ellos se sabe algo que el
Entregable 2 no podía separar: cuánto de la mejora viene del modelo y cuánto de la información de vecindad.

Para el proyecto, esto significa que el aporte que más mejora la predicción del estrato en una zona sin datos no
es el modelo sino **describir el entorno** de cada vivienda: el área, los baños, el piso y la proporción de
propiedad horizontal de sus 8 vecinos, más su posición norte-sur y su distancia al centro (capítulo 8). El XGBoost
geográfico es una buena opción para **completar el mapa en zonas ya muestreadas**, no para extrapolar a barrios
nuevos.

## 10.2 Qué enseñó el pipeline combinatorio

1. **El modelo importa mucho más que el balanceo y el optimizador.** Entre modelos, el mejor F1 macro externo va de
   0.452 (Naive Bayes) a 0.658 (Random Forest); dentro de un mismo modelo, las 16 combinaciones de balanceo y
   optimizador se separan apenas 0.02 a 0.05 (0.08 en el árbol). Los tres ensambles de árboles (Random Forest,
   XGBoost y XGBoost geográfico) forman el grupo de cabeza en validación, sin diferencias demostrables entre ellos.
2. **Balanceo:** `class_weight` es la mejor opción y la más barata (rango medio 2.25 de 4; 136 s por búsqueda frente a
   259 s de SMOTE). SMOTE y no balancear quedan casi empatados, y **ADASYN es la peor** (Friedman p = 2·10⁻⁶): al
   concentrar los ejemplos sintéticos en las viviendas «difíciles», que aquí son las que están en la frontera
   espacial entre estratos, refuerza el ruido. Todas las técnicas de balanceo empeoran algo la calibración.
3. **Optimizadores:** con el mismo presupuesto (12 evaluaciones), Grid, Random, bayesiana (TPE) y genética llegan
   al mismo F1 externo (0.586 a 0.590; Friedman p = 0.74). La diferencia está en la eficiencia: la bayesiana y Random
   alcanzan el 99 % de su mejor valor en 5 o 6 evaluaciones y Grid necesita 10. El genético converge
   prematuramente con una población de 4. Successive Halving da el mismo F1 que Random Search en XGBoost con un
   cuarto del cómputo, pero pierde en Random Forest, cuyos hiperparámetros dependen del tamaño de la muestra.
4. **Regresión:** tratar el estrato como número no ayuda a clasificarlo. El mejor regresor (Random Forest, RMSE 0.57
   estratos) comprime los extremos: solo el 19 % de las viviendas de estrato 6 se redondea a 6. Sus residuos son
   heterocedásticos y están autocorrelacionados en el espacio (I de Moran 0.35), otra señal de que queda
   estructura espacial por explicar.

## 10.3 Validación frente a test

La validación anidada con kNNDM y el test no cuentan la misma historia, y esa diferencia es en sí un resultado:

- **Todo baja en el test.** Los ensambles pasan de un F1 de ≈ 0.66 en el bucle externo a 0.52–0.57 en el test.
  kNNDM reproduce la *distancia* entre el test y el entrenamiento, pero no su *composición*: los 7 bloques de test
  tienen muy pocas viviendas de estrato 6 (1.2 %, frente al 4.8 % de la submuestra), y un recall bajo en esa clase
  pesa un sexto del F1 macro.
- **El orden cambia.** En validación ganan los árboles; en el test, la logística con vecindad empata con XGBoost y
  Random Forest (0.572 frente a 0.571 y 0.571 de media entre semillas) porque reconoce mucho mejor los estratos 1
  y 6.
- **El test tiene 7 bloques.** Sus intervalos BCa son anchos y la mayoría de las diferencias entre los mejores
  modelos no son concluyentes. Las conclusiones firmes son las que aparecen en las dos miradas.

## 10.4 Cómputo

- **La complejidad teórica predice bien el escalamiento:** el SVM de núcleo completo crece casi
  cuadráticamente (b = 1.88), Nyström y SGD casi linealmente, y XGBoost `hist` es 9 veces más rápido que `exact`
  con el mismo F1.
- **No toda «optimización» de la guía conviene en estos datos:** con 35 variables densas, lbfgs es 7 veces más
  rápido que SAGA y la búsqueda exhaustiva de KNN es 17 veces más rápida que KD-Tree; los árboles de búsqueda solo
  ayudan con las 2 coordenadas.
- **El perfil manda:** el 91 % del tiempo de una evaluación es construir árboles; el preprocesamiento y SMOTE no
  llegan al 5 %.
- **Paralelizar entre tareas grandes:** las 640 tareas (38.4 horas de tarea) terminaron en 4.8 horas con 8
  procesos; paralelizar pliegues de un segundo no acelera nada.

## 10.5 Limitaciones

- **Submuestra.** Para que las 156 combinaciones con validación anidada cupieran en un portátil (38 horas de
  cómputo en tareas, frente a unas 630 con todos los datos), se entrenó con 10 001 de las 257 273 viviendas.
  Reentrenar los finalistas con todos los datos (sección 9.4) mueve el F1 de test entre +0.006 y +0.025 en los
  modelos cuyos hiperparámetros no dependen de n, y los tres mejores siguen empatados; pero los hiperparámetros
  elegidos con 10 000 viviendas no se trasladan a los modelos que cuentan viviendas (el árbol, KNN) ni al XGBoost
  geográfico.
- **Presupuesto de búsqueda corto** (12 evaluaciones): suficiente para llegar a la meseta en estos espacios, pero
  no para que el genético o la multi-fidelidad muestren todo su potencial.
- **Test de 7 bloques.** Los intervalos del test son anchos y asimétricos; con tan pocos bloques, diferencias de
  0.01 a 0.05 entre los mejores modelos no son concluyentes.
- **Tiempos.** Los tiempos de búsqueda se midieron con 8 procesos simultáneos, y los de la regresión con dos
  copias de la corrida en paralelo por un error de ejecución: solo son comparables dentro de cada tabla.

## 10.6 Trabajo futuro

- Repetir la **búsqueda** del XGBoost geográfico con todos los datos (o con más viviendas por ancla): con los
  hiperparámetros de la submuestra pierde frente a XGBoost global, mientras que con los del Entregable 2
  (α = 0.75, profundidad 4) superaba al modelo base.
- Recalibrar con validación cruzada como paso final del pipeline: en el test mejoró la calibración y, en varios
  modelos, también el F1.
- Para la multi-fidelidad, expresar los frenos de los árboles como fracciones de n o usar Hyperband con una
  fidelidad mínima más alta.
- Un ensamble apilado de la logística con vecindad y los árboles: ordenan mejor estratos distintos (la logística
  los altos, los árboles los bajos y medios; capítulo 7).
