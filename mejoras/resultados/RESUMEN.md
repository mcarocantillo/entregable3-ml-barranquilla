# Resumen de las mejoras (19-oct-2026)

Protocolo: 257 273 viviendas; los 5 pliegues kNNDM de zonas nuevas del Entregable 2 v3 (el modelo base reproduce su
F1 de 0.460 y el XGBoost geográfico con la configuración del E2 su 0.572); elección por el F1 macro medio en la
validación; test de 7 bloques mirado una sola vez; IC 95 % BCa remuestreando bloques. Criterio del proyecto: diferencia
pareada > 1 SE en la validación y confirmación en el test. Cifras de `mejoras.json`.

| Modelo | Validación | vs base (SE, pliegues) | Test [IC 95 %] | vs base en test [IC 95 %] (bloques) | vs logística con vecindad en test |
|---|---|---|---|---|---|
| Random Forest (referencia de la guía) | 0.576 | +0.116 (0.039, 4/5) | 0.601 [0.559, 0.783] | +0.108 [+0.022, +0.170] (6/7) | +0.027 [−0.009, +0.073] |
| Stacking S1: logística, XGBoost y RF (candidato 8) | 0.598 | +0.138 (0.020, 5/5) | 0.596 [0.535, 0.761] | +0.102 [−0.006, +0.173] (5/7) | +0.021 [−0.032, +0.080] |
| **Rotation Forest (candidato 5)** | **0.598** | +0.138 (0.029, 5/5) | 0.593 [0.560, 0.801] | **+0.099 [+0.024, +0.170] (6/7)** | +0.018 [−0.019, +0.064] |
| XGBoost geográfico, configuración del E2 | 0.572 | +0.112 (0.029, 5/5) | 0.585 [0.530, 0.793] | +0.092 [−0.024, +0.172] (5/7) | +0.011 [−0.045, +0.074] |
| Logística con vecindad | 0.575 | +0.115 (0.017, 5/5) | 0.574 [0.549, 0.607] | +0.081 [+0.003, +0.091] (6/7) | — |
| Stacking S2: S1 + XGBoost geográfico | 0.583 | +0.123 (0.017, 5/5) | 0.568 [0.509, 0.764] | +0.075 [−0.053, +0.145] (5/7) | −0.007 [−0.061, +0.061] |
| XGBoost geográfico afinado (todos los datos) | 0.581 | +0.121 (0.032, 5/5) | 0.566 [0.509, 0.805] | +0.073 [−0.119, +0.149] (5/7) | −0.008 [−0.111, +0.069] |
| XGBoost afinado (ablación, α = 1) | 0.575 | +0.115 (0.031, 5/5) | 0.565 [0.522, 0.778] | +0.072 [−0.015, +0.150] (5/7) | −0.009 [−0.046, +0.037] |
| RF + SVM en dos etapas (candidato 4) | 0.553 | +0.093 (0.030, 4/5) | 0.517 [0.470, 0.562] | +0.024 [−0.060, +0.088] (4/7) | −0.057 [−0.148, +0.013] |
| SVM sola (referencia del RF + SVM) | 0.550 | +0.090 (0.025, 5/5) | 0.493 [0.413, 0.548] | −0.001 [−0.192, +0.063] (3/7) | −0.082 [−0.187, −0.008] |
| Modelo base (logística B, sin vecindad) | 0.460 | — | 0.493 [0.466, 0.532] | — | −0.081 [−0.091, −0.003] |

## Lectura

1. **Hiperparámetros del XGBoost geográfico.** La búsqueda con todos los datos eligió un XGBoost global de profundidad 6
   sin pesos de clase, modelos locales en una rejilla de 1 km con el 5 % de los vecinos y α = 0.5. Mejora a la
   configuración del E2 en la validación (+0.009, SE 0.007, 4 de 5 pliegues), pero **no en el test** (0.566 frente a
   0.585). Frente al XGBoost con el mismo global, la parte geográfica aporta +0.005 en la validación (menos de 1 SE) y
   +0.001 [−0.027, +0.068] en el test: **no se distingue**. La configuración del E2 sigue siendo su mejor versión.
2. **Modelos originales de la revisión.** El **Rotation Forest** es el único que cumple el criterio del proyecto frente
   al modelo base en las dos miradas: +0.138 en la validación (SE 0.029, 5 de 5 pliegues) y **+0.099 en el test, IC
   [+0.024, +0.170], 6 de 7 bloques**. El stacking S1 tiene la misma validación, pero su intervalo de test toca el 0
   ([−0.006, +0.173]). Agregar el XGBoost geográfico al stacking (S2) lo empeora. El RF + SVM no se confirma en el
   test; la selección de variables del Random Forest sí mejora a la SVM sola (+0.025 en el test).
3. **Frente a la logística con las mismas variables de vecindad, ningún modelo se confirma en el test.** El Rotation
   Forest y el stacking S1 la superan en la validación por apenas 1 SE (+0.023), y en el test sus intervalos incluyen
   el 0. El Random Forest de la guía, que no es original, tiene el mejor F1 de test (0.601).
4. **Advertencia sobre los intervalos.** Con el método de este entregable (BCa por bloques), el +0.092 de la
   configuración del E2 frente al modelo base tiene IC [−0.024, +0.172]; el Entregable 2 publicó [+0.034, +0.199] con
   bootstrap de percentiles. La cifra puntual es la misma; con solo 7 bloques, la confirmación depende del método del
   intervalo.
5. **Comparaciones múltiples.** El test se miró una sola vez, pero se comparan 10 modelos contra el base: un intervalo
   que excluye el 0 por poco (Rotation Forest, +0.024) debe leerse con esa cautela.

## Significancia estadística (`04_estadistica.py`, `estadistica.json`)

Secuencia del capítulo 7, con Holm dentro de cada familia de comparaciones:

- **30 bloques espaciales de 2 km** (validación, predicciones fuera de muestra): Friedman χ² = 39.0, gl = 10,
  p = 2.6·10⁻⁵; Nemenyi CD = 2.76 rangos. **Frente al modelo base, Wilcoxon con Holm p < 0.01 para todos los modelos
  con vecindad física** (logística con vecindad, XGBoost, Random Forest, XGBoost geográfico del E2 y afinado, Rotation
  Forest, stacking S1 y S2), con δ de Cliff de 0.35 a 0.51 (mediano a grande); RF + SVM y SVM sola no (p 0.10).
  **Frente a la logística con vecindad, ninguna diferencia es significativa** (p de Holm ≥ 0.54, |δ| ≤ 0.24); el
  geográfico frente al XGBoost afinado, δ = +0.02.
- **5 pliegues** (t corregida de Nadeau y Bengio): ninguna diferencia significativa después de Holm (la menor, 0.078).
- **7 bloques de test** (permutación exacta): p crudo 0.031 para Rotation Forest, Random Forest y logística con
  vecindad frente al base; 0.31 con Holm. Con 7 bloques el p mínimo posible es 0.016: ninguna diferencia puede ser
  significativa después de corregir por 10 comparaciones.

**Conclusión.** La mejora sobre el modelo base es estadísticamente significativa y de efecto mediano a grande para
todos los modelos con vecindad física, incluidos el XGBoost geográfico y el Rotation Forest. Ningún modelo original
supera significativamente a los convencionales con la misma información: lo significativo es la vecindad física.
