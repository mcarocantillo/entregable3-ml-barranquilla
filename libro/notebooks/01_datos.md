# 1. Datos: ETL y análisis exploratorio mejorado

*Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)*

```{admonition} Qué responde este capítulo
:class: tip
**Pregunta.** ¿De dónde salen los datos, cómo se prepararon y qué se corrigió desde la primera entrega?

**Respuesta corta.** Del catastro abierto de Barranquilla: 382 597 registros que, tras la limpieza y la
ubicación, quedan en 257 273 viviendas de entrenamiento y 65 145 de test (7 bloques de 2 km). Desde la primera
entrega se ubicaron por su manzana las viviendas sin coordenadas, se cambió la validación por kNNDM y se agregó
la vecindad física, que usan todos los modelos de este entregable.
```

El detalle completo está en el libro del Entregable 1 (versión 3); aquí se resume el flujo y, como pide la
sección 7.1 de la guía, las **correcciones y los hallazgos nuevos** que se incorporaron después de la primera
entrega.

## 1.1 ETL: de la descarga al universo de modelado

| Paso | Filas | Qué se hace |
|---|---|---|
| Extracción | 382 597 | Servicio abierto de catastro de la Alcaldía de Barranquilla (descarga del 15-sep-2026) |
| Transformación | 332 718 | Solo viviendas con estrato residencial válido (1–6); reglas por fila (áreas imposibles, años fuera de rango); variables derivadas (antigüedad, distancia al centro, identificador del predio matriz) |
| Ubicación | 322 418 (96.9 %) | 278 824 con coordenadas propias y 43 594 ubicadas por el centroide de su manzana (versión 2); 10 300 quedan sin ubicar |
| Carga | 257 273 + 65 145 | Partición por bloques espaciales de 2 km: entrenamiento y prueba (7 bloques). Se hace **antes** de cualquier paso que aprenda de los datos |

El preprocesamiento que aprende de los datos (winsorización p0.1/p99.9, imputación por mediana con indicador de
faltante, log1p de las variables asimétricas, estandarización y one-hot con categorías infrecuentes < 1 %) va
**dentro** del Pipeline, así que en cada pliegue se ajusta solo con su entrenamiento.

## 1.2 Correcciones y mejoras incorporadas

1. **Viviendas sin coordenadas (v2).** El 16 % de las viviendas, casi todas informales de estratos 1 y 2, no tenía
   coordenadas. Se ubicaron con el centroide de su manzana cuando la manzana es compacta (error de posición
   mediano de 31 m). La cobertura del estrato 1 pasó del 67 % al 92 %.
2. **Grupo del ICC.** Las mejoras en terreno ajeno se cuentan cada una como su propio grupo: ICC = 0.992, efecto de
   diseño ≈ 144, tamaño efectivo ≈ 2 300 viviendas independientes.
3. **Variables de pisos.** `planta_ubicacion` es el piso de la unidad, `total_plantas` los pisos de la propia
   vivienda (1 en casi todos los apartamentos) y `altura` es casi constante (3 m en el 99.7 % de las filas).
4. **Validación (v3).** El buffer fijo de 1 km evaluaba a 1.4–1.5 km del dato más cercano, más lejos que el
   escenario declarado (una zona nueva de 2 km con su entorno conocido: 0.45 km). Se reemplazó por la validación
   **kNNDM**, que arma los pliegues para reproducir esa distancia.

## 1.3 Hallazgos nuevos que usa este entregable

- **El estrato es una variable de zona.** I de Moran 0.92, alcance del correlograma 2.6 km y ICC 0.99: la
  ubicación domina y los errores son casi siempre de un estrato, como en la metodología del DANE (zonas
  homogéneas y viviendas atípicas).
- **Describir el entorno ayuda.** Con la validación correcta, la **vecindad física** (promedios de área, baños,
  habitaciones, antigüedad, piso y proporción de propiedad horizontal de los 8 edificios más cercanos, sin usar el
  estrato) sube el F1 de test de la logística de 0.49 a 0.57–0.60 (Entregable 2 v3, cap. 7). Todas las
  combinaciones de este entregable usan estas variables.
- **El entorno dice más que la vivienda misma.** En los seis atributos, el promedio de los 8 vecinos se relaciona con el
  estrato más que el atributo de la propia vivienda (η² de Kruskal-Wallis: baños 0.60 frente a 0.33; área 0.45
  frente a 0.17).
- **Las fronteras concentran la dificultad.** El 22 % de las viviendas tiene la mitad o más de sus 8 edificios
  vecinos en otro estrato (el estrato 5, el 44 %), y el 96 % de esas diferencias es de un solo estrato: por eso los
  errores son casi siempre de un estrato y ADASYN, que refuerza los casos «difíciles», es el peor balanceo (cap. 5).
- **Desbalance.** El estrato 1 es el 30 % del entrenamiento y el 6 menos del 5 % (6.2 a 1): por eso la métrica
  principal es el F1 macro y se comparan cuatro técnicas de balanceo.
- **Composición del test** (diagnóstico hecho después de evaluar, no usado para decidir). El test tiene 1.2 % de
  estrato 6 y sus 793 viviendas de ese estrato están en **un solo bloque**: el F1 del estrato 6 depende de un bloque,
  los intervalos de test son anchos y, con 7 bloques, ninguna diferencia pequeña puede ser significativa después de
  corregir por comparaciones múltiples (cap. 11).

```{admonition} El EDA mejorado completo
:class: note
Las correcciones de la primera entrega y estos hallazgos, con sus tablas y figuras, están en el **capítulo 15 del
Entregable 1** (versión 3): https://mcarocantillo.github.io/entregable1-ml-barranquilla/v3/notebooks/15_eda_mejorado.html.
Es la respuesta a la sección 7.1 de la guía («realizar todas las correcciones solicitadas en la primera entrega y
mejorar el análisis exploratorio con base en nuevos hallazgos»).
```

## 1.4 Variables del modelo

| Grupo | Variables |
|---|---|
| Físicas (numéricas) | área construida, área del terreno, habitaciones, baños, pisos de la vivienda, piso de ubicación, altura, antigüedad |
| Categóricas | régimen de propiedad (condición del predio), uso, tipo de vivienda |
| Ubicación | x, y (km) y distancia al centro |
| Vecindad física (grafo KNN, k = 8) | promedios de los 6 atributos en los 8 edificios más cercanos y la distancia media a ellos |
