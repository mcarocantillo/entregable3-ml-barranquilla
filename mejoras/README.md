# Mejoras posteriores al Entregable 3

Experimentos pedidos después de publicar el Entregable 3. No forman parte del libro publicado ni se han subido a
GitHub.

## Preguntas

1. ¿Mejora el **XGBoost geográfico** si sus hiperparámetros se buscan con **todos los datos**? En el E2 quedaron
   fijos (α = 0.75, 15 % de vecinos por ancla, profundidad 4) y en el E3 se buscaron con 10 001 viviendas (α = 0.97).
2. ¿Funcionan mejor otros **modelos originales de la revisión bibliográfica** (Tabla III de
   `Revision_Bibliografica_ML_Estrato_IEEE.docx`)? Se prueban tres:
   - **Stacking** (candidato 8): metamodelo sobre la logística con vecindad, XGBoost, Random Forest y el geográfico;
   - **Rotation Forest** (candidato 5): PCA por grupos de variables antes de cada árbol;
   - **RF + SVM en dos etapas** (candidato 4): Random Forest elige las variables y una SVM clasifica.
   Bagged TAO Trees (candidato 3) no tiene una implementación abierta confiable en Python; GWO-XGBoost (2) y TPE-SVM (7)
   son optimizadores (el E3 mostró que los cuatro optimizadores empatan; TPE-SVM ya está en el E3: SVM + bayesiana).

## Protocolo

- Las 257 273 viviendas de entrenamiento y las variables del E3 (modelo B + vecindad física KNN, k = 8).
- Los mismos 5 pliegues kNNDM de zonas nuevas del E2 v3 (`datos/knndm_e2v3.pkl`, q = 30, semilla 42). El modelo
  base reproduce su F1 del E2 v3 (0.460): es la verificación de que los pliegues son los mismos.
- Hiperparámetros elegidos por la media del F1 macro en esos pliegues; el test de 7 bloques se mira una sola vez, al
  final, con los modelos ya elegidos; IC 95 % BCa remuestreando bloques.
- Criterio de reemplazo del proyecto: diferencia pareada por pliegue > 1 SE y confirmación en el test.

## Scripts (desde `e3/`, en este orden)

```
set PYTHONPATH=%CD%;%CD%\mejoras
..\venv_e2\Scripts\python -u mejoras\01_geo_busqueda.py   # 6 globales × 9 locales × 21 α (75 ajustes)
..\venv_e2\Scripts\python -u mejoras\02_revision.py       # base, logística con vecindad, RF, Rotation Forest, RF + SVM
..\venv_e2\Scripts\python -u mejoras\03_test.py           # stacking, comparaciones y test
..\venv_e2\Scripts\python -u mejoras\04_estadistica.py    # Friedman, Nemenyi, Wilcoxon, t corregida, permutación, Holm
```

`N_TRABAJOS` (3) procesos en paralelo, `HILOS` (2) hilos por XGBoost; `RAPIDO=1` es una prueba de humo (pocas filas,
carpetas `chk_rapido/` y `resultados_rapido/`). Las probabilidades de cada pliegue quedan en `chk/` (se reanuda solo).

## Resultados

`resultados/geo_busqueda.json`, `resultados/revision_cv.json`, `resultados/mejoras.json`,
`resultados/estadistica.json` y `resultados/figuras/cd_mejoras.png`; resumen en `resultados/RESUMEN.md`; registros en
`ml_trabajo/log_mejoras.txt` y `ml_trabajo/log_mejoras_estadistica.txt`. El capítulo 11 del libro los presenta.
