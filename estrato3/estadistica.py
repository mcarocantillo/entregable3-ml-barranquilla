# -*- coding: utf-8 -*-
"""Comparación estadística de modelos (sección 6 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Secuencia jerárquica para clasificación:

1. **Friedman** (prueba ómnibus) sobre los rangos de cada combinación en cada pliegue externo.
2. **Nemenyi** (post-hoc con control del error familiar) y **diagrama de diferencia crítica**.
3. **DeLong** solo entre un subconjunto reducido (los finalistas), por estrato (uno contra el resto, porque la
   prueba de DeLong compara AUC binarias), con corrección de **Holm-Bonferroni**.
4. **Tamaño del efecto**: delta de Cliff sobre los F1 por pliegue.
5. **Intervalos de confianza** por bootstrap BCa remuestreando bloques espaciales (las viviendas de un bloque no
   son independientes).
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Friedman y Nemenyi
# ---------------------------------------------------------------------------
def friedman(tabla):
    """Prueba de Friedman.

    Parámetros
    ----------
    tabla : pandas.DataFrame
        Filas = pliegues (bloques de la prueba), columnas = modelos, valores = métrica (más alto es mejor).

    Devuelve
    --------
    dict
        Estadístico χ²_F, grados de libertad, p-valor y rangos promedio (1 = mejor).
    """
    est, p = stats.friedmanchisquare(*[tabla[c].to_numpy() for c in tabla.columns])
    rangos = tabla.rank(axis=1, ascending=False).mean().sort_values()
    return {"chi2": float(est), "gl": tabla.shape[1] - 1, "p": float(p), "rangos": rangos}


def diferencia_critica(k, n, alpha=0.05):
    """Diferencia crítica de Nemenyi: CD = q_α · sqrt(k(k+1) / (6n)) (Demšar, 2006).

    q_α es el cuantil del rango studentizado con k grupos e infinitos grados de libertad, dividido por √2.

    Parámetros
    ----------
    k : int
        Número de modelos comparados.
    n : int
        Número de bloques de la prueba de Friedman (pliegues o bloques espaciales).
    alpha : float
        Nivel de significancia familiar.

    Devuelve
    --------
    float
        Diferencia mínima de rango promedio para declarar distintos a dos modelos.
    """
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def nemenyi(tabla):
    """Matriz de p-valores de Nemenyi (scikit-posthocs) sobre la misma tabla de Friedman.

    Parámetros
    ----------
    tabla : pandas.DataFrame
        Filas = bloques, columnas = modelos, valores = métrica (más alto es mejor).

    Devuelve
    --------
    pandas.DataFrame
        Matriz simétrica (modelos × modelos) de p-valores ajustados por el error familiar.

    Supuestos
    ---------
    Solo tiene sentido si Friedman rechazó la hipótesis nula (secuencia jerárquica de la sección 6 de la guía).
    """
    import scikit_posthocs as sp
    return sp.posthoc_nemenyi_friedman(tabla.to_numpy()).set_axis(tabla.columns, axis=0).set_axis(tabla.columns, axis=1)


def diagrama_cd(rangos, cd, titulo="Diagrama de diferencia crítica (Nemenyi)", ax=None, max_modelos=None):
    """Diagrama de diferencia crítica: modelos por rango promedio; las barras unen los indistinguibles.

    Parámetros
    ----------
    rangos : pandas.Series
        Rango promedio por modelo (1 = mejor), ordenado de menor a mayor.
    cd : float
        Diferencia crítica.
    titulo : str
        Título del gráfico.
    ax : matplotlib.axes.Axes, opcional
        Ejes donde dibujar; si no se dan, se crea una figura.
    max_modelos : int, opcional
        Dibuja solo los ``max_modelos`` mejores (útil con 128 combinaciones).

    Devuelve
    --------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    r = rangos.sort_values()
    if max_modelos:
        r = r.iloc[:max_modelos]
    k = len(r)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.35 * k + 2))
    lo, hi = np.floor(r.min()), np.ceil(r.max())
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-k - 1, 2)
    ax.hlines(0, lo, hi, color="black")
    for v in np.arange(lo, hi + 1):
        ax.vlines(v, 0, 0.2, color="black")
        ax.text(v, 0.4, f"{v:.0f}", ha="center", fontsize=9)
    for i, (nom, v) in enumerate(r.items()):
        y = -(i + 1)
        ax.plot([v, v], [0, y], color="#888", lw=0.8)
        ax.text(hi + 0.6 if i >= k / 2 else lo - 0.6, y, f"{nom} ({v:.1f})", va="center",
                ha="left" if i >= k / 2 else "right", fontsize=8)
        ax.plot([v, hi + 0.5 if i >= k / 2 else lo - 0.5], [y, y], color="#888", lw=0.8)
    # barras de grupos indistinguibles (|rango_i − rango_j| < CD)
    vals = r.to_numpy()
    nivel = 0.9
    vistos = set()
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and (i, j) not in vistos and not any(a <= i and j <= b for a, b in vistos):
            ax.hlines(nivel, vals[i], vals[j], color="#e34948", lw=3)
            vistos.add((i, j))
            nivel += 0.25
    ax.hlines(1.6, lo, lo + cd, color="black", lw=2)
    ax.text(lo + cd / 2, 1.75, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.axis("off")
    ax.set_title(titulo)
    return ax


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _componentes_delong(pos, neg):
    """Componentes estructurales V10 y V01 de DeLong para una AUC (comparación con empates = 0.5)."""
    m, n = len(pos), len(neg)
    todos = np.concatenate([pos, neg])
    r = stats.rankdata(todos)
    r_pos = stats.rankdata(pos)
    r_neg = stats.rankdata(neg)
    v10 = (r[:m] - r_pos) / n
    v01 = 1 - (r[m:] - r_neg) / m
    return v10, v01


def delong(y_bin, s1, s2):
    """Prueba de DeLong para dos AUC correlacionadas (mismas observaciones).

    Parámetros
    ----------
    y_bin : array (n,) de 0/1
        Clase positiva (un estrato contra el resto).
    s1, s2 : array (n,)
        Puntajes de los dos modelos para la clase positiva.

    Devuelve
    --------
    dict
        AUC de cada modelo, diferencia, error estándar, z y p-valor bilateral.
    """
    y_bin = np.asarray(y_bin).astype(bool)
    v = []
    aucs = []
    for s in (s1, s2):
        v10, v01 = _componentes_delong(np.asarray(s)[y_bin], np.asarray(s)[~y_bin])
        v.append((v10, v01))
        aucs.append(v10.mean())
    s10 = np.cov(np.vstack([v[0][0], v[1][0]]))
    s01 = np.cov(np.vstack([v[0][1], v[1][1]]))
    S = s10 / y_bin.sum() + s01 / (~y_bin).sum()
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    dif = aucs[0] - aucs[1]
    z = dif / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diferencia": float(dif), "se": float(np.sqrt(max(var, 0))),
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def holm(pvals):
    """p-valores ajustados por Holm-Bonferroni (mismo orden de entrada).

    Parámetros
    ----------
    pvals : sequence of float
        p-valores crudos de una familia de pruebas.

    Devuelve
    --------
    numpy.ndarray
        p-valores ajustados (controlan el error familiar), acotados a 1 y monótonos en el orden de los crudos.
    """
    p = np.asarray(pvals, float)
    orden = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, (m - r) * p[i])
        aj[i] = min(1.0, acum)
    return aj


def benjamini_hochberg(pvals):
    """p-valores ajustados por Benjamini-Hochberg (tasa de falsos descubrimientos).

    Parámetros
    ----------
    pvals : sequence of float
        p-valores crudos de una familia de pruebas.

    Devuelve
    --------
    numpy.ndarray
        p-valores ajustados en el orden de entrada.

    Supuestos
    ---------
    Pruebas independientes o con dependencia positiva (las AUC por estrato de los mismos modelos lo son).
    """
    p = np.asarray(pvals, float)
    m = len(p)
    orden = np.argsort(p)[::-1]
    aj = np.empty(m)
    acum = 1.0
    for r, i in enumerate(orden):
        acum = min(acum, p[i] * m / (m - r))
        aj[i] = acum
    return aj


# ---------------------------------------------------------------------------
# Tamaño del efecto y bootstrap
# ---------------------------------------------------------------------------
def delta_cliff(a, b):
    """Delta de Cliff: P(a > b) − P(a < b). |δ| < 0.147 despreciable, < 0.33 pequeño, < 0.474 mediano, mayor grande.

    Parámetros
    ----------
    a, b : array
        Métricas de dos modelos (aquí, el F1 macro por bloque espacial).

    Devuelve
    --------
    (float, str)
        δ en [−1, 1] (positivo = ``a`` tiende a ser mayor) y su magnitud según Romano et al. (2006).

    Supuestos
    ---------
    No paramétrico: no supone normalidad; compara todos los pares (a_i, b_j), sin emparejar.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mayor = (a[:, None] > b[None, :]).sum()
    menor = (a[:, None] < b[None, :]).sum()
    d = (mayor - menor) / (len(a) * len(b))
    mag = abs(d)
    etiqueta = "despreciable" if mag < 0.147 else "pequeño" if mag < 0.33 else "mediano" if mag < 0.474 else "grande"
    return float(d), etiqueta


def bootstrap_bca_bloques(estadistico, bloques, n_resamples=999, seed=42, confianza=0.95):
    """IC BCa remuestreando BLOQUES espaciales completos.

    Parámetros
    ----------
    estadistico : callable
        f(indices_de_filas) -> float, calculado sobre las filas de los bloques remuestreados.
    bloques : array (n,)
        Bloque de cada fila.
    n_resamples : int
        Réplicas bootstrap.
    seed : int
        Semilla del remuestreo.
    confianza : float
        Nivel del intervalo.

    Supuestos
    ---------
    Las filas de un mismo bloque espacial están correlacionadas y los bloques son aproximadamente independientes;
    por eso se remuestrean bloques completos. Con pocos bloques (7 en el test) el intervalo es ancho y asimétrico.

    Devuelve
    --------
    (float, float)
        Límites del intervalo.
    """
    ub, inv = np.unique(np.asarray(bloques), return_inverse=True)
    filas = [np.where(inv == g)[0] for g in range(len(ub))]

    def f(idx_bloques, axis=-1):
        """Estadístico sobre las filas de los bloques remuestreados (vectorizado para scipy)."""
        a = np.asarray(idx_bloques)
        if a.ndim == 1:
            return estadistico(np.concatenate([filas[int(g)] for g in a]))
        return np.array([estadistico(np.concatenate([filas[int(g)] for g in fila])) for fila in a])

    r = stats.bootstrap((np.arange(len(ub)),), f, vectorized=True, paired=False, n_resamples=n_resamples,
                        confidence_level=confianza, method="BCa", random_state=seed)
    return float(r.confidence_interval.low), float(r.confidence_interval.high)
