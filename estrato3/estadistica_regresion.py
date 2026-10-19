# -*- coding: utf-8 -*-
"""Pruebas para la rama de regresión (secciones 5.2.2 y 6.3 de la guía).

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

Las pruebas de independencia de la guía (BDS, Ljung-Box, ACF) y las de comparación predictiva (Diebold-Mariano,
Giacomini-White, bootstrap estacionario) se diseñaron para series de tiempo. Aquí no hay tiempo: los datos son
una foto del catastro. Se aplican sobre los residuos ordenados en el ESPACIO (de sur a norte, la dirección del
gradiente del estrato), de modo que la «dependencia serial» se lee como dependencia entre viviendas vecinas en esa
dirección; se complementan con el I de Moran de los residuos, que es la prueba natural de dependencia espacial.
"""
import numpy as np
from scipy import stats


def dm_hln(e1, e2, h=1, potencia=2):
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997) para muestras finitas.

    d_t = |e1_t|^potencia − |e2_t|^potencia; DM = d̄ / sqrt(V̂(d̄)) con varianza de largo plazo (Newey-West con
    h − 1 rezagos); HLN multiplica por sqrt((n + 1 − 2h + h(h − 1)/n) / n) y usa la t de Student con n − 1 gl.
    Valor negativo = el modelo 1 tiene menor pérdida.
    """
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    n = len(d)
    dm = d.mean()
    gamma = [np.mean((d[k:] - dm) * (d[:n - k] - dm)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    est = dm / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = est * corr
    return {"diferencia_perdida": float(dm), "estadistico": float(t), "p": float(2 * stats.t.sf(abs(t), n - 1))}


def clark_west(y, pred_restringido, pred_general):
    """Clark-West (2007) para modelos ANIDADOS (el restringido es caso particular del general).

    f_t = (y − ŷ_r)² − [(y − ŷ_g)² − (ŷ_r − ŷ_g)²]; se prueba H0: E[f] = 0 contra E[f] > 0 (el general predice
    mejor) con una t de una cola. El término (ŷ_r − ŷ_g)² corrige el sesgo del DM a favor del modelo grande.
    """
    f = (y - pred_restringido) ** 2 - ((y - pred_general) ** 2 - (pred_restringido - pred_general) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return {"media_ajustada": float(f.mean()), "estadistico": float(t), "p_una_cola": float(stats.norm.sf(t))}


def giacomini_white(e1, e2, instrumentos=None):
    """Giacomini-White (2006), versión condicional: H0: E[h_t · d_t] = 0 con d_t = e1² − e2².

    ``instrumentos``: matriz (n, q) de variables que podrían predecir cuál modelo es mejor (aquí: constante y la
    diferencia de pérdida de la vivienda anterior en el orden espacial, análogo del «estado del ciclo»). Con solo
    la constante es la versión incondicional. Estadístico n·Z̄'Ω⁻¹Z̄ ~ χ²_q.
    """
    d = e1 ** 2 - e2 ** 2
    if instrumentos is None:
        instrumentos = np.c_[np.ones(len(d) - 1), d[:-1]]
        d = d[1:]
    Z = instrumentos * d[:, None]
    zb = Z.mean(0)
    omega = np.cov(Z, rowvar=False)
    omega = np.atleast_2d(omega)
    est = len(d) * zb @ np.linalg.solve(omega, zb)
    q = Z.shape[1]
    return {"estadistico": float(est), "gl": int(q), "p": float(stats.chi2.sf(est, q)),
            "signo": "modelo 1 mejor" if d.mean() < 0 else "modelo 2 mejor"}


def bootstrap_estacionario_ic(d, bloque_medio=50, n=999, seed=42):
    """IC 95 % de la media de una diferencia de pérdidas con bootstrap estacionario (Politis y Romano, 1994)."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(bloque_medio, np.asarray(d), seed=seed)
    ic = bs.conf_int(np.mean, reps=n, method="percentile")
    return float(ic[0, 0]), float(ic[1, 0])


def mcs(perdidas, alpha=0.10, bloque=50, reps=999, seed=42):
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con arch.

    Parámetros
    ----------
    perdidas : pandas.DataFrame
        Filas = observaciones (en orden espacial), columnas = modelos, valores = pérdida (p. ej. error²).

    Devuelve
    --------
    pandas.DataFrame
        p-valor MCS de cada modelo y si queda en el conjunto de confianza.
    """
    from arch.bootstrap import MCS
    m = MCS(perdidas, size=alpha, block_size=bloque, reps=reps, method="R", seed=seed)
    m.compute()
    pv = m.pvalues.copy()
    pv["en_el_conjunto"] = pv.index.isin(m.included)
    pv["perdida_media"] = perdidas.mean()[pv.index]
    pv["rango_perdida"] = perdidas.rank(axis=1).mean()[pv.index]
    return pv.sort_values("perdida_media")


def supuestos_residuos(residuos, X_white=None, lags=20):
    """White (homocedasticidad), BDS (independencia no lineal), Ljung-Box (autocorrelación) y normalidad.

    ``residuos`` debe venir ordenado en el espacio (sur → norte). ``X_white``: regresores para la prueba de White.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
    from statsmodels.tsa.stattools import acf, bds
    r = np.asarray(residuos, float)
    out = {}
    if X_white is not None:
        import statsmodels.api as sm
        lm, p_lm, f, p_f = het_white(r, sm.add_constant(X_white))
        out["white"] = {"lm": float(lm), "p": float(p_lm)}
    est, p = bds(r[:4000], max_dim=3)
    out["bds"] = {"estadistico_dim2": float(np.atleast_1d(est)[0]), "p_dim2": float(np.atleast_1d(p)[0])}
    lb = acorr_ljungbox(r, lags=[lags], return_df=True)
    out["ljung_box"] = {"lags": lags, "q": float(lb.lb_stat.iloc[0]), "p": float(lb.lb_pvalue.iloc[0])}
    out["acf"] = acf(r, nlags=lags).tolist()
    out["normalidad"] = {"asimetria": float(stats.skew(r)), "curtosis_exceso": float(stats.kurtosis(r)),
                         "p_dagostino": float(stats.normaltest(r).pvalue)}
    return out
