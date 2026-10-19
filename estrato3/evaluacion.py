# -*- coding: utf-8 -*-
"""Métricas de clasificación, ordinales y de calibración.

Autores: María Carolina Cantillo Orozco (200179105) y Juan Camilo Oñoro Araujo (200177329)

La métrica principal de validación es el **F1 macro**: promedia el F1 de los
seis estratos sin ponderar por su tamaño, así que el estrato 6 (menos del 5 %
del entrenamiento) pesa lo mismo que el 1. Con un desbalance de 6.2 a 1 la
accuracy premiaría predecir solo los estratos grandes. Se reportan también las
métricas ordinales (MAE, accuracy ±1, kappa cuadrático) y las de probabilidad
(AUC, log loss, Brier, ECE).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

from . import configuracion as CF


def proba_completa(est, X, clases=CF.CLASES):
    """Probabilidades en las seis columnas de los estratos aunque el modelo no haya visto alguno."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), len(clases)))
    for j, k in enumerate(est.classes_):
        out[:, int(np.where(clases == k)[0][0])] = p[:, j]
    return out


def ece(y, proba, clases=CF.CLASES, n_bins=10):
    """Error de calibración esperado (top-label): Σ_b (n_b/n)·|precisión_b − confianza_b|.

    Parámetros
    ----------
    y : array (n,)
        Estrato real.
    proba : array (n, K)
        Probabilidades.
    n_bins : int
        Número de intervalos de confianza de igual ancho.
    """
    conf = proba.max(1)
    acierto = clases[proba.argmax(1)] == np.asarray(y)
    bordes = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    return float(total)


def metricas(y, proba, clases=CF.CLASES):
    """Tablero de métricas a partir de las probabilidades (la clase predicha es el argmax).

    Devuelve
    --------
    dict
        f1_macro, accuracy, balanced_accuracy, precision_macro, recall_macro, auc_ovr, log_loss, brier, ece,
        mae_ordinal, acc_pm1 y kappa_cuadratico.
    """
    y = np.asarray(y)
    pred = clases[proba.argmax(1)]
    Y = (y[:, None] == clases[None, :]).astype(float)
    p = np.clip(proba, 1e-12, 1)
    p = p / p.sum(1, keepdims=True)
    presentes = np.isin(clases, np.unique(y))
    try:
        auc = roc_auc_score(y, p[:, presentes] / p[:, presentes].sum(1, keepdims=True), multi_class="ovr",
                            average="macro", labels=clases[presentes])
    except ValueError:
        auc = np.nan
    return {"f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
            "auc_ovr": auc, "log_loss": log_loss(y, p, labels=clases), "brier": float(np.mean(((p - Y) ** 2).sum(1))),
            "ece": ece(y, p, clases), "mae_ordinal": float(np.mean(np.abs(y - pred))),
            "acc_pm1": float(np.mean(np.abs(y - pred) <= 1)),
            "kappa_cuadratico": cohen_kappa_score(y, pred, weights="quadratic")}


def metricas_clase(y, pred):
    """F1 macro de una predicción de clase (p. ej. la regresión redondeada al estrato más cercano)."""
    return float(f1_score(np.asarray(y), np.asarray(pred), average="macro", labels=CF.CLASES, zero_division=0))
