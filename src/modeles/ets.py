"""Modèle ETS sur le logarithme des prix agricoles."""

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def ajuster_ets(prix, tendance=None, saisonnalite=None, amortie=False):
    """Estime un modèle ETS sur une série mensuelle d'entraînement.

    prix : série pandas indexée par date, contenant des prix positifs.
    tendance : None ou "add".
    saisonnalite : None ou "add".
    amortie : amortit la tendance si True.
    """
    prix = prix.sort_index().astype(float)

    if not isinstance(prix.index, pd.DatetimeIndex):
        raise ValueError("Les prix doivent être indexés par date.")

    if prix.index.has_duplicates:
        raise ValueError("La série contient des dates en double.")

    # Rend explicites les éventuels mois absents.
    prix = prix.asfreq("MS")

    if prix.empty or not np.isfinite(prix.to_numpy()).all():
        raise ValueError("La série est vide ou contient des prix manquants ou invalides.")

    if (prix <= 0).any():
        raise ValueError("Les prix doivent être positifs pour calculer le logarithme.")

    if amortie and tendance is None:
        raise ValueError("L'amortissement nécessite une tendance.")

    modele = ETSModel(
        np.log(prix),
        error="add",
        trend=tendance,
        damped_trend=amortie,
        seasonal=saisonnalite,
        seasonal_periods=12 if saisonnalite is not None else None,
        initialization_method="estimated",
    )

    return modele.fit(disp=False)


def prevoir_ets(resultat, pas=8):
    """Produit les prévisions médianes et les intervalles en dollars."""
    debut = resultat.nobs
    fin = debut + pas - 1

    prediction = resultat.get_prediction(
        start=debut,
        end=fin,
        method="exact",
    )

    intervalle80 = prediction.summary_frame(alpha=0.20)
    intervalle95 = prediction.summary_frame(alpha=0.05)

    return pd.DataFrame({
        "prevision": np.exp(intervalle80["mean"]),
        "inf80": np.exp(intervalle80["pi_lower"]),
        "sup80": np.exp(intervalle80["pi_upper"]),
        "inf95": np.exp(intervalle95["pi_lower"]),
        "sup95": np.exp(intervalle95["pi_upper"]),
    })