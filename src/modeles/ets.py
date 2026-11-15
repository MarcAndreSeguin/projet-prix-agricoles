"""ETS additif sur les logarithmes, compatible avec l'interface commune.

Le module ne lit aucun fichier. Le choix de configuration est fourni par
l'appelant après sélection sur la validation. Le retour reste en logarithmes.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from src import config

CONFIGURATIONS = {
    "sans_tendance": dict(tendance=None, saisonnalite=None, amortie=False),
    "tendance": dict(tendance="add", saisonnalite=None, amortie=False),
    "tendance_amortie": dict(tendance="add", saisonnalite=None, amortie=True),
    "saisonnalite": dict(tendance=None, saisonnalite="add", amortie=False),
    "tendance_saisonnalite": dict(tendance="add", saisonnalite="add", amortie=False),
    "tendance_amortie_saisonnalite": dict(tendance="add", saisonnalite="add", amortie=True),
}
COLONNES = ["pas", "prevision", "inf80", "sup80", "inf95", "sup95"]


def ajuster_log(historique, configuration):
    """Estime une configuration ETS sur l'historique déjà logarithmique."""
    if not isinstance(historique, pd.Series) or not isinstance(historique.index, pd.DatetimeIndex):
        raise ValueError("historique doit être une Series avec un DatetimeIndex.")
    index = historique.index
    if historique.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Historique vide, non trié ou contenant des doublons.")
    attendu = pd.date_range(index[0], periods=len(index), freq="MS")
    if not index.equals(attendu):
        raise ValueError("Les dates doivent être mensuelles, consécutives et au premier du mois.")
    valeurs = historique.astype(float).asfreq("MS")
    if not np.isfinite(valeurs.to_numpy()).all():
        raise ValueError("L'historique contient des valeurs manquantes ou non finies.")
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Configuration ETS inconnue : {configuration}")
    options = CONFIGURATIONS[configuration]
    minimum = 24 if options["saisonnalite"] else 3
    if len(valeurs) < minimum:
        raise ValueError(f"Au moins {minimum} observations sont nécessaires.")

    # Les valeurs sont déjà des logarithmes : ne pas appliquer np.log ici.
    modele = ETSModel(
        valeurs, error="add", trend=options["tendance"],
        damped_trend=options["amortie"], seasonal=options["saisonnalite"],
        seasonal_periods=12 if options["saisonnalite"] else None,
        initialization_method="estimated",
    )
    resultat = modele.fit(disp=False, maxiter=2000)
    if not resultat.mle_retvals.get("converged", False):
        raise RuntimeError(f"Échec de convergence ETS : {configuration}")
    return resultat


def prevoir(historique, pas, configuration="sans_tendance"):
    """Retourne les prévisions et intervalles en log, dans l'ordre des pas.

    configuration est fournie par l'évaluateur. La valeur par défaut désigne
    un ETS(A,N,N) de référence et ne réalise aucune sélection automatique.
    """
    pas = list(pas)
    if not pas or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or p < 1 for p in pas):
        raise ValueError("pas doit contenir des entiers strictement positifs.")
    resultat = ajuster_log(historique, configuration)
    prediction = resultat.get_prediction(
        start=resultat.nobs, end=resultat.nobs + max(pas) - 1, method="exact",
    )
    i80 = prediction.summary_frame(alpha=1 - config.NIVEAU_AFFICHAGE)
    i95 = prediction.summary_frame(alpha=1 - config.NIVEAU_ALERTE)
    toutes = pd.DataFrame({
        "pas": np.arange(1, max(pas) + 1),
        "prevision": np.asarray(i80["mean"]),
        "inf80": np.asarray(i80["pi_lower"]), "sup80": np.asarray(i80["pi_upper"]),
        "inf95": np.asarray(i95["pi_lower"]), "sup95": np.asarray(i95["pi_upper"]),
    })
    sortie = toutes.iloc[np.asarray(pas) - 1].reset_index(drop=True)
    if not np.isfinite(sortie.to_numpy()).all():
        raise ValueError("Prévisions ETS non finies.")
    if not ((sortie.inf95 <= sortie.inf80) & (sortie.inf80 <= sortie.prevision)
            & (sortie.prevision <= sortie.sup80) & (sortie.sup80 <= sortie.sup95)).all():
        raise ValueError("Intervalles ETS incohérents.")
    return sortie[COLONNES]
