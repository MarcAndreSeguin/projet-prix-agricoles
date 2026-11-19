"""Modèles ETS sur prix bruts ou logarithmes.

prevoir_prix : interface en dollars pour l'évaluation révisée.
prevoir et ajuster_log : interface historique conservée pour les anciens scripts.
Le module n'effectue aucune sélection et ne lit aucun fichier.
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


VARIANTES = {
    "log_additif": dict(error="add", trend="add", seasonal="add", log=True),
    "brut_additif": dict(error="add", trend="add", seasonal="add", log=False),
    "brut_multiplicatif": dict(error="mul", trend="mul", seasonal="mul", log=False),
}


def prevoir_prix(historique, pas, configuration="saisonnalite", variante="log_additif"):
    """Entrée et sortie en dollars ; l'ancienne fonction prevoir reste en log.

    Intervalles analytiques pour les modèles additifs ; 10 000 simulations
    gaussiennes reproductibles pour les modèles multiplicatifs. Les paramètres
    sont considérés comme estimés : leur incertitude n'est pas intégrée.
    """
    import hashlib
    if variante not in VARIANTES or configuration not in CONFIGURATIONS:
        raise ValueError("Variante ou configuration ETS inconnue.")
    if not isinstance(historique, pd.Series) or not isinstance(historique.index, pd.DatetimeIndex):
        raise ValueError("Historique mensuel requis.")
    y = historique.astype(float).asfreq("MS")
    if historique.empty or historique.index.has_duplicates or not historique.index.equals(y.index):
        raise ValueError("Dates mensuelles consécutives et triées requises.")
    if not np.isfinite(y.to_numpy()).all() or (y <= 0).any():
        raise ValueError("Prix strictement positifs et finis requis.")
    pas = list(pas)
    if not pas or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or p < 1 for p in pas):
        raise ValueError("Les pas doivent être des entiers strictement positifs.")
    v, c = VARIANTES[variante], CONFIGURATIONS[configuration]
    if len(y) < (24 if c["saisonnalite"] else 3):
        raise ValueError("Historique trop court.")
    modele = ETSModel(
        np.log(y) if v["log"] else y, error=v["error"],
        trend=v["trend"] if c["tendance"] else None,
        damped_trend=c["amortie"],
        seasonal=v["seasonal"] if c["saisonnalite"] else None,
        seasonal_periods=12 if c["saisonnalite"] else None,
        initialization_method="estimated",
    )
    r = modele.fit(disp=False, maxiter=2000)
    if not r.mle_retvals.get("converged", False):
        raise RuntimeError(f"Non convergence : {variante}/{configuration}")
    options = dict(start=r.nobs, end=r.nobs + max(pas) - 1)
    if v["error"] == "add":
        prediction = r.get_prediction(**options, method="exact")
    else:
        # Une graine par historique : les appels répétés donnent les mêmes bornes.
        seed = int.from_bytes(hashlib.sha256(y.to_numpy().tobytes()).digest()[:4], "little")
        prediction = r.get_prediction(**options, method="simulated",
                                      simulate_repetitions=10000, random_state=seed)
    i80 = prediction.summary_frame(alpha=1-config.NIVEAU_AFFICHAGE)
    i95 = prediction.summary_frame(alpha=1-config.NIVEAU_ALERTE)
    t = pd.DataFrame(dict(pas=np.arange(1, max(pas)+1),
        prevision=np.asarray(r.forecast(max(pas))),
        inf80=np.asarray(i80.pi_lower), sup80=np.asarray(i80.pi_upper),
        inf95=np.asarray(i95.pi_lower), sup95=np.asarray(i95.pi_upper)))
    if v["log"]:
        t[COLONNES[1:]] = np.exp(t[COLONNES[1:]])
    t = t.iloc[np.asarray(pas)-1].reset_index(drop=True)
    if not np.isfinite(t.to_numpy()).all():
        raise ValueError("Prévisions ou intervalles non finis.")
    if not ((t.inf95 <= t.inf80) & (t.inf80 <= t.sup80) & (t.sup80 <= t.sup95)).all():
        raise ValueError("Intervalles non emboîtés.")
    return t[COLONNES]
