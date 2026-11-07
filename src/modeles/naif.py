"""Modèle naïf (référence) : marche aléatoire appliquée au log des prix.

Interface commune des modèles (docs/protocole.md) :
    prevoir(historique, pas) -> DataFrame (pas, prevision, inf80, sup80, inf95, sup95), en log.

Notations
---------
- y_t = log(prix_t), pour t = 1, ..., T ; T est le dernier mois de l'historique
  fourni (l'origine de la prévision).
- d_t = y_t - y_(t-1), pour t = 2, ..., T : variations mensuelles du log
  (environ la variation relative du prix d'un mois à l'autre).
- k : nombre de pas après la dernière donnée (3, 5 ou 8 selon config.PAS_PAR_HORIZON).

Modèle
------
Marche aléatoire : y_t = y_(t-1) + e_t, où les chocs e_t sont indépendants et
suivent une loi normale N(0, sigma²).

Prévision ponctuelle à k pas : la dernière valeur observée,
    ŷ_(T+k) = y_T.

Erreur de prévision : y_(T+k) - y_T = e_(T+1) + ... + e_(T+k), somme de k chocs
indépendants, donc de variance k · sigma² et d'écart-type sigma · racine(k).
L'incertitude croît comme la racine du nombre de pas.

Estimation de sigma : écart-type empirique des variations mensuelles de
l'historique fourni (n = T - 1 variations, d̄ leur moyenne),
    s = racine( somme des (d_t - d̄)² / (n - 1) ).
C'est l'écart-type usuel, centré sur d̄ ; le modèle sans dérive suppose une
moyenne nulle des chocs, mais la prévision reste y_T (aucune dérive n'est ajoutée).

Intervalle de prévision de niveau c (config.NIVEAUX_INTERVALLE : 80 % et 95 %) :
    y_T ± z_c · s · racine(k),   avec z_c = Φ⁻¹(0,5 + c / 2),
où Φ⁻¹ est le quantile de la loi normale centrée réduite
(z ≈ 1,2816 pour 80 %, z ≈ 1,9600 pour 95 %).

Retour en dollars (fait par le script d'évaluation, pas ici) : la prévision est
le dernier prix observé P_T, et l'intervalle devient
[P_T · exp(-z_c · s · racine(k)), P_T · exp(z_c · s · racine(k))], asymétrique
autour de P_T.

Absence de fuite d'information
------------------------------
La fonction ne reçoit que l'historique jusqu'à l'origine et ne lit aucun
fichier : la prévision et s ne dépendent d'aucune donnée postérieure à la
dernière date de l'historique. s est recalculé à chaque appel, donc à chaque
origine (fenêtre croissante depuis le début de la série).
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import config

# Colonnes du résultat : pas, prevision, puis inf/sup pour chaque niveau (inf80, sup80, inf95, sup95)
COLONNES = ["pas", "prevision"] + [
    f"{borne}{int(round(niveau * 100))}"
    for niveau in config.NIVEAUX_INTERVALLE
    for borne in ("inf", "sup")
]


def verifier_historique(historique):
    """Vérifie que l'historique respecte l'interface commune et le retourne en float.

    Exige un DatetimeIndex mensuel (premier du mois), croissant, sans trou ni
    doublon, sans valeur manquante, et au moins 3 valeurs (2 variations pour
    l'écart-type).
    """
    if not isinstance(historique, pd.Series) or not isinstance(historique.index, pd.DatetimeIndex):
        raise TypeError("historique doit être une pd.Series indexée par un DatetimeIndex")
    if len(historique) < 3:
        raise ValueError(f"historique trop court : {len(historique)} valeur(s), au moins 3 requises")
    if historique.isna().any():
        raise ValueError("historique contient des valeurs manquantes")
    attendues = pd.date_range(historique.index[0], periods=len(historique), freq="MS")
    if not historique.index.equals(attendues):
        raise ValueError("l'index de historique doit être mensuel (premier du mois), "
                         "croissant, sans trou ni doublon")
    return historique.astype(float)


def verifier_pas(pas):
    """Vérifie que pas est une liste non vide d'entiers ≥ 1 et la retourne."""
    pas = list(pas)
    if not pas or any(int(k) != k or k < 1 for k in pas):
        raise ValueError(f"pas doit être une liste non vide d'entiers ≥ 1 (reçu : {pas})")
    return [int(k) for k in pas]


def ecart_type_variations(historique):
    """Écart-type empirique (n - 1 au dénominateur) des variations mensuelles du log."""
    return float(historique.diff().dropna().std(ddof=1))


def prevoir(historique, pas):
    """Prévision naïve (marche aléatoire) du log des prix, avec intervalles.

    historique : log des prix d'une série, index mensuel, connu jusqu'à l'origine.
    pas : nombres de pas après la dernière donnée (ex. [3, 5, 8]).
    Retourne un DataFrame, une ligne par pas dans l'ordre reçu, colonnes
    pas, prevision, inf80, sup80, inf95, sup95 (en log) :
      prevision = y_T ;  bornes = y_T ± z_c · s · racine(k).
    """
    y = verifier_historique(historique)
    pas = verifier_pas(pas)

    derniere = y.iloc[-1]
    s = ecart_type_variations(y)
    racine_k = np.sqrt(np.asarray(pas, dtype=float))

    resultat = {"pas": pas, "prevision": np.full(len(pas), derniere)}
    for niveau in config.NIVEAUX_INTERVALLE:
        z = stats.norm.ppf(0.5 + niveau / 2)
        demi_largeur = z * s * racine_k
        pct = int(round(niveau * 100))
        resultat[f"inf{pct}"] = derniere - demi_largeur
        resultat[f"sup{pct}"] = derniere + demi_largeur
    return pd.DataFrame(resultat, columns=COLONNES)
