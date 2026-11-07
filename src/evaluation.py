"""Métriques (MAE, RMSE, MAPE), couverture des intervalles et test de Diebold-Mariano.

Conventions :
- mape est exprimée en pourcentage ; couverture80 et couverture95 en proportion (0 à 1) ;
- le test de Diebold-Mariano utilise la perte absolue (cohérente avec le critère
  de sélection fondé sur la MAE), une variance de long terme avec h - 1 retards,
  où h est le nombre de pas entre l'origine et la cible, et la correction de
  Harvey, Leybourne et Newbold (1997) avec une loi de Student à n - 1 degrés de liberté ;
- stat_dm < 0 signifie que modele_a a une perte moyenne plus faible que modele_b ;
- meilleur vaut le modèle à plus faible perte si p_valeur < config.SEUIL_DM, sinon « aucun ».
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def generer_paires_origine_cible(debut, fin, horizons=config.HORIZONS):
    """Liste les prévisions à produire pour une période (validation ou test).

    Origine glissante : chaque mois de debut à fin (inclus) est un mois cible,
    prévu à chaque horizon depuis l'origine date_cible - PAS_PAR_HORIZON[horizon].
    Le modèle est estimé sur les données jusqu'à date_origine incluse, puis
    réestimé à l'origine suivante (config.REESTIMATION_A_CHAQUE_ORIGINE).

    Retourne un DataFrame trié (date_origine, date_cible, horizon, pas).
    Exemple : cible 2023-01, horizon 6 (8 pas) -> origine 2022-05.
    """
    cibles = pd.date_range(debut, fin, freq="MS")
    lignes = [
        {
            "date_origine": cible - pd.DateOffset(months=config.PAS_PAR_HORIZON[h]),
            "date_cible": cible,
            "horizon": h,
            "pas": config.PAS_PAR_HORIZON[h],
        }
        for cible in cibles
        for h in horizons
    ]
    paires = pd.DataFrame(lignes, columns=["date_origine", "date_cible", "horizon", "pas"])
    return paires.sort_values(["date_origine", "date_cible"]).reset_index(drop=True)


def origines_a_estimer(debut, fin, horizons=config.HORIZONS):
    """Origines distinctes d'une période : une estimation du modèle par origine."""
    return sorted(generer_paires_origine_cible(debut, fin, horizons)["date_origine"].unique())


def calculer_metriques(previsions):
    """Calcule les métriques par modèle, série et horizon (format de metriques.csv)."""
    p = previsions.dropna(subset=["reel"]).copy()
    p["erreur"] = p["reel"] - p["prevision"]
    p["dans80"] = (p["reel"] >= p["inf80"]) & (p["reel"] <= p["sup80"])
    p["dans95"] = (p["reel"] >= p["inf95"]) & (p["reel"] <= p["sup95"])

    def resumer(g):
        e = g["erreur"]
        return pd.Series({
            "mae": e.abs().mean(),
            "rmse": np.sqrt((e ** 2).mean()),
            "mape": (e.abs() / g["reel"].abs()).mean() * 100,
            "couverture80": g["dans80"].mean(),
            "couverture95": g["dans95"].mean(),
            "n": len(g),
        })

    metriques = (
        p.groupby(["modele", "serie", "horizon"])[["erreur", "reel", "dans80", "dans95"]]
        .apply(resumer)
        .reset_index()
    )
    metriques["n"] = metriques["n"].astype(int)
    return metriques[config.COLONNES_METRIQUES]


def test_diebold_mariano(erreurs_a, erreurs_b, pas):
    """Test de Diebold-Mariano (perte absolue, correction HLN).

    erreurs_a, erreurs_b : erreurs alignées sur les mêmes origines.
    pas : nombre de pas entre l'origine et la cible.
    Retourne (stat_dm, p_valeur) ; (nan, nan) si le test n'est pas calculable.
    """
    d = np.abs(np.asarray(erreurs_a, float)) - np.abs(np.asarray(erreurs_b, float))
    n = len(d)
    if n < 2 * pas + 1:
        return np.nan, np.nan
    d_centre = d - d.mean()
    gamma = [np.dot(d_centre[k:], d_centre[: n - k]) / n for k in range(pas)]
    variance = gamma[0] + 2 * sum(gamma[1:])
    if variance <= 0:
        variance = gamma[0]
    if variance <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * pas + pas * (pas - 1) / n) / n)
    stat *= correction
    p_valeur = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p_valeur)


def choisir_modele_final(metriques, tests_dm, serie):
    """Applique la règle de sélection du protocole à une série.

    Modèle final : plus faible MAE moyenne sur les 3 horizons, à condition de
    battre le naïf (Diebold-Mariano, 5 %). Interprétation retenue :
    - les candidats sont classés par MAE moyenne croissante ;
    - un candidat est retenu si sa MAE moyenne est inférieure à celle du naïf et
      si le test DM contre le naïf le désigne meilleur sur au moins un horizon ;
    - si aucun candidat ne remplit ces conditions, le modèle final est le naïf.
    Retourne None si les métriques de la série ou du naïf sont absentes.
    """
    reference = config.MODELE_REFERENCE
    mae_moyenne = metriques.loc[metriques["serie"] == serie].groupby("modele")["mae"].mean()
    if reference not in mae_moyenne:
        return None
    tests = tests_dm.loc[tests_dm["serie"] == serie]
    for modele in mae_moyenne.sort_values().index:
        if modele == reference or mae_moyenne[modele] >= mae_moyenne[reference]:
            continue
        contre_naif = tests[
            ((tests["modele_a"] == modele) & (tests["modele_b"] == reference))
            | ((tests["modele_a"] == reference) & (tests["modele_b"] == modele))
        ]
        if (contre_naif["meilleur"] == modele).any():
            return modele
    return reference


def calculer_tests_dm(previsions, modeles=config.MODELES):
    """Compare chaque paire de modèles, par série et horizon (format de tests_dm.csv)."""
    p = previsions.dropna(subset=["reel"]).copy()
    p["erreur"] = p["reel"] - p["prevision"]
    lignes = []
    for (serie, horizon), g in p.groupby(["serie", "horizon"]):
        erreurs = g.pivot(index="date_origine", columns="modele", values="erreur")
        pas = config.PAS_PAR_HORIZON[horizon]
        for modele_a, modele_b in combinations(modeles, 2):
            if modele_a not in erreurs or modele_b not in erreurs:
                continue
            paire = erreurs[[modele_a, modele_b]].dropna()
            stat, p_valeur = test_diebold_mariano(paire[modele_a], paire[modele_b], pas)
            if np.isnan(p_valeur) or p_valeur >= config.SEUIL_DM:
                meilleur = "aucun"
            else:
                meilleur = modele_a if stat < 0 else modele_b
            lignes.append({
                "serie": serie, "horizon": horizon, "modele_a": modele_a,
                "modele_b": modele_b, "stat_dm": stat, "p_valeur": p_valeur,
                "meilleur": meilleur,
            })
    return pd.DataFrame(lignes, columns=config.COLONNES_TESTS_DM)
