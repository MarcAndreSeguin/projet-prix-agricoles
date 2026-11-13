"""Paires origine-cible, métriques (MAE, RMSE, MAPE), couverture des intervalles et
choix du modèle final.

Conventions :
- mape est exprimée en pourcentage ; couverture80 et couverture95 en proportion (0 à 1) ;
- chaque prévision appartient à une seule période de previsions.csv
  (config.PERIODES_PREVISIONS : validation, test_1, test_2, test_3) ;
- les métriques sont calculées séparément pour chaque période, plus test_global,
  calculé sur l'ensemble des prévisions de test_1, test_2 et test_3
  (config.COMPOSITION_PERIODES) ; validation et test ne sont jamais mélangés ;
- la règle du modèle final s'applique sur config.PERIODE_SELECTION (validation).
"""

import numpy as np
import pandas as pd

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


def bornes_periodes():
    """Mois cibles (début, fin) de chaque période de previsions.csv.

    La validation est découpée en blocs (config.PERIODES_VALIDATION) ; chaque
    période de test est un seul intervalle (config.PERIODES_TEST).
    Retourne une liste de (periode, debut, fin).
    """
    bornes = [(config.PERIODE_VALIDATION, debut, fin) for debut, fin in config.PERIODES_VALIDATION]
    bornes += [(periode, debut, fin) for periode, (debut, fin) in config.PERIODES_TEST.items()]
    return bornes


def periode_de(date_cible):
    """Période de previsions.csv d'un mois cible (validation, test_1, ...), ou None."""
    date_cible = pd.Timestamp(date_cible)
    for periode, debut, fin in bornes_periodes():
        if pd.Timestamp(debut) <= date_cible <= pd.Timestamp(fin):
            return periode
    return None


def generer_paires_evaluation(horizons=config.HORIZONS):
    """Paires origine-cible de toutes les périodes, avec la colonne periode.

    Chaque couple (date_cible, horizon) n'apparaît qu'une fois : lève ValueError
    si un mois cible appartient à deux périodes.
    """
    paires = pd.concat(
        [generer_paires_origine_cible(debut, fin, horizons).assign(periode=periode)
         for periode, debut, fin in bornes_periodes()],
        ignore_index=True,
    )
    if paires.duplicated(["date_cible", "horizon"]).any():
        raise ValueError("Un même mois cible appartient à deux périodes : vérifier le découpage")
    return paires


def numero_periode_validation(date_cible):
    """Numéro (1 à len(PERIODES_VALIDATION)) du bloc de validation d'une date cible.

    Retourne None si la date n'appartient à aucun bloc (par exemple le test).
    Sert à rapporter les erreurs bloc par bloc, donc par régime de prix.
    """
    date_cible = pd.Timestamp(date_cible)
    for numero, (debut, fin) in enumerate(config.PERIODES_VALIDATION, start=1):
        if pd.Timestamp(debut) <= date_cible <= pd.Timestamp(fin):
            return numero
    return None


def origines_a_estimer(debut, fin, horizons=config.HORIZONS):
    """Origines distinctes d'une période : une estimation du modèle par origine."""
    return sorted(generer_paires_origine_cible(debut, fin, horizons)["date_origine"].unique())


def _exiger_periode(previsions):
    """Lève ValueError si la colonne periode est absente ou contient une valeur inconnue."""
    if "periode" not in previsions:
        raise ValueError(f"previsions doit contenir la colonne periode ({config.PERIODES_PREVISIONS})")
    inconnues = set(previsions["periode"].dropna()) - set(config.PERIODES_PREVISIONS)
    if inconnues or previsions["periode"].isna().any():
        raise ValueError(f"Valeurs de periode non conformes : {sorted(inconnues) or 'manquantes'}")


def filtrer_periode(tableau, periode):
    """Lignes de previsions.csv couvertes par une période de metriques.csv.

    test_global regroupe test_1, test_2 et test_3 ; les autres périodes se
    couvrent elles-mêmes.
    """
    return tableau[tableau["periode"].isin(config.COMPOSITION_PERIODES[periode])]


def calculer_metriques(previsions):
    """Calcule les métriques par modèle, série, période et horizon (format de metriques.csv).

    Périodes : celles de previsions.csv, plus test_global (union des trois
    périodes de test).
    """
    _exiger_periode(previsions)
    p = previsions.dropna(subset=["reel"]).copy()
    p["erreur"] = p["reel"] - p["prevision"]
    p["dans80"] = (p["reel"] >= p["inf80"]) & (p["reel"] <= p["sup80"])
    p["dans95"] = (p["reel"] >= p["inf95"]) & (p["reel"] <= p["sup95"])
    p = pd.concat(
        [filtrer_periode(p, periode).assign(periode=periode) for periode in config.PERIODES_METRIQUES],
        ignore_index=True,
    )

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
        p.groupby(["modele", "serie", "periode", "horizon"])[["erreur", "reel", "dans80", "dans95"]]
        .apply(resumer)
        .reset_index()
    )
    metriques["n"] = metriques["n"].astype(int)
    return metriques[config.COLONNES_METRIQUES]


def mae_moyenne(metriques, serie, periode=config.PERIODE_SELECTION):
    """MAE moyenne sur les horizons, par modèle, pour une série et une période."""
    choisies = (metriques["serie"] == serie) & (metriques["periode"] == periode)
    return metriques.loc[choisies].groupby("modele")["mae"].mean()


def choisir_modele_final(metriques, serie, periode=config.PERIODE_SELECTION):
    """Applique la règle de sélection du protocole à une série, sur une seule période.

    Modèle final : plus faible MAE moyenne sur les horizons, calculée sur la
    période de validation (config.PERIODE_SELECTION) uniquement ; le test ne
    sert qu'à rapporter les performances. Le naïf est retenu s'il est le
    meilleur ; en cas d'égalité, le naïf est préféré, puis l'ordre de config.MODELES.
    Retourne None si aucune métrique n'est disponible pour cette série et cette période.
    """
    moyennes = mae_moyenne(metriques, serie, periode)
    if moyennes.empty:
        return None
    ordre = [config.MODELE_REFERENCE] + [m for m in config.MODELES if m != config.MODELE_REFERENCE]
    rang = {modele: i for i, modele in enumerate(ordre)}
    candidats = sorted(moyennes.index, key=lambda m: (moyennes[m], rang.get(m, len(ordre))))
    return candidats[0]
