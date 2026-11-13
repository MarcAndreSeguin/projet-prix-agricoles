"""Lecture et validation de data/series.csv et des fichiers de resultats/."""

import pandas as pd

from src import config


def charger_series(chemin=None):
    """Lit data/series.csv et retourne un DataFrame au format long du contrat.

    Colonnes : date (datetime64, premier du mois), serie, produit, province, prix,
    prix_impute.
    Les lignes sont triées par série puis par date.
    """
    chemin = chemin or config.FICHIER_SERIES
    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}. "
            "Exécuter d'abord : python scripts/preparer_donnees.py"
        )
    series = pd.read_csv(chemin, parse_dates=["date"], encoding="utf-8")
    colonnes_manquantes = set(config.COLONNES_SERIES) - set(series.columns)
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin.name} : {sorted(colonnes_manquantes)}")
    series = series[config.COLONNES_SERIES]
    return series.sort_values(["serie", "date"]).reset_index(drop=True)


# Colonnes attendues, colonnes de dates et valeurs de periode admises pour
# chaque fichier de resultats/
RESULTATS = {
    "previsions": (lambda: config.FICHIER_PREVISIONS, config.COLONNES_PREVISIONS,
                   ["date_origine", "date_cible"], config.PERIODES_PREVISIONS),
    "previsions_futures": (lambda: config.FICHIER_PREVISIONS_FUTURES,
                           config.COLONNES_PREVISIONS_FUTURES, ["date_origine", "date_cible"], None),
    "metriques": (lambda: config.FICHIER_METRIQUES, config.COLONNES_METRIQUES, [],
                  config.PERIODES_METRIQUES),
}


def chemin_resultat(nom):
    """Retourne le chemin du fichier de resultats/ désigné par nom (clé de RESULTATS)."""
    return RESULTATS[nom][0]()


def charger_resultat(nom):
    """Lit un fichier de resultats/ et vérifie que ses colonnes sont celles du contrat.

    nom : « previsions », « previsions_futures » ou « metriques ».
    Vérifie aussi que la colonne periode, si elle existe, ne contient que les
    valeurs admises pour ce fichier (config.PERIODES_PREVISIONS pour
    previsions.csv, config.PERIODES_METRIQUES pour metriques.csv).
    """
    _, colonnes, colonnes_dates, periodes = RESULTATS[nom]
    chemin = chemin_resultat(nom)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    tableau = pd.read_csv(chemin, parse_dates=colonnes_dates, encoding="utf-8")
    verifier_colonnes(tableau, colonnes, chemin.name, periodes)
    return tableau


def verifier_colonnes(tableau, colonnes, nom_fichier, periodes=None):
    """Lève ValueError si les colonnes ou les valeurs de periode ne respectent pas le contrat.

    periodes : valeurs admises dans la colonne periode (None si le fichier n'en a pas).
    """
    if list(tableau.columns) != colonnes:
        raise ValueError(
            f"Colonnes de {nom_fichier} non conformes au contrat : "
            f"{list(tableau.columns)} au lieu de {colonnes}"
        )
    if "periode" in tableau:
        inconnues = set(tableau["periode"].dropna()) - set(periodes or [])
        if inconnues or tableau["periode"].isna().any():
            raise ValueError(
                f"Valeurs de periode non conformes dans {nom_fichier} : "
                f"{sorted(inconnues) or 'valeurs manquantes'} (attendu : {periodes})"
            )
