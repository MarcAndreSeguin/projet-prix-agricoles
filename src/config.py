"""Constantes du projet, tirées de docs/protocole.md.

Toute modification de ces valeurs doit être reportée dans le protocole et
approuvée par l'équipe.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "data"
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_FIGURES = RACINE / "rapport" / "figures"

FICHIER_SERIES = DOSSIER_DATA / "series.csv"
FICHIER_PREVISIONS = DOSSIER_RESULTATS / "previsions.csv"
FICHIER_PREVISIONS_FUTURES = DOSSIER_RESULTATS / "previsions_futures.csv"
FICHIER_METRIQUES = DOSSIER_RESULTATS / "metriques.csv"

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
TABLEAU_STATCAN = "32-10-0077-01"
UNITE_PRIX = "$/tonne"

# ---------------------------------------------------------------------------
# Séries
# ---------------------------------------------------------------------------
# Fichiers sources extraits du tableau 32-10-0077-01 (versionnés dans data/).
# Ils sont encodés en cp1252, avec des dates au format mois/jour/année.
FICHIERS_SOURCES = {
    "mais": DOSSIER_DATA / "mais.csv",
    "canola": DOSSIER_DATA / "canola.csv",
}
ENCODAGE_SOURCES = "cp1252"
FORMAT_DATE_SOURCES = "%m/%d/%Y"
UNITE_SOURCES = "Dollars par tonne métrique"

# Correspondance série -> (culture et province telles qu'elles apparaissent
# dans les fichiers sources, nom du produit dans le tableau de Statistique Canada).
# Décision de l'équipe : une seule série par produit, dans la principale province
# productrice (Ontario pour le maïs-grain, Saskatchewan pour le canola). Les
# provinces voisines (Québec, Alberta) ont des marchés pratiquement jumeaux et
# ne sont pas retenues.
SERIES = {
    "mais_ON": {"culture": "Maïs-grain", "province": "Ontario",
                "produit_statcan": "Maïs-grain"},
    "canola_SK": {"culture": "Canola", "province": "Saskatchewan",
                  "produit_statcan": "Canola (y compris le colza)"},
}
LISTE_SERIES = list(SERIES)

# Code de produit court (préfixe de la série)
PRODUIT_PAR_SERIE = {serie: serie.split("_")[0] for serie in SERIES}
PROVINCE_PAR_SERIE = {serie: serie.split("_")[1] for serie in SERIES}

# Une série par produit : produit -> série
SERIE_PAR_PRODUIT = {PRODUIT_PAR_SERIE[serie]: serie for serie in SERIES}
LISTE_PRODUITS = list(SERIE_PAR_PRODUIT)

# Valeurs manquantes connues dans les sources des séries retenues, estimées par
# interpolation linéaire entre les deux mois voisins. Les deux séries retenues
# sont complètes ; la seule valeur manquante des sources (canola, Alberta,
# 2012-05) appartient à une série non retenue. La colonne prix_impute reste
# dans le contrat (toutes ses valeurs sont False).
# Toute autre valeur manquante fait échouer scripts/preparer_donnees.py.
VALEURS_A_IMPUTER = []

# ---------------------------------------------------------------------------
# Période et découpage
# ---------------------------------------------------------------------------
DATE_DEBUT = "1985-01-01"
DATE_FIN = "2026-08-01"
NB_MOIS_ATTENDU = 500

# Découpage chronologique (décision de l'équipe), défini par mois CIBLES.
# Communs aux deux séries et aux trois méthodes.
# - Entraînement : depuis DATE_DEBUT_ENTRAINEMENT, fenêtre croissante jusqu'à
#   chaque origine.
# - Validation : six blocs contigus de 12 mois, de 2014-09 à 2020-08 ; sert
#   seulement au choix du modèle final.
# - Test : trois périodes contiguës de 24 mois (test_1, test_2, test_3), de
#   2020-09 à 2026-08 ; test_global fait le bilan sur les 72 mois de test.
DATE_DEBUT_ENTRAINEMENT = "1985-01-01"
PERIODES_VALIDATION = [
    ("2014-09-01", "2015-08-01"),
    ("2015-09-01", "2016-08-01"),
    ("2016-09-01", "2017-08-01"),
    ("2017-09-01", "2018-08-01"),
    ("2018-09-01", "2019-08-01"),
    ("2019-09-01", "2020-08-01"),
]
PERIODES_TEST = {
    "test_1": ("2020-09-01", "2022-08-01"),
    "test_2": ("2022-09-01", "2024-08-01"),
    "test_3": ("2024-09-01", "2026-08-01"),
}
DATE_DEBUT_VALIDATION = PERIODES_VALIDATION[0][0]
DATE_FIN_VALIDATION = PERIODES_VALIDATION[-1][1]
DATE_DEBUT_TEST = PERIODES_TEST["test_1"][0]
DATE_FIN_TEST = PERIODES_TEST["test_3"][1]

# Manière de parcourir chaque période : origine glissante, modèle réestimé à
# chaque origine. Les mois des périodes de validation et de test sont des
# mois CIBLES : chaque mois cible est prévu
# à chaque horizon depuis l'origine date_cible - PAS_PAR_HORIZON[horizon].
# Les origines peuvent donc précéder la période, mais aucune cible n'en sort
# (pas de fuite d'une période vers la suivante).
MODE_ORIGINE = "glissante"
REESTIMATION_A_CHAQUE_ORIGINE = True

# ---------------------------------------------------------------------------
# Horizons (À CONFIRMER PAR L'ÉQUIPE : repris de la première version du protocole)
# ---------------------------------------------------------------------------
DELAI_PUBLICATION = 2  # mois entre la période de référence et la publication
HORIZONS = [1, 3, 6]  # horizons « consultation », en mois
# Nombre de pas après la dernière donnée connue pour chaque horizon
PAS_PAR_HORIZON = {h: h + DELAI_PUBLICATION for h in HORIZONS}  # {1: 3, 3: 5, 6: 8}
HORIZON_ALERTE = 1

# ---------------------------------------------------------------------------
# Intervalles
# ---------------------------------------------------------------------------
NIVEAU_AFFICHAGE = 0.80
NIVEAU_ALERTE = 0.95
NIVEAUX_INTERVALLE = [NIVEAU_AFFICHAGE, NIVEAU_ALERTE]

# ---------------------------------------------------------------------------
# Modèles et sélection
# ---------------------------------------------------------------------------
MODELES = ["naif", "ets", "sarima"]
MODELE_REFERENCE = "naif"

# ---------------------------------------------------------------------------
# Colonnes des fichiers (contrat)
# ---------------------------------------------------------------------------
COLONNES_SERIES = ["date", "serie", "produit", "province", "prix", "prix_impute"]
# Colonne periode.
# - previsions.csv : validation, test_1, test_2 ou test_3 ; chaque prévision
#   n'apparaît qu'une fois.
# - metriques.csv : les mêmes valeurs, plus test_global, calculé sur l'ensemble
#   des prévisions de test_1, test_2 et test_3.
# Validation et test ne sont jamais mélangés dans un calcul.
PERIODE_VALIDATION = "validation"
PERIODE_TEST_GLOBAL = "test_global"
PERIODES_PREVISIONS = [PERIODE_VALIDATION] + list(PERIODES_TEST)
PERIODES_METRIQUES = PERIODES_PREVISIONS + [PERIODE_TEST_GLOBAL]
# Périodes de previsions.csv couvertes par chaque période de metriques.csv
COMPOSITION_PERIODES = {periode: [periode] for periode in PERIODES_PREVISIONS}
COMPOSITION_PERIODES[PERIODE_TEST_GLOBAL] = list(PERIODES_TEST)
PERIODE_SELECTION = PERIODE_VALIDATION   # période sur laquelle la règle du modèle final est appliquée
PERIODE_RAPPORT = PERIODE_TEST_GLOBAL    # période des performances rapportées par défaut
COLONNES_PREVISIONS = [
    "serie", "modele", "periode", "date_origine", "date_cible", "horizon",
    "prevision", "inf80", "sup80", "inf95", "sup95", "reel",
]
COLONNES_PREVISIONS_FUTURES = [c for c in COLONNES_PREVISIONS if c not in ("periode", "reel")]
COLONNES_METRIQUES = [
    "modele", "serie", "periode", "horizon", "mae", "rmse", "mape",
    "couverture80", "couverture95", "n",
]

# ---------------------------------------------------------------------------
# Libellés français
# ---------------------------------------------------------------------------
LIBELLES_PRODUITS = {
    "mais": "Maïs-grain",
    "canola": "Canola",
}
LIBELLES_PROVINCES = {
    "ON": "Ontario",
    "SK": "Saskatchewan",
}
LIBELLES_MODELES = {
    "naif": "Naïf (référence)",
    "ets": "ETS / Holt-Winters",
    "sarima": "SARIMA",
}
LIBELLES_SERIES = {
    serie: f"{LIBELLES_PRODUITS[PRODUIT_PAR_SERIE[serie]]} – "
           f"{LIBELLES_PROVINCES[PROVINCE_PAR_SERIE[serie]]}"
    for serie in SERIES
}
LIBELLES_HORIZONS = {1: "1 mois", 3: "3 mois", 6: "6 mois"}
LIBELLES_PERIODES = {
    "validation": "Validation",
    "test_1": "Test 1",
    "test_2": "Test 2",
    "test_3": "Test 3",
    "test_global": "Test global",
}
