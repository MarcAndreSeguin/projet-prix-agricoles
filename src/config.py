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
FICHIER_TESTS_DM = DOSSIER_RESULTATS / "tests_dm.csv"

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
SERIES = {
    "mais_QC": {"culture": "Maïs-grain", "province": "Québec",
                "produit_statcan": "Maïs-grain"},
    "mais_ON": {"culture": "Maïs-grain", "province": "Ontario",
                "produit_statcan": "Maïs-grain"},
    "canola_SK": {"culture": "Canola", "province": "Saskatchewan",
                  "produit_statcan": "Canola (y compris le colza)"},
    "canola_AB": {"culture": "Canola", "province": "Alberta",
                  "produit_statcan": "Canola (y compris le colza)"},
}
LISTE_SERIES = list(SERIES)

# Code de produit court (préfixe de la série)
PRODUIT_PAR_SERIE = {serie: serie.split("_")[0] for serie in SERIES}
PROVINCE_PAR_SERIE = {serie: serie.split("_")[1] for serie in SERIES}

# Provinces suivies pour chaque produit (le canola n'est pas suivi au Québec
# ni en Ontario, le maïs-grain ne l'est pas dans les Prairies).
PROVINCES_PAR_PRODUIT = {}
for _serie in SERIES:
    PROVINCES_PAR_PRODUIT.setdefault(PRODUIT_PAR_SERIE[_serie], []).append(PROVINCE_PAR_SERIE[_serie])

# Valeurs manquantes connues dans les sources, estimées par interpolation
# linéaire entre les deux mois voisins (notebooks/01_pretraitement.ipynb).
# Toute autre valeur manquante fait échouer scripts/preparer_donnees.py.
VALEURS_A_IMPUTER = [("canola_AB", "2012-05-01")]

# ---------------------------------------------------------------------------
# Période et découpage
# ---------------------------------------------------------------------------
DATE_DEBUT = "1985-01-01"
DATE_FIN = "2026-08-01"
NB_MOIS_ATTENDU = 500

# Découpage proposé dans notebooks/01_pretraitement.ipynb (À VALIDER PAR
# L'ÉQUIPE) : trois validations de 12 mois avec entraînement croissant, puis
# 12 mois de test final réservés, communs aux quatre séries et aux trois méthodes.
DATE_DEBUT_ENTRAINEMENT = "1985-01-01"
PERIODES_VALIDATION = [
    ("2022-09-01", "2023-08-01"),
    ("2023-09-01", "2024-08-01"),
    ("2024-09-01", "2025-08-01"),
]
DATE_DEBUT_TEST = "2025-09-01"
DATE_FIN_TEST = "2026-08-01"

# Manière de parcourir chaque période (choix de Marc, À VALIDER PAR L'ÉQUIPE) :
# origine glissante, modèle réestimé à chaque origine. Les mois des périodes
# de validation et de test sont des mois CIBLES : chaque mois cible est prévu
# à chaque horizon depuis l'origine date_cible - PAS_PAR_HORIZON[horizon].
# Les origines peuvent donc précéder la période, mais aucune cible n'en sort
# (pas de fuite d'une validation vers la suivante ni vers le test).
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
SEUIL_DM = 0.05

# ---------------------------------------------------------------------------
# Colonnes des fichiers (contrat)
# ---------------------------------------------------------------------------
COLONNES_SERIES = ["date", "serie", "produit", "province", "prix", "prix_impute"]
COLONNES_PREVISIONS = [
    "serie", "modele", "date_origine", "date_cible", "horizon",
    "prevision", "inf80", "sup80", "inf95", "sup95", "reel",
]
COLONNES_PREVISIONS_FUTURES = [c for c in COLONNES_PREVISIONS if c != "reel"]
COLONNES_METRIQUES = [
    "modele", "serie", "horizon", "mae", "rmse", "mape",
    "couverture80", "couverture95", "n",
]
COLONNES_TESTS_DM = [
    "serie", "horizon", "modele_a", "modele_b", "stat_dm", "p_valeur", "meilleur",
]

# ---------------------------------------------------------------------------
# Libellés français
# ---------------------------------------------------------------------------
LIBELLES_PRODUITS = {
    "mais": "Maïs-grain",
    "canola": "Canola",
}
LIBELLES_PROVINCES = {
    "QC": "Québec",
    "ON": "Ontario",
    "SK": "Saskatchewan",
    "AB": "Alberta",
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
