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
DOSSIER_BRUT = DOSSIER_DATA / "brut"
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_FIGURES = RACINE / "rapport" / "figures"

FICHIER_BRUT = DOSSIER_BRUT / "32100077.csv"
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
# Correspondance série -> (nom du produit Statistique Canada sans le code
# entre crochets, province telle qu'elle apparaît dans la colonne GEO).
SERIES = {
    "mais_QC": {"produit_statcan": "Corn for grain", "province": "Quebec"},
    "mais_ON": {"produit_statcan": "Corn for grain", "province": "Ontario"},
    "soya_QC": {"produit_statcan": "Soybeans", "province": "Quebec"},
    "soya_ON": {"produit_statcan": "Soybeans", "province": "Ontario"},
    "ble_QC": {"produit_statcan": "Wheat (except durum wheat)", "province": "Quebec"},
    "ble_ON": {"produit_statcan": "Ontario wheat excluding payments", "province": "Ontario"},
}
LISTE_SERIES = list(SERIES)

# Code de produit court (préfixe de la série)
PRODUIT_PAR_SERIE = {serie: serie.split("_")[0] for serie in SERIES}
PROVINCE_PAR_SERIE = {serie: serie.split("_")[1] for serie in SERIES}

# Séries dont la définition diffère et doit être signalée à l'utilisateur
SERIES_DEFINITION_DIFFERENTE = ["ble_ON"]

# ---------------------------------------------------------------------------
# Période et découpage
# ---------------------------------------------------------------------------
DATE_DEBUT = "1992-01-01"
DATE_FIN = "2026-08-01"
NB_MOIS_ATTENDU = 416

DATE_DEBUT_ENTRAINEMENT = "1992-01-01"
DATE_DEBUT_TEST = "2019-01-01"
DATE_FIN_TEST = "2026-08-01"

# ---------------------------------------------------------------------------
# Horizons
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
MODELES = ["naif", "ets", "sarima", "gbm"]
MODELE_REFERENCE = "naif"
SEUIL_DM = 0.05

# ---------------------------------------------------------------------------
# Colonnes des fichiers (contrat)
# ---------------------------------------------------------------------------
COLONNES_SERIES = ["date", "serie", "produit", "province", "prix"]
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
    "soya": "Soya",
    "ble": "Blé",
}
LIBELLES_PROVINCES = {
    "QC": "Québec",
    "ON": "Ontario",
}
LIBELLES_MODELES = {
    "naif": "Naïf (référence)",
    "ets": "ETS / Holt-Winters",
    "sarima": "SARIMA",
    "gbm": "Gradient boosting (LightGBM)",
}
LIBELLES_SERIES = {
    serie: f"{LIBELLES_PRODUITS[PRODUIT_PAR_SERIE[serie]]} – "
           f"{LIBELLES_PROVINCES[PROVINCE_PAR_SERIE[serie]]}"
    for serie in SERIES
}
LIBELLES_HORIZONS = {1: "1 mois", 3: "3 mois", 6: "6 mois"}
