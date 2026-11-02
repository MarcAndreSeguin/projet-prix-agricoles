"""Prépare data/series.csv à partir du fichier brut data/brut/32100077.csv.

Usage : python scripts/preparer_donnees.py

Étapes :
1. lecture du CSV complet de Statistique Canada (tableau 32-10-0077-01) ;
2. retrait des codes entre crochets dans les noms de produits ;
3. sélection des 6 séries de config.SERIES, de config.DATE_DEBUT à la dernière date disponible ;
4. vérifications (nombre de mois, valeurs manquantes, unité, dates consécutives) ;
5. écriture de data/series.csv au format long du contrat.

Le script se termine avec le code 1 si une vérification échoue ; dans ce cas
data/series.csv n'est pas écrit.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

UNITE_ATTENDUE = "Dollars per metric tonne"

# Colonnes standard des CSV complets de Statistique Canada ; la colonne de
# dimension restante est celle des produits.
COLONNES_STANDARD = {
    "REF_DATE", "GEO", "DGUID", "UOM", "UOM_ID", "SCALAR_FACTOR", "SCALAR_ID",
    "VECTOR", "COORDINATE", "VALUE", "STATUS", "SYMBOL", "TERMINATED", "DECIMALS",
}


def lire_brut(chemin):
    """Lit le CSV brut et retourne les colonnes utiles renommées."""
    if not chemin.exists():
        sys.exit(
            f"Fichier brut introuvable : {chemin}\n"
            "Télécharger le tableau 32-10-0077-01 (CSV, tableau complet) sur le site "
            "de Statistique Canada et placer 32100077.csv dans data/brut/."
        )
    brut = pd.read_csv(chemin, encoding="utf-8-sig", dtype=str)

    colonnes_produit = [c for c in brut.columns if c not in COLONNES_STANDARD]
    if len(colonnes_produit) != 1:
        sys.exit(f"Impossible d'identifier la colonne des produits parmi : {colonnes_produit}")
    colonne_produit = colonnes_produit[0]

    brut = brut.rename(columns={
        "REF_DATE": "date", "GEO": "geo", colonne_produit: "produit_statcan",
        "UOM": "unite", "VALUE": "prix",
    })
    brut["produit_statcan"] = (
        brut["produit_statcan"].str.replace(r"\s*\[[^\]]*\]", "", regex=True).str.strip()
    )
    brut["date"] = pd.to_datetime(brut["date"], format="%Y-%m")
    brut["prix"] = pd.to_numeric(brut["prix"], errors="coerce")
    return brut[["date", "geo", "produit_statcan", "unite", "prix"]]


def selectionner_series(brut):
    """Garde les 6 séries de config.SERIES à partir de config.DATE_DEBUT."""
    morceaux = []
    for serie, definition in config.SERIES.items():
        masque = (
            (brut["produit_statcan"] == definition["produit_statcan"])
            & (brut["geo"] == definition["province"])
            & (brut["date"] >= config.DATE_DEBUT)
        )
        morceau = brut.loc[masque].copy()
        morceau["serie"] = serie
        morceau["produit"] = config.PRODUIT_PAR_SERIE[serie]
        morceau["province"] = config.PROVINCE_PAR_SERIE[serie]
        morceaux.append(morceau)
    return pd.concat(morceaux, ignore_index=True).sort_values(["serie", "date"])


def verifier(series):
    """Affiche les vérifications et retourne True si toutes réussissent."""
    succes = True

    def rapporter(condition, message):
        nonlocal succes
        succes &= bool(condition)
        print(f"  [{'OK' if condition else 'ÉCHEC'}] {message}")

    print("\nVérifications par série")
    print(f"  {'série':<9} {'début':<8} {'fin':<8} {'mois':>5} {'manquants':>10} {'doublons':>9} {'trous':>6}")
    for serie in config.LISTE_SERIES:
        s = series[series["serie"] == serie]
        if s.empty:
            rapporter(False, f"{serie} : aucune ligne trouvée dans le fichier brut")
            continue
        dates = s["date"]
        attendues = pd.date_range(dates.min(), dates.max(), freq="MS")
        nb_trous = len(attendues.difference(dates))
        nb_doublons = int(dates.duplicated().sum())
        nb_manquants = int(s["prix"].isna().sum())
        print(f"  {serie:<9} {dates.min():%Y-%m}  {dates.max():%Y-%m}  {len(s):>5} "
              f"{nb_manquants:>10} {nb_doublons:>9} {nb_trous:>6}")

    print("\nVérifications globales")
    trouvees = set(series["serie"])
    rapporter(trouvees == set(config.LISTE_SERIES),
              f"6 séries présentes ({len(trouvees)} trouvées)")

    nb_manquants = int(series["prix"].isna().sum())
    rapporter(nb_manquants == 0, f"aucune valeur manquante ({nb_manquants} trouvée(s))")

    unites = sorted(series["unite"].dropna().unique())
    rapporter(unites == [UNITE_ATTENDUE], f"unité unique « {UNITE_ATTENDUE} » (trouvé : {unites})")

    sans_trou = True
    for serie, s in series.groupby("serie"):
        dates = s["date"]
        attendues = pd.date_range(dates.min(), dates.max(), freq="MS")
        if dates.duplicated().any() or len(attendues.difference(dates)) > 0:
            sans_trou = False
    rapporter(sans_trou, "dates consécutives, sans trou ni doublon, dans chaque série")

    debuts = series.groupby("serie")["date"].min()
    fins = series.groupby("serie")["date"].max()
    rapporter(debuts.nunique() == 1 and debuts.iloc[0] == pd.Timestamp(config.DATE_DEBUT),
              f"toutes les séries débutent en {config.DATE_DEBUT[:7]}")
    rapporter(fins.nunique() == 1, f"même date de fin pour toutes les séries ({sorted(fins.dt.strftime('%Y-%m').unique())})")

    nb_mois = series.groupby("serie").size()
    rapporter(nb_mois.nunique() == 1,
              f"même nombre de mois pour toutes les séries ({sorted(int(n) for n in nb_mois.unique())})")
    if nb_mois.nunique() == 1 and nb_mois.iloc[0] != config.NB_MOIS_ATTENDU:
        print(f"  [INFO] {nb_mois.iloc[0]} mois au lieu des {config.NB_MOIS_ATTENDU} "
              "indiqués dans le protocole (nouvelle publication ?)")

    return succes


def main():
    brut = lire_brut(config.FICHIER_BRUT)
    print(f"Fichier brut : {config.FICHIER_BRUT.name} ({len(brut)} lignes)")
    series = selectionner_series(brut)

    if not verifier(series):
        sys.exit("\nAu moins une vérification a échoué : data/series.csv n'a pas été écrit.")

    sortie = series[config.COLONNES_SERIES].copy()
    sortie["date"] = sortie["date"].dt.strftime("%Y-%m-01")
    config.FICHIER_SERIES.parent.mkdir(parents=True, exist_ok=True)
    sortie.to_csv(config.FICHIER_SERIES, index=False, encoding="utf-8")
    print(f"\nÉcrit : {config.FICHIER_SERIES} ({len(sortie)} lignes)")


if __name__ == "__main__":
    main()
