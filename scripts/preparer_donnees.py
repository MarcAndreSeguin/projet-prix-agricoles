"""Prépare data/series.csv à partir de data/mais.csv et data/canola.csv.

Usage : python scripts/preparer_donnees.py

Les deux fichiers sources ont été extraits du tableau 32-10-0077-01 de
Statistique Canada. Ce script reproduit le prétraitement du notebook
notebooks/01_pretraitement.ipynb, qui reste la référence (contrôles,
statistiques descriptives, graphiques). Le notebook n'écrit aucun fichier :
c'est ce script qui produit data/series.csv, le fichier commun aux trois
méthodes et à l'application. Le découpage chronologique du notebook n'est pas
repris ici ; les périodes utilisées sont celles de src/config.py.

Étapes :

1. lecture des sources (cp1252, dates mois/jour/année) ;
2. sélection des 4 séries de config.SERIES, de config.DATE_DEBUT à la dernière date ;
3. estimation des valeurs manquantes connues (config.VALEURS_A_IMPUTER) par
   interpolation linéaire entre les deux mois voisins, signalée dans la colonne
   prix_impute ;
4. vérifications (nombre de mois, valeurs manquantes, unité, dates consécutives) ;
5. écriture de data/series.csv au format long du contrat (UTF-8, dates AAAA-MM-01).

Le script se termine avec le code 1 si une vérification échoue ; dans ce cas
data/series.csv n'est pas écrit.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def lire_sources():
    """Lit les fichiers sources et retourne un seul tableau aux colonnes normalisées."""
    morceaux = []
    for chemin in config.FICHIERS_SOURCES.values():
        if not chemin.exists():
            sys.exit(f"Fichier source introuvable : {chemin}")
        source = pd.read_csv(chemin, encoding=config.ENCODAGE_SOURCES)
        source = source.rename(columns={"unité": "unite"})
        source["date"] = pd.to_datetime(source["date"], format=config.FORMAT_DATE_SOURCES)
        source["prix"] = pd.to_numeric(source["prix"], errors="coerce")
        morceaux.append(source[["date", "culture", "province", "unite", "prix"]])
    return pd.concat(morceaux, ignore_index=True)


def selectionner_series(sources):
    """Garde les 4 séries de config.SERIES à partir de config.DATE_DEBUT."""
    morceaux = []
    for serie, definition in config.SERIES.items():
        masque = (
            (sources["culture"] == definition["culture"])
            & (sources["province"] == definition["province"])
            & (sources["date"] >= config.DATE_DEBUT)
        )
        morceau = sources.loc[masque].copy()
        morceau["serie"] = serie
        morceau["produit"] = config.PRODUIT_PAR_SERIE[serie]
        morceau["province"] = config.PROVINCE_PAR_SERIE[serie]
        morceaux.append(morceau)
    return pd.concat(morceaux, ignore_index=True).sort_values(["serie", "date"]).reset_index(drop=True)


def imputer(series):
    """Estime les valeurs manquantes connues par interpolation linéaire entre les
    deux mois voisins et les signale dans prix_impute.

    Les autres valeurs manquantes sont laissées telles quelles : la vérification
    « aucune valeur manquante » échouera alors.
    """
    series = series.copy()
    series["prix_impute"] = False
    print("\nValeurs estimées")
    for serie, date in config.VALEURS_A_IMPUTER:
        date = pd.Timestamp(date)
        s = series[series["serie"] == serie].set_index("date")["prix"]
        avant, apres = date - pd.DateOffset(months=1), date + pd.DateOffset(months=1)
        if date not in s.index or not pd.isna(s[date]):
            print(f"  [INFO] {serie} {date:%Y-%m} : valeur présente dans la source, rien à estimer")
            continue
        if pd.isna(s.get(avant)) or pd.isna(s.get(apres)):
            print(f"  [ÉCHEC] {serie} {date:%Y-%m} : mois voisins absents, estimation impossible")
            continue
        estimation = (s[avant] + s[apres]) / 2
        masque = (series["serie"] == serie) & (series["date"] == date)
        series.loc[masque, "prix"] = estimation
        series.loc[masque, "prix_impute"] = True
        print(f"  {serie} {date:%Y-%m} : {estimation:.3f} "
              f"(moyenne de {s[avant]} et {s[apres]})")
    return series


def verifier(series):
    """Affiche les vérifications et retourne True si toutes réussissent."""
    succes = True

    def rapporter(condition, message):
        nonlocal succes
        succes &= bool(condition)
        print(f"  [{'OK' if condition else 'ÉCHEC'}] {message}")

    print("\nVérifications par série")
    print(f"  {'série':<10} {'début':<8} {'fin':<8} {'mois':>5} {'manquants':>10} "
          f"{'estimés':>8} {'doublons':>9} {'trous':>6}")
    for serie in config.LISTE_SERIES:
        s = series[series["serie"] == serie]
        if s.empty:
            rapporter(False, f"{serie} : aucune ligne trouvée dans les sources")
            continue
        dates = s["date"]
        attendues = pd.date_range(dates.min(), dates.max(), freq="MS")
        print(f"  {serie:<10} {dates.min():%Y-%m}  {dates.max():%Y-%m}  {len(s):>5} "
              f"{int(s['prix'].isna().sum()):>10} {int(s['prix_impute'].sum()):>8} "
              f"{int(dates.duplicated().sum()):>9} {len(attendues.difference(dates)):>6}")

    print("\nVérifications globales")
    trouvees = set(series["serie"])
    rapporter(trouvees == set(config.LISTE_SERIES),
              f"{len(config.LISTE_SERIES)} séries présentes ({len(trouvees)} trouvées)")

    nb_manquants = int(series["prix"].isna().sum())
    rapporter(nb_manquants == 0, f"aucune valeur manquante après estimation ({nb_manquants} trouvée(s))")

    unites = sorted(series["unite"].dropna().unique())
    rapporter(unites == [config.UNITE_SOURCES],
              f"unité unique « {config.UNITE_SOURCES} » (trouvé : {unites})")

    sans_trou = True
    for _, s in series.groupby("serie"):
        dates = s["date"]
        attendues = pd.date_range(dates.min(), dates.max(), freq="MS")
        if dates.duplicated().any() or len(attendues.difference(dates)) > 0:
            sans_trou = False
    rapporter(sans_trou, "dates consécutives, sans trou ni doublon, dans chaque série")

    debuts = series.groupby("serie")["date"].min()
    fins = series.groupby("serie")["date"].max()
    rapporter(debuts.nunique() == 1 and debuts.iloc[0] == pd.Timestamp(config.DATE_DEBUT),
              f"toutes les séries débutent en {config.DATE_DEBUT[:7]}")
    rapporter(fins.nunique() == 1,
              f"même date de fin pour toutes les séries ({sorted(fins.dt.strftime('%Y-%m').unique())})")

    nb_mois = series.groupby("serie").size()
    rapporter(nb_mois.nunique() == 1,
              f"même nombre de mois pour toutes les séries ({sorted(int(n) for n in nb_mois.unique())})")
    if nb_mois.nunique() == 1 and nb_mois.iloc[0] != config.NB_MOIS_ATTENDU:
        print(f"  [INFO] {nb_mois.iloc[0]} mois au lieu des {config.NB_MOIS_ATTENDU} "
              "indiqués dans le protocole (nouvelle publication ?)")

    return succes


def main():
    sources = lire_sources()
    print(f"Sources : {', '.join(c.name for c in config.FICHIERS_SOURCES.values())} "
          f"({len(sources)} lignes)")
    series = imputer(selectionner_series(sources))

    if not verifier(series):
        sys.exit("\nAu moins une vérification a échoué : data/series.csv n'a pas été écrit.")

    sortie = series[config.COLONNES_SERIES].copy()
    sortie["date"] = sortie["date"].dt.strftime("%Y-%m-01")
    config.FICHIER_SERIES.parent.mkdir(parents=True, exist_ok=True)
    sortie.to_csv(config.FICHIER_SERIES, index=False, encoding="utf-8")
    print(f"\nÉcrit : {config.FICHIER_SERIES} ({len(sortie)} lignes)")


if __name__ == "__main__":
    main()
