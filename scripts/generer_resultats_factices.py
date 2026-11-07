"""Génère des résultats FACTICES dans resultats/ pour développer l'interface.

Usage : python scripts/generer_resultats_factices.py [--forcer]

Les prévisions sont fabriquées à partir des vraies séries de data/series.csv,
sur les mêmes paires origine-cible que l'évaluation réelle (six blocs de
validation et test, src/evaluation.py, generer_paires_origine_cible()), avec la
colonne periode : dernier prix connu à l'origine, plus un écart aléatoire propre
à chaque modèle (le modèle naïf n'a aucun écart). Les intervalles sont calculés
en log, avec une largeur proportionnelle à l'écart-type des variations
mensuelles avant config.DATE_DEBUT_TEST et à la racine du nombre de pas, puis
ramenés en dollars. Quelques prévisions de test à l'horizon 1 sont décalées
volontairement pour que le prix observé sorte de l'intervalle à 95 % et
déclenche une alerte.

Fichiers produits : previsions.csv, previsions_futures.csv, metriques.csv,
tests_dm.csv et le marqueur FACTICE.txt. Ces résultats n'ont aucune valeur
d'analyse et ne doivent pas être utilisés dans le rapport.

Par sécurité, le script refuse d'écraser des résultats existants si
FACTICE.txt est absent (résultats réels), sauf avec --forcer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chargement import charger_series  # noqa: E402
from src.evaluation import (  # noqa: E402
    calculer_metriques,
    calculer_tests_dm,
    generer_paires_origine_cible,
)

GRAINE = 5103
NB_HORS_INTERVALLE_PAR_SERIE = 3

# Paramètres factices par modèle : (biais, écart-type de l'écart par pas, facteur de largeur)
# Le biais et l'écart sont en log ; le facteur multiplie la largeur des intervalles.
PARAMETRES = {
    "naif": (0.0, 0.0, 1.00),
    "ets": (0.002, 0.020, 0.95),
    "sarima": (-0.003, 0.025, 1.05),
}

Z80 = stats.norm.ppf(0.5 + config.NIVEAU_AFFICHAGE / 2)
Z95 = stats.norm.ppf(0.5 + config.NIVEAU_ALERTE / 2)


def construire_ligne(serie, modele, origine, cible, horizon, log_prevision, sigma):
    """Construit une ligne de prévision en dollars à partir du log et de sigma."""
    return {
        "serie": serie, "modele": modele, "date_origine": origine, "date_cible": cible,
        "horizon": horizon,
        "prevision": np.exp(log_prevision),
        "inf80": np.exp(log_prevision - Z80 * sigma),
        "sup80": np.exp(log_prevision + Z80 * sigma),
        "inf95": np.exp(log_prevision - Z95 * sigma),
        "sup95": np.exp(log_prevision + Z95 * sigma),
    }


def prevoir(rng, serie, origine, cible, horizon, pas, prix_origine, sigma_mensuel):
    """Prévisions factices des 3 modèles pour une paire origine-cible."""
    lignes = []
    for modele, (biais, ecart, facteur) in PARAMETRES.items():
        log_prevision = np.log(prix_origine) + biais * pas + rng.normal(0, ecart * np.sqrt(pas))
        sigma = facteur * sigma_mensuel * np.sqrt(pas)
        ligne = construire_ligne(serie, modele, origine, cible, horizon, log_prevision, sigma)
        ligne["sigma"] = sigma
        lignes.append(ligne)
    return lignes


def paires_a_evaluer():
    """Paires origine-cible des blocs de validation et du test, avec leur période."""
    morceaux = [generer_paires_origine_cible(debut, fin).assign(periode="validation")
                for debut, fin in config.PERIODES_VALIDATION]
    morceaux.append(generer_paires_origine_cible(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)
                    .assign(periode="test"))
    return pd.concat(morceaux, ignore_index=True)


def forcer_hors_intervalle(rng, previsions):
    """Décale quelques prévisions de test à l'horizon 1 pour que le réel sorte de l'IC 95 %.

    Le décalage est appliqué aux 3 modèles pour la même série et la même date
    cible, afin que l'alerte apparaisse quel que soit le modèle retenu.
    Retourne les prévisions modifiées et la liste des (serie, date_cible) visées.
    """
    visees = []
    h1 = previsions[(previsions["horizon"] == config.HORIZON_ALERTE)
                    & (previsions["periode"] == config.PERIODE_RAPPORT)]
    for serie, g in h1.groupby("serie"):
        cibles = g["date_cible"].drop_duplicates()
        choix = rng.choice(cibles.to_numpy(), size=min(NB_HORS_INTERVALLE_PAR_SERIE, len(cibles)),
                           replace=False)
        for cible in choix:
            signe = rng.choice([-1, 1])
            masque = (previsions["serie"] == serie) & (previsions["date_cible"] == cible) \
                & (previsions["horizon"] == config.HORIZON_ALERTE) \
                & (previsions["periode"] == config.PERIODE_RAPPORT)
            for indice in previsions.index[masque]:
                ligne = previsions.loc[indice]
                log_prevision = np.log(ligne["reel"]) - signe * 1.5 * Z95 * ligne["sigma"]
                nouvelle = construire_ligne(serie, ligne["modele"], ligne["date_origine"], cible,
                                            ligne["horizon"], log_prevision, ligne["sigma"])
                for colonne in ["prevision", "inf80", "sup80", "inf95", "sup95"]:
                    previsions.loc[indice, colonne] = nouvelle[colonne]
            visees.append((serie, pd.Timestamp(cible)))
    return previsions, visees


def mettre_en_forme(previsions, colonnes):
    """Arrondit les prix et formate les dates selon le contrat."""
    sortie = previsions[colonnes].copy()
    for colonne in ["date_origine", "date_cible"]:
        sortie[colonne] = pd.to_datetime(sortie[colonne]).dt.strftime("%Y-%m-01")
    colonnes_prix = [c for c in ["prevision", "inf80", "sup80", "inf95", "sup95", "reel"] if c in sortie]
    sortie[colonnes_prix] = sortie[colonnes_prix].round(2)
    tri = [c for c in ["serie", "modele", "periode", "date_cible", "horizon"] if c in sortie]
    return sortie.sort_values(tri).reset_index(drop=True)


def generer(series, rng):
    """Retourne (previsions, previsions_futures, metriques, tests_dm, visees)."""
    lignes, lignes_futures = [], []
    paires = paires_a_evaluer()
    for serie, s in series.groupby("serie"):
        prix_par_date = s.set_index("date")["prix"]
        avant_test = prix_par_date[prix_par_date.index < config.DATE_DEBUT_TEST]
        sigma_mensuel = np.log(avant_test).diff().std()

        for paire in paires.itertuples():
            nouvelles = prevoir(rng, serie, paire.date_origine, paire.date_cible, paire.horizon,
                                paire.pas, prix_par_date[paire.date_origine], sigma_mensuel)
            for ligne in nouvelles:
                ligne["periode"] = paire.periode
                ligne["reel"] = prix_par_date[paire.date_cible]
            lignes += nouvelles

        derniere = prix_par_date.index.max()
        for horizon, pas in config.PAS_PAR_HORIZON.items():
            lignes_futures += prevoir(rng, serie, derniere, derniere + pd.DateOffset(months=pas),
                                      horizon, pas, prix_par_date[derniere], sigma_mensuel)

    previsions = pd.DataFrame(lignes)
    previsions, visees = forcer_hors_intervalle(rng, previsions)
    previsions_futures = pd.DataFrame(lignes_futures)

    metriques = calculer_metriques(previsions)
    tests_dm = calculer_tests_dm(previsions)

    previsions = mettre_en_forme(previsions, config.COLONNES_PREVISIONS)
    previsions_futures = mettre_en_forme(previsions_futures, config.COLONNES_PREVISIONS_FUTURES)
    metriques = metriques.round({"mae": 4, "rmse": 4, "mape": 4, "couverture80": 4, "couverture95": 4})
    tests_dm = tests_dm.round({"stat_dm": 4, "p_valeur": 4})
    return previsions, previsions_futures, metriques, tests_dm, visees


def verifier_colonnes(fichiers):
    """Relit chaque fichier et compare ses colonnes au contrat. Retourne True si tout concorde."""
    attendu = {
        config.FICHIER_PREVISIONS.name: config.COLONNES_PREVISIONS,
        config.FICHIER_PREVISIONS_FUTURES.name: config.COLONNES_PREVISIONS_FUTURES,
        config.FICHIER_METRIQUES.name: config.COLONNES_METRIQUES,
        config.FICHIER_TESTS_DM.name: config.COLONNES_TESTS_DM,
    }
    succes = True
    print("\nVérification des colonnes (relecture des fichiers)")
    for chemin in fichiers:
        colonnes = list(pd.read_csv(chemin, nrows=0).columns)
        concorde = colonnes == attendu[chemin.name]
        succes &= concorde
        print(f"  [{'OK' if concorde else 'ÉCHEC'}] {chemin.name} : {', '.join(colonnes)}")
    return succes


def main():
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--forcer", action="store_true",
                           help="écraser des résultats existants même sans FACTICE.txt")
    args = analyseur.parse_args()

    dossier = config.DOSSIER_RESULTATS
    marqueur = dossier / "FACTICE.txt"
    fichiers = [config.FICHIER_PREVISIONS, config.FICHIER_PREVISIONS_FUTURES,
                config.FICHIER_METRIQUES, config.FICHIER_TESTS_DM]
    if any(f.exists() for f in fichiers) and not marqueur.exists() and not args.forcer:
        sys.exit("resultats/ contient des résultats sans FACTICE.txt (résultats réels ?). "
                 "Arrêt sans rien écrire ; utiliser --forcer pour les écraser.")

    series = charger_series()
    rng = np.random.default_rng(GRAINE)
    previsions, previsions_futures, metriques, tests_dm, visees = generer(series, rng)

    dossier.mkdir(parents=True, exist_ok=True)
    for tableau, chemin in zip([previsions, previsions_futures, metriques, tests_dm], fichiers):
        tableau.to_csv(chemin, index=False, encoding="utf-8")
        print(f"Écrit : {chemin.name} ({len(tableau)} lignes)")
    marqueur.write_text(
        "Les fichiers de ce dossier sont FACTICES (scripts/generer_resultats_factices.py).\n"
        "Ils servent uniquement à développer l'interface et n'ont aucune valeur d'analyse.\n"
        "Ce fichier sera supprimé par scripts/executer_evaluation.py.\n",
        encoding="utf-8",
    )
    print(f"Écrit : {marqueur.name}")

    print(f"\nObservations forcées hors de l'IC 95 % à l'horizon {config.HORIZON_ALERTE} :")
    for serie, cible in visees:
        print(f"  {serie} {cible:%Y-%m}")

    if not verifier_colonnes(fichiers):
        sys.exit("\nLes colonnes ne correspondent pas au contrat.")


if __name__ == "__main__":
    main()
