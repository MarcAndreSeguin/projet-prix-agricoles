"""Exécute l'évaluation sur le découpage commun et produit les fichiers de resultats/.

Usage : python scripts/executer_evaluation.py [--modeles naif ets sarima]

Déroulement, pour chaque série de config.LISTE_SERIES et chaque modèle disponible :
1. paires origine-cible des six blocs de validation (config.PERIODES_VALIDATION,
   periode = « validation ») et du test final (config.DATE_DEBUT_TEST à
   config.DATE_FIN_TEST, periode = « test »), produites par
   src/evaluation.py, generer_paires_origine_cible() : chaque mois d'une période
   est un mois cible, prévu à chaque horizon depuis date_cible - pas ;
2. à chaque origine, appel de prevoir(historique, pas) sur le log des prix
   connus jusqu'à l'origine incluse (fenêtre croissante depuis le début de la
   série) : le modèle est réestimé à chaque origine et ne voit aucune donnée
   postérieure. Une origine qui sert à plusieurs périodes (par exemple une
   cible de validation à 3 pas et une cible de test à 8 pas) n'est estimée
   qu'une fois ;
3. retour en dollars par exponentielle et ajout du prix observé (reel) ;
4. prévisions futures : origine = dernière donnée, historique complet,
   pas de config.PAS_PAR_HORIZON ;
5. métriques et tests de Diebold-Mariano par période (src/evaluation.py), sans
   jamais mélanger validation et test.

Retour en dollars
-----------------
Les modèles prévoient le log du prix. Si le log du prix futur suit une loi
N(mu, sigma²), exp(mu) est la MÉDIANE du prix prévu, et non sa moyenne, qui
vaut exp(mu + sigma² / 2). La colonne prevision contient donc une médiane.
Les bornes, elles, restent exactes après l'exponentielle : la fonction
exponentielle est croissante, donc les quantiles du log deviennent les
quantiles du prix (l'intervalle en dollars est asymétrique).

Modèles disponibles
-------------------
Un modèle de config.MODELES est disponible si son module src/modeles/<modele>.py
définit une fonction prevoir() qui ne lève pas NotImplementedError lors d'un appel
d'essai (historique de la première série jusqu'à la première origine évaluée,
un pas). Les autres
sont ignorés avec un message ; il suffit d'écrire leur prevoir() (interface
commune de docs/protocole.md) et de relancer ce script. Toute autre erreur
d'un modèle arrête le script sans rien écrire.

Écriture
--------
Les colonnes de chaque fichier sont vérifiées avant l'écriture. Les fichiers
sont d'abord écrits sous un nom temporaire, puis renommés. resultats/FACTICE.txt
n'est supprimé que si toute l'exécution a réussi.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chargement import charger_series, verifier_colonnes  # noqa: E402
from src.evaluation import (  # noqa: E402
    calculer_metriques,
    calculer_tests_dm,
    generer_paires_origine_cible,
)

COLONNES_MODELE = ["pas", "prevision", "inf80", "sup80", "inf95", "sup95"]
COLONNES_PRIX = ["prevision", "inf80", "sup80", "inf95", "sup95"]
DECIMALES_PRIX = 4
DECIMALES_METRIQUES = 6
FICHIER_FACTICE = config.DOSSIER_RESULTATS / "FACTICE.txt"


class ModeleNonDisponible(Exception):
    """Le modèle n'a pas encore de prevoir() utilisable."""


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def charger_modele(nom, historique_essai):
    """Retourne la fonction prevoir du module src/modeles/<nom>.py.

    Lève ModeleNonDisponible si le module n'a pas de prevoir() ou si un appel
    d'essai lève NotImplementedError.
    """
    module = importlib.import_module(f"src.modeles.{nom}")
    prevoir = getattr(module, "prevoir", None)
    if not callable(prevoir):
        raise ModeleNonDisponible(f"src/modeles/{nom}.py ne définit pas prevoir()")
    try:
        prevoir(historique_essai, [1])
    except NotImplementedError as erreur:
        raise ModeleNonDisponible(f"prevoir() lève NotImplementedError ({erreur})") from erreur
    return prevoir


def appeler(prevoir, nom, serie, historique, pas):
    """Appelle prevoir() et vérifie que le résultat respecte l'interface commune."""
    resultat = prevoir(historique, pas)
    contexte = f"{nom}, {serie}, origine {historique.index[-1]:%Y-%m}"
    if not isinstance(resultat, pd.DataFrame) or list(resultat.columns) != COLONNES_MODELE:
        colonnes = list(resultat.columns) if isinstance(resultat, pd.DataFrame) else type(resultat)
        raise ValueError(f"{contexte} : colonnes {colonnes} au lieu de {COLONNES_MODELE}")
    if resultat["pas"].tolist() != list(pas):
        raise ValueError(f"{contexte} : pas {resultat['pas'].tolist()} au lieu de {list(pas)}")
    if resultat[COLONNES_PRIX].isna().any().any():
        raise ValueError(f"{contexte} : valeurs manquantes dans les prévisions")
    return resultat


# ---------------------------------------------------------------------------
# Paires origine-cible
# ---------------------------------------------------------------------------
def paires_a_evaluer():
    """Paires origine-cible des blocs de validation et du test, avec leur période."""
    morceaux = [
        generer_paires_origine_cible(debut, fin).assign(periode="validation")
        for debut, fin in config.PERIODES_VALIDATION
    ]
    morceaux.append(generer_paires_origine_cible(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)
                    .assign(periode="test"))
    paires = pd.concat(morceaux, ignore_index=True)
    doublons = paires.duplicated(["date_cible", "horizon"])
    if doublons.any():
        raise ValueError("Un même mois cible appartient à deux périodes : vérifier le découpage")
    return paires


# ---------------------------------------------------------------------------
# Prévisions
# ---------------------------------------------------------------------------
def en_dollars(tableau):
    """Convertit les colonnes de prix du log en dollars (exp : médiane du prix prévu)."""
    tableau = tableau.copy()
    tableau[COLONNES_PRIX] = np.exp(tableau[COLONNES_PRIX])
    return tableau


def evaluer_serie(nom, prevoir, serie, prix, paires):
    """Prévisions d'un modèle pour une série, sur toutes les paires origine-cible."""
    log_prix = np.log(prix)
    morceaux = []
    for origine, groupe in paires.groupby("date_origine"):
        if origine not in log_prix.index:
            raise ValueError(f"{serie} : origine {origine:%Y-%m} absente de data/series.csv")
        pas = sorted(groupe["pas"].unique().tolist())
        resultat = appeler(prevoir, nom, serie, log_prix[:origine], pas)
        morceaux.append(groupe.merge(resultat, on="pas", how="left"))
    previsions = en_dollars(pd.concat(morceaux, ignore_index=True))
    absentes = ~previsions["date_cible"].isin(prix.index)
    if absentes.any():
        raise ValueError(f"{serie} : {int(absentes.sum())} date(s) cible(s) absente(s) de data/series.csv")
    previsions["reel"] = previsions["date_cible"].map(prix)
    return previsions.assign(serie=serie, modele=nom)


def prevoir_futur(nom, prevoir, serie, prix):
    """Prévisions futures : origine = dernière donnée, historique complet."""
    origine = prix.index.max()
    pas = list(config.PAS_PAR_HORIZON.values())
    resultat = en_dollars(appeler(prevoir, nom, serie, np.log(prix), pas))
    horizon_par_pas = {p: h for h, p in config.PAS_PAR_HORIZON.items()}
    return resultat.assign(
        serie=serie, modele=nom, date_origine=origine,
        date_cible=[origine + pd.DateOffset(months=p) for p in resultat["pas"]],
        horizon=resultat["pas"].map(horizon_par_pas),
    )


# ---------------------------------------------------------------------------
# Mise en forme et écriture
# ---------------------------------------------------------------------------
def mettre_en_forme(tableau, colonnes, tri):
    """Ordonne les colonnes, formate les dates (AAAA-MM-01) et arrondit les prix."""
    sortie = tableau[colonnes].copy()
    for colonne in ("date_origine", "date_cible"):
        if colonne in sortie:
            sortie[colonne] = pd.to_datetime(sortie[colonne]).dt.strftime("%Y-%m-01")
    prix = [c for c in COLONNES_PRIX + ["reel"] if c in sortie]
    sortie[prix] = sortie[prix].round(DECIMALES_PRIX)
    return sortie.sort_values(tri).reset_index(drop=True)


def ecrire(fichiers):
    """Vérifie les colonnes de chaque tableau, puis écrit (nom temporaire, puis renommage)."""
    for chemin, (tableau, colonnes) in fichiers.items():
        verifier_colonnes(tableau, colonnes, chemin.name)
    config.DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
    temporaires = {}
    for chemin, (tableau, _) in fichiers.items():
        temporaire = chemin.with_name(chemin.name + ".tmp")
        tableau.to_csv(temporaire, index=False, encoding="utf-8")
        temporaires[chemin] = temporaire
    for chemin, temporaire in temporaires.items():
        temporaire.replace(chemin)
        print(f"  Écrit : resultats/{chemin.name} ({len(fichiers[chemin][0])} lignes)")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("--modeles", nargs="+", choices=config.MODELES, default=config.MODELES,
                           help="modèles à exécuter (par défaut : tous les modèles disponibles)")
    args = analyseur.parse_args()
    debut_chrono = time.perf_counter()

    series = charger_series()
    prix_par_serie = {
        serie: series.loc[series["serie"] == serie].set_index("date")["prix"].asfreq("MS")
        for serie in config.LISTE_SERIES
    }
    paires = paires_a_evaluer()

    # Modèles disponibles : appel d'essai sur l'historique de la première origine évaluée
    # (même longueur minimale que dans l'évaluation, pour ne pas pénaliser les modèles saisonniers)
    premiere_origine = paires["date_origine"].min()
    historique_essai = np.log(prix_par_serie[config.LISTE_SERIES[0]][:premiere_origine])
    modeles, ignores = {}, {}
    for nom in args.modeles:
        try:
            modeles[nom] = charger_modele(nom, historique_essai)
        except ModeleNonDisponible as raison:
            ignores[nom] = str(raison)
    print("Modèles exécutés :", ", ".join(modeles) or "aucun")
    for nom, raison in ignores.items():
        print(f"Modèle ignoré : {nom} – {raison}")
    if not modeles:
        sys.exit("Aucun modèle disponible : rien n'est écrit dans resultats/.")

    # Évaluation et prévisions futures
    morceaux, futurs = [], []
    for nom, prevoir in modeles.items():
        for serie, prix in prix_par_serie.items():
            debut_serie = time.perf_counter()
            morceaux.append(evaluer_serie(nom, prevoir, serie, prix, paires))
            futurs.append(prevoir_futur(nom, prevoir, serie, prix))
            print(f"  {nom:<7} {serie:<10} {paires['date_origine'].nunique()} origines "
                  f"({time.perf_counter() - debut_serie:.1f} s)")
    previsions = pd.concat(morceaux, ignore_index=True)
    previsions_futures = pd.concat(futurs, ignore_index=True)

    # Métriques et tests par période, calculés avant l'arrondi des prix
    metriques = calculer_metriques(previsions)
    tests_dm = calculer_tests_dm(previsions, modeles=list(modeles))
    metriques = metriques.round({c: DECIMALES_METRIQUES for c in
                                 ["mae", "rmse", "mape", "couverture80", "couverture95"]})
    tests_dm = tests_dm.round({"stat_dm": DECIMALES_METRIQUES, "p_valeur": DECIMALES_METRIQUES})

    fichiers = {
        config.FICHIER_PREVISIONS: (
            mettre_en_forme(previsions, config.COLONNES_PREVISIONS,
                            ["serie", "modele", "periode", "date_cible", "horizon"]),
            config.COLONNES_PREVISIONS),
        config.FICHIER_PREVISIONS_FUTURES: (
            mettre_en_forme(previsions_futures, config.COLONNES_PREVISIONS_FUTURES,
                            ["serie", "modele", "horizon"]),
            config.COLONNES_PREVISIONS_FUTURES),
        config.FICHIER_METRIQUES: (
            metriques.sort_values(["serie", "modele", "periode", "horizon"]).reset_index(drop=True),
            config.COLONNES_METRIQUES),
        config.FICHIER_TESTS_DM: (
            tests_dm.sort_values(["serie", "periode", "horizon"]).reset_index(drop=True),
            config.COLONNES_TESTS_DM),
    }
    print("\nÉcriture")
    ecrire(fichiers)

    if FICHIER_FACTICE.exists():
        FICHIER_FACTICE.unlink()
        print(f"  Supprimé : resultats/{FICHIER_FACTICE.name}")

    print("\nRésumé")
    print("  Modèles exécutés :", ", ".join(modeles))
    print("  Modèles ignorés :", ", ".join(ignores) or "aucun")
    for periode, nombre in previsions.groupby("periode").size().reindex(config.PERIODES_RESULTATS).items():
        print(f"  Prévisions ({periode}) : {int(nombre)}")
    print(f"  Prévisions futures : {len(previsions_futures)}")
    print(f"  Durée : {time.perf_counter() - debut_chrono:.1f} s")


if __name__ == "__main__":
    main()
