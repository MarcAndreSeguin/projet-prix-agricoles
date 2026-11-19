"""Évaluation commune : ETS en dollars, naïf et SARIMA via leur interface log.

ETS : configuration choisie sur validation uniquement par selectionner_ets_revise.
Les prévisions ETS brutes ne sont pas exponentiées. Les intervalles additifs
peuvent comporter des bornes négatives ; aucune troncature n'est appliquée.
"""

import argparse
import json
import hashlib
from functools import partial
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chargement import charger_series, verifier_colonnes  # noqa: E402
from src.evaluation import calculer_metriques, generer_paires_evaluation  # noqa: E402

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
def charger_modele(nom, historique_essai, configuration_ets=None):
    """Retourne la fonction prevoir du module src/modeles/<nom>.py.

    Lève ModeleNonDisponible si le module n'a pas de prevoir() ou si un appel
    d'essai lève NotImplementedError.
    """
    module = importlib.import_module(f"src.modeles.{nom}")
    prevoir = getattr(module, "prevoir", None)
    if not callable(prevoir):
        raise ModeleNonDisponible(f"src/modeles/{nom}.py ne définit pas prevoir()")
    if nom == "ets":
        prevoir = partial(module.prevoir_prix, **configuration_ets)
    try:
        prevoir(np.exp(historique_essai) if nom == "ets" else historique_essai, [1])
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
    if not np.isfinite(resultat[COLONNES_PRIX].to_numpy()).all():
        raise ValueError(f"{contexte} : valeurs manquantes dans les prévisions")
    return resultat


# ---------------------------------------------------------------------------
# Prévisions
# ---------------------------------------------------------------------------
def en_dollars(tableau, nom):
    """ETS est déjà en dollars ; les autres modèles reviennent du log par exp."""
    tableau = tableau.copy()
    if nom != "ets":
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
        resultat = appeler(prevoir, nom, serie, prix.loc[config.DATE_DEBUT_ENTRAINEMENT:origine] if nom == "ets" else log_prix.loc[config.DATE_DEBUT_ENTRAINEMENT:origine], pas)
        morceaux.append(groupe.merge(resultat, on="pas", how="left"))
    previsions = en_dollars(pd.concat(morceaux, ignore_index=True), nom)
    absentes = ~previsions["date_cible"].isin(prix.index)
    if absentes.any():
        raise ValueError(f"{serie} : {int(absentes.sum())} date(s) cible(s) absente(s) de data/series.csv")
    previsions["reel"] = previsions["date_cible"].map(prix)
    return previsions.assign(serie=serie, modele=nom)


def prevoir_futur(nom, prevoir, serie, prix):
    """Prévisions futures : origine = dernière donnée, historique complet."""
    prix = prix.loc[config.DATE_DEBUT_ENTRAINEMENT:]
    origine = prix.index.max()
    pas = list(config.PAS_PAR_HORIZON.values())
    resultat = en_dollars(appeler(prevoir, nom, serie, prix if nom == "ets" else np.log(prix), pas), nom)
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
    """Vérifie les colonnes et les périodes de chaque tableau, puis écrit (nom
    temporaire, puis renommage)."""
    for chemin, (tableau, colonnes, periodes) in fichiers.items():
        verifier_colonnes(tableau, colonnes, chemin.name, periodes)
    config.DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
    temporaires = {}
    for chemin, (tableau, _, _) in fichiers.items():
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
    paires = generer_paires_evaluation()

    # Les configurations ETS viennent uniquement de la sélection en validation.
    choix_ets = {}
    if "ets" in args.modeles:
        chemin_choix = config.DOSSIER_RESULTATS / "ets_selection_revisee" / "configurations_retenues.json"
        if not chemin_choix.is_file():
            sys.exit("Lancer d'abord : python -m scripts.selectionner_ets_revise")
        document = json.loads(chemin_choix.read_text(encoding="utf-8"))
        if (document["periodes_validation"] != [list(p) for p in config.PERIODES_VALIDATION]
                or document["horizons"] != config.HORIZONS
                or document["delai_publication"] != config.DELAI_PUBLICATION
                or document["date_debut_entrainement"] != config.DATE_DEBUT_ENTRAINEMENT):
            raise ValueError("Le protocole a changé : relancer la sélection ETS.")
        if document["empreinte_series"] != hashlib.sha256(config.FICHIER_SERIES.read_bytes()).hexdigest():
            raise ValueError("series.csv a changé : relancer la sélection ETS.")
        choix_ets = document["choix"]
        if set(choix_ets) != set(config.LISTE_SERIES):
            raise ValueError("Les choix ETS ne correspondent pas aux séries actuelles.")
        from src.modeles.ets import CONFIGURATIONS, VARIANTES
        if document["configurations_candidates"] != CONFIGURATIONS or document["variantes"] != VARIANTES:
            raise ValueError("Les candidats ETS ont changé : relancer la sélection.")
        for choix in choix_ets.values():
            if choix["configuration"] not in CONFIGURATIONS or choix["variante"] not in VARIANTES:
                raise ValueError("Choix ETS invalide.")

    # Modèles disponibles : appel d'essai sur l'historique de la première origine évaluée
    # (même longueur minimale que dans l'évaluation, pour ne pas pénaliser les modèles saisonniers)
    premiere_origine = paires["date_origine"].min()
    historique_essai = np.log(prix_par_serie[config.LISTE_SERIES[0]][:premiere_origine])
    modeles, ignores = {}, {}
    for nom in args.modeles:
        try:
            modeles[nom] = charger_modele(
                nom, historique_essai, choix_ets.get(config.LISTE_SERIES[0]))
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
            fonction = partial(importlib.import_module("src.modeles.ets").prevoir_prix, **choix_ets[serie]) if nom == "ets" else prevoir
            morceaux.append(evaluer_serie(nom, fonction, serie, prix, paires))
            futurs.append(prevoir_futur(nom, fonction, serie, prix))
            print(f"  {nom:<7} {serie:<10} {paires['date_origine'].nunique()} origines "
                  f"({time.perf_counter() - debut_serie:.1f} s)")
    previsions = pd.concat(morceaux, ignore_index=True)
    previsions_futures = pd.concat(futurs, ignore_index=True)

    # Métriques par période (et test_global), calculées avant l'arrondi des prix
    metriques = calculer_metriques(previsions)
    metriques = metriques.round({c: DECIMALES_METRIQUES for c in
                                 ["mae", "rmse", "mape", "couverture80", "couverture95"]})
    ordre_periodes = {periode: rang for rang, periode in enumerate(config.PERIODES_METRIQUES)}

    fichiers = {
        config.FICHIER_PREVISIONS: (
            mettre_en_forme(previsions, config.COLONNES_PREVISIONS,
                            ["serie", "modele", "date_cible", "horizon"]),
            config.COLONNES_PREVISIONS, config.PERIODES_PREVISIONS),
        config.FICHIER_PREVISIONS_FUTURES: (
            mettre_en_forme(previsions_futures, config.COLONNES_PREVISIONS_FUTURES,
                            ["serie", "modele", "horizon"]),
            config.COLONNES_PREVISIONS_FUTURES, None),
        config.FICHIER_METRIQUES: (
            metriques.sort_values(["serie", "modele", "periode", "horizon"],
                                  key=lambda c: c.map(ordre_periodes) if c.name == "periode" else c)
            .reset_index(drop=True),
            config.COLONNES_METRIQUES, config.PERIODES_METRIQUES),
    }
    print("\nÉcriture")
    ecrire(fichiers)

    if FICHIER_FACTICE.exists():
        FICHIER_FACTICE.unlink()
        print(f"  Supprimé : resultats/{FICHIER_FACTICE.name}")

    print("\nRésumé")
    print("  Modèles exécutés :", ", ".join(modeles))
    print("  Modèles ignorés :", ", ".join(ignores) or "aucun")
    for periode, nombre in previsions.groupby("periode").size().reindex(config.PERIODES_PREVISIONS).items():
        print(f"  Prévisions ({periode}) : {int(nombre)}")
    print(f"  Prévisions futures : {len(previsions_futures)}")
    print(f"  Durée : {time.perf_counter() - debut_chrono:.1f} s")


if __name__ == "__main__":
    main()
