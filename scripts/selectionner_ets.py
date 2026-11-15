"""Sélectionne ETS sur la validation uniquement ; ne calcule aucun test.

Usage : python -m scripts.selectionner_ets
Sorties dans resultats/ets_selection/, sans modifier les CSV communs.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.evaluation import generer_paires_evaluation
from src.modeles.ets import CONFIGURATIONS, ajuster_log


def main():
    donnees = pd.read_csv(config.FICHIER_SERIES, parse_dates=["date"])
    paires = generer_paires_evaluation()
    paires = paires.loc[paires.periode.eq(config.PERIODE_VALIDATION)].copy()
    if paires.empty:
        raise ValueError("Aucune paire de validation.")
    groupes = list(paires.groupby("date_origine"))
    tableaux, erreurs, classements, choix = [], [], [], {}
    for serie in config.LISTE_SERIES:
        prix = donnees.loc[donnees.serie.eq(serie)].set_index("date")["prix"].sort_index()
        if prix.empty or prix.index.has_duplicates:
            raise ValueError(f"Série absente ou dupliquée : {serie}")
        prix = prix.asfreq("MS")
        if not np.isfinite(prix.to_numpy()).all() or not prix.gt(0).all():
            raise ValueError(f"Prix invalides : {serie}")
        if not paires.date_cible.isin(prix.index).all():
            raise ValueError(f"Mois cibles absents : {serie}")
        lignes = []
        for nom in CONFIGURATIONS:
            print(f"{serie} — {nom}", flush=True)
            candidat, echec = [], False
            for numero, (origine, groupe) in enumerate(groupes, start=1):
                try:
                    if origine not in prix.index:
                        raise ValueError("Origine absente de l'historique")
                    historique = np.log(prix.loc[config.DATE_DEBUT_ENTRAINEMENT:origine])
                    resultat = ajuster_log(historique, nom)
                    predictions = np.exp(np.asarray(resultat.forecast(int(groupe.pas.max()))))
                    if not np.isfinite(predictions).all():
                        raise ValueError("Prévisions non finies")
                    for paire in groupe.itertuples(index=False):
                        candidat.append(dict(serie=serie, configuration=nom, date_origine=origine,
                            date_cible=paire.date_cible, horizon=paire.horizon,
                            prevision=predictions[int(paire.pas)-1], reel=prix.loc[paire.date_cible]))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    erreurs.append(dict(serie=serie, configuration=nom, origine=origine, raison=str(exc)))
                    echec = True
                    break
                if numero % 24 == 0 or numero == len(groupes):
                    print(f"  {numero}/{len(groupes)} origines", flush=True)
            if not echec and len(candidat) == len(paires):
                lignes.extend(candidat)
        if not lignes:
            raise RuntimeError(f"Aucune configuration complète pour {serie}")
        tableau = pd.DataFrame(lignes)
        tableau["erreur_absolue"] = (tableau.reel - tableau.prevision).abs()
        classement = tableau.pivot_table(index="configuration", columns="horizon", values="erreur_absolue", aggfunc="mean")
        classement["mae_moyenne"] = classement[config.HORIZONS].mean(axis=1)
        # L'ordre déclaré départage seulement les égalités exactes.
        classement["rang"] = [list(CONFIGURATIONS).index(x) for x in classement.index]
        classement = classement.sort_values(["mae_moyenne", "rang"]).drop(columns="rang")
        print(classement.round(4).to_string(), flush=True)
        choix[serie] = classement.index[0]
        classements.append(classement.reset_index().assign(serie=serie))
        tableaux.append(tableau)
    dossier = config.DOSSIER_RESULTATS / "ets_selection"
    dossier.mkdir(parents=True, exist_ok=True)
    pd.concat(tableaux).to_csv(dossier / "previsions_validation_configurations.csv", index=False, date_format="%Y-%m-%d")
    pd.concat(classements).to_csv(dossier / "classement_validation.csv", index=False)
    pd.DataFrame(erreurs, columns=["serie", "configuration", "origine", "raison"]).to_csv(dossier / "echecs.csv", index=False)
    document = {"empreinte_series": hashlib.sha256(config.FICHIER_SERIES.read_bytes()).hexdigest(),
        "choix": choix, "periodes_validation": config.PERIODES_VALIDATION,
        "horizons": config.HORIZONS, "delai_publication": config.DELAI_PUBLICATION,
        "date_debut_entrainement": config.DATE_DEBUT_ENTRAINEMENT,
        "configurations_candidates": CONFIGURATIONS,
        "critere": "MAE moyenne non arrondie sur les horizons de validation"}
    (dossier / "configurations_retenues.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nChoix retenus :", choix)
    print("Aucun test évalué. Résultats :", dossier)

if __name__ == "__main__":
    main()
