"""Évalue les ETS sélectionnés sur la validation, sans utiliser le test."""

import json

import numpy as np
import pandas as pd

from src import config
from src.evaluation import (
    calculer_metriques,
    generer_paires_origine_cible,
)
from src.modeles.ets import ajuster_ets, prevoir_ets


# Choix obtenus par comparaison des six configurations en validation.
CHOIX_ETS = {
    "mais_QC": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": True,
    },
    "mais_ON": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": False,
    },
    "canola_SK": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": True,
    },
    "canola_AB": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": True,
    },
}


def main():
    donnees = pd.read_csv(
        config.FICHIER_SERIES,
        parse_dates=["date"],
    )

    blocs = [
        generer_paires_origine_cible(debut, fin)
        for debut, fin in config.PERIODES_VALIDATION
    ]
    paires = pd.concat(blocs, ignore_index=True)
    groupes = list(paires.groupby("date_origine"))

    # Dossier propre à ETS : préserver les résultats communs de l'équipe.
    dossier = config.DOSSIER_RESULTATS / "ets"
    dossier.mkdir(parents=True, exist_ok=True)

    with (dossier / "configurations_retenues.json").open(
        "w", encoding="utf-8"
    ) as fichier:
        json.dump(CHOIX_ETS, fichier, indent=4, ensure_ascii=False)

    lignes = []

    for serie in config.LISTE_SERIES:
        print(f"\nSérie : {serie}", flush=True)

        prix = (
            donnees.loc[donnees["serie"].eq(serie)]
            .set_index("date")["prix"]
            .sort_index()
        )

        if prix.empty or prix.index.has_duplicates:
            raise ValueError(f"Série absente ou dates en double : {serie}")

        prix = prix.asfreq("MS")

        if (
            not np.isfinite(prix.to_numpy(dtype=float)).all()
            or (prix <= 0).any()
        ):
            raise ValueError(f"Prix manquants ou invalides : {serie}")

        for numero, (origine, groupe) in enumerate(groupes, start=1):
            entrainement = prix.loc[
                config.DATE_DEBUT_ENTRAINEMENT:origine
            ]

            resultat = ajuster_ets(
                entrainement,
                **CHOIX_ETS[serie],
            )

            if not resultat.mle_retvals.get("converged", False):
                raise RuntimeError(
                    f"Estimation non convergée : {serie}, {origine}"
                )

            previsions = prevoir_ets(
                resultat,
                pas=int(groupe["pas"].max()),
            )

            if not np.isfinite(
                previsions.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Prévisions invalides : {serie}, {origine}"
                )

            for paire in groupe.itertuples(index=False):
                # Le premier pas se trouve à la position zéro.
                estimation = previsions.iloc[int(paire.pas) - 1]

                lignes.append({
                    "serie": serie,
                    "modele": "ets",
                    "date_origine": origine,
                    "date_cible": paire.date_cible,
                    "horizon": paire.horizon,
                    "prevision": estimation["prevision"],
                    "inf80": estimation["inf80"],
                    "sup80": estimation["sup80"],
                    "inf95": estimation["inf95"],
                    "sup95": estimation["sup95"],
                    "reel": prix.loc[paire.date_cible],
                })

            if numero % 12 == 0 or numero == len(groupes):
                print(
                    f"  {numero}/{len(groupes)} origines traitées",
                    flush=True,
                )

    previsions = pd.DataFrame(
        lignes,
        columns=config.COLONNES_PREVISIONS,
    )

    if len(previsions) != len(paires) * len(config.LISTE_SERIES):
        raise RuntimeError("Nombre de prévisions incohérent.")

    metriques = calculer_metriques(previsions)

    # Conserver toute la précision dans les CSV.
    previsions.to_csv(
        dossier / "previsions_validation.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    metriques.to_csv(
        dossier / "metriques_validation.csv",
        index=False,
    )

    print("\nMétriques de validation des configurations retenues :")
    print(metriques.round(3).to_string(index=False))
    print(f"\nFichiers enregistrés dans : {dossier}")
    print("Le test final n'a pas été évalué.")


if __name__ == "__main__":
    main()