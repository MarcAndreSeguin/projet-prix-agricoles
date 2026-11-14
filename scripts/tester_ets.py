"""Évalue les configurations ETS figées sur le test final."""

import json

import numpy as np
import pandas as pd

from src import config
from src.evaluation import (
    calculer_metriques,
    generer_paires_origine_cible,
)
from src.modeles.ets import ajuster_ets, prevoir_ets


def main():
    dossier = config.DOSSIER_RESULTATS / "ets"
    fichier_choix = dossier / "configurations_retenues.json"

    # Relire les choix enregistrés après la validation.
    with fichier_choix.open("r", encoding="utf-8") as fichier:
        choix = json.load(fichier)

    donnees = pd.read_csv(
        config.FICHIER_SERIES,
        parse_dates=["date"],
    )

    # Les mois cibles du test viennent de la configuration commune.
    paires = generer_paires_origine_cible(
        config.DATE_DEBUT_TEST,
        config.DATE_FIN_TEST,
    )
    groupes = list(paires.groupby("date_origine"))

    lignes = []

    for serie in config.LISTE_SERIES:
        print(f"\nTest final — série : {serie}", flush=True)

        prix = (
            donnees.loc[donnees["serie"].eq(serie)]
            .set_index("date")["prix"]
            .sort_index()
        )

        if prix.empty or prix.index.has_duplicates:
            raise ValueError(
                f"Série absente ou dates en double : {serie}"
            )

        prix = prix.asfreq("MS")

        if (
            not np.isfinite(prix.to_numpy(dtype=float)).all()
            or (prix <= 0).any()
        ):
            raise ValueError(f"Prix manquants ou invalides : {serie}")

        if not paires["date_cible"].isin(prix.index).all():
            raise ValueError(f"Dates cibles absentes : {serie}")

        for numero, (origine, groupe) in enumerate(groupes, start=1):
            # Historique croissant, limité aux données connues à l'origine.
            entrainement = prix.loc[
                config.DATE_DEBUT_ENTRAINEMENT:origine
            ]

            # Même configuration, paramètres réestimés à chaque origine.
            resultat = ajuster_ets(
                entrainement,
                **choix[serie],
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

    # Vérifier les effectifs de chaque série et horizon.
    attendu = paires.groupby("horizon").size()
    effectifs = previsions.groupby(
        ["serie", "horizon"]
    ).size().unstack()

    if not effectifs.eq(attendu, axis="columns").all().all():
        raise RuntimeError("Effectifs du test incohérents.")

    metriques = calculer_metriques(previsions)

    previsions.to_csv(
        dossier / "previsions_test.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    metriques.to_csv(
        dossier / "metriques_test.csv",
        index=False,
    )

    print("\nMétriques ETS sur le test final :")
    print(metriques.round(3).to_string(index=False))

    print(f"\nRésultats enregistrés dans : {dossier}")
    print("Les configurations ETS retenues n'ont pas été modifiées.")


if __name__ == "__main__":
    main()