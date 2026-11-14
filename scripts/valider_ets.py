"""Compare six configurations ETS sur la validation des quatre séries."""

import numpy as np
import pandas as pd

from src import config
from src.evaluation import generer_paires_origine_cible
from src.modeles.ets import ajuster_ets


CONFIGURATIONS = {
    "sans_tendance": {
        "tendance": None,
        "saisonnalite": None,
        "amortie": False,
    },
    "tendance": {
        "tendance": "add",
        "saisonnalite": None,
        "amortie": False,
    },
    "tendance_amortie": {
        "tendance": "add",
        "saisonnalite": None,
        "amortie": True,
    },
    "saisonnalite": {
        "tendance": None,
        "saisonnalite": "add",
        "amortie": False,
    },
    "tendance_saisonnalite": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": False,
    },
    "tendance_amortie_saisonnalite": {
        "tendance": "add",
        "saisonnalite": "add",
        "amortie": True,
    },
}


def construire_paires_validation():
    """Construit les origines et les cibles des périodes communes."""
    blocs = []

    for numero, (debut, fin) in enumerate(
        config.PERIODES_VALIDATION, start=1
    ):
        paires = generer_paires_origine_cible(debut, fin)
        paires["bloc"] = numero
        blocs.append(paires)

    return pd.concat(blocs, ignore_index=True)


def valider_serie(donnees, serie, paires):
    """Compare les configurations pour une seule série."""
    print("\n" + "=" * 70, flush=True)
    print(f"SÉRIE : {serie}", flush=True)
    print("=" * 70, flush=True)

    prix = (
        donnees.loc[donnees["serie"].eq(serie)]
        .set_index("date")["prix"]
        .sort_index()
    )

    if prix.empty:
        raise ValueError(f"Aucune observation pour la série {serie}.")

    if prix.index.has_duplicates:
        raise ValueError(f"Dates en double pour la série {serie}.")

    prix = prix.asfreq("MS")

    if not np.isfinite(prix.to_numpy(dtype=float)).all():
        raise ValueError(f"Prix absents ou invalides pour {serie}.")

    if (prix <= 0).any():
        raise ValueError(f"Prix non positifs pour {serie}.")

    # Vérifier la présence des prix réels de toutes les cibles.
    if not paires["date_cible"].isin(prix.index).all():
        raise ValueError(f"Certaines dates cibles sont absentes pour {serie}.")

    groupes = list(paires.groupby("date_origine"))
    lignes = []
    echecs = []

    for nom, options in CONFIGURATIONS.items():
        print(f"\nConfiguration : {nom}", flush=True)
        resultats_configuration = []

        for numero, (origine, groupe) in enumerate(groupes, start=1):
            # Utiliser uniquement les données connues à cette origine.
            entrainement = prix.loc[
                config.DATE_DEBUT_ENTRAINEMENT:origine
            ]

            try:
                resultat = ajuster_ets(entrainement, **options)

                if not resultat.mle_retvals.get("converged", False):
                    raise RuntimeError("L'estimation n'a pas convergé.")

                pas_max = int(groupe["pas"].max())

                # Retour en dollars : prévisions médianes.
                previsions = np.exp(
                    np.asarray(resultat.forecast(steps=pas_max))
                )

                if not np.isfinite(previsions).all():
                    raise ValueError("Prévisions non finies.")

                for ligne in groupe.itertuples(index=False):
                    resultats_configuration.append({
                        "serie": serie,
                        "configuration": nom,
                        "bloc": ligne.bloc,
                        "horizon": ligne.horizon,
                        "date_origine": origine,
                        "date_cible": ligne.date_cible,
                        "prevision": previsions[int(ligne.pas) - 1],
                        "reel": prix.loc[ligne.date_cible],
                    })

            except (
                ValueError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as erreur:
                echecs.append({
                    "serie": serie,
                    "configuration": nom,
                    "origine": origine,
                    "raison": str(erreur),
                })

            if numero % 12 == 0 or numero == len(groupes):
                print(
                    f"  {numero}/{len(groupes)} origines traitées",
                    flush=True,
                )

        # Une configuration incomplète ne participe pas au classement.
        if len(resultats_configuration) == len(paires):
            lignes.extend(resultats_configuration)
        else:
            print(
                "  Configuration exclue du classement : "
                "certaines prévisions ont échoué.",
                flush=True,
            )

    if echecs:
        print(f"\nÉchecs d'estimation pour {serie} :")
        print(pd.DataFrame(echecs).to_string(index=False))

    if not lignes:
        print(
            f"\nAucune configuration complète pour {serie}.",
            flush=True,
        )
        return None

    comparaison = pd.DataFrame(lignes)

    comparaison["erreur_absolue"] = (
        comparaison["reel"] - comparaison["prevision"]
    ).abs()

    mae = comparaison.pivot_table(
        index="configuration",
        columns="horizon",
        values="erreur_absolue",
        aggfunc="mean",
    )

    # Vérifier les effectifs avant de comparer les configurations.
    effectifs = comparaison.groupby(
        ["configuration", "horizon"]
    ).size().unstack()

    attendu = paires.groupby("horizon").size()

    if not effectifs.eq(attendu, axis="columns").all().all():
        raise RuntimeError(
            f"Effectifs de validation incohérents pour {serie}."
        )

    mae["mae_moyenne"] = mae[config.HORIZONS].mean(axis=1)
    mae = mae.sort_values("mae_moyenne")

    print(f"\nMAE de validation pour {serie}, en dollars par tonne :")
    print(mae.round(2).to_string())

    print("\nNombre de prévisions par configuration et horizon :")
    print(effectifs.to_string())

    # Retenir la meilleure configuration de cette série.
    meilleure = mae.iloc[0]

    return {
        "serie": serie,
        "configuration": mae.index[0],
        **{
            f"mae_h{horizon}": meilleure[horizon]
            for horizon in config.HORIZONS
        },
        "mae_moyenne": meilleure["mae_moyenne"],
    }


def main():
    # Charger les données une seule fois pour les quatre séries.
    donnees = pd.read_csv(
        config.FICHIER_SERIES,
        parse_dates=["date"],
    )

    paires = construire_paires_validation()
    meilleures = []

    # Traiter successivement les quatre séries du projet.
    for serie in config.LISTE_SERIES:
        meilleure = valider_serie(donnees, serie, paires)

        if meilleure is not None:
            meilleures.append(meilleure)

    if meilleures:
        print("\n" + "=" * 70)
        print("MEILLEURE CONFIGURATION ETS PAR SÉRIE")
        print("=" * 70)

        resume = pd.DataFrame(meilleures).set_index("serie")
        print(resume.round(2).to_string())

    print(
        "\nValidation terminée. "
        "Le test final n'a pas été évalué.",
        flush=True,
    )


if __name__ == "__main__":
    main()