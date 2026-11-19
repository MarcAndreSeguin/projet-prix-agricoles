from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

racine = Path(__file__).resolve().parents[1]
sortie = racine / "rapport" / "figures"
sortie.mkdir(parents=True, exist_ok=True)

donnees = pd.read_csv(
    racine / "resultats" / "previsions.csv",
    parse_dates=["date_cible"],
)

series = {
    "canola_SK": ("Canola en Saskatchewan", "canola"),
    "mais_ON": ("Maïs-grain en Ontario", "mais"),
}

for serie, (titre, nom_fichier) in series.items():
    p = donnees[
        donnees["serie"].eq(serie)
        & donnees["modele"].eq("ets")
        & donnees["periode"].isin(["test_1", "test_2", "test_3"])
    ].copy()

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 11), sharex=True, sharey=True
    )

    for ax, horizon in zip(axes, [1, 3, 6]):
        t = p[p["horizon"].eq(horizon)].sort_values("date_cible")

        if len(t) != 72 or t["date_cible"].duplicated().any():
            raise ValueError(
                f"Données incomplètes ou dupliquées : {serie}, horizon {horizon}"
            )

        dates = t["date_cible"]
        hors = (t["reel"] < t["inf95"]) | (t["reel"] > t["sup95"])

        ax.fill_between(
            dates,
            t["inf95"].to_numpy(),
            t["sup95"].to_numpy(),
            color="steelblue",
            alpha=0.2,
            label="Intervalle à 95 %",
        )
        ax.plot(
            dates, t["reel"],
            color="black", label="Prix observé",
        )
        ax.plot(
            dates, t["prevision"],
            color="darkorange", label="Prévision ETS",
        )
        ax.scatter(
            dates[hors], t.loc[hors, "reel"],
            color="crimson", s=18, zorder=3,
            label="Observation hors intervalle",
        )

        for limite in ["2022-09-01", "2024-09-01"]:
            ax.axvline(
                pd.Timestamp(limite),
                color="gray", linestyle="--",
            )

        for milieu, etiquette in [
            ("2021-09-01", "Test 1"),
            ("2023-09-01", "Test 2"),
            ("2025-09-01", "Test 3"),
        ]:
            ax.text(
                pd.Timestamp(milieu), 0.95, etiquette,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="dimgray",
            )

        couverture = 100 * (~hors).mean()
        ax.set_title(
            f"Horizon : {horizon} mois - "
            f"couverture à 95 % : {couverture:.1f} %"
        )
        ax.set_ylabel("Prix ($ CA/tonne)")
        ax.grid(alpha=0.2)

    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Mois cible")

    fig.suptitle(f"{titre} - prévisions ETS et intervalles")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.text(
        0.5, 0.01,
        "Horizons de consultation : 1, 3 et 6 mois ; "
        "pas de prévision avec délai de publication : 3, 5 et 8 mois.",
        ha="center", fontsize=9,
    )

    for extension in ["png", "pdf"]:
        fichier = sortie / f"{nom_fichier}_ets_intervalles.{extension}"
        fig.savefig(fichier, dpi=300, bbox_inches="tight")
        print(f"Figure enregistrée : {fichier}")

    plt.close(fig)