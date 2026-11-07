"""Page d'accueil de l'application Streamlit (point d'entrée : streamlit run app.py).

Tableau de bord : date de la dernière donnée, cartes des 4 séries (dernier prix,
variations sur 1 et 12 mois, indicateur d'alerte) et résumé chiffré des alertes.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ic.configurer_page("Tableau de bord")
series = ic.exiger_series()
ic.panneau_lateral()

st.title("Suivi et prévision des prix agricoles")
st.caption(f"Maïs-grain (Québec, Ontario) et canola (Saskatchewan, Alberta) – Statistique Canada, "
           f"tableau {config.TABLEAU_STATCAN} ({config.UNITE_PRIX})")

# ---------------------------------------------------------------------------
# Dernière donnée disponible
# ---------------------------------------------------------------------------
derniere_date = series["date"].max()
st.markdown(f"## Dernière donnée disponible : {ic.formater_mois(derniere_date)}")
st.info(
    f"Statistique Canada publie ces prix avec un délai d'environ "
    f"{config.DELAI_PUBLICATION} mois. Les prévisions « à 1, 3 et 6 mois » portent "
    f"donc sur {', '.join(str(p) for p in config.PAS_PAR_HORIZON.values())} mois "
    f"après la dernière donnée.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Alertes du modèle final
# ---------------------------------------------------------------------------
previsions = ic.lire_resultat("previsions")
finaux = ic.modeles_finaux()
if previsions is not None and finaux:
    alertes = ic.calculer_alertes(previsions, finaux)
    en_cours = ic.alertes_en_cours(alertes, series)
else:
    alertes = en_cours = None


def variation(prix, mois):
    """Variation relative entre le dernier prix et celui d'il y a « mois » mois."""
    derniere = prix.index.max()
    reference = derniere - pd.DateOffset(months=mois)
    if reference not in prix.index:
        return float("nan")
    return prix[derniere] / prix[reference] - 1


# ---------------------------------------------------------------------------
# Cartes : une ligne par produit, une colonne par province suivie
# ---------------------------------------------------------------------------
st.markdown("## Situation par série")
for produit, provinces in config.PROVINCES_PAR_PRODUIT.items():
    st.markdown(f"### {config.LIBELLES_PRODUITS[produit]}")
    colonnes = st.columns(len(provinces))
    for colonne, province in zip(colonnes, provinces):
        serie = f"{produit}_{province}"
        prix = series.loc[series["serie"] == serie].set_index("date")["prix"]
        with colonne.container(border=True):
            st.markdown(f"**{config.LIBELLES_PROVINCES[province]}**")
            if prix.empty:
                st.caption("Aucune donnée.")
                continue
            derniere = prix.index.max()
            var_1 = variation(prix, 1)
            st.metric(
                label=f"Dernier prix ({ic.formater_mois(derniere)})",
                value=ic.formater_prix(prix[derniere]),
                delta=f"{ic.formater_pct(var_1)} sur 1 mois" if pd.notna(var_1) else None,
                delta_color="off",
            )
            st.caption(f"Sur 12 mois : {ic.formater_pct(variation(prix, 12))}")

            if en_cours is None:
                st.markdown("Alerte : indisponible (aucun résultat)")
            elif serie in set(en_cours["serie"]):
                sens = en_cours.loc[en_cours["serie"] == serie, "sens"].iloc[0]
                st.markdown(f":red[**⚠ Alerte** : prix {sens} de l'intervalle à {ic.NIVEAU_ALERTE_PCT} %]")
            else:
                st.markdown("✓ Aucune alerte")

# ---------------------------------------------------------------------------
# Résumé chiffré des alertes
# ---------------------------------------------------------------------------
st.markdown("## Alertes")
if alertes is None:
    st.caption("Aucun résultat de prévision disponible dans resultats/.")
else:
    st.caption(
        f"Prix observé hors de l'intervalle à {ic.NIVEAU_ALERTE_PCT} % de la prévision à l'horizon "
        f"{config.HORIZON_ALERTE} mois, selon le modèle final de chaque série."
    )
    debut_12_mois = derniere_date - pd.DateOffset(months=11)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Séries en alerte ({ic.formater_mois(derniere_date)})",
              f"{en_cours['serie'].nunique()} sur {len(config.LISTE_SERIES)}")
    c2.metric("Alertes sur les 12 derniers mois",
              int((alertes["date_cible"] >= debut_12_mois).sum()))
    c3.metric("Alertes depuis le début du test", len(alertes))

    if not en_cours.empty:
        tableau = pd.DataFrame({
            "Série": en_cours["serie"].map(ic.libelle_serie),
            "Mois": en_cours["date_cible"].map(ic.formater_mois),
            "Modèle": en_cours["modele"].map(ic.libelle_modele),
            "Prix observé": en_cours["reel"].map(ic.formater_prix),
            f"Intervalle à {ic.NIVEAU_ALERTE_PCT} %": [
                f"{ic.formater_prix(a)} – {ic.formater_prix(b)}"
                for a, b in zip(en_cours[ic.INF_ALERTE], en_cours[ic.SUP_ALERTE])],
            "Sens": en_cours["sens"],
            "Écart à la borne": en_cours["ecart_pct"].map(ic.formater_pct),
        })
        st.dataframe(tableau, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Commentaire de l'équipe
# ---------------------------------------------------------------------------
st.markdown("## Commentaire")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")
