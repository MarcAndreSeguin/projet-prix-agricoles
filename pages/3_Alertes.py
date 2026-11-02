"""Page Alertes : prix observés hors de l'intervalle à 95 % de la prévision à l'horizon 1.

- graphique des prix observés sur la période de test, avec l'intervalle à 95 %
  de la prévision à l'horizon 1 et les épisodes hors intervalle marqués ;
- tableau des alertes (série, date, prix observé, borne dépassée, écart, sens),
  filtrable par série.

Avec « Modèle final » dans le panneau latéral, chaque série utilise son propre
modèle final ; sinon, le modèle choisi est appliqué à toutes les séries.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ic.configurer_page("Alertes")
series = ic.exiger_series()
previsions = ic.exiger_resultat("previsions")
selection = ic.panneau_lateral(avec_niveau=False)

st.title("Alertes")
st.caption(f"Alerte : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon "
           f"{config.HORIZON_ALERTE} mois ({config.PAS_PAR_HORIZON[config.HORIZON_ALERTE]} mois "
           "après la dernière donnée connue à l'origine).")
if selection.modele is None:
    st.error("Aucun modèle disponible dans resultats/.")
    st.stop()

if selection.choix_final:
    modeles_par_serie = ic.modeles_finaux()
    description_modele = "modèle final de chaque série"
else:
    modeles_par_serie = {serie: selection.modele for serie in config.LISTE_SERIES}
    description_modele = ic.libelle_modele(selection.modele)
alertes = ic.calculer_alertes(previsions, modeles_par_serie)

# ---------------------------------------------------------------------------
# Graphique de la série choisie
# ---------------------------------------------------------------------------
libelle = ic.libelle_serie(selection.serie)
modele = modeles_par_serie.get(selection.serie)
st.header(libelle)
h1 = previsions[(previsions["serie"] == selection.serie) & (previsions["modele"] == modele)
                & (previsions["horizon"] == config.HORIZON_ALERTE)].sort_values("date_cible")
s = series[(series["serie"] == selection.serie) & (series["date"] >= h1["date_cible"].min())]
alertes_serie = alertes[alertes["serie"] == selection.serie]

figure = ic.graphique_serie(s, h1, niveau=95, modele=modele, alertes=alertes_serie)
figure.update_traces(selector=dict(name=ic.libelle_modele(modele)), mode="lines",
                     name=f"Prévision à {config.HORIZON_ALERTE} mois ({ic.libelle_modele(modele)})")
ic.afficher_graphique(figure)

c1, c2, c3 = st.columns(3)
c1.metric("Mois évalués", len(h1))
c2.metric("Alertes", len(alertes_serie))
c3.metric("Proportion hors intervalle",
          ic.formater_pct(len(alertes_serie) / len(h1), signe=False) if len(h1) else "–")
st.caption(f"Modèle : {ic.libelle_modele(modele)}. Les croix marquent les mois en alerte. "
           f"Proportion attendue hors de l'intervalle à 95 % : environ 5 %.")

# ---------------------------------------------------------------------------
# Tableau des alertes
# ---------------------------------------------------------------------------
st.header("Tableau des alertes")
filtre = st.multiselect("Séries", config.LISTE_SERIES, default=config.LISTE_SERIES,
                        format_func=ic.libelle_serie, key="alertes_filtre")
choisies = alertes[alertes["serie"].isin(filtre)]
st.caption(f"{len(choisies)} alerte(s) – {description_modele}.")

if not choisies.empty:
    au_dessus = choisies["sens"] == "au-dessus"
    borne = choisies["sup95"].where(au_dessus, choisies["inf95"])
    tableau = pd.DataFrame({
        "Série": choisies["serie"].map(ic.libelle_serie),
        "Date": choisies["date_cible"].map(ic.formater_mois),
        "Modèle": choisies["modele"].map(ic.libelle_modele),
        "Prix observé": choisies["reel"].map(ic.formater_prix),
        "Borne dépassée": [f"{'supérieure' if h else 'inférieure'} : {ic.formater_prix(b)}"
                           for h, b in zip(au_dessus, borne)],
        "Écart à la borne": choisies["ecart_pct"].map(ic.formater_pct),
        "Sens": au_dessus.map({True: "▲ Hausse", False: "▼ Baisse"}),
    })
    st.dataframe(tableau, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Commentaire de l'équipe
# ---------------------------------------------------------------------------
st.header("Commentaire")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")
