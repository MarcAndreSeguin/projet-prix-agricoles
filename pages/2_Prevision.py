"""Page Prévision : prévisions futures aux horizons 1, 3 et 6 mois avec intervalles.

- historique des 5 dernières années prolongé par la prévision du modèle choisi,
  avec la bande d'intervalle au niveau choisi (80 % ou 95 %) ;
- tableau des prévisions et de leurs dates cibles ;
- probabilité que le prix dépasse le dernier prix observé, pour chaque horizon ;
- export CSV des prévisions affichées.

Les prévisions sont lues dans resultats/previsions_futures.csv.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ANNEES_HISTORIQUE = 5
# La probabilité de dépassement est toujours déduite de l'intervalle à 80 %,
# pour qu'elle ne change pas avec le niveau affiché.
NIVEAU_PROBABILITE = int(round(config.NIVEAU_AFFICHAGE * 100))

ic.configurer_page("Prévision")
series = ic.exiger_series()
futures = ic.exiger_resultat("previsions_futures")
selection = ic.panneau_lateral()
libelle = ic.libelle_serie(selection.serie)

st.title("Prévision")
if selection.modele is None:
    st.error("Aucun modèle disponible pour cette série dans resultats/.")
    st.stop()

s = series[series["serie"] == selection.serie]
derniere_date = s["date"].max()
dernier_prix = s.loc[s["date"] == derniere_date, "prix"].iloc[0]
f = futures[(futures["serie"] == selection.serie) & (futures["modele"] == selection.modele)]
f = f.sort_values("horizon")

nom_modele = ic.libelle_modele(selection.modele)
if selection.modele == selection.modele_final:
    nom_modele += " – modèle final"
st.caption(f"{libelle} – {nom_modele} – intervalle à {selection.niveau} %")

if f.empty:
    st.error(f"Aucune prévision future pour {libelle} avec le modèle {nom_modele}.")
    st.stop()
origines = f["date_origine"].unique()
if len(origines) != 1 or pd.Timestamp(origines[0]) != derniere_date:
    st.warning(
        f"Les prévisions partent de {', '.join(ic.formater_mois(o) for o in origines)}, "
        f"alors que la dernière donnée de data/series.csv est {ic.formater_mois(derniere_date)}. "
        "Les résultats ne sont pas à jour."
    )

c1, c2 = st.columns(2)
c1.metric("Dernière donnée", ic.formater_mois(derniere_date))
c2.metric("Dernier prix observé", ic.formater_prix(dernier_prix))

# ---------------------------------------------------------------------------
# Graphique : historique récent + prévision
# ---------------------------------------------------------------------------
inf, sup = ic.colonnes_intervalle(selection.niveau)
debut = derniere_date - pd.DateOffset(years=ANNEES_HISTORIQUE)
historique = s[s["date"] > debut]
# Le dernier prix observé sert de point de départ à la courbe et à la bande.
depart = pd.DataFrame({"date_cible": [derniere_date], "prevision": [dernier_prix],
                       inf: [dernier_prix], sup: [dernier_prix]})
trace = pd.concat([depart, f[["date_cible", "prevision", inf, sup]]], ignore_index=True)
figure = ic.graphique_serie(historique, trace, niveau=selection.niveau, modele=selection.modele)
figure.add_vline(x=derniere_date, line=dict(color="#999999", width=1, dash="dot"))
ic.afficher_graphique(figure)
st.caption(f"Historique des {ANNEES_HISTORIQUE} dernières années ; la ligne pointillée verticale "
           "marque la dernière donnée disponible.")

# ---------------------------------------------------------------------------
# Tableau des prévisions
# ---------------------------------------------------------------------------
st.header("Prévisions par horizon")
inf_p, sup_p = ic.colonnes_intervalle(NIVEAU_PROBABILITE)
f = f.assign(
    pas=f["horizon"].map(config.PAS_PAR_HORIZON),
    probabilite=ic.probabilite_depassement(f["prevision"], f[inf_p], f[sup_p],
                                           dernier_prix, NIVEAU_PROBABILITE),
)
tableau = pd.DataFrame({
    "Horizon": f["horizon"].map(config.LIBELLES_HORIZONS),
    "Date cible": f["date_cible"].map(ic.formater_mois),
    "Mois après la dernière donnée": f["pas"],
    "Prévision": f["prevision"].map(ic.formater_prix),
    f"Borne inférieure ({selection.niveau} %)": f[inf].map(ic.formater_prix),
    f"Borne supérieure ({selection.niveau} %)": f[sup].map(ic.formater_prix),
    "P(prix > dernier prix)": f["probabilite"].map(lambda x: ic.formater_pct(x, 0, signe=False)),
})
st.dataframe(tableau, hide_index=True, use_container_width=True)
st.caption(
    f"P(prix > dernier prix) : probabilité que le prix à la date cible dépasse le dernier prix "
    f"observé ({ic.formater_prix(dernier_prix)}), déduite de l'intervalle à {NIVEAU_PROBABILITE} % "
    "en supposant une loi normale du log du prix. C'est une approximation qui dépend de la "
    "fiabilité des intervalles (voir la page Comparaison des modèles)."
)

# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------
export = pd.DataFrame({
    "serie": selection.serie,
    "modele": selection.modele,
    "date_origine": f["date_origine"].dt.strftime("%Y-%m-01"),
    "date_cible": f["date_cible"].dt.strftime("%Y-%m-01"),
    "horizon": f["horizon"],
    "prevision": f["prevision"],
    f"inf{selection.niveau}": f[inf],
    f"sup{selection.niveau}": f[sup],
    "prob_depassement_dernier_prix": f["probabilite"].round(4),
    "dernier_prix": dernier_prix,
})
# Point-virgule et virgule décimale : ouverture directe dans Excel en français.
contenu = export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
st.download_button(
    "Exporter ces prévisions (CSV)",
    data=contenu,
    file_name=f"previsions_{selection.serie}_{selection.modele}_{derniere_date:%Y-%m}.csv",
    mime="text/csv",
)
