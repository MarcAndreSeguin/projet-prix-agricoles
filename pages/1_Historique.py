"""Page Historique : évolution des prix observés par série.

- courbe des prix depuis 1985, avec comparaison entre provinces ou entre toutes les séries,
  périodes de validation et de test ombrées et valeur estimée marquée ;
- décomposition STL (tendance, saisonnalité, résidu) de la série choisie ;
- profil saisonnier moyen (prix moyen par mois de l'année) ;
- statistiques descriptives (série choisie ou toutes les séries), hors valeur estimée.

La décomposition STL est descriptive ; elle ne produit aucune prévision.
"""

import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import STL

from src import config
from src import interface_commune as ic

ic.configurer_page("Historique")
series = ic.exiger_series()
selection = ic.panneau_lateral(avec_modele=False, avec_niveau=False)
libelle = ic.libelle_serie(selection.serie)
s = series[series["serie"] == selection.serie]

st.title("Historique des prix")
st.caption(f"{libelle} – {ic.formater_mois(s['date'].min())} à {ic.formater_mois(s['date'].max())} "
           f"({len(s)} mois, {config.UNITE_PRIX})")

# ---------------------------------------------------------------------------
# Courbe des prix et comparaison
# ---------------------------------------------------------------------------
st.header("Évolution des prix")
COMPARAISONS = {
    "aucune": "Série choisie seulement",
    "provinces": f"Entre provinces ({config.LIBELLES_PRODUITS[selection.produit]})",
    "toutes": "Toutes les séries",
}
comparaison = st.radio("Comparer", list(COMPARAISONS), format_func=COMPARAISONS.get,
                       horizontal=True, key="historique_comparaison")
if comparaison == "provinces":
    liste = [f"{selection.produit}_{p}" for p in config.PROVINCES_PAR_PRODUIT[selection.produit]]
elif comparaison == "toutes":
    liste = list(config.LISTE_SERIES)
else:
    liste = [selection.serie]
liste = [serie for serie in liste if serie in set(series["serie"])]

figure = ic.graphique_comparaison(series, liste)
ic.ajouter_periodes(figure)
ic.ajouter_valeurs_estimees(figure, series, liste)
ic.afficher_graphique(figure)
validations = " ; ".join(f"{ic.formater_mois(d)} à {ic.formater_mois(f)}"
                         for d, f in config.PERIODES_VALIDATION)
st.caption(f"Bandes grises : {len(config.PERIODES_VALIDATION)} blocs de validation de 12 mois "
           f"({validations}). Bande orange : test final, de "
           f"{ic.formater_mois(config.DATE_DEBUT_TEST)} à {ic.formater_mois(config.DATE_FIN_TEST)}.")
if series.loc[series["serie"].isin(liste), "prix_impute"].any():
    estimes = series[series["serie"].isin(liste) & series["prix_impute"]]
    st.caption("Losange : valeur estimée par interpolation linéaire (absente de la source) – "
               + ", ".join(f"{ic.libelle_serie(r.serie)}, {ic.formater_mois(r.date)}"
                           for r in estimes.itertuples()) + ".")

# ---------------------------------------------------------------------------
# Décomposition STL
# ---------------------------------------------------------------------------
st.header("Décomposition STL")


@st.cache_data(show_spinner=False)
def decomposer(dates, prix):
    """Décomposition STL robuste (période de 12 mois) des prix en dollars."""
    serie = pd.Series(prix, index=pd.DatetimeIndex(dates, freq="MS"))
    resultat = STL(serie, period=12, robust=True).fit()
    return pd.DataFrame({
        "date": serie.index, "observe": serie.to_numpy(), "tendance": resultat.trend.to_numpy(),
        "saisonnier": resultat.seasonal.to_numpy(), "residu": resultat.resid.to_numpy(),
    })


decomposition = decomposer(s["date"].to_numpy(), s["prix"].to_numpy())
ic.afficher_graphique(ic.graphique_decomposition(decomposition))
st.caption("STL robuste, période de 12 mois, appliquée aux prix en dollars : "
           "observé = tendance + saisonnalité + résidu.")

# ---------------------------------------------------------------------------
# Profil saisonnier
# ---------------------------------------------------------------------------
st.header("Profil saisonnier moyen")
annee_min, annee_max = int(s["date"].dt.year.min()), int(s["date"].dt.year.max())
col_periode, col_mesure = st.columns(2)
debut, fin = col_periode.slider("Années incluses", annee_min, annee_max, (annee_min, annee_max),
                                key="historique_annees")
MESURES = {
    "prix": f"Prix moyen ({config.UNITE_PRIX})",
    "ecart": "Écart moyen à la moyenne de l'année (%)",
}
mesure = col_mesure.radio("Mesure", list(MESURES), format_func=MESURES.get,
                          key="historique_mesure")

p = s[(s["date"].dt.year >= debut) & (s["date"].dt.year <= fin)].copy()
p["mois"] = p["date"].dt.month
if mesure == "prix":
    p["valeur"] = p["prix"]
    titre_y, format_y = config.UNITE_PRIX, ",.2f"
else:
    p["valeur"] = p["prix"] / p.groupby(p["date"].dt.year)["prix"].transform("mean") * 100 - 100
    titre_y, format_y = "%", "+,.1f"
profil = p.groupby("mois")["valeur"].agg(["mean", "count"]).reindex(range(1, 13)).reset_index()
profil["libelle_mois"] = [ic.MOIS_COURTS_FR[m - 1] for m in profil["mois"]]

figure = ic.graphique_barres(profil, x="libelle_mois", y="mean", titre_y=titre_y, format_y=format_y)
figure.update_traces(name=MESURES[mesure], showlegend=False)
figure.update_xaxes(categoryorder="array", categoryarray=ic.MOIS_COURTS_FR)
ic.afficher_graphique(figure)
st.caption(f"Moyenne par mois de l'année, {debut} à {fin} "
           f"({int(profil['count'].min())} à {int(profil['count'].max())} observations par mois).")

# ---------------------------------------------------------------------------
# Statistiques descriptives
# ---------------------------------------------------------------------------
st.header("Statistiques descriptives")
PORTEES = {"serie": f"Série choisie ({libelle})", "toutes": "Toutes les séries"}
portee = st.radio("Séries", list(PORTEES), format_func=PORTEES.get, horizontal=True,
                  key="historique_statistiques")
liste_statistiques = [selection.serie] if portee == "serie" else config.LISTE_SERIES
statistiques = ic.statistiques_descriptives(series, liste_statistiques)
st.dataframe(ic.tableau_statistiques(statistiques), hide_index=True, use_container_width=True)
st.caption(f"Prix en {config.UNITE_PRIX}, de {ic.formater_mois(series['date'].min())} à "
           f"{ic.formater_mois(series['date'].max())}, hors valeur estimée (même calcul que "
           "notebooks/01_pretraitement.ipynb).")

# ---------------------------------------------------------------------------
# Commentaire de l'équipe
# ---------------------------------------------------------------------------
st.header("Commentaire")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")
