"""Page Comparaison des modèles : métriques par horizon, couverture des intervalles et
tests de Diebold-Mariano, pour la série choisie et une période (test par défaut, ou
validation, où le modèle final est choisi). Les deux périodes ne sont jamais mélangées.

- tableau des métriques (MAE, RMSE, MAPE) par modèle et par horizon, meilleure valeur mise en évidence ;
- tableau des tests de Diebold-Mariano ;
- couverture réelle des intervalles contre la couverture visée ;
- prévisions contre valeurs réelles sur la période de test.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ic.configurer_page("Comparaison des modèles")
ic.exiger_series()
metriques = ic.exiger_resultat("metriques")
tests_dm = ic.exiger_resultat("tests_dm")
previsions = ic.exiger_resultat("previsions")
selection = ic.panneau_lateral(avec_modele=False, avec_niveau=False)
libelle = ic.libelle_serie(selection.serie)

st.title("Comparaison des modèles")
PERIODES = {
    "test": f"Test ({ic.formater_mois(config.DATE_DEBUT_TEST)} à "
            f"{ic.formater_mois(config.DATE_FIN_TEST)})",
    "validation": f"Validation ({len(config.PERIODES_VALIDATION)} blocs, "
                  f"{ic.formater_mois(config.PERIODES_VALIDATION[0][0])} à "
                  f"{ic.formater_mois(config.PERIODES_VALIDATION[-1][1])})",
}
periode = st.radio("Période évaluée", list(PERIODES), format_func=PERIODES.get, horizontal=True,
                   key="comparaison_periode")
m = metriques[(metriques["serie"] == selection.serie) & (metriques["periode"] == periode)]
if m.empty:
    st.error(f"Aucune métrique pour {libelle} sur cette période dans resultats/metriques.csv.")
    st.stop()
modeles = [x for x in config.MODELES if x in set(m["modele"])]
final = selection.modele_final
st.caption(f"{libelle} – {PERIODES[periode]}, mois cibles prévus par origine glissante – "
           f"modèle final (choisi sur la validation) : "
           f"{ic.libelle_modele(final) if final else 'indéterminé'}")


def nom_ligne(modele):
    return ic.libelle_modele(modele) + (" ★" if modele == final else "")


# ---------------------------------------------------------------------------
# Tableau des métriques
# ---------------------------------------------------------------------------
st.header("Erreurs de prévision")
METRIQUES = {"mae": "MAE ($)", "rmse": "RMSE ($)", "mape": "MAPE (%)"}
colonnes = {}
for horizon in config.HORIZONS:
    mh = m[m["horizon"] == horizon].set_index("modele")
    for code, nom in METRIQUES.items():
        colonnes[f"{nom} – {config.LIBELLES_HORIZONS[horizon]}"] = mh[code]
colonnes["MAE moyenne ($)"] = m.groupby("modele")["mae"].mean()
tableau = pd.DataFrame(colonnes).reindex(modeles)
tableau.index = [nom_ligne(x) for x in tableau.index]
tableau.index.name = "Modèle"

style = tableau.style.apply(ic.style_meilleur, axis=0).format(lambda v: ic.formater_nombre(v, 2))
st.dataframe(style, use_container_width=True)
n = m["n"]
st.caption(f"Meilleure valeur (la plus faible) de chaque colonne en gras sur fond bleu. "
           f"★ : modèle final selon la règle du protocole. "
           f"Nombre de prévisions évaluées par modèle et horizon : {int(n.min())} à {int(n.max())}.")

# ---------------------------------------------------------------------------
# Tests de Diebold-Mariano
# ---------------------------------------------------------------------------
st.header("Tests de Diebold-Mariano")
d = tests_dm[(tests_dm["serie"] == selection.serie) & (tests_dm["periode"] == periode)]
d = d.sort_values(["horizon", "modele_a", "modele_b"])
seulement_naif = st.checkbox("Afficher seulement les comparaisons avec le modèle naïf",
                             key="comparaison_naif")
if seulement_naif:
    d = d[(d["modele_a"] == config.MODELE_REFERENCE) | (d["modele_b"] == config.MODELE_REFERENCE)]
tableau_dm = pd.DataFrame({
    "Horizon": d["horizon"].map(config.LIBELLES_HORIZONS),
    "Modèle A": d["modele_a"].map(ic.libelle_modele),
    "Modèle B": d["modele_b"].map(ic.libelle_modele),
    "Statistique DM": d["stat_dm"].map(lambda v: ic.formater_nombre(v, 3)),
    "p-valeur": d["p_valeur"].map(lambda v: ic.formater_nombre(v, 4)),
    "Significatif (5 %)": (d["p_valeur"] < config.SEUIL_DM).map({True: "Oui", False: "Non"}),
    "Meilleur": d["meilleur"].map(lambda x: "aucun" if x == "aucun" else ic.libelle_modele(x)),
})
if d.empty:
    st.caption("Aucun test : au moins deux modèles sont nécessaires.")
else:
    st.dataframe(tableau_dm, hide_index=True, use_container_width=True)
st.caption("Perte absolue, correction de Harvey, Leybourne et Newbold. Statistique négative : "
           "le modèle A a une erreur absolue moyenne plus faible que le modèle B.")

# ---------------------------------------------------------------------------
# Couverture des intervalles
# ---------------------------------------------------------------------------
st.header("Couverture des intervalles")
colonnes_graphiques = st.columns(2)
for colonne_page, niveau in zip(colonnes_graphiques, config.NIVEAUX_INTERVALLE):
    pct = int(round(niveau * 100))
    donnees = m.assign(
        horizon_libelle=m["horizon"].map(config.LIBELLES_HORIZONS),
        couverture=m[f"couverture{pct}"] * 100,
        ordre=m["modele"].map({x: i for i, x in enumerate(modeles)}),
    ).sort_values(["ordre", "horizon"])
    figure = ic.graphique_barres(donnees, x="horizon_libelle", y="couverture", couleur="modele",
                                 titre=f"Intervalle à {pct} %", titre_y="Couverture réelle (%)",
                                 format_y=",.1f")
    figure.add_hline(y=pct, line=dict(color="#000000", dash="dash", width=1.5),
                     annotation_text=f"Visée : {pct} %", annotation_position="top left")
    figure.update_yaxes(range=[0, 105])
    with colonne_page:
        ic.afficher_graphique(figure)
st.caption("Couverture réelle : proportion des prix observés situés dans l'intervalle. "
           "La ligne pointillée indique la couverture visée.")

# ---------------------------------------------------------------------------
# Prévisions contre valeurs réelles
# ---------------------------------------------------------------------------
st.header("Prévisions et valeurs réelles sur la période évaluée")
c1, c2 = st.columns([1, 2])
horizon = c1.radio("Horizon", config.HORIZONS, format_func=config.LIBELLES_HORIZONS.get,
                   horizontal=True, key="comparaison_horizon")
modeles_affiches = c2.multiselect("Modèles", modeles, default=modeles,
                                  format_func=ic.libelle_modele, key="comparaison_modeles")
p = previsions[(previsions["serie"] == selection.serie) & (previsions["horizon"] == horizon)
               & (previsions["periode"] == periode) & previsions["reel"].notna()]
ic.afficher_graphique(ic.graphique_previsions_reel(p, modeles_affiches))
st.caption(f"Chaque point est la prévision faite {config.PAS_PAR_HORIZON[horizon]} mois avant la "
           "date cible (horizon de consultation "
           f"{config.LIBELLES_HORIZONS[horizon]}), comparée au prix observé à cette date.")

# ---------------------------------------------------------------------------
# Commentaire de l'équipe
# ---------------------------------------------------------------------------
st.header("Commentaire")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")
