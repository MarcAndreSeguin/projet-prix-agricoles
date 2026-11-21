"""Page Comparaison des modèles : métriques par horizon et couverture des intervalles,
pour la série choisie et une période (test global par défaut, test 1, test 2, test 3,
ou validation, où le modèle final est choisi). Validation et test ne sont jamais mélangés.

- tableau des métriques (MAE, RMSE, MAPE) par modèle et par horizon, meilleure valeur mise en évidence ;
- couverture réelle des intervalles contre la couverture visée ;
- MAE de chaque modèle par période de test (test 1, test 2, test 3), en tableau et en graphique ;
- prévisions contre valeurs réelles sur la période choisie.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ic.configurer_page("Comparaison des modèles")
ic.exiger_series()
metriques = ic.exiger_resultat("metriques")
previsions = ic.exiger_resultat("previsions")
selection = ic.panneau_lateral(avec_modele=False, avec_niveau=False)
libelle = ic.libelle_serie(selection.serie)

st.title("Comparaison des modèles")
PERIODES = {periode: ic.libelle_periode(periode) for periode in ic.PERIODES_AFFICHEES}
periode = st.selectbox("Période évaluée", list(PERIODES), format_func=PERIODES.get,
                       key="comparaison_periode")
metriques_serie = metriques[metriques["serie"] == selection.serie]
m = metriques_serie[metriques_serie["periode"] == periode]
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
           f"★ : modèle final selon la règle du protocole (plus faible MAE moyenne sur la "
           f"validation). Nombre de prévisions évaluées par modèle et horizon : "
           f"{int(n.min())} à {int(n.max())}.")

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
# MAE par période de test
# ---------------------------------------------------------------------------
st.header("MAE par période de test")
MOYENNE = "moyenne"
choix_horizons = [MOYENNE] + config.HORIZONS
horizon_mae = st.selectbox(
    "Horizon", choix_horizons, key="comparaison_horizon_mae",
    format_func=lambda h: "Moyenne des horizons" if h == MOYENNE else config.LIBELLES_HORIZONS[h],
)
mt = metriques_serie[metriques_serie["periode"].isin(list(config.PERIODES_TEST))]
if horizon_mae != MOYENNE:
    mt = mt[mt["horizon"] == horizon_mae]
mae_tests = mt.groupby(["modele", "periode"], as_index=False)["mae"].mean()
modeles_tests = [x for x in config.MODELES if x in set(mae_tests["modele"])]

tableau_tests = (mae_tests.pivot(index="modele", columns="periode", values="mae")
                 .reindex(index=modeles_tests, columns=list(config.PERIODES_TEST)))
tableau_tests.columns = [ic.libelle_periode(p) for p in tableau_tests.columns]
tableau_tests.index = [nom_ligne(x) for x in tableau_tests.index]
tableau_tests.index.name = "Modèle"
st.dataframe(tableau_tests.style.format(lambda v: ic.formater_nombre(v, 2)),
             use_container_width=True)

donnees_tests = mae_tests.assign(
    periode_libelle=mae_tests["periode"].map(lambda p: ic.libelle_periode(p, avec_dates=False)),
    ordre_modele=mae_tests["modele"].map({x: i for i, x in enumerate(modeles_tests)}),
    ordre_periode=mae_tests["periode"].map({p: i for i, p in enumerate(config.PERIODES_TEST)}),
).sort_values(["ordre_modele", "ordre_periode"])
figure = ic.graphique_barres(donnees_tests, x="periode_libelle", y="mae", couleur="modele",
                             titre_y="MAE ($/tonne)")
ic.afficher_graphique(figure)
st.caption("MAE en dollars par tonne, calculée séparément sur chaque période de test de 24 mois "
           + ("(moyenne des MAE aux trois horizons)." if horizon_mae == MOYENNE
              else f"(horizon {config.LIBELLES_HORIZONS[horizon_mae]}).")
           + " Les périodes : " + " ; ".join(ic.libelle_periode(p) for p in config.PERIODES_TEST)
           + ".")

# ---------------------------------------------------------------------------
# Prévisions contre valeurs réelles
# ---------------------------------------------------------------------------
st.header("Prévisions et valeurs réelles sur la période évaluée")
c1, c2 = st.columns([1, 2])
horizon = c1.radio("Horizon", config.HORIZONS, format_func=config.LIBELLES_HORIZONS.get,
                   horizontal=True, key="comparaison_horizon")
modeles_affiches = c2.multiselect("Modèles", modeles, default=modeles,
                                  format_func=ic.libelle_modele, key="comparaison_modeles")
p = ic.filtrer_periode(previsions, periode)
p = p[(p["serie"] == selection.serie) & (p["horizon"] == horizon) & p["reel"].notna()]
ic.afficher_graphique(ic.graphique_previsions_reel(p, modeles_affiches))
st.caption(f"Chaque point est la prévision faite {config.PAS_PAR_HORIZON[horizon]} mois avant la "
           "date cible (horizon de consultation "
           f"{config.LIBELLES_HORIZONS[horizon]}), comparée au prix observé à cette date.")
