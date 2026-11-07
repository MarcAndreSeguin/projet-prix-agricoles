"""Page Méthodologie et limites : description des données, des modèles et des limites de l'outil.

Le contenu descriptif reprend docs/protocole.md ; les sections « Limites » et
« Recommandations » sont rédigées par l'équipe.
"""

import pandas as pd
import streamlit as st

from src import config
from src import interface_commune as ic

ic.configurer_page("Méthodologie et limites")
ic.panneau_lateral(avec_modele=False, avec_niveau=False)
series = ic.lire_series()

st.title("Méthodologie et limites")

# ---------------------------------------------------------------------------
# Source des données
# ---------------------------------------------------------------------------
st.header("Source des données")
if series is not None:
    periode = f"{ic.formater_mois(series['date'].min())} à {ic.formater_mois(series['date'].max())}"
else:
    periode = (f"{ic.formater_mois(config.DATE_DEBUT)} à {ic.formater_mois(config.DATE_FIN)} "
               "(selon le protocole ; data/series.csv absent)")
st.markdown(f"""
- **Source** : Statistique Canada, tableau {config.TABLEAU_STATCAN},
  *Prix des produits agricoles, mensuel*.
- **Licence** : Licence du gouvernement ouvert – Canada.
- **Unité** : dollars par tonne métrique ({config.UNITE_PRIX}), prix nominaux.
- **Période utilisée** : {periode}.
- **Délai de publication** : environ {config.DELAI_PUBLICATION} mois entre le mois de référence
  et la publication.
""")

# ---------------------------------------------------------------------------
# Définition des séries
# ---------------------------------------------------------------------------
st.header("Définition des séries")
definitions = pd.DataFrame({
    "Série": [ic.libelle_serie(s) for s in config.LISTE_SERIES],
    "Code": config.LISTE_SERIES,
    "Produit dans le tableau de Statistique Canada": [config.SERIES[s]["produit_statcan"]
                                                       for s in config.LISTE_SERIES],
    "Géographie": [config.SERIES[s]["province"] for s in config.LISTE_SERIES],
})
st.dataframe(definitions, hide_index=True, use_container_width=True)
st.markdown(
    "Le canola n'est retenu que pour la Saskatchewan et l'Alberta, ce qui permet une période "
    "commune aux quatre séries à partir de janvier 1985 (au Québec, la série du canola ne "
    "commence qu'en 2000)."
)
estimes = ", ".join(f"{ic.libelle_serie(s)} ({ic.formater_mois(d)})" for s, d in config.VALEURS_A_IMPUTER)
st.info(
    f"**Valeur manquante estimée** : {estimes}. Le prix est absent de la source ; il est "
    "estimé par la moyenne des deux mois voisins (interpolation linéaire) et signalé dans la "
    "colonne prix_impute de data/series.csv.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Protocole d'évaluation
# ---------------------------------------------------------------------------
st.header("Protocole d'évaluation")
horizons = ", ".join(f"{config.LIBELLES_HORIZONS[h]} ({p} mois après la dernière donnée)"
                     for h, p in config.PAS_PAR_HORIZON.items())
validations = " ; ".join(f"{ic.formater_mois(d)} à {ic.formater_mois(f)}"
                         for d, f in config.PERIODES_VALIDATION)
st.markdown(f"""
**Modèles comparés**
- {config.LIBELLES_MODELES['naif']} : la prévision est le dernier prix observé.
- {config.LIBELLES_MODELES['ets']} : lissage exponentiel.
- {config.LIBELLES_MODELES['sarima']} : modèle autorégressif intégré à moyenne mobile saisonnier.

**Transformation** : les modèles sont estimés sur le logarithme des prix nominaux ;
les prévisions sont affichées en dollars.

**Horizons** : compte tenu du délai de publication, les horizons de consultation sont
{horizons}.

**Découpage chronologique** (commun aux quatre séries et aux trois méthodes) : entraînement à
partir de {ic.formater_mois(config.DATE_DEBUT_ENTRAINEMENT)} ; {len(config.PERIODES_VALIDATION)} périodes de validation de
12 mois ({validations}), chacune précédée de toutes les données antérieures ; test final de
{ic.formater_mois(config.DATE_DEBUT_TEST)} à {ic.formater_mois(config.DATE_FIN_TEST)}, réservé et
non utilisé pour choisir les modèles. Les prévisions sont produites avec les seules données
connues à leur date d'origine, puis comparées aux prix observés.

**Métriques**, par horizon :
- MAE (erreur absolue moyenne, en dollars) ;
- RMSE (racine de l'erreur quadratique moyenne, en dollars) ;
- MAPE (erreur absolue moyenne en pourcentage) ;
- taux de couverture des intervalles à 80 % et à 95 % (proportion des prix observés situés
  dans l'intervalle).

**Test de Diebold-Mariano** : comparaison des erreurs de deux modèles sur les mêmes origines
(perte absolue, variance de long terme avec h − 1 retards où h est le nombre de pas,
correction de Harvey, Leybourne et Newbold, seuil de 5 %).

**Intervalles** : 80 % pour l'affichage, 95 % pour les alertes.

**Choix du modèle final**, pour chaque série : plus faible MAE moyenne sur les 3 horizons, à
condition de battre le modèle naïf. Un modèle est retenu si sa MAE moyenne est inférieure à
celle du naïf et si le test de Diebold-Mariano le désigne meilleur que le naïf (5 %) sur au
moins un horizon ; sinon, le modèle naïf est retenu.

**Alerte** : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon
{config.LIBELLES_HORIZONS[config.HORIZON_ALERTE]}.

**Probabilité de dépassement** (page Prévision) : déduite de l'intervalle à 80 %, en supposant
que le logarithme du prix futur suit une loi normale centrée sur le logarithme de la prévision.
""")

# ---------------------------------------------------------------------------
# Sections rédigées par l'équipe
# ---------------------------------------------------------------------------
st.header("Limites")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")

st.header("Recommandations")
st.markdown("> **À RÉDIGER PAR L'ÉQUIPE**")
