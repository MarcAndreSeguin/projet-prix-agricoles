"""Page Méthodologie et limites : description des données, du prétraitement, des modèles et
des limites de l'outil.

Le contenu descriptif reprend docs/protocole.md et notebooks/01_pretraitement.ipynb
(contrôles, valeur estimée et statistiques recalculés à partir de data/series.csv) ;
les sections « Limites » et
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

# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------
st.header("Prétraitement")
st.markdown("""
Le prétraitement de référence est le notebook **notebooks/01_pretraitement.ipynb**. Il lit les
fichiers sources **data/mais.csv** et **data/canola.csv** (extraits du tableau de Statistique
Canada, encodage cp1252, dates au format mois/jour/année), contrôle les quatre séries, calcule
les statistiques descriptives et estime la valeur manquante. Le script
**scripts/preparer_donnees.py** reproduit ces traitements et écrit **data/series.csv**, seul
fichier de données lu par l'application.
""")


def controler(series):
    """Contrôles par série sur data/series.csv : calendrier mensuel, doublons, valeurs estimées."""
    lignes = []
    for serie in config.LISTE_SERIES:
        s = series[series["serie"] == serie]
        if s.empty:
            continue
        dates = pd.DatetimeIndex(s["date"])
        calendrier = pd.date_range(config.DATE_DEBUT, dates.max(), freq="MS")
        lignes.append({
            "Série": ic.libelle_serie(serie),
            "Début": ic.formater_mois(dates.min()),
            "Fin": ic.formater_mois(dates.max()),
            "Mois": len(s),
            "Mois absents": len(calendrier.difference(dates)),
            "Dates hors calendrier": len(dates.difference(calendrier)),
            "Doublons": int(dates.duplicated().sum()),
            "Prix manquants dans la source (estimés)": int(s["prix_impute"].sum()),
        })
    return pd.DataFrame(lignes)


def detailler_estimations(series):
    """Valeurs estimées avec les prix des deux mois voisins."""
    lignes = []
    for r in series[series["prix_impute"]].itertuples():
        prix = series[series["serie"] == r.serie].set_index("date")["prix"]
        avant = prix.get(r.date - pd.DateOffset(months=1))
        apres = prix.get(r.date + pd.DateOffset(months=1))
        lignes.append({
            "Série": ic.libelle_serie(r.serie),
            "Mois estimé": ic.formater_mois(r.date),
            "Mois précédent": f"{ic.formater_mois(r.date - pd.DateOffset(months=1))} : "
                              f"{ic.formater_prix(avant, 3)}",
            "Mois suivant": f"{ic.formater_mois(r.date + pd.DateOffset(months=1))} : "
                            f"{ic.formater_prix(apres, 3)}",
            "Valeur estimée": ic.formater_prix(r.prix, 3),
        })
    return pd.DataFrame(lignes)


if series is None:
    st.warning("data/series.csv est absent : les contrôles et les statistiques ne peuvent pas "
               "être affichés. Exécuter `python scripts/preparer_donnees.py`.")
else:
    st.subheader("Contrôles réalisés")
    st.markdown(f"""
- **Calendrier** : chaque série est comparée au calendrier mensuel attendu, de
  {ic.formater_mois(config.DATE_DEBUT)} à son dernier mois (mois absents, dates hors calendrier).
- **Doublons** : une seule observation par série et par mois (combinaison date, province, culture).
- **Valeurs manquantes** : nombre de prix vides par série dans les fichiers sources.
""")
    st.dataframe(controler(series), hide_index=True, use_container_width=True)

    st.subheader("Valeur manquante estimée")
    estimations = detailler_estimations(series)
    if estimations.empty:
        st.markdown("Aucune valeur estimée dans data/series.csv.")
    else:
        premiere_validation = ic.formater_mois(config.PERIODES_VALIDATION[0][0])
        st.markdown(f"""
Le prix absent de la source est remplacé par la moyenne des prix des deux mois voisins, ce qui
correspond à une interpolation linéaire pour un seul mois manquant. La valeur est signalée dans la
colonne **prix_impute** de data/series.csv. L'estimation est faite avant le découpage chronologique ;
les mois voisins précèdent la première période de validation ({premiere_validation}).
""")
        st.dataframe(estimations, hide_index=True, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(ic.tableau_statistiques(ic.statistiques_descriptives(series)),
                 hide_index=True, use_container_width=True)
    st.caption(f"Prix en {config.UNITE_PRIX}, de {ic.formater_mois(series['date'].min())} à "
               f"{ic.formater_mois(series['date'].max())}, hors valeur estimée (même calcul que "
               "notebooks/01_pretraitement.ipynb).")

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

**Choix du modèle final**, pour chaque série, sur la période de validation uniquement : plus
faible MAE moyenne sur les horizons, à condition de battre le modèle naïf. Un modèle est retenu
si sa MAE moyenne est inférieure à celle du naïf et si le test de Diebold-Mariano le désigne
meilleur que le naïf (5 %) sur au moins un horizon ; sinon, le modèle naïf est retenu. Les
performances sont rapportées sur la période de test ; les métriques et les tests sont calculés
séparément pour chaque période.

**Prévision en dollars** : les modèles prévoient le logarithme du prix ; l'exponentielle de la
prévision correspond à la médiane du prix prévu (et non à sa moyenne). Les bornes des
intervalles restent des quantiles du prix.

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
