"""Page Méthodologie et limites : description des données, du prétraitement, des modèles et
des limites de l'outil.

Le contenu descriptif reprend docs/protocole.md et notebooks/01_pretraitement.ipynb
(contrôles, valeurs estimées et statistiques recalculés à partir de data/series.csv) :
séries et choix des provinces, découpage chronologique, métriques et leurs formules,
règle du modèle final. Les sections « Limites » et « Recommandations » reprennent le texte
rédigé par l'équipe pour le rapport.
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
st.subheader("Choix des provinces")
st.markdown(
    "Une seule série est suivie par produit : le maïs-grain en Ontario et le canola en "
    "Saskatchewan. Ce sont les principales provinces productrices de ces cultures, et les "
    "provinces voisines (Québec pour le maïs-grain, Alberta pour le canola) ont des marchés "
    "pratiquement jumeaux. Les deux séries retenues couvrent la même période, à partir "
    f"de {ic.formater_mois(config.DATE_DEBUT)}."
)

# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------
st.header("Prétraitement")
st.markdown("""
Le prétraitement de référence est le notebook **notebooks/01_pretraitement.ipynb**. Il lit les
fichiers sources **data/mais.csv** et **data/canola.csv** (extraits du tableau de Statistique
Canada, encodage cp1252, dates au format mois/jour/année), contrôle les séries et calcule
les statistiques descriptives. Le script **scripts/preparer_donnees.py** reproduit ces
traitements pour les deux séries retenues et écrit **data/series.csv**, seul fichier de
données lu par l'application.
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

    st.subheader("Valeurs manquantes")
    estimations = detailler_estimations(series)
    if estimations.empty:
        st.markdown(
            "Aucune valeur estimée : les deux séries retenues sont complètes dans les fichiers "
            "sources. La seule valeur manquante des sources (canola en Alberta, mai 2012) "
            "appartient à une série non retenue. La colonne **prix_impute** de data/series.csv "
            "est conservée (toutes ses valeurs valent False)."
        )
    else:
        premiere_validation = ic.formater_mois(config.PERIODES_VALIDATION[0][0])
        st.markdown(f"""
Le prix absent de la source est remplacé par la moyenne des prix des deux mois voisins, ce qui
correspond à une interpolation linéaire pour un seul mois manquant. La valeur est signalée dans la
colonne **prix_impute** de data/series.csv. L'estimation est faite avant le découpage chronologique
(première période de validation : {premiere_validation}).
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
st.markdown(f"""
**Modèles comparés**
- {config.LIBELLES_MODELES['naif']} : la prévision est le dernier prix observé.
- {config.LIBELLES_MODELES['ets']} : lissage exponentiel.
- {config.LIBELLES_MODELES['sarima']} : modèle autorégressif intégré à moyenne mobile saisonnier.

**Transformation** : les modèles sont estimés sur le logarithme des prix nominaux ;
les prévisions sont affichées en dollars.

**Horizons** : compte tenu du délai de publication, les horizons de consultation sont
{horizons}.
""")

# ---------------------------------------------------------------------------
# Découpage chronologique
# ---------------------------------------------------------------------------
st.subheader("Découpage chronologique")
st.markdown(f"""
Le découpage est commun aux deux séries et aux trois méthodes. Les périodes de validation et de
test sont définies par leurs mois **cibles** : chaque mois cible est prévu à chaque horizon depuis
l'origine située {", ".join(str(p) for p in config.PAS_PAR_HORIZON.values())} mois plus tôt
(origine glissante), avec un modèle réestimé à chaque origine sur toutes les données disponibles
depuis {ic.formater_mois(config.DATE_DEBUT_ENTRAINEMENT)} (fenêtre croissante). Les origines
peuvent précéder une période, mais aucune cible n'en sort : chaque prévision appartient à une
seule période.
""")


def nb_mois(debut, fin):
    return len(pd.date_range(debut, fin, freq="MS"))


lignes_decoupage = [{
    "Période": "Entraînement",
    "Mois cibles": f"depuis {ic.formater_mois(config.DATE_DEBUT_ENTRAINEMENT)}, jusqu'à chaque origine",
    "Nombre de mois": "fenêtre croissante",
    "Rôle": "Estimation des modèles",
}]
for numero, (debut, fin) in enumerate(config.PERIODES_VALIDATION, start=1):
    lignes_decoupage.append({
        "Période": f"Validation, bloc {numero}",
        "Mois cibles": f"{ic.formater_mois(debut)} à {ic.formater_mois(fin)}",
        "Nombre de mois": str(nb_mois(debut, fin)),
        "Rôle": "Choix du modèle final",
    })
for periode, (debut, fin) in config.PERIODES_TEST.items():
    lignes_decoupage.append({
        "Période": ic.libelle_periode(periode, avec_dates=False),
        "Mois cibles": f"{ic.formater_mois(debut)} à {ic.formater_mois(fin)}",
        "Nombre de mois": str(nb_mois(debut, fin)),
        "Rôle": "Performances rapportées",
    })
lignes_decoupage.append({
    "Période": ic.libelle_periode(config.PERIODE_TEST_GLOBAL, avec_dates=False),
    "Mois cibles": f"{ic.formater_mois(config.DATE_DEBUT_TEST)} à {ic.formater_mois(config.DATE_FIN_TEST)}",
    "Nombre de mois": str(nb_mois(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)),
    "Rôle": "Bilan sur l'ensemble des trois tests",
})
st.dataframe(pd.DataFrame(lignes_decoupage), hide_index=True, use_container_width=True)
st.caption(f"Nombre de prévisions évaluées par série, modèle et horizon : "
           f"{nb_mois(config.DATE_DEBUT_VALIDATION, config.DATE_FIN_VALIDATION)} en validation, "
           f"{nb_mois(*config.PERIODES_TEST['test_1'])} par période de test, "
           f"{nb_mois(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)} pour le test global. "
           "Le test n'est jamais utilisé pour choisir les modèles.")

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
st.subheader("Métriques")
st.markdown(
    "Les métriques sont calculées pour chaque série, chaque modèle, chaque horizon et chaque "
    "période (validation, test 1, test 2, test 3 et test global), sans jamais mélanger validation "
    "et test. Pour une période et un horizon, $y_t$ est le prix observé au mois cible $t$, "
    "$\\hat{y}_t$ la prévision (en dollars), $[L_t^{(c)}, U_t^{(c)}]$ l'intervalle de prévision "
    "de niveau $c$ et $n$ le nombre de mois cibles."
)
st.markdown("**MAE** (erreur absolue moyenne, en dollars par tonne)")
st.latex(r"\mathrm{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left| y_t - \hat{y}_t \right|")
st.markdown("**RMSE** (racine de l'erreur quadratique moyenne, en dollars par tonne)")
st.latex(r"\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} \left( y_t - \hat{y}_t \right)^2}")
st.markdown("**MAPE** (erreur absolue moyenne en pourcentage)")
st.latex(r"\mathrm{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \frac{\left| y_t - \hat{y}_t \right|}{\left| y_t \right|}")
st.markdown("**Couverture** de l'intervalle de niveau $c$ (80 % ou 95 %) : proportion des prix "
            "observés situés dans l'intervalle, à comparer au niveau visé $c$")
st.latex(r"\mathrm{Couverture}_c = \frac{1}{n} \sum_{t=1}^{n} \mathbf{1}\left\{ L_t^{(c)} \le y_t \le U_t^{(c)} \right\}")

# ---------------------------------------------------------------------------
# Modèle final et autres règles
# ---------------------------------------------------------------------------
st.subheader("Choix du modèle final")
st.markdown("""
Pour chaque série, le modèle final est celui qui a la plus faible MAE moyenne sur les trois
horizons, calculée sur la **période de validation uniquement** :
""")
st.latex(r"\text{modèle final} = \arg\min_{m} \; \frac{1}{3} \sum_{h \in \{1, 3, 6\}} \mathrm{MAE}_{m,h}^{\text{validation}}")
st.markdown("""
Le modèle naïf est retenu s'il est le meilleur (en cas d'égalité, le naïf est préféré). Les
métriques des trois méthodes sont ensuite présentées pour chaque période de test et pour le test
global ; elles ne modifient pas le choix du modèle final.
""")

st.subheader("Autres règles")
st.markdown(f"""
**Intervalles** : 80 % pour l'affichage, 95 % pour les alertes.

**Prévision en dollars** : les modèles prévoient le logarithme du prix ; l'exponentielle de la
prévision correspond à la médiane du prix prévu (et non à sa moyenne). Les bornes des
intervalles restent des quantiles du prix.

**Alerte** : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon
{config.LIBELLES_HORIZONS[config.HORIZON_ALERTE]}.

**Probabilité de dépassement** (page Prévision) : déduite de l'intervalle à 80 %, en supposant
que le logarithme du prix futur suit une loi normale centrée sur le logarithme de la prévision.
""")

# ---------------------------------------------------------------------------
# Limites et recommandations (texte de l'équipe, repris du rapport)
# ---------------------------------------------------------------------------
st.header("Limites")
st.markdown("""
Les résultats doivent être interprétés en tenant compte de plusieurs limites.
Premièrement, l'étude porte sur deux séries seulement : le maïs-grain en Ontario et le
canola en Saskatchewan. Les conclusions ne peuvent donc pas être généralisées à toutes les
cultures, à toutes les provinces ou à tous les marchés agricoles. Les prix utilisés sont
des prix mensuels nominaux en dollars par tonne métrique : ils ne sont pas corrigés de
l'inflation et ne représentent pas nécessairement le prix obtenu par chaque producteur
selon la qualité, la localisation, le contrat ou les coûts de transport.

Deuxièmement, les prix mensuels sont publiés avec un délai d'environ deux mois. L'outil
travaille donc sur des données qui ne reflètent pas immédiatement les conditions du marché
au moment de la consultation. En outre, les modèles utilisent seulement l'historique des
prix : ils n'intègrent pas de variables explicatives comme les précipitations, les
rendements, les stocks, les coûts des engrais et du carburant, les taux de change, les
prix à terme ou les événements géopolitiques.

Troisièmement, les modèles naïf, ETS et SARIMA reposent sur des régularités historiques
qui peuvent être perturbées par des changements de régime ou des événements
exceptionnels ; les performances varient d'ailleurs nettement entre les trois périodes de
test. Les intervalles présentent aussi des limites : pour l'ETS du canola, la couverture
observée est nettement inférieure aux niveaux de 80 % et de 95 %, ce qui montre que
l'incertitude est sous-estimée dans cette configuration.

Enfin, le protocole de sélection repose sur la MAE moyenne des trois horizons. Ce critère
est simple et comparable, mais il ne résume pas toutes les qualités d'un modèle : il ne
mesure ni la calibration des intervalles, ni le coût des erreurs selon leur sens, ni
l'utilité économique d'une prévision pour une décision donnée. Les résultats constituent
donc une évaluation dans le cadre défini par le projet, et non une garantie de performance
future.
""")

st.header("Recommandations")
st.markdown("""
Pour l'utilisation courante, il est recommandé à l'analyste de lire les prévisions comme
des ordres de grandeur et de toujours examiner les intervalles associés. Les prévisions à
un mois peuvent être privilégiées pour le suivi à court terme, tandis que celles à trois
et six mois demandent plus de prudence, puisque l'erreur augmente généralement avec
l'horizon. L'analyste devrait aussi vérifier la date de la dernière donnée disponible et
tenir compte du délai de publication.

Les alertes devraient servir de signaux invitant à examiner un prix, à vérifier les
données et à rechercher un contexte explicatif, plutôt que de déclencheurs de décisions
automatiques. Cette précaution est particulièrement importante pour le canola, dont les
intervalles ETS sont trop étroits au test global. L'analyste peut compléter l'information
de l'application par les bulletins de marché, les données météorologiques et les
informations sur les stocks et les marchés internationaux.

Plusieurs pistes peuvent améliorer l'outil. La première consiste à recalibrer les
intervalles de prévision, notamment pour le canola, afin que leur couverture se rapproche
des niveaux annoncés. La deuxième consiste à enrichir les modèles avec des variables
externes : conditions météorologiques, rendements, stocks, coûts des intrants, taux de
change ou prix de référence internationaux. La troisième consiste à suivre régulièrement
les performances sur les nouvelles données et à réévaluer les modèles lorsque les
conditions du marché changent. Enfin, d'autres modèles pourraient être comparés, à
condition de conserver une séparation chronologique rigoureuse entre la sélection et
l'évaluation.
""")
