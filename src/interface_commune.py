"""Éléments partagés par les pages Streamlit : lecture en cache, panneau latéral,
bandeau des résultats factices, mise en forme, graphiques Plotly, alertes,
probabilité de dépassement et statistiques du prétraitement.

L'interface ne calcule aucun modèle : elle lit uniquement data/series.csv et
les fichiers de resultats/.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from src import config
from src.chargement import charger_resultat, charger_series, chemin_resultat
from src.evaluation import choisir_modele_final

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Palette Okabe-Ito, lisible par les personnes ayant une déficience de la
# perception des couleurs. Les modèles se distinguent aussi par le tracé.
COULEUR_OBSERVE = "#000000"
COULEUR_ALERTE = "#B00020"
COULEURS_MODELES = {
    "naif": "#666666",
    "ets": "#0072B2",
    "sarima": "#D55E00",
}
TRACES_MODELES = {"naif": "dot", "ets": "dash", "sarima": "dashdot"}
# Courbes comparées (provinces ou toutes les séries), dans l'ordre d'affichage
COULEURS_COMPARAISON = ["#000000", "#0072B2", "#D55E00", "#009E73"]
TRACES_COMPARAISON = ["solid", "dash", "dot", "dashdot"]
# Périodes du découpage et valeurs estimées
COULEUR_VALIDATION = "#999999"
OPACITES_VALIDATION = [0.12, 0.24]  # alternées d'un bloc à l'autre
COULEUR_TEST = "#E69F00"
OPACITE_TEST = 0.25
COULEUR_ESTIMEE = "#CC79A7"
POLICE = "Source Sans Pro, Arial, sans-serif"

MOIS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
           "août", "septembre", "octobre", "novembre", "décembre"]
MOIS_COURTS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.",
                  "août", "sept.", "oct.", "nov.", "déc."]

ESPACE_INSECABLE = "\u00a0"
ESPACE_FINE = "\u202f"  # espace fine insécable, séparateur des milliers

FICHIER_FACTICE = "FACTICE.txt"


# ---------------------------------------------------------------------------
# Mise en forme des nombres et des dates (conventions françaises)
# ---------------------------------------------------------------------------
def formater_nombre(valeur, decimales=2):
    """Formate un nombre à la française : 1 234,56."""
    if pd.isna(valeur):
        return "–"
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", ESPACE_FINE).replace(".", ",")


def formater_prix(valeur, decimales=2):
    """Formate un prix en dollars : 1 234,56 $."""
    if pd.isna(valeur):
        return "–"
    return f"{formater_nombre(valeur, decimales)}{ESPACE_INSECABLE}$"


def formater_pct(proportion, decimales=1, signe=True):
    """Formate une proportion (0,023) en pourcentage (+2,3 %)."""
    if pd.isna(proportion):
        return "–"
    texte = formater_nombre(abs(proportion) * 100, decimales)
    prefixe = ("+" if proportion > 0 else "−" if proportion < 0 else "") if signe else \
        ("−" if proportion < 0 else "")
    return f"{prefixe}{texte}{ESPACE_INSECABLE}%"


def formater_mois(date):
    """Formate une date en « août 2026 »."""
    date = pd.Timestamp(date)
    return f"{MOIS_FR[date.month - 1]} {date.year}"


def libelle_serie(serie):
    """Libellé français d'une série : « Maïs-grain – Québec »."""
    return config.LIBELLES_SERIES.get(serie, serie)


def libelle_modele(modele):
    """Libellé français d'un modèle."""
    return config.LIBELLES_MODELES.get(modele, modele)


def colonnes_intervalle(niveau):
    """Noms des colonnes de bornes pour un niveau (80 ou 95) : (« inf80 », « sup80 »)."""
    return f"inf{niveau}", f"sup{niveau}"


# ---------------------------------------------------------------------------
# Page et bandeau
# ---------------------------------------------------------------------------
def configurer_page(titre):
    """Configure la page (titre d'onglet, mise en page large) et affiche le bandeau
    des résultats factices. Doit être appelée en premier dans chaque page."""
    st.set_page_config(page_title=f"{titre} – Prix agricoles", layout="wide")
    afficher_bandeau_factice()


def resultats_factices():
    """Indique si resultats/ contient des résultats factices (présence de FACTICE.txt)."""
    return (config.DOSSIER_RESULTATS / FICHIER_FACTICE).exists()


def afficher_bandeau_factice():
    """Affiche un avertissement si les résultats sont factices."""
    if resultats_factices():
        st.warning(
            "**Résultats factices.** Les prévisions, métriques et alertes affichées "
            "sont générées aléatoirement pour le développement de l'interface. "
            "Elles ne doivent pas être utilisées pour l'analyse.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Lecture en cache
# ---------------------------------------------------------------------------
# La date de modification du fichier fait partie de la clé du cache : un
# fichier régénéré est relu automatiquement.
def _horodatage(chemin):
    return chemin.stat().st_mtime if chemin.exists() else None


@st.cache_data(show_spinner=False)
def _lire_series_en_cache(chemin, horodatage):
    return charger_series(chemin)


@st.cache_data(show_spinner=False)
def _lire_resultat_en_cache(nom, chemin, horodatage):
    return charger_resultat(nom)


def lire_series():
    """Retourne data/series.csv, ou None si le fichier est absent."""
    chemin = config.FICHIER_SERIES
    if not chemin.exists():
        return None
    return _lire_series_en_cache(chemin, _horodatage(chemin))


def lire_resultat(nom):
    """Retourne un fichier de resultats/ (previsions, previsions_futures, metriques,
    tests_dm), ou None s'il est absent."""
    chemin = chemin_resultat(nom)
    if not chemin.exists():
        return None
    return _lire_resultat_en_cache(nom, chemin, _horodatage(chemin))


def exiger_series():
    """Retourne les séries ou arrête la page avec un message si elles sont absentes."""
    series = lire_series()
    if series is None:
        st.error(
            "Le fichier **data/series.csv** est introuvable. Vérifier que "
            "data/mais.csv et data/canola.csv sont présents, puis exécuter :\n\n"
            "`python scripts/preparer_donnees.py`"
        )
        st.stop()
    return series


def exiger_resultat(nom):
    """Retourne un fichier de resultats/ ou arrête la page avec un message s'il est absent."""
    tableau = lire_resultat(nom)
    if tableau is None:
        st.error(
            f"Le fichier **resultats/{chemin_resultat(nom).name}** est introuvable. "
            "Exécuter le script d'évaluation (ou, en développement, "
            "`python scripts/generer_resultats_factices.py`)."
        )
        st.stop()
    return tableau


def modeles_finaux():
    """Retourne {série: modèle final} selon la règle du protocole, ou {} sans résultats."""
    metriques = lire_resultat("metriques")
    tests_dm = lire_resultat("tests_dm")
    if metriques is None or tests_dm is None:
        return {}
    finaux = {}
    for serie in config.LISTE_SERIES:
        modele = choisir_modele_final(metriques, tests_dm, serie)
        if modele is not None:
            finaux[serie] = modele
    return finaux


# ---------------------------------------------------------------------------
# Panneau latéral commun
# ---------------------------------------------------------------------------
MODELE_FINAL = "final"


@dataclass
class Selection:
    """Choix courants du panneau latéral."""
    produit: str
    province: str
    serie: str
    modele: str | None          # modèle effectivement affiché (None si aucun résultat)
    modele_final: str | None    # modèle final de la série selon le protocole
    niveau: int                 # 80 ou 95
    choix_final: bool = True    # True si l'utilisateur a choisi « Modèle final »


def _widget_persistant(fonction, cle, libelle, options, defaut, **options_widget):
    """Crée un widget dont la valeur est conservée d'une page à l'autre.

    Streamlit efface l'état des widgets quand on change de page ; la valeur est
    donc recopiée dans une clé de st.session_state indépendante du widget.
    """
    etat = st.session_state
    cle_widget = f"_widget_{cle}"
    if etat.get(cle) not in options:
        etat[cle] = defaut
    etat[cle_widget] = etat[cle]

    def memoriser():
        etat[cle] = etat[cle_widget]

    return fonction(libelle, options, key=cle_widget, on_change=memoriser, **options_widget)


def panneau_lateral(avec_modele=True, avec_niveau=True):
    """Affiche le panneau latéral commun et retourne la Selection courante.

    Le modèle proposé par défaut est le modèle final de la série choisie
    (règle du protocole, calculée à partir de metriques.csv et tests_dm.csv).
    """
    barre = st.sidebar
    barre.header("Sélection")

    produits = list(config.PROVINCES_PAR_PRODUIT)
    produit = _widget_persistant(barre.selectbox, "produit", "Produit", produits, produits[0],
                                 format_func=config.LIBELLES_PRODUITS.get)
    # Seules les provinces suivies pour ce produit sont proposées
    provinces = config.PROVINCES_PAR_PRODUIT[produit]
    province = _widget_persistant(barre.selectbox, "province", "Province", provinces, provinces[0],
                                  format_func=config.LIBELLES_PROVINCES.get)
    serie = f"{produit}_{province}"

    finaux = modeles_finaux()
    modele_final = finaux.get(serie)
    modele = None
    choix_final = True
    if avec_modele:
        if modele_final is None:
            barre.caption("Modèle : aucun résultat disponible.")
        else:
            options = [MODELE_FINAL] + config.MODELES

            def libeller(option):
                if option == MODELE_FINAL:
                    return f"Modèle final ({libelle_modele(modele_final)})"
                return libelle_modele(option)

            choix = _widget_persistant(barre.selectbox, "modele", "Modèle", options, MODELE_FINAL,
                                       format_func=libeller)
            choix_final = choix == MODELE_FINAL
            modele = modele_final if choix_final else choix

    niveau = int(round(config.NIVEAU_AFFICHAGE * 100))
    if avec_niveau:
        niveaux = [int(round(n * 100)) for n in config.NIVEAUX_INTERVALLE]
        niveau = _widget_persistant(barre.radio, "niveau", "Intervalle de prévision", niveaux,
                                    niveau, format_func=lambda n: f"{n}{ESPACE_INSECABLE}%",
                                    horizontal=True)

    return Selection(produit, province, serie, modele, modele_final, niveau, choix_final)


# ---------------------------------------------------------------------------
# Alertes
# ---------------------------------------------------------------------------
def calculer_alertes(previsions, modeles_par_serie=None):
    """Retourne les alertes : prix observé hors de l'intervalle à 95 % de la
    prévision à l'horizon 1 (config.HORIZON_ALERTE).

    modeles_par_serie : {série: modèle} pour ne garder qu'un modèle par série
    (par exemple le modèle final) ; None pour garder tous les modèles.
    Colonnes ajoutées : sens (« au-dessus » ou « en dessous ») et ecart_pct,
    écart relatif entre le prix observé et la borne franchie.
    """
    p = previsions[(previsions["horizon"] == config.HORIZON_ALERTE) & previsions["reel"].notna()]
    if modeles_par_serie is not None:
        p = p[p["modele"] == p["serie"].map(modeles_par_serie)]
    au_dessus = p["reel"] > p["sup95"]
    en_dessous = p["reel"] < p["inf95"]
    alertes = p[au_dessus | en_dessous].copy()
    alertes["sens"] = np.where(au_dessus[alertes.index], "au-dessus", "en dessous")
    borne = np.where(alertes["sens"] == "au-dessus", alertes["sup95"], alertes["inf95"])
    alertes["ecart_pct"] = alertes["reel"] / borne - 1
    return alertes.sort_values(["date_cible", "serie", "modele"], ascending=[False, True, True])


def alertes_en_cours(alertes, series):
    """Garde les alertes portant sur le dernier mois observé de chaque série."""
    derniers = series.groupby("serie")["date"].max()
    return alertes[alertes["date_cible"] == alertes["serie"].map(derniers)]


# ---------------------------------------------------------------------------
# Probabilité de dépassement
# ---------------------------------------------------------------------------
def probabilite_depassement(prevision, borne_inf, borne_sup, seuil, niveau):
    """Probabilité que le prix dépasse un seuil, déduite d'un intervalle de prévision.

    Hypothèse : le log du prix futur suit une loi normale N(mu, sigma²), avec
      - mu = log(prevision) (la prévision est traitée comme la médiane) ;
      - sigma = (log(borne_sup) - log(borne_inf)) / (2 z), où z est le quantile
        de la loi normale centrée réduite d'ordre 0,5 + niveau / 2
        (z ≈ 1,2816 pour 80 %, z ≈ 1,9600 pour 95 %).
    Alors P(prix > seuil) = 1 - Φ((log(seuil) - mu) / sigma).

    Si l'intervalle n'est pas symétrique en log (par exemple des quantiles
    estimés séparément), sigma correspond à la demi-largeur moyenne. Le
    résultat n'est qu'une approximation : il hérite de la qualité de
    l'intervalle (voir les taux de couverture dans metriques.csv).

    Les arguments peuvent être des scalaires ou des tableaux ; niveau est une
    proportion (0,80) ou un pourcentage (80).
    """
    niveau = niveau / 100 if niveau > 1 else niveau
    z = stats.norm.ppf(0.5 + niveau / 2)
    mu = np.log(prevision)
    sigma = (np.log(borne_sup) - np.log(borne_inf)) / (2 * z)
    return 1 - stats.norm.cdf((np.log(seuil) - mu) / sigma)


# ---------------------------------------------------------------------------
# Statistiques descriptives (prétraitement)
# ---------------------------------------------------------------------------
STATISTIQUES = {
    "prix_observes": "Observations",
    "moyenne": "Moyenne",
    "ecart_type": "Écart-type",
    "minimum": "Minimum",
    "Q1": "Q1",
    "mediane": "Médiane",
    "Q3": "Q3",
    "maximum": "Maximum",
}


def statistiques_descriptives(series, liste_series=None):
    """Statistiques descriptives des prix par série, hors valeurs imputées.

    Même calcul que notebooks/01_pretraitement.ipynb, où les statistiques sont
    produites avant l'imputation : les lignes où prix_impute vaut True sont
    exclues (canola_AB compte donc 499 observations au lieu de 500).

    series : DataFrame au format de data/series.csv.
    liste_series : séries à garder (None pour toutes).
    Retourne un DataFrame indexé par série, dans l'ordre de config.LISTE_SERIES,
    avec les colonnes de STATISTIQUES.
    """
    s = series if liste_series is None else series[series["serie"].isin(liste_series)]
    s = s[~s["prix_impute"].astype(bool)]
    tableau = s.groupby("serie")["prix"].describe().rename(columns={
        "count": "prix_observes", "mean": "moyenne", "std": "ecart_type", "min": "minimum",
        "25%": "Q1", "50%": "mediane", "75%": "Q3", "max": "maximum",
    })
    ordre = [serie for serie in config.LISTE_SERIES if serie in tableau.index]
    return tableau.reindex(ordre)[list(STATISTIQUES)]


def tableau_statistiques(statistiques):
    """Met en forme le résultat de statistiques_descriptives() pour st.dataframe.

    Les valeurs sont arrondies avec pandas, comme dans le notebook, avant la mise
    en forme (les deux arrondis peuvent différer sur un cas limite comme 383,185).
    """
    tableau = pd.DataFrame({"Série": [libelle_serie(s) for s in statistiques.index]})
    for colonne, nom in STATISTIQUES.items():
        if colonne == "prix_observes":
            tableau[nom] = statistiques[colonne].astype(int).to_numpy()
        else:
            tableau[nom] = statistiques[colonne].round(2).map(formater_nombre).to_numpy()
    return tableau


# ---------------------------------------------------------------------------
# Graphiques Plotly
# ---------------------------------------------------------------------------
def mettre_en_forme(figure, titre=None, titre_y=f"Prix ({config.UNITE_PRIX})", hauteur=440):
    """Applique le style commun à une figure Plotly."""
    figure.update_layout(
        template="simple_white",
        title=dict(text=titre, x=0, xanchor="left", font=dict(size=18)) if titre else None,
        font=dict(family=POLICE, size=14, color="#222222"),
        separators=", ",  # virgule décimale, espace pour les milliers
        hovermode="x unified",
        height=hauteur,
        margin=dict(l=10, r=10, t=60 if titre else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    figure.update_xaxes(showgrid=False, ticks="outside")
    figure.update_yaxes(title=titre_y, showgrid=True, gridcolor="#E5E5E5", ticks="outside")
    return figure


def afficher_graphique(figure):
    """Affiche une figure Plotly sur toute la largeur, sans logo Plotly."""
    st.plotly_chart(figure, use_container_width=True,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def graphique_serie(historique, previsions=None, niveau=80, modele=None, alertes=None,
                    titre=None, hauteur=440):
    """Série observée, avec en option une prévision et sa bande d'intervalle.

    historique : DataFrame avec les colonnes date et prix.
    previsions : DataFrame avec date_cible, prevision et les bornes du niveau
                 choisi (inf80/sup80 ou inf95/sup95), trié par date_cible.
    alertes    : DataFrame avec date_cible et reel, marqué par des croix.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=historique["date"], y=historique["prix"], mode="lines", name="Prix observé",
        line=dict(color=COULEUR_OBSERVE, width=2),
        hovertemplate="%{y:,.2f} $<extra>Observé</extra>",
    ))

    if previsions is not None and not previsions.empty:
        inf, sup = colonnes_intervalle(niveau)
        couleur = COULEURS_MODELES.get(modele, "#0072B2")
        nom = libelle_modele(modele) if modele else "Prévision"
        figure.add_trace(go.Scatter(
            x=previsions["date_cible"], y=previsions[sup], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=previsions["date_cible"], y=previsions[inf], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor=_transparence(couleur, 0.2),
            name=f"Intervalle à {niveau}{ESPACE_INSECABLE}%", hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=previsions["date_cible"], y=previsions["prevision"], mode="lines+markers",
            name=nom, line=dict(color=couleur, width=2, dash=TRACES_MODELES.get(modele, "dash")),
            marker=dict(size=6),
            customdata=np.stack([previsions[inf], previsions[sup]], axis=-1),
            hovertemplate=(f"%{{y:,.2f}} $ [%{{customdata[0]:,.2f}} ; %{{customdata[1]:,.2f}}]"
                           f"<extra>{nom}</extra>"),
        ))

    if alertes is not None and not alertes.empty:
        figure.add_trace(go.Scatter(
            x=alertes["date_cible"], y=alertes["reel"], mode="markers", name="Alerte",
            marker=dict(symbol="x", size=11, color=COULEUR_ALERTE, line=dict(width=1)),
            hovertemplate="%{y:,.2f} $<extra>Alerte</extra>",
        ))

    return mettre_en_forme(figure, titre=titre, hauteur=hauteur)


def graphique_barres(tableau, x, y, couleur=None, titre=None, titre_y=None, format_y=",.2f",
                     hauteur=380):
    """Diagramme en barres groupées ; couleur désigne la colonne des modèles."""
    figure = go.Figure()
    groupes = tableau.groupby(couleur, sort=False) if couleur else [(None, tableau)]
    for groupe, donnees in groupes:
        figure.add_trace(go.Bar(
            x=donnees[x].astype(str), y=donnees[y],
            name=libelle_modele(groupe) if groupe else y,
            marker_color=COULEURS_MODELES.get(groupe, "#0072B2"),
            hovertemplate=f"%{{y:{format_y}}}<extra>{libelle_modele(groupe) if groupe else ''}</extra>",
        ))
    figure.update_layout(barmode="group")
    return mettre_en_forme(figure, titre=titre, titre_y=titre_y, hauteur=hauteur)


def ajouter_selecteur_periode(figure):
    """Ajoute des boutons de période (5 ans, 10 ans, 20 ans, tout) sur l'axe des dates."""
    figure.update_xaxes(rangeselector=dict(
        buttons=[
            dict(count=5, label="5 ans", step="year", stepmode="backward"),
            dict(count=10, label="10 ans", step="year", stepmode="backward"),
            dict(count=20, label="20 ans", step="year", stepmode="backward"),
            dict(step="all", label="Tout"),
        ],
        x=0, y=1.02, xanchor="left", yanchor="bottom",
    ))
    figure.update_layout(legend=dict(y=1.12))
    return figure


def ajouter_periodes(figure):
    """Ombre les blocs de validation et la période de test (config.py) sur l'axe des dates.

    Les blocs de validation alternent deux gris pour rester distincts ; le test
    est en orange. Chaque bande couvre ses mois en entier (du premier jour du
    premier mois au premier jour du mois suivant le dernier). Deux entrées de
    légende décrivent les bandes.
    """
    for numero, (debut, fin) in enumerate(config.PERIODES_VALIDATION):
        figure.add_vrect(
            x0=debut, x1=pd.Timestamp(fin) + pd.DateOffset(months=1), layer="below", line_width=0,
            fillcolor=COULEUR_VALIDATION, opacity=OPACITES_VALIDATION[numero % 2],
        )
    figure.add_vrect(
        x0=config.DATE_DEBUT_TEST, x1=pd.Timestamp(config.DATE_FIN_TEST) + pd.DateOffset(months=1),
        layer="below", line_width=0, fillcolor=COULEUR_TEST, opacity=OPACITE_TEST,
    )
    nb_blocs = len(config.PERIODES_VALIDATION)
    debut_validation = config.PERIODES_VALIDATION[0][0]
    fin_validation = config.PERIODES_VALIDATION[-1][1]
    legendes = [
        (COULEUR_VALIDATION, OPACITES_VALIDATION[1],
         f"Validation ({nb_blocs} blocs, {formater_mois(debut_validation)} à "
         f"{formater_mois(fin_validation)})"),
        (COULEUR_TEST, OPACITE_TEST,
         f"Test ({formater_mois(config.DATE_DEBUT_TEST)} à {formater_mois(config.DATE_FIN_TEST)})"),
    ]
    for couleur, opacite, nom in legendes:
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=nom, hoverinfo="skip",
            marker=dict(symbol="square", size=14, color=_transparence(couleur, opacite)),
        ))
    return figure


def ajouter_valeurs_estimees(figure, series, liste_series):
    """Marque les valeurs estimées (prix_impute) des séries affichées."""
    estimes = series[series["serie"].isin(liste_series) & series["prix_impute"].astype(bool)]
    if estimes.empty:
        return figure
    figure.add_trace(go.Scatter(
        x=estimes["date"], y=estimes["prix"], mode="markers", name="Valeur estimée (interpolation)",
        marker=dict(symbol="diamond-open", size=12, color=COULEUR_ESTIMEE, line=dict(width=2)),
        customdata=estimes["serie"].map(libelle_serie),
        hovertemplate="%{customdata} : %{y:,.2f} $ (valeur estimée)<extra></extra>",
    ))
    return figure


def graphique_comparaison(series, liste_series, titre=None, hauteur=480):
    """Prix de plusieurs séries sur un même graphique (comparaison de provinces ou de toutes les séries).

    series : DataFrame au format de data/series.csv.
    """
    figure = go.Figure()
    for rang, serie in enumerate(liste_series):
        s = series[series["serie"] == serie]
        figure.add_trace(go.Scatter(
            x=s["date"], y=s["prix"], mode="lines", name=libelle_serie(serie),
            line=dict(color=COULEURS_COMPARAISON[rang % len(COULEURS_COMPARAISON)],
                      dash=TRACES_COMPARAISON[rang % len(TRACES_COMPARAISON)], width=2),
            hovertemplate=f"%{{y:,.2f}} $<extra>{libelle_serie(serie)}</extra>",
        ))
    mettre_en_forme(figure, titre=titre, hauteur=hauteur)
    return ajouter_selecteur_periode(figure)


def graphique_decomposition(decomposition, titre=None, hauteur=720):
    """Décomposition en 4 panneaux superposés : observé, tendance, saisonnalité, résidu.

    decomposition : DataFrame avec les colonnes date, observe, tendance, saisonnier, residu.
    """
    panneaux = [("observe", "Observé"), ("tendance", "Tendance"),
                ("saisonnier", "Saisonnalité"), ("residu", "Résidu")]
    figure = make_subplots(rows=len(panneaux), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                           subplot_titles=[nom for _, nom in panneaux])
    for rang, (colonne, nom) in enumerate(panneaux, start=1):
        figure.add_trace(go.Scatter(
            x=decomposition["date"], y=decomposition[colonne], mode="lines", name=nom,
            line=dict(color=COULEUR_OBSERVE if colonne == "observe" else "#0072B2", width=1.5),
            hovertemplate=f"%{{y:,.2f}} $<extra>{nom}</extra>", showlegend=False,
        ), row=rang, col=1)
        if colonne in ("saisonnier", "residu"):
            figure.add_hline(y=0, line=dict(color="#999999", width=1), row=rang, col=1)
    mettre_en_forme(figure, titre=titre, titre_y=None, hauteur=hauteur)
    figure.update_layout(hovermode="x")
    figure.update_yaxes(title_text=config.UNITE_PRIX)
    for annotation in figure.layout.annotations:
        annotation.update(x=0, xanchor="left", font=dict(size=14))
    return figure


def graphique_previsions_reel(previsions, modeles, titre=None, hauteur=480):
    """Valeurs réelles et prévisions de plusieurs modèles sur la période de test.

    previsions : lignes de previsions.csv pour une série et un horizon.
    """
    figure = go.Figure()
    reel = previsions.drop_duplicates("date_cible").sort_values("date_cible")
    figure.add_trace(go.Scatter(
        x=reel["date_cible"], y=reel["reel"], mode="lines", name="Prix observé",
        line=dict(color=COULEUR_OBSERVE, width=2.5),
        hovertemplate="%{y:,.2f} $<extra>Observé</extra>",
    ))
    for modele in modeles:
        p = previsions[previsions["modele"] == modele].sort_values("date_cible")
        figure.add_trace(go.Scatter(
            x=p["date_cible"], y=p["prevision"], mode="lines", name=libelle_modele(modele),
            line=dict(color=COULEURS_MODELES.get(modele), dash=TRACES_MODELES.get(modele), width=1.8),
            hovertemplate=f"%{{y:,.2f}} $<extra>{libelle_modele(modele)}</extra>",
        ))
    return mettre_en_forme(figure, titre=titre, hauteur=hauteur)


def style_meilleur(colonne, sens="min"):
    """Style pandas : met en évidence la meilleure valeur d'une colonne (fond et gras).

    sens : « min » (plus petite valeur) ou « max » (plus grande valeur).
    """
    cible = colonne.min() if sens == "min" else colonne.max()
    return ["background-color: #CCE5F5; font-weight: bold" if v == cible else ""
            for v in colonne]


def _transparence(couleur_hex, alpha):
    """Convertit une couleur #RRGGBB en rgba avec la transparence alpha."""
    r, g, b = (int(couleur_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"
