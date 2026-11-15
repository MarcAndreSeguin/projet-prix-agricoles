"""Modèle SARIMA (responsable : Marc). ÉBAUCHE à compléter et à valider.

Interface commune des modèles (docs/protocole.md), la même que src/modeles/naif.py :
    prevoir(historique, pas) -> DataFrame (pas, prevision, inf80, sup80, inf95, sup95), en log.

Notations
---------
- y_t = log(prix_t) ; T est le dernier mois de l'historique fourni (l'origine).
- k : nombre de pas après la dernière donnée (3, 5 ou 8 selon config.PAS_PAR_HORIZON).

Modèle
------
SARIMA(p,d,q)(P,D,Q)12 : on différencie d fois (y_t - y_(t-1)) et D fois à
12 mois (y_t - y_(t-12)) pour rendre la série stationnaire, puis la valeur du
mois t s'explique par :
- les p valeurs précédentes (partie AR) et les P valeurs 12, 24... mois avant ;
- les q erreurs précédentes (partie MA) et les Q erreurs 12, 24... mois avant.
Les chocs sont supposés indépendants et de loi normale N(0, sigma²).
Les coefficients et sigma² sont estimés par maximum de vraisemblance
(statsmodels, SARIMAX sans variable exogène).

Prévision et intervalles : statsmodels calcule la prévision à k pas et sa
variance, qui tient compte de l'accumulation des chocs futurs. L'intervalle de
niveau c est prevision ± z_c · écart-type, comme pour le naïf.

Retour en dollars : fait par le script d'évaluation, pas ici (comme le naïf).

Absence de fuite d'information
------------------------------
La fonction ne reçoit que l'historique jusqu'à l'origine et ne lit aucun
fichier. Le modèle est réestimé à chaque appel, donc à chaque origine.

Essai rapide (depuis la racine du dépôt) :
    python -m src.modeles.sarima --serie mais_ON --blocs 1

Le choix de l'ordre final et l'interprétation des résultats reviennent à
l'équipe : ce fichier ne fait que calculer les erreurs de validation.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config
from src.chargement import charger_series
from src.evaluation import generer_paires_origine_cible, numero_periode_validation

# Mêmes colonnes et mêmes vérifications que le naïf de Sabra (maintenant dans main)
from src.modeles import naif
from src.modeles.naif import COLONNES, verifier_historique, verifier_pas

# ---------------------------------------------------------------------------
# Ordres candidats : ((p, d, q), (P, D, Q, 12))
# ---------------------------------------------------------------------------
# Une poignée d'ordres simples, à comparer en validation (pas de recherche
# exhaustive). d = 1 parce que les prix « dérivent » sans revenir à une
# moyenne fixe ; à vérifier avec un test de racine unitaire (ADF/KPSS) et les
# graphiques ACF/PACF de la série différenciée.
ORDRES_CANDIDATS = {
    "ARIMA(1,1,0)": ((1, 1, 0), (0, 0, 0, 0)),  # sans saisonnalité, le plus simple
    "ARIMA(0,1,1)": ((0, 1, 1), (0, 0, 0, 0)),
    "ARIMA(1,1,1)": ((1, 1, 1), (0, 0, 0, 0)),
    "SARIMA(1,1,0)(1,0,0)12": ((1, 1, 0), (1, 0, 0, 12)),  # saisonnalité légère
    "SARIMA(0,1,1)(0,1,1)12": ((0, 1, 1), (0, 1, 1, 12)),  # modèle « airline »
}

# Ordre utilisé par prevoir(), donc par scripts/executer_evaluation.py.
# CONFIRMÉ par la validation : c'est l'ordre qui donne le plus petit MAE moyen sur les blocs de 
# # validation. L'interface commune ne reçoit pas le nom de la série : c'est donc
# un seul ordre pour les deux séries (mais_ON et canola_SK).
ORDRE_RETENU = ORDRES_CANDIDATS["SARIMA(0,1,1)(0,1,1)12"]


def ajuster(historique, ordre=ORDRE_RETENU):
    """Estime un SARIMA sur l'historique (log des prix) ; retourne le résultat statsmodels.

    Utile seul pour examiner les coefficients : ajuster(y).summary().
    """
    (p, d, q), saison = ordre
    modele = SARIMAX(historique, order=(p, d, q), seasonal_order=saison)
    # L'optimiseur avertit parfois qu'il n'a pas parfaitement convergé ; on
    # masque ces messages pour garder la console lisible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return modele.fit(disp=False)


def prevoir(historique, pas, ordre=ORDRE_RETENU):
    """Prévision SARIMA du log des prix, avec intervalles.

    historique : log des prix d'une série, index mensuel, connu jusqu'à l'origine.
    pas : nombres de pas après la dernière donnée (ex. [3, 5, 8]).
    ordre : ((p, d, q), (P, D, Q, 12)) ; par défaut ORDRE_RETENU.
    Retourne un DataFrame, une ligne par pas dans l'ordre reçu, colonnes
    pas, prevision, inf80, sup80, inf95, sup95 (en log).
    """
    y = verifier_historique(historique).asfreq("MS")
    pas = verifier_pas(pas)

    resultat = ajuster(y, ordre)
    # Une seule prévision jusqu'au plus grand pas, puis on garde les pas demandés.
    prevision = resultat.get_forecast(steps=max(pas))
    lignes = [k - 1 for k in pas]  # le pas k est la ligne k - 1 (on compte depuis 0)

    sortie = {"pas": pas, "prevision": prevision.predicted_mean.to_numpy()[lignes]}
    for niveau in config.NIVEAUX_INTERVALLE:
        bornes = prevision.conf_int(alpha=1 - niveau).to_numpy()[lignes]
        pct = int(round(niveau * 100))
        sortie[f"inf{pct}"] = bornes[:, 0]
        sortie[f"sup{pct}"] = bornes[:, 1]
    return pd.DataFrame(sortie, columns=COLONNES)


def comparer_ordres_en_validation(prix, ordres=ORDRES_CANDIDATS, nb_blocs=None):
    """MAE de validation (en $/t) de chaque ordre candidat et du naïf, par bloc et par horizon.

    prix : pd.Series de prix en dollars, index mensuel (toute la série).
    nb_blocs : limite aux nb_blocs derniers blocs de validation (essai rapide) ;
        None = les six blocs de config.PERIODES_VALIDATION.
    Retourne un DataFrame : ordre, bloc, horizon, mae, n.
    Le naïf de Sabra est ajouté comme ligne de référence (« Naïf (référence) »).
    """
    # Une fonction de prévision par ligne du tableau : les ordres candidats,
    # puis le naïf. ordre=ordre fige l'ordre de chaque ligne dans la boucle.
    fonctions = {nom: (lambda h, p, ordre=ordre: prevoir(h, p, ordre))
                 for nom, ordre in ordres.items()}
    fonctions["Naïf (référence)"] = naif.prevoir

    log_prix = np.log(prix)
    blocs = config.PERIODES_VALIDATION[-nb_blocs:] if nb_blocs else config.PERIODES_VALIDATION
    lignes = []
    for nom, fonction in fonctions.items():
        for debut, fin in blocs:
            paires = generer_paires_origine_cible(debut, fin)
            erreurs = []
            # Origine glissante : une estimation par origine, sur les données
            # jusqu'à l'origine incluse seulement.
            for origine, groupe in paires.groupby("date_origine"):
                p = fonction(log_prix.loc[:origine], groupe["pas"])
                prevision_dollars = np.exp(p["prevision"].to_numpy())  # retour en $
                reel = prix.loc[groupe["date_cible"]].to_numpy()
                erreurs.append(pd.DataFrame({
                    "horizon": groupe["horizon"].to_numpy(),
                    "erreur_abs": np.abs(reel - prevision_dollars),
                }))
            erreurs = pd.concat(erreurs)
            for horizon, g in erreurs.groupby("horizon"):
                lignes.append({"ordre": nom, "bloc": numero_periode_validation(debut),
                               "horizon": horizon, "mae": g["erreur_abs"].mean(),
                               "n": len(g)})
    return pd.DataFrame(lignes)


def principal():
    """Essai en ligne de commande : compare les ordres candidats sur une série."""
    parseur = argparse.ArgumentParser(description="Essai du modèle SARIMA")
    parseur.add_argument("--serie", default=config.LISTE_SERIES[0], choices=config.LISTE_SERIES)
    parseur.add_argument("--blocs", type=int, default=6,
                         help="nombre de blocs de validation (1 à 6) ; 1 = essai rapide")
    parseur.add_argument("--csv", help="fichier où enregistrer le tableau détaillé (facultatif)")
    args = parseur.parse_args()

    series = charger_series()
    prix = series.loc[series["serie"] == args.serie].set_index("date")["prix"].asfreq("MS")

    # 1) Coefficients de chaque ordre candidat, estimés sur tout l'entraînement
    #    (avant le premier bloc de validation).
    fin_entrainement = pd.Timestamp(config.PERIODES_VALIDATION[0][0]) - pd.DateOffset(months=1)
    y = np.log(prix.loc[:fin_entrainement])
    for nom, ordre in ORDRES_CANDIDATS.items():
        print(f"\n--- {nom}, {args.serie}, estimé sur {y.index[0]:%Y-%m} à {fin_entrainement:%Y-%m} ---")
        print(ajuster(y, ordre).summary().tables[1])

    # 2) Comparaison des ordres candidats (et du naïf) en validation.
    tableau = comparer_ordres_en_validation(prix, nb_blocs=args.blocs)
    print(f"\nMAE de validation ($/t), {args.serie}, moyenne sur {args.blocs} bloc(s) :")
    print(tableau.pivot_table(index="ordre", columns="horizon", values="mae").round(2))
    print(f"\nMAE de validation ($/t), {args.serie}, par bloc (moyenne des 3 horizons) :")
    print(tableau.pivot_table(index="ordre", columns="bloc", values="mae").round(2))
    if args.csv:
        tableau.assign(serie=args.serie).to_csv(args.csv, index=False)
        print(f"\nTableau détaillé enregistré : {args.csv}")


if __name__ == "__main__":
    principal()
