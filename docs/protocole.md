# Projet INF-5103 (UQO) : outil de suivi et de prévision des prix agricoles
Projet de session en équipe de trois. Remise du rapport : 22 novembre 2026.
Client : analyste d'une coopérative ou d'une association de producteurs, qui suit les prix pour conseiller ses membres et doit repérer les mouvements de prix anormaux.
Données : Statistique Canada, tableau 32-10-0077-01 (prix des produits agricoles, mensuel, $/tonne), Licence du gouvernement ouvert – Canada. Fichiers sources versionnés : data/mais.csv et data/canola.csv (extraits du tableau, cp1252, dates mois/jour/année).

## Séries (4)
- mais_QC, mais_ON : maïs-grain, Québec et Ontario
- canola_SK, canola_AB : canola (y compris le colza), Saskatchewan et Alberta. Le canola au Québec ne commence qu'en 2000-01 ; les Prairies permettent une période commune plus longue.
- Période commune : 1985-01 à 2026-08 (500 mois, aucun mois absent).
- Une seule valeur manquante : canola_AB 2012-05, estimée par interpolation linéaire (moyenne d'avril et de juin 2012) et signalée dans la colonne prix_impute.
- La rupture d'août 2012 (fin du monopole de la Commission canadienne du blé) ne touche pas ces séries.
- Prétraitement détaillé : notebooks/01_pretraitement.ipynb. Fichier commun aux trois méthodes : data/series.csv, produit par scripts/preparer_donnees.py.

## Décisions méthodologiques
- Modèles : naïf (référence), ETS/Holt-Winters, SARIMA ; une personne par modèle (Sabra : naïf, Hanane : ETS, Marc : SARIMA)
- Découpage chronologique (structure proposée dans le notebook de prétraitement, test allongé et six blocs de validation proposés par Marc, **à valider par l'équipe**) : entraînement depuis 1985-01 ; six validations contiguës de 12 mois avec entraînement croissant, de 2017-09 à 2023-08 (2017-09 à 2018-08, 2018-09 à 2019-08, 2019-09 à 2020-08, 2020-09 à 2021-08, 2021-09 à 2022-08, 2022-09 à 2023-08) ; test final réservé de 2023-09 à 2026-08 (36 mois). Mêmes périodes pour les quatre séries et les trois méthodes.
- Pourquoi six blocs de validation : avec trois blocs (2020-09 à 2023-08), la validation était dominée par la flambée de 2021-2022 (hausse de +42 % à +63 % sur le premier bloc), alors que le test 2023-09 à 2026-08 est plutôt stable (de −6 % à +3 %). Les six années précédant le test couvrent un régime stable (2017-2020), la flambée, le sommet de 2022 et le début de la baisse. Les erreurs de validation sont aussi rapportées bloc par bloc (src/evaluation.py, numero_periode_validation()) pour montrer l'effet du régime.
- Pourquoi 36 mois de test plutôt que 12 : le test de Diebold-Mariano (src/evaluation.py) exige au moins 2 × pas + 1 erreurs, soit 7, 11 et 17 aux horizons 1, 3 et 6 mois. Avec 12 mois de test, il est impossible à 6 mois, alors que la règle de sélection (battre le naïf) en dépend. Avec 36 mois, il est calculable à tous les horizons, et le classement ne repose plus sur une seule année. Le notebook de prétraitement garde pour l'instant le découpage d'origine (12 mois de test) ; à mettre à jour si l'équipe retient ce choix.
- Parcours de chaque période (choix proposé par Marc, **à valider par l'équipe**) : origine glissante, modèle réestimé à chaque origine. Les 12 mois d'une période sont des mois **cibles** : chaque mois cible est prévu à chaque horizon depuis l'origine date_cible − pas (pas = horizon + délai de publication). Exemple : cible 2023-01 à l'horizon 6 mois (8 pas) → origine 2022-05, entraînement de 1985-01 à 2022-05. Les origines peuvent précéder la période, mais aucune cible n'en sort, donc pas de fuite vers la validation suivante ni vers le test. Avec 3 horizons, cela donne 12 erreurs par horizon, par série et par période, soit 72 en validation (6 blocs) et 36 en test. Fonction commune : src/evaluation.py, generer_paires_origine_cible().

### À confirmer par l'équipe (repris de la première version du protocole)
- Transformation : log des prix nominaux ; prévisions affichées en dollars
- Délai de publication : environ 2 mois. Horizons « consultation » 1, 3, 6 mois = 3, 5, 8 pas après la dernière donnée. Les mois de validation et de test étant des mois cibles, chaque horizon est évalué sur les mêmes 12 mois par période (les origines à 8 pas commencent 8 mois avant la période).
- Métriques : MAE, RMSE, MAPE par horizon ; test de Diebold-Mariano ; taux de couverture des intervalles
- Intervalles : 80 % (affichage), 95 % (alertes)
- Modèle final : plus faible MAE moyenne sur les horizons, à condition de battre le naïf (Diebold-Mariano, 5 %) ; règle appliquée sur la période de validation uniquement, performances rapportées sur le test (**Proposition – à valider par l'équipe**)
- Alerte : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon 1

## Contrat des fichiers (ne pas modifier sans accord de l'équipe)
- data/series.csv : date (AAAA-MM-01), serie, produit, province, prix, prix_impute (True si la valeur est estimée)
- resultats/previsions.csv : serie, modele, periode, date_origine, date_cible, horizon, prevision, inf80, sup80, inf95, sup95, reel
- resultats/previsions_futures.csv : mêmes colonnes sans periode ni reel, date_origine = dernière donnée
- resultats/metriques.csv : modele, serie, periode, horizon, mae, rmse, mape, couverture80, couverture95, n
- resultats/tests_dm.csv : serie, periode, horizon, modele_a, modele_b, stat_dm, p_valeur, meilleur
- Colonne periode (**Proposition – à valider par l'équipe**) : « validation » (mois cibles des six blocs de validation) ou « test » (mois cibles du test final). Les métriques et les tests de Diebold-Mariano sont calculés séparément pour chaque période, sans jamais les mélanger. La règle du modèle final est appliquée sur la période de validation uniquement ; les performances sont rapportées sur le test.
- date_origine = dernier mois de données connu ; horizon ∈ {1, 3, 6}
- Valeurs de modele : naif, ets, sarima
- Valeurs de serie : mais_QC, mais_ON, canola_SK, canola_AB

## Interface commune des modèles (Proposition – à valider par l'équipe)
- Chaque modèle est un module src/modeles/<modele>.py (naif, ets, sarima) qui expose la même fonction :
  `prevoir(historique: pd.Series, pas: list[int]) -> pd.DataFrame`
- historique : log des prix d'une seule série, index mensuel (DatetimeIndex, premier du mois, croissant, sans trou ni valeur manquante), données connues jusqu'à l'origine seulement (dernière valeur = date_origine).
- pas : nombres de pas après la dernière donnée à prévoir, entiers ≥ 1 (config.PAS_PAR_HORIZON : 3, 5, 8 pour les horizons 1, 3, 6 mois).
- Retour : une ligne par pas, dans l'ordre reçu, colonnes pas, prevision, inf80, sup80, inf95, sup95, exprimées en log ; niveaux tirés de config.NIVEAUX_INTERVALLE.
- Le modèle n'utilise que l'historique reçu : il ne lit aucun fichier et n'a accès à aucune donnée postérieure à l'origine. Il est réestimé à chaque appel (une origine = un appel).
- Le script d'évaluation construit les paires origine-cible (src/evaluation.py, generer_paires_origine_cible()), appelle prevoir() à chaque origine, convertit en dollars (exponentielle) et associe chaque pas à son horizon et à sa date cible pour produire les fichiers de resultats/.

## Conventions
- Python 3.12, Windows. Interface Streamlit multipage + Plotly, entièrement en français.
- L'application ne calcule aucun modèle : elle lit uniquement data/series.csv et resultats/.
- Commande unique : streamlit run app.py
- Code et commentaires en français, noms de fichiers sans accents.
- Les textes d'analyse, d'interprétation et de recommandation au client sont rédigés par l'équipe.
