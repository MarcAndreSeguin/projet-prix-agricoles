# Projet INF-5103 (UQO) : outil de suivi et de prévision des prix agricoles
Projet de session en équipe de trois. Remise du rapport : 22 novembre 2026.
Client : analyste d'une coopérative ou d'une association de producteurs, qui suit les prix pour conseiller ses membres et doit repérer les mouvements de prix anormaux.
Données : Statistique Canada, tableau 32-10-0077-01 (prix des produits agricoles, mensuel, $/tonne), Licence du gouvernement ouvert – Canada. Fichiers sources versionnés : data/mais.csv et data/canola.csv (extraits du tableau, cp1252, dates mois/jour/année).

## Séries (2)
- mais_ON : maïs-grain, Ontario
- canola_SK : canola (y compris le colza), Saskatchewan
- Choix des provinces (décision de l'équipe) : ce sont les principales provinces productrices de ces cultures, et les provinces voisines (Québec pour le maïs-grain, Alberta pour le canola) ont des marchés pratiquement jumeaux. Le Québec et l'Alberta ne sont donc pas retenus.
- Période commune : 1985-01 à 2026-08 (500 mois, aucun mois absent).
- Aucune valeur manquante dans les deux séries retenues : aucune valeur n'est estimée. La seule valeur manquante des fichiers sources (canola, Alberta, 2012-05) appartient à une série non retenue. La colonne prix_impute est conservée dans le contrat (toutes ses valeurs valent False).
- La rupture d'août 2012 (fin du monopole de la Commission canadienne du blé) ne touche pas ces séries.
- Prétraitement détaillé : notebooks/01_pretraitement.ipynb. Fichier commun aux trois méthodes : data/series.csv, produit par scripts/preparer_donnees.py.

## Décisions méthodologiques
- Modèles : naïf (référence), ETS/Holt-Winters, SARIMA ; une personne par modèle (Sabra : naïf, Hanane : ETS, Marc : SARIMA)
- Découpage chronologique (décision de l'équipe), défini par mois **cibles**, commun aux deux séries et aux trois méthodes (src/config.py) :

| Période | Mois cibles | Mois | Rôle |
| :--- | :--- | ---: | :--- |
| Entraînement | depuis 1985-01, fenêtre croissante jusqu'à chaque origine | – | estimation des modèles |
| validation | 6 blocs de 12 mois : 2014-09 à 2015-08, 2015-09 à 2016-08, 2016-09 à 2017-08, 2017-09 à 2018-08, 2018-09 à 2019-08, 2019-09 à 2020-08 | 72 | choix du modèle final |
| test_1 | 2020-09 à 2022-08 | 24 | performances rapportées |
| test_2 | 2022-09 à 2024-08 | 24 | performances rapportées |
| test_3 | 2024-09 à 2026-08 | 24 | performances rapportées |
| test_global | 2020-09 à 2026-08 (test_1 + test_2 + test_3) | 72 | bilan sur l'ensemble des trois tests |

- Parcours de chaque période : origine glissante, modèle réestimé à chaque origine. Chaque mois cible est prévu à chaque horizon depuis l'origine date_cible − pas (pas = horizon + délai de publication). Exemple : cible 2023-01 à l'horizon 6 mois (8 pas) → origine 2022-05, entraînement de 1985-01 à 2022-05. Les origines peuvent précéder la période, mais aucune cible n'en sort : chaque prévision appartient à une seule période (validation, test_1, test_2 ou test_3). Par série, modèle et horizon : 72 prévisions en validation, 24 par période de test, 72 pour test_global. Fonctions communes : src/evaluation.py, generer_paires_origine_cible() et generer_paires_evaluation().
- Les erreurs de validation peuvent aussi être rapportées bloc par bloc (src/evaluation.py, numero_periode_validation()).
- Pas de test de Diebold-Mariano (décision de l'équipe) : les métriques des trois méthodes sont présentées clairement pour chaque période.

### Métriques
Pour chaque série, modèle, horizon et période (validation, test_1, test_2, test_3, test_global), sans jamais mélanger validation et test ; y_t est le prix observé au mois cible t, ŷ_t la prévision en dollars, [L_t, U_t] l'intervalle de niveau c, n le nombre de mois cibles :
- MAE = (1/n) Σ |y_t − ŷ_t| ($/tonne)
- RMSE = √[(1/n) Σ (y_t − ŷ_t)²] ($/tonne)
- MAPE = (100/n) Σ |y_t − ŷ_t| / |y_t| (%)
- couverture80, couverture95 = (1/n) Σ 1{L_t ≤ y_t ≤ U_t}, proportion des prix observés dans l'intervalle à 80 % ou 95 %
- test_global est calculé sur l'ensemble des prévisions de test_1, test_2 et test_3 (et non comme la moyenne des trois périodes).

### Modèle final (décision de l'équipe)
- Pour chaque série : plus faible MAE moyenne sur les horizons (1, 3 et 6 mois), calculée sur la période de validation uniquement. Le naïf est retenu s'il est le meilleur (en cas d'égalité, le naïf est préféré). Les périodes de test ne servent qu'à rapporter les performances. Fonction : src/evaluation.py, choisir_modele_final().

### À confirmer par l'équipe (repris de la première version du protocole)
- Transformation : log des prix nominaux ; prévisions affichées en dollars
- Délai de publication : environ 2 mois. Horizons « consultation » 1, 3, 6 mois = 3, 5, 8 pas après la dernière donnée. Les mois de validation et de test étant des mois cibles, chaque horizon est évalué sur les mêmes mois par période (les origines à 8 pas commencent 8 mois avant la période).
- Intervalles : 80 % (affichage), 95 % (alertes)
- Alerte : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon 1

## Contrat des fichiers (ne pas modifier sans accord de l'équipe)
- data/series.csv : date (AAAA-MM-01), serie, produit, province, prix, prix_impute (True si la valeur est estimée)
- resultats/previsions.csv : serie, modele, periode, date_origine, date_cible, horizon, prevision, inf80, sup80, inf95, sup95, reel
- resultats/previsions_futures.csv : mêmes colonnes sans periode ni reel, date_origine = dernière donnée
- resultats/metriques.csv : modele, serie, periode, horizon, mae, rmse, mape, couverture80, couverture95, n
- Colonne periode :
  - previsions.csv : validation, test_1, test_2 ou test_3 ; chaque prévision n'apparaît qu'une fois ;
  - metriques.csv : les mêmes valeurs, plus test_global, calculé sur l'ensemble des trois périodes de test.
- date_origine = dernier mois de données connu ; horizon ∈ {1, 3, 6}
- Valeurs de modele : naif, ets, sarima
- Valeurs de serie : mais_ON, canola_SK

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
