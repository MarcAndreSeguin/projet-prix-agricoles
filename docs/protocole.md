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
- Découpage chronologique (proposé dans le notebook de prétraitement, **à valider par l'équipe**) : entraînement depuis 1985-01 ; trois validations de 12 mois avec entraînement croissant (2022-09 à 2023-08, 2023-09 à 2024-08, 2024-09 à 2025-08) ; test final réservé de 2025-09 à 2026-08. Mêmes périodes pour les quatre séries et les trois méthodes.

### À confirmer par l'équipe (repris de la première version du protocole)
- Transformation : log des prix nominaux ; prévisions affichées en dollars
- Délai de publication : environ 2 mois. Horizons « consultation » 1, 3, 6 mois = 3, 5, 8 pas après la dernière donnée. Avec un test de 12 mois, l'horizon de 8 pas ne laisse que 4 origines évaluables.
- Manière de prévoir sur une période de 12 mois : une seule origine (1 à 12 pas) ou origine glissante (un pas à la fois)
- Métriques : MAE, RMSE, MAPE par horizon ; test de Diebold-Mariano ; taux de couverture des intervalles
- Intervalles : 80 % (affichage), 95 % (alertes)
- Modèle final : plus faible MAE moyenne sur les horizons, à condition de battre le naïf (Diebold-Mariano, 5 %)
- Alerte : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon 1

## Contrat des fichiers (ne pas modifier sans accord de l'équipe)
- data/series.csv : date (AAAA-MM-01), serie, produit, province, prix, prix_impute (True si la valeur est estimée)
- resultats/previsions.csv : serie, modele, date_origine, date_cible, horizon, prevision, inf80, sup80, inf95, sup95, reel
- resultats/previsions_futures.csv : mêmes colonnes sans reel, date_origine = dernière donnée
- resultats/metriques.csv : modele, serie, horizon, mae, rmse, mape, couverture80, couverture95, n
- resultats/tests_dm.csv : serie, horizon, modele_a, modele_b, stat_dm, p_valeur, meilleur
- date_origine = dernier mois de données connu ; horizon ∈ {1, 3, 6}
- Valeurs de modele : naif, ets, sarima
- Valeurs de serie : mais_QC, mais_ON, canola_SK, canola_AB

## Conventions
- Python 3.12, Windows. Interface Streamlit multipage + Plotly, entièrement en français.
- L'application ne calcule aucun modèle : elle lit uniquement data/series.csv et resultats/.
- Commande unique : streamlit run app.py
- Code et commentaires en français, noms de fichiers sans accents.
- Les textes d'analyse, d'interprétation et de recommandation au client sont rédigés par l'équipe.
