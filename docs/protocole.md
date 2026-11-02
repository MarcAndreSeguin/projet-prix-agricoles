# Projet INF-5103 (UQO) : outil de suivi et de prévision des prix agricoles
Projet de session en équipe de trois. Remise du rapport : 22 novembre 2026.
Client : analyste d'Agriculture et Agroalimentaire Canada qui rédige la note mensuelle de conjoncture et doit repérer les mouvements de prix anormaux.
Données : Statistique Canada, tableau 32-10-0077-01 (prix des produits agricoles, mensuel, $/tonne), Licence du gouvernement ouvert – Canada. Fichier brut : data/brut/32100077.csv (non versionné).

## Séries (6)
- mais_QC, mais_ON : "Corn for grain"
- soya_QC, soya_ON : "Soybeans"
- ble_QC : "Wheat (except durum wheat)" ; ble_ON : "Ontario wheat excluding payments" (définition différente, à signaler)
- Les noms de produits du CSV contiennent un code entre crochets à retirer. Période commune : 1992-01 à 2026-08 (416 mois, aucun trou).

## Décisions méthodologiques
- Modèles : naïf (référence), ETS/Holt-Winters, SARIMA, gradient boosting (LightGBM, modèle global sur les 6 séries)
- Transformation : log des prix nominaux ; prévisions affichées en dollars
- Délai de publication : environ 2 mois. Horizons « consultation » 1, 3, 6 mois = 3, 5, 8 pas après la dernière donnée
- Entraînement depuis 1992-01 ; test par origine glissante de 2019-01 à 2026-08
- Métriques : MAE, RMSE, MAPE par horizon ; test de Diebold-Mariano ; taux de couverture des intervalles
- Intervalles : 80 % (affichage), 95 % (alertes)
- Modèle final : plus faible MAE moyenne sur les 3 horizons, à condition de battre le naïf (Diebold-Mariano, 5 %)
- Alerte : prix observé hors de l'intervalle à 95 % de la prévision à l'horizon 1

## Contrat des fichiers (ne pas modifier sans accord de l'équipe)
- data/series.csv : date (AAAA-MM-01), serie, produit, province, prix
- resultats/previsions.csv : serie, modele, date_origine, date_cible, horizon, prevision, inf80, sup80, inf95, sup95, reel
- resultats/previsions_futures.csv : mêmes colonnes sans reel, date_origine = dernière donnée
- resultats/metriques.csv : modele, serie, horizon, mae, rmse, mape, couverture80, couverture95, n
- resultats/tests_dm.csv : serie, horizon, modele_a, modele_b, stat_dm, p_valeur, meilleur
- date_origine = dernier mois de données connu ; horizon ∈ {1, 3, 6}
- Valeurs de modele : naif, ets, sarima, gbm

## Conventions
- Python 3.12, Windows. Interface Streamlit multipage + Plotly, entièrement en français.
- L'application ne calcule aucun modèle : elle lit uniquement data/series.csv et resultats/.
- Commande unique : streamlit run app.py
- Code et commentaires en français, noms de fichiers sans accents.
- Les textes d'analyse, d'interprétation et de recommandation au client sont rédigés par l'équipe.
