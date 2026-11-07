# Suivi et prévision des prix agricoles

Outil de suivi et de prévision des prix mensuels du maïs-grain (Québec, Ontario) et du canola (Saskatchewan, Alberta), à partir du tableau 32-10-0077-01 de Statistique Canada.
Il compare trois méthodes de prévision (naïf, ETS, SARIMA) et signale les prix observés qui sortent des intervalles de prévision.
Projet de session du cours INF-5103 (UQO) ; le protocole complet se trouve dans [docs/protocole.md](docs/protocole.md).

## Fonctionnalités

L'application Streamlit est entièrement en français. Elle ne calcule aucun modèle : elle lit uniquement `data/series.csv` et les fichiers de `resultats/`.

- **Tableau de bord** (accueil) : date de la dernière donnée disponible et rappel du délai de publication ; pour les 4 séries, dernier prix, variations sur 1 et 12 mois et indicateur d'alerte ; résumé chiffré des alertes.
- **Historique** : prix depuis 1985, comparaison entre provinces ou entre toutes les séries, décomposition STL (tendance, saisonnalité, résidu), profil saisonnier moyen.
- **Prévision** : 5 dernières années prolongées par la prévision du modèle choisi et son intervalle (80 % ou 95 %), tableau des prévisions à 1, 3 et 6 mois, probabilité de dépasser le dernier prix observé, export CSV.
- **Alertes** : prix observés hors de l'intervalle à 95 % de la prévision à l'horizon 1, sur graphique et dans un tableau filtrable par série.
- **Comparaison des modèles** : MAE, RMSE et MAPE par horizon, tests de Diebold-Mariano, couverture réelle des intervalles contre la couverture visée, prévisions contre valeurs réelles sur la période de test.
- **Méthodologie et limites** : source et licence des données, définition des séries, protocole d'évaluation.

Un panneau latéral commun (produit, province, modèle, niveau d'intervalle) conserve les choix d'une page à l'autre. Le modèle proposé par défaut est le modèle final de la série, selon la règle du protocole. Un bandeau d'avertissement s'affiche tant que les résultats sont factices (présence de `resultats/FACTICE.txt`).

## Prérequis

- Windows
- **Python 3.12** (requis)

## Installation

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

Commande unique :

```bat
streamlit run app.py
```

## Régénérer les données et les résultats

Toutes les commandes se lancent depuis la racine du dépôt, avec l'environnement virtuel activé.

### 1. Données : `data/series.csv`

Les fichiers sources `data/mais.csv` et `data/canola.csv` (extraits du tableau 32-10-0077-01) sont versionnés. Exécuter :

```bat
python scripts\preparer_donnees.py
```

Le script garde les 4 séries à partir de 1985-01, estime la valeur manquante de mai 2012 (canola, Alberta) par interpolation linéaire, comme dans `notebooks/01_pretraitement.ipynb`, et affiche les vérifications : nombre de mois par série, valeurs manquantes, unité ($/tonne métrique), dates consécutives. Si une vérification échoue, `data/series.csv` n'est pas écrit.

### 2. Résultats : `resultats/`

Évaluation réelle des modèles (**à venir** : le script ne contient pour l'instant qu'une docstring) :

```bat
python scripts\executer_evaluation.py
```

En développement, des résultats factices au format du contrat peuvent être générés à partir des vraies séries. Ils créent `resultats/FACTICE.txt`, qui déclenche le bandeau d'avertissement dans l'application :

```bat
python scripts\generer_resultats_factices.py
```

Ce script refuse d'écraser des résultats réels (dossier sans `FACTICE.txt`), sauf avec l'option `--forcer`.

## Structure du dépôt

```
app.py                          page d'accueil (tableau de bord)
pages/
  1_Historique.py
  2_Prevision.py
  3_Alertes.py
  4_Comparaison_modeles.py
  5_Methodologie_et_limites.py
src/
  config.py                     constantes du protocole (séries, horizons, dates, libellés)
  chargement.py                 lecture et validation de data/series.csv et de resultats/
  evaluation.py                 métriques, test de Diebold-Mariano, choix du modèle final
  interface_commune.py          panneau latéral, cache, graphiques, alertes
  modeles/                      naif.py, ets.py, sarima.py
scripts/
  preparer_donnees.py           data/mais.csv + data/canola.csv -> data/series.csv
  executer_evaluation.py        évaluation sur le découpage commun -> resultats/ (à venir)
  generer_resultats_factices.py résultats factices pour le développement
data/
  mais.csv, canola.csv          fichiers sources extraits de Statistique Canada
  series.csv                    séries au format long (généré)
resultats/                      previsions.csv, previsions_futures.csv, metriques.csv, tests_dm.csv
docs/protocole.md               protocole et contrat des fichiers
notebooks/                      01_pretraitement.ipynb (contrôles, statistiques, découpage)
rapport/figures/                figures du rapport
requirements.txt                dépendances (versions figées)
```

## Données

Statistique Canada, tableau 32-10-0077-01, *Prix des produits agricoles, mensuel*, diffusé sous la Licence du gouvernement ouvert – Canada.
