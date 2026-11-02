# Suivi et prévision des prix agricoles

Outil de suivi et de prévision des prix mensuels du maïs-grain, du soya et du blé au Québec et en Ontario, à partir du tableau 32-10-0077-01 de Statistique Canada.
Il compare plusieurs modèles de prévision (naïf, ETS, SARIMA, gradient boosting) et signale les prix observés qui sortent des intervalles de prévision.
Projet de session du cours INF-5103 (UQO) ; le protocole complet se trouve dans [docs/protocole.md](docs/protocole.md).

## Fonctionnalités

L'application Streamlit est entièrement en français. Elle ne calcule aucun modèle : elle lit uniquement `data/series.csv` et les fichiers de `resultats/`.

- **Tableau de bord** (accueil) : date de la dernière donnée disponible et rappel du délai de publication ; pour les 6 séries, dernier prix, variations sur 1 et 12 mois et indicateur d'alerte ; résumé chiffré des alertes.
- **Historique** : prix depuis 1992, comparaison entre provinces ou entre produits, décomposition STL (tendance, saisonnalité, résidu), profil saisonnier moyen.
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

1. Télécharger le tableau 32-10-0077-01 de Statistique Canada au format CSV (tableau complet).
2. Placer le fichier `32100077.csv` dans `data/brut/` (ce dossier n'est pas versionné).
3. Exécuter :

```bat
python scripts\preparer_donnees.py
```

Le script retire les codes entre crochets des noms de produits, garde les 6 séries à partir de 1992-01 et affiche les vérifications : nombre de mois par série, valeurs manquantes, unité ($/tonne métrique), dates consécutives. Si une vérification échoue, `data/series.csv` n'est pas écrit.

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
  modeles/                      naif.py, ets.py, sarima.py, gbm.py
scripts/
  preparer_donnees.py           data/brut/32100077.csv -> data/series.csv
  executer_evaluation.py        évaluation par origine glissante -> resultats/ (à venir)
  generer_resultats_factices.py résultats factices pour le développement
data/
  series.csv                    séries au format long (généré)
  brut/                         fichier brut de Statistique Canada (non versionné)
resultats/                      previsions.csv, previsions_futures.csv, metriques.csv, tests_dm.csv
docs/protocole.md               protocole et contrat des fichiers
notebooks/                      explorations
rapport/figures/                figures du rapport
requirements.txt                dépendances (versions figées)
```

## Données

Statistique Canada, tableau 32-10-0077-01, *Prix des produits agricoles, mensuel*, diffusé sous la Licence du gouvernement ouvert – Canada.
