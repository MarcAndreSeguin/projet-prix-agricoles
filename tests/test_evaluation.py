"""Tests du découpage, du contrat des fichiers de resultats/ et de la règle du modèle final."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.chargement import charger_resultat, chemin_resultat
from src.evaluation import (
    calculer_metriques,
    calculer_tests_dm,
    choisir_modele_final,
    generer_paires_origine_cible,
)

FICHIERS = ["previsions", "previsions_futures", "metriques", "tests_dm"]


# ---------------------------------------------------------------------------
# Découpage
# ---------------------------------------------------------------------------
def test_aucune_cible_de_validation_dans_le_test():
    debut_test = pd.Timestamp(config.DATE_DEBUT_TEST)
    for debut, fin in config.PERIODES_VALIDATION:
        paires = generer_paires_origine_cible(debut, fin)
        assert (paires["date_cible"] < debut_test).all()
        assert paires["date_cible"].between(pd.Timestamp(debut), pd.Timestamp(fin)).all()


def test_cibles_de_test_dans_le_test():
    paires = generer_paires_origine_cible(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)
    assert paires["date_cible"].between(pd.Timestamp(config.DATE_DEBUT_TEST),
                                        pd.Timestamp(config.DATE_FIN_TEST)).all()


def test_blocs_de_validation_disjoints():
    cibles = pd.concat([generer_paires_origine_cible(d, f)["date_cible"]
                        for d, f in config.PERIODES_VALIDATION]).drop_duplicates()
    nb_mois = sum(len(pd.date_range(d, f, freq="MS")) for d, f in config.PERIODES_VALIDATION)
    assert len(cibles) == nb_mois


def test_origine_egale_cible_moins_pas():
    paires = generer_paires_origine_cible(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)
    for ligne in paires.itertuples():
        assert ligne.date_origine == ligne.date_cible - pd.DateOffset(months=ligne.pas)
        assert ligne.pas == config.PAS_PAR_HORIZON[ligne.horizon]


# ---------------------------------------------------------------------------
# Fichiers de resultats/
# ---------------------------------------------------------------------------
@pytest.fixture(params=FICHIERS)
def nom_fichier(request):
    if not chemin_resultat(request.param).exists():
        pytest.skip(f"resultats/{request.param}.csv absent")
    return request.param


def test_colonnes_conformes_au_contrat(nom_fichier):
    # charger_resultat lève ValueError si les colonnes ou les valeurs de periode
    # ne respectent pas le contrat
    tableau = charger_resultat(nom_fichier)
    attendues = {
        "previsions": config.COLONNES_PREVISIONS,
        "previsions_futures": config.COLONNES_PREVISIONS_FUTURES,
        "metriques": config.COLONNES_METRIQUES,
        "tests_dm": config.COLONNES_TESTS_DM,
    }[nom_fichier]
    assert list(tableau.columns) == attendues


@pytest.fixture
def resultats():
    if not all(chemin_resultat(nom).exists() for nom in ("previsions", "metriques")):
        pytest.skip("resultats/ incomplet")
    return charger_resultat("previsions"), charger_resultat("metriques")


def test_periode_coherente_avec_les_dates(resultats):
    previsions, _ = resultats
    dans_test = previsions["date_cible"] >= pd.Timestamp(config.DATE_DEBUT_TEST)
    assert (previsions.loc[dans_test, "periode"] == "test").all()
    assert (previsions.loc[~dans_test, "periode"] == "validation").all()


def test_metriques_sans_melange_des_periodes(resultats):
    """Chaque ligne de metriques.csv ne compte que les prévisions de sa période."""
    previsions, metriques = resultats
    cles = ["modele", "serie", "periode", "horizon"]
    comptes = previsions.dropna(subset=["reel"]).groupby(cles).size().rename("attendu")
    comparaison = metriques.set_index(cles)["n"].to_frame().join(comptes)
    assert (comparaison["n"] == comparaison["attendu"]).all()
    assert not metriques.duplicated(cles).any()
    nb_validation = sum(len(pd.date_range(d, f, freq="MS")) for d, f in config.PERIODES_VALIDATION)
    nb_test = len(pd.date_range(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST, freq="MS"))
    attendu = metriques["periode"].map({"validation": nb_validation, "test": nb_test})
    assert (metriques["n"] == attendu).all()


# ---------------------------------------------------------------------------
# Calculs par période
# ---------------------------------------------------------------------------
def _previsions_synthetiques():
    """Deux modèles, deux périodes : ets meilleur en validation, naïf meilleur en test.

    Les erreurs varient d'un mois à l'autre (sinon la variance des écarts de perte est
    nulle et le test de Diebold-Mariano n'est pas calculable).
    """
    lignes = []
    for periode, debut, erreurs in [("validation", "2020-01-01", {"naif": 10.0, "ets": 2.0}),
                                    ("test", "2024-01-01", {"naif": 1.0, "ets": 30.0})]:
        for rang, date in enumerate(pd.date_range(debut, periods=24, freq="MS")):
            for modele, erreur in erreurs.items():
                signe = 1 if date.month % 2 else -1
                variation = 1 + 0.3 * np.sin(rang + (0 if modele == "naif" else 1.5))
                lignes.append({
                    "serie": "mais_QC", "modele": modele, "periode": periode,
                    "date_origine": date - pd.DateOffset(months=3), "date_cible": date, "horizon": 1,
                    "prevision": 100 + signe * erreur * variation, "inf80": 0.0, "sup80": 1e9,
                    "inf95": 0.0, "sup95": 1e9, "reel": 100.0,
                })
    return pd.DataFrame(lignes)


def test_calculs_separes_par_periode():
    previsions = _previsions_synthetiques()
    metriques = calculer_metriques(previsions).set_index(["modele", "periode"])
    erreurs = previsions.assign(e=(previsions["reel"] - previsions["prevision"]).abs())
    attendu = erreurs.groupby(["modele", "periode"])["e"].mean()
    assert metriques.loc[("ets", "validation"), "mae"] == pytest.approx(attendu[("ets", "validation")])
    assert metriques.loc[("ets", "test"), "mae"] == pytest.approx(attendu[("ets", "test")])
    assert metriques.loc[("ets", "validation"), "mae"] < 5 < metriques.loc[("ets", "test"), "mae"]
    assert set(calculer_tests_dm(previsions, ["naif", "ets"])["periode"]) == {"validation", "test"}


def test_calculs_refusent_une_periode_absente():
    previsions = _previsions_synthetiques().drop(columns="periode")
    with pytest.raises(ValueError, match="periode"):
        calculer_metriques(previsions)
    with pytest.raises(ValueError, match="periode"):
        calculer_tests_dm(previsions)


def test_modele_final_choisi_sur_la_validation():
    """ets bat le naïf en validation et perd en test : il est retenu quand même."""
    previsions = _previsions_synthetiques()
    metriques = calculer_metriques(previsions)
    tests_dm = calculer_tests_dm(previsions, ["naif", "ets"])
    validation = tests_dm[tests_dm["periode"] == "validation"]
    assert (validation["meilleur"] == "ets").all()
    assert choisir_modele_final(metriques, tests_dm, "mais_QC") == "ets"


def test_modele_final_ignore_le_test():
    """Un modèle meilleur sur le test seulement n'est pas retenu."""
    previsions = _previsions_synthetiques()
    # Échange des modèles : ets devient pire en validation et meilleur en test
    previsions = previsions.assign(modele=previsions["modele"].map({"naif": "ets", "ets": "naif"}))
    metriques = calculer_metriques(previsions)
    tests_dm = calculer_tests_dm(previsions, ["naif", "ets"])
    assert choisir_modele_final(metriques, tests_dm, "mais_QC") == config.MODELE_REFERENCE
