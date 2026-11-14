"""Tests des séries, du découpage, du contrat des fichiers de resultats/ et de la règle
du modèle final."""

import numpy as np
import pandas as pd
import pytest

from src import config, evaluation
from src.chargement import RESULTATS, charger_resultat, charger_series, chemin_resultat
from src.evaluation import (
    calculer_metriques,
    choisir_modele_final,
    generer_paires_evaluation,
    generer_paires_origine_cible,
    periode_de,
)

FICHIERS = ["previsions", "previsions_futures", "metriques"]


def _mois(debut, fin):
    return pd.date_range(debut, fin, freq="MS")


# ---------------------------------------------------------------------------
# Séries
# ---------------------------------------------------------------------------
def test_deux_series():
    assert config.LISTE_SERIES == ["mais_ON", "canola_SK"]
    assert config.SERIE_PAR_PRODUIT == {"mais": "mais_ON", "canola": "canola_SK"}


def test_series_csv_contient_les_deux_series():
    if not config.FICHIER_SERIES.exists():
        pytest.skip("data/series.csv absent")
    series = charger_series()
    assert list(series.columns) == config.COLONNES_SERIES
    assert set(series["serie"]) == set(config.LISTE_SERIES)
    assert not series["prix"].isna().any()
    assert (series.groupby("serie").size() == config.NB_MOIS_ATTENDU).all()


# ---------------------------------------------------------------------------
# Découpage
# ---------------------------------------------------------------------------
def test_validation_six_blocs_de_douze_mois():
    assert len(config.PERIODES_VALIDATION) == 6
    for debut, fin in config.PERIODES_VALIDATION:
        assert len(_mois(debut, fin)) == 12
    assert config.PERIODES_VALIDATION[0][0] == "2014-09-01"
    assert config.PERIODES_VALIDATION[-1][1] == "2020-08-01"


def test_periodes_de_test():
    assert config.PERIODES_TEST == {
        "test_1": ("2020-09-01", "2022-08-01"),
        "test_2": ("2022-09-01", "2024-08-01"),
        "test_3": ("2024-09-01", "2026-08-01"),
    }
    assert len(_mois(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST)) == 72


def test_periodes_contigues_et_disjointes():
    """Validation puis tests, sans trou ni chevauchement, de 2014-09 à 2026-08."""
    cibles = pd.concat([pd.Series(_mois(d, f)) for _, d, f in evaluation.bornes_periodes()])
    assert not cibles.duplicated().any()
    attendues = _mois(config.DATE_DEBUT_VALIDATION, config.DATE_FIN_TEST)
    assert pd.DatetimeIndex(cibles.sort_values()).equals(attendues)


def test_test_global_union_des_trois_tests():
    assert config.COMPOSITION_PERIODES[config.PERIODE_TEST_GLOBAL] == ["test_1", "test_2", "test_3"]
    union = pd.DatetimeIndex(sorted(m for d, f in config.PERIODES_TEST.values() for m in _mois(d, f)))
    assert union.equals(_mois(config.DATE_DEBUT_TEST, config.DATE_FIN_TEST))


def test_aucune_prevision_de_validation_dans_un_test():
    paires = generer_paires_evaluation()
    validation = paires[paires["periode"] == config.PERIODE_VALIDATION]
    assert (validation["date_cible"] <= pd.Timestamp(config.DATE_FIN_VALIDATION)).all()
    assert (validation["date_cible"] < pd.Timestamp(config.DATE_DEBUT_TEST)).all()
    for periode, (debut, fin) in config.PERIODES_TEST.items():
        cibles = paires.loc[paires["periode"] == periode, "date_cible"]
        assert cibles.between(pd.Timestamp(debut), pd.Timestamp(fin)).all()
        assert len(cibles) == len(_mois(debut, fin)) * len(config.HORIZONS)


def test_chaque_prevision_dans_une_seule_periode():
    paires = generer_paires_evaluation()
    assert set(paires["periode"]) == set(config.PERIODES_PREVISIONS)
    assert not paires.duplicated(["date_cible", "horizon"]).any()
    assert (paires["date_cible"].map(periode_de) == paires["periode"]).all()


def test_origine_egale_cible_moins_pas():
    for ligne in generer_paires_evaluation().itertuples():
        assert ligne.date_origine == ligne.date_cible - pd.DateOffset(months=ligne.pas)
        assert ligne.pas == config.PAS_PAR_HORIZON[ligne.horizon]


def test_blocs_de_validation_disjoints():
    cibles = pd.concat([generer_paires_origine_cible(d, f)["date_cible"]
                        for d, f in config.PERIODES_VALIDATION]).drop_duplicates()
    assert len(cibles) == sum(len(_mois(d, f)) for d, f in config.PERIODES_VALIDATION)


# ---------------------------------------------------------------------------
# Plus de Diebold-Mariano
# ---------------------------------------------------------------------------
def test_aucune_trace_de_diebold_mariano():
    assert "tests_dm" not in RESULTATS
    for nom in ("FICHIER_TESTS_DM", "COLONNES_TESTS_DM", "SEUIL_DM"):
        assert not hasattr(config, nom)
    for nom in ("calculer_tests_dm", "test_diebold_mariano"):
        assert not hasattr(evaluation, nom)
    assert not (config.DOSSIER_RESULTATS / "tests_dm.csv").exists()


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
    }[nom_fichier]
    assert list(tableau.columns) == attendues


@pytest.fixture
def resultats():
    if not all(chemin_resultat(nom).exists() for nom in ("previsions", "metriques")):
        pytest.skip("resultats/ incomplet")
    return charger_resultat("previsions"), charger_resultat("metriques")


def test_periodes_des_fichiers(resultats):
    previsions, metriques = resultats
    assert set(previsions["periode"]) == set(config.PERIODES_PREVISIONS)
    assert config.PERIODE_TEST_GLOBAL not in set(previsions["periode"])
    assert set(metriques["periode"]) == set(config.PERIODES_METRIQUES)
    assert set(previsions["serie"]) == set(metriques["serie"]) == set(config.LISTE_SERIES)


def test_periode_coherente_avec_les_dates(resultats):
    previsions, _ = resultats
    assert (previsions["date_cible"].map(periode_de) == previsions["periode"]).all()


def test_prevision_unique(resultats):
    previsions, _ = resultats
    assert not previsions.duplicated(["serie", "modele", "date_cible", "horizon"]).any()


def test_metriques_sans_melange_des_periodes(resultats):
    """Chaque ligne de metriques.csv ne compte que les prévisions de sa période ;
    test_global compte celles des trois tests."""
    previsions, metriques = resultats
    cles = ["modele", "serie", "periode", "horizon"]
    assert not metriques.duplicated(cles).any()
    p = previsions.dropna(subset=["reel"])
    comptes = pd.concat([evaluation.filtrer_periode(p, periode).assign(periode=periode)
                         for periode in config.PERIODES_METRIQUES])
    comptes = comptes.groupby(cles).size().rename("attendu")
    comparaison = metriques.set_index(cles)["n"].to_frame().join(comptes)
    assert (comparaison["n"] == comparaison["attendu"]).all()
    attendu = metriques["periode"].map({"validation": 72, "test_1": 24, "test_2": 24,
                                        "test_3": 24, "test_global": 72})
    assert (metriques["n"] == attendu).all()


def test_test_global_recalcule_sur_les_trois_tests(resultats):
    previsions, metriques = resultats
    tests = previsions[previsions["periode"].isin(list(config.PERIODES_TEST))].dropna(subset=["reel"])
    erreur = (tests["reel"] - tests["prevision"]).abs()
    mae = erreur.groupby([tests["modele"], tests["serie"], tests["horizon"]]).mean()
    globale = metriques[metriques["periode"] == config.PERIODE_TEST_GLOBAL]
    globale = globale.set_index(["modele", "serie", "horizon"])["mae"]
    # Les prix de previsions.csv sont arrondis à l'écriture : tolérance de 1e-3 $
    assert np.allclose(globale.sort_index(), mae.reindex(globale.index).sort_index(), atol=1e-3)


# ---------------------------------------------------------------------------
# Calculs par période
# ---------------------------------------------------------------------------
ERREURS = {
    # periode : (premier mois cible, {modèle : erreur absolue})
    "validation": ("2015-01-01", {"naif": 10.0, "ets": 2.0}),
    "test_1": ("2021-01-01", {"naif": 1.0, "ets": 30.0}),
    "test_2": ("2023-01-01", {"naif": 3.0, "ets": 20.0}),
    "test_3": ("2025-01-01", {"naif": 5.0, "ets": 10.0}),
}


def _previsions_synthetiques(erreurs=ERREURS, nb_mois=12):
    """Deux modèles : ets meilleur en validation, naïf meilleur sur chaque test."""
    lignes = []
    for periode, (debut, par_modele) in erreurs.items():
        for date in pd.date_range(debut, periods=nb_mois, freq="MS"):
            for modele, erreur in par_modele.items():
                signe = 1 if date.month % 2 else -1
                lignes.append({
                    "serie": "mais_ON", "modele": modele, "periode": periode,
                    "date_origine": date - pd.DateOffset(months=3), "date_cible": date, "horizon": 1,
                    "prevision": 100 + signe * erreur, "inf80": 0.0, "sup80": 1e9,
                    "inf95": 0.0, "sup95": 1e9, "reel": 100.0,
                })
    return pd.DataFrame(lignes)


def test_calculs_separes_par_periode():
    metriques = calculer_metriques(_previsions_synthetiques()).set_index(["modele", "periode"])
    assert set(metriques.index.get_level_values("periode")) == set(config.PERIODES_METRIQUES)
    for periode, (_, par_modele) in ERREURS.items():
        for modele, erreur in par_modele.items():
            assert metriques.loc[(modele, periode), "mae"] == pytest.approx(erreur)
            assert metriques.loc[(modele, periode), "n"] == 12


def test_test_global_union_des_tests_dans_les_metriques():
    metriques = calculer_metriques(_previsions_synthetiques()).set_index(["modele", "periode"])
    assert metriques.loc[("ets", "test_global"), "mae"] == pytest.approx((30 + 20 + 10) / 3)
    assert metriques.loc[("naif", "test_global"), "mae"] == pytest.approx((1 + 3 + 5) / 3)
    assert metriques.loc[("ets", "test_global"), "n"] == 36
    # La validation n'entre pas dans test_global
    assert metriques.loc[("ets", "test_global"), "mae"] > 5


def test_formules_des_metriques():
    previsions = _previsions_synthetiques({"validation": ("2015-01-01", {"naif": 10.0})}, nb_mois=4)
    previsions.loc[0, ["inf80", "sup80", "inf95", "sup95"]] = [0.0, 50.0, 0.0, 50.0]
    m = calculer_metriques(previsions).set_index("periode").loc["validation"]
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(10.0)
    assert m["mape"] == pytest.approx(10.0)
    assert m["couverture80"] == pytest.approx(0.75)
    assert m["couverture95"] == pytest.approx(0.75)


def test_calculs_refusent_une_periode_absente_ou_inconnue():
    with pytest.raises(ValueError, match="periode"):
        calculer_metriques(_previsions_synthetiques().drop(columns="periode"))
    with pytest.raises(ValueError, match="periode"):
        calculer_metriques(_previsions_synthetiques().assign(periode="test"))
    with pytest.raises(ValueError, match="periode"):
        calculer_metriques(_previsions_synthetiques().assign(periode="test_global"))


# ---------------------------------------------------------------------------
# Modèle final
# ---------------------------------------------------------------------------
def test_modele_final_choisi_sur_la_validation():
    """ets a la plus faible MAE en validation et perd sur tous les tests : il est retenu."""
    metriques = calculer_metriques(_previsions_synthetiques())
    assert choisir_modele_final(metriques, "mais_ON") == "ets"


def test_naif_retenu_s_il_est_le_meilleur():
    erreurs = {p: (d, {"naif": e["ets"], "ets": e["naif"]}) for p, (d, e) in ERREURS.items()}
    metriques = calculer_metriques(_previsions_synthetiques(erreurs))
    assert choisir_modele_final(metriques, "mais_ON") == config.MODELE_REFERENCE


def test_naif_prefere_en_cas_d_egalite():
    erreurs = {"validation": ("2015-01-01", {"naif": 4.0, "ets": 4.0, "sarima": 4.0})}
    metriques = calculer_metriques(_previsions_synthetiques(erreurs))
    assert choisir_modele_final(metriques, "mais_ON") == config.MODELE_REFERENCE


def test_modele_final_ignore_le_test():
    """Un modèle meilleur sur les tests seulement n'est pas retenu."""
    erreurs = dict(ERREURS)
    erreurs["validation"] = ("2015-01-01", {"naif": 2.0, "ets": 10.0})
    metriques = calculer_metriques(_previsions_synthetiques(erreurs))
    # ets devient meilleur sur test_global en changeant seulement les tests
    tests = metriques["periode"].isin([*config.PERIODES_TEST, config.PERIODE_TEST_GLOBAL])
    metriques.loc[tests & (metriques["modele"] == "ets"), "mae"] = 0.1
    assert choisir_modele_final(metriques, "mais_ON") == config.MODELE_REFERENCE


def test_modele_final_moyenne_des_horizons():
    lignes = []
    for horizon, (naif, ets) in {1: (1.0, 2.0), 3: (5.0, 2.0), 6: (9.0, 8.0)}.items():
        lignes += [{"modele": "naif", "serie": "mais_ON", "periode": "validation",
                    "horizon": horizon, "mae": naif},
                   {"modele": "ets", "serie": "mais_ON", "periode": "validation",
                    "horizon": horizon, "mae": ets}]
    metriques = pd.DataFrame(lignes)
    # Moyennes : naïf 5, ets 4 ; le naïf est meilleur à 1 mois mais ets est retenu
    assert choisir_modele_final(metriques, "mais_ON") == "ets"
    assert choisir_modele_final(metriques, "canola_SK") is None
