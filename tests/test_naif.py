"""Tests du modèle naïf (src/modeles/naif.py)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import config
from src.modeles import naif

PAS = list(config.PAS_PAR_HORIZON.values())  # [3, 5, 8]


@pytest.fixture
def historique():
    """Log d'une marche aléatoire synthétique de 120 mois."""
    rng = np.random.default_rng(5103)
    index = pd.date_range("2000-01-01", periods=120, freq="MS")
    return pd.Series(np.log(200) + np.cumsum(rng.normal(0, 0.05, len(index))), index=index)


def test_colonnes_et_prevision(historique):
    resultat = naif.prevoir(historique, PAS)
    assert list(resultat.columns) == ["pas", "prevision", "inf80", "sup80", "inf95", "sup95"]
    assert resultat["pas"].tolist() == PAS
    assert np.allclose(resultat["prevision"], historique.iloc[-1])
    assert not resultat.isna().any().any()


def test_ordre_des_bornes(historique):
    r = naif.prevoir(historique, [1, 2, 3, 5, 8, 12])
    assert ((r["inf95"] <= r["inf80"]) & (r["inf80"] <= r["prevision"])
            & (r["prevision"] <= r["sup80"]) & (r["sup80"] <= r["sup95"])).all()


def test_elargissement_avec_le_pas(historique):
    r = naif.prevoir(historique, [1, 3, 5, 8])
    for niveau in ("80", "95"):
        largeur = (r[f"sup{niveau}"] - r[f"inf{niveau}"]).to_numpy()
        assert np.all(np.diff(largeur) > 0)
        # La largeur est proportionnelle à la racine du nombre de pas
        assert np.allclose(largeur / largeur[0], np.sqrt([1, 3, 5, 8]))


def test_formule_des_intervalles(historique):
    r = naif.prevoir(historique, PAS)
    s = np.std(np.diff(historique.to_numpy()), ddof=1)
    for niveau in config.NIVEAUX_INTERVALLE:
        z = stats.norm.ppf(0.5 + niveau / 2)
        pct = int(round(niveau * 100))
        assert np.allclose(r[f"sup{pct}"], historique.iloc[-1] + z * s * np.sqrt(PAS))
        assert np.allclose(r[f"inf{pct}"], historique.iloc[-1] - z * s * np.sqrt(PAS))


def test_ordre_des_pas_conserve(historique):
    assert naif.prevoir(historique, [8, 3, 5])["pas"].tolist() == [8, 3, 5]


def test_absence_de_fuite(historique):
    """Les données postérieures à l'origine n'influencent pas la prévision."""
    origine = historique.index[80]
    modifie = historique.copy()
    modifie[modifie.index > origine] += 10.0
    attendu = naif.prevoir(historique[:origine], PAS)
    pd.testing.assert_frame_equal(naif.prevoir(modifie[:origine], PAS), attendu)
    # Même résultat que si la série s'arrêtait physiquement à l'origine
    tronque = pd.Series(historique[:origine].to_numpy(), index=historique[:origine].index)
    pd.testing.assert_frame_equal(naif.prevoir(tronque, PAS), attendu)


def test_entree_non_series():
    with pytest.raises(TypeError):
        naif.prevoir([1.0, 2.0, 3.0], PAS)


def test_index_non_temporel():
    with pytest.raises(TypeError):
        naif.prevoir(pd.Series([1.0, 2.0, 3.0]), PAS)


def test_valeur_manquante(historique):
    h = historique.copy()
    h.iloc[10] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        naif.prevoir(h, PAS)


def test_mois_absent(historique):
    with pytest.raises(ValueError, match="mensuel"):
        naif.prevoir(historique.drop(historique.index[50]), PAS)


def test_index_non_croissant(historique):
    with pytest.raises(ValueError, match="mensuel"):
        naif.prevoir(historique.iloc[::-1], PAS)


def test_historique_trop_court(historique):
    with pytest.raises(ValueError, match="trop court"):
        naif.prevoir(historique.iloc[:2], PAS)


@pytest.mark.parametrize("pas", [[], [0], [-1], [1.5], [3, 0]])
def test_pas_invalides(historique, pas):
    with pytest.raises(ValueError, match="pas"):
        naif.prevoir(historique, pas)
