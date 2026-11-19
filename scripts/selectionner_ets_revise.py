"""Sélection ETS sur validation : trois variantes, six structures par variante."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.modeles.ets import CONFIGURATIONS, VARIANTES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    racine = config.FICHIER_SERIES.parent.parent
    sortie = config.DOSSIER_RESULTATS / 'ets_selection_revisee'
    empreinte = hashlib.sha256(config.FICHIER_SERIES.read_bytes()).hexdigest()
    subprocess.run([sys.executable, '-m', 'scripts.comparer_ets_echelles_v2',
        '--racine', str(racine), '--sortie', str(sortie),
        '--workers', str(args.workers), '--validation-seulement'], cwd=racine, check=True)
    if hashlib.sha256(config.FICHIER_SERIES.read_bytes()).hexdigest() != empreinte:
        raise ValueError('Données modifiées pendant la sélection.')
    classement = pd.read_csv(sortie / 'classement_validation.csv')
    if len(classement) != len(config.LISTE_SERIES)*len(VARIANTES)*len(CONFIGURATIONS):
        raise ValueError('Certains candidats sont incomplets : examiner echecs_validation.csv.')
    gagnants = classement.sort_values(['serie','mae_moyenne','variante','configuration']).groupby('serie').head(1)
    if set(gagnants.serie) != set(config.LISTE_SERIES):
        raise ValueError('Séries incomplètes.')
    choix = {r.serie: dict(variante=r.variante, configuration=r.configuration)
             for r in gagnants.itertuples(index=False)}
    document = dict(choix=choix, empreinte_series=empreinte,
        periodes_validation=[list(p) for p in config.PERIODES_VALIDATION],
        horizons=config.HORIZONS, delai_publication=config.DELAI_PUBLICATION,
        date_debut_entrainement=config.DATE_DEBUT_ENTRAINEMENT,
        configurations_candidates=CONFIGURATIONS, variantes=VARIANTES,
        critere='Moyenne non pondérée des MAE par horizon, en dollars ; validation seule',
        remarque='Protocole révisé après consultation de tests antérieurs ; évaluation historique.')
    (sortie / 'configurations_retenues.json').write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    gagnants.to_csv(sortie / 'choix_final_validation.csv',index=False)
    print('\nCHOIX ETS FINAL SUR VALIDATION\n'+gagnants.to_string(index=False))


if __name__ == '__main__':
    main()
