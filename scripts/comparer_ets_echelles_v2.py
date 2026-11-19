"""Compare trois variantes ETS sur six structures par série.

Variantes : additive sur logarithmes, additive sur prix bruts et
multiplicative sur prix bruts. Le type s'applique à toutes les composantes
présentes. La tendance peut être absente, présente ou amortie ; la
saisonnalité, lorsqu'elle est présente, a une période de 12 mois.

Avec --validation-seulement, seule la validation est évaluée.
Sinon, le meilleur candidat de chaque variante, choisi sur validation,
est également évalué sur les périodes de test.

Le script selectionner_ets_revise utilise les résultats de validation
pour choisir la variante et la configuration finales de chaque série.
Ce script ne calcule pas les intervalles de prévision.
"""
import argparse,json,os,sys,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
CONFIGS={
 'sans_tendance':(None,False,False),'tendance':('add',False,False),
 'tendance_amortie':('add',True,False),'saisonnalite':(None,False,True),
 'tendance_saisonnalite':('add',False,True),'tendance_amortie_saisonnalite':('add',True,True)}
VARIANTES={
 'log_additif':dict(error='add',trend='add',seasonal='add',log=True),
 'brut_additif':dict(error='add',trend='add',seasonal='add',log=False),
 'brut_multiplicatif':dict(error='mul',trend='mul',seasonal='mul',log=False),
}
def options_modele(variante, configuration):
 """Les six structures sont identiques ; le type des composantes dépend de la variante."""
 tendance_presente,amortie,saison_presente=CONFIGS[configuration]
 v=VARIANTES[variante]
 return dict(error=v['error'],trend=v['trend'] if tendance_presente else None,
             damped_trend=amortie,seasonal=v['seasonal'] if saison_presente else None,
             seasonal_periods=12 if saison_presente else None,
             initialization_method='estimated')
def calculer(job):
 serie,variante,nom,prix,paires=job
 log=VARIANTES[variante]['log']
 options=options_modele(variante,nom)
 lignes=[];echecs=[];avert=[]
 for origine,g in paires.groupby('date_origine'):
  y=prix.loc[prix.attrs['debut_entrainement']:origine].asfreq('MS')
  if log:y=np.log(y)
  try:
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    mod=ETSModel(y,**options)
    r=mod.fit(disp=False,maxiter=2000)
    if not r.mle_retvals.get('converged',False):raise RuntimeError('Non convergence')
    f=np.asarray(r.forecast(int(g.pas.max())),dtype=float)
    if log:f=np.exp(f)
   for w in ws:avert.append(dict(serie=serie,variante=variante,configuration=nom,origine=str(origine),raison=str(w.message)))
   if not np.isfinite(f).all():raise ValueError('Prévisions non finies')
   for x in g.itertuples(index=False):
    lignes.append(dict(serie=serie,variante=variante,configuration=nom,periode=x.periode,horizon=x.horizon,date_origine=origine,date_cible=x.date_cible,prevision=float(f[int(x.pas)-1]),reel=float(prix.loc[x.date_cible])))
  except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError) as e:
   echecs.append(dict(serie=serie,variante=variante,configuration=nom,origine=str(origine),raison=str(e)))
 return serie,variante,nom,lignes,echecs,avert
def evaluer(jobs,out,phase,workers):
 lignes=[];echecs=[];avert=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  fs={pool.submit(calculer,j):j for j in jobs}
  for f in as_completed(fs):
   serie,var,nom,ls,es,ws=f.result();lignes+=ls;echecs+=es;avert+=ws
   print(f'{phase}: {serie} / {var} / {nom}: {len(ls)} prévisions, {len(es)} échecs',flush=True)
 pd.DataFrame(lignes).to_csv(out/f'previsions_{phase}.csv',index=False)
 pd.DataFrame(echecs,columns=['serie','variante','configuration','origine','raison']).to_csv(out/f'echecs_{phase}.csv',index=False)
 pd.DataFrame(avert,columns=['serie','variante','configuration','origine','raison']).to_csv(out/f'avertissements_{phase}.csv',index=False)
 return pd.DataFrame(lignes)
def metriques(p):
 p=p.copy();p['ea']=(p.reel-p.prevision).abs();p['ec']=(p.reel-p.prevision)**2;p['ape']=100*p.ea/p.reel.abs()
 return p.groupby(['serie','variante','configuration','periode','horizon']).agg(mae=('ea','mean'),rmse=('ec',lambda s:np.sqrt(s.mean())),mape=('ape','mean'),n=('ea','size'),previsions_negatives=('prevision',lambda s:int((s<0).sum()))).reset_index()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--racine',default='.');ap.add_argument('--sortie',default='resultats/ets_exploratoire_v2');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--validation-seulement',action='store_true');args=ap.parse_args()
 if args.workers<1:ap.error('--workers doit être supérieur ou égal à 1')
 root=Path(args.racine).resolve();out=Path(args.sortie).resolve();out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(root));from src import config
 from src.evaluation import generer_paires_evaluation
 paires=generer_paires_evaluation();vp=paires[paires.periode.eq('validation')];tp=paires[~paires.periode.eq('validation')]
 data=pd.read_csv(root/'data/series.csv',parse_dates=['date']);prix={s:g.set_index('date').prix.sort_index() for s,g in data.groupby('serie') if s in config.LISTE_SERIES}
 for s,y in prix.items():
  y.attrs['debut_entrainement']=config.DATE_DEBUT_ENTRAINEMENT
  if not np.isfinite(y.to_numpy()).all() or (y<=0).any() or y.index.duplicated().any():raise ValueError(f'Série invalide {s}')
 if set(prix)!=set(config.LISTE_SERIES):raise ValueError('Certaines séries configurées sont absentes des données.')
 definitions=[dict(variante=v,configuration=c,**options_modele(v,c),log=VARIANTES[v]['log']) for v in VARIANTES for c in CONFIGS]
 pd.DataFrame(definitions).to_csv(out/'definitions_configurations.csv',index=False)
 jobs=[(s,v,c,y,vp) for s,y in prix.items() for v in VARIANTES for c in CONFIGS]
 pv=evaluer(jobs,out,'validation',args.workers);mv=metriques(pv)
 attendu=vp.groupby('horizon').size().to_dict();complets=[]
 for keys,g in mv.groupby(['serie','variante','configuration']):
  if g.set_index('horizon').n.to_dict()==attendu:complets.append(keys)
 mv['complet']=mv.apply(lambda r:(r.serie,r.variante,r.configuration) in complets,axis=1)
 mv.to_csv(out/'metriques_validation.csv',index=False)
 classement=mv[mv.complet].pivot(index=['serie','variante','configuration'],columns='horizon',values='mae').reset_index();classement['mae_moyenne']=classement[config.HORIZONS].mean(axis=1)
 classement=classement.sort_values(['serie','variante','mae_moyenne','configuration']);classement.to_csv(out/'classement_validation.csv',index=False)
 retenus=classement.groupby(['serie','variante'],sort=False).head(1);retenus.to_csv(out/'configurations_retenues.csv',index=False)
 jobs=[] if args.validation_seulement else [(r.serie,r.variante,r.configuration,prix[r.serie],tp) for r in retenus.itertuples(index=False)]
 pt=evaluer(jobs,out,'tests',args.workers) if jobs else pd.DataFrame()
 if len(pt):
  mt=metriques(pd.concat([pt,pt.assign(periode='test_global')],ignore_index=True));mt.to_csv(out/'metriques_tests.csv',index=False)
  resume=mt.groupby(['serie','variante','configuration','periode']).agg(mae_moyenne=('mae','mean'),n_total=('n','sum')).reset_index();resume.to_csv(out/'resume_tests.csv',index=False)
  effectifs=tp.groupby('horizon').size().to_dict()
  for _,g in mt[mt.periode.eq('test_global')].groupby(['serie','variante']):
   if g.set_index('horizon').n.to_dict()!=effectifs:print('ATTENTION : test incomplet',flush=True)
 info={'version_script':2,'statsmodels':statsmodels.__version__,'numpy':np.__version__,'pandas':pd.__version__,'validation_cibles':[str(vp.date_cible.min()),str(vp.date_cible.max())],'horizons':config.HORIZONS,'pas':config.PAS_PAR_HORIZON,'maxiter':2000,'but':'exploratoire, ne remplace pas le protocole officiel','previsions':'forecast direct pour brut ; exp(forecast) sans correction de biais pour log ; pas de troncature des prévisions négatives','variantes':VARIANTES}
 (out/'protocole_execution.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 print('\nSÉLECTION VALIDATION\n'+retenus.to_string(index=False),flush=True)
if __name__=='__main__':main()
