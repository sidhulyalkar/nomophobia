#!/usr/bin/env python
"""Aggregate mature diversity forced-weight tests across three S3 fold seeds."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.evaluate import paired_compare
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--base-runs',nargs='+',required=True);p.add_argument('--candidate-runs',nargs='+',required=True);p.add_argument('--candidate',required=True);p.add_argument('--baseline',default='blend');p.add_argument('--weights',nargs='+',type=float,default=[.05,.10,.15]);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--out',default='artifacts/aggregate_forced_diversity.json');a=p.parse_args()
if len(a.base_runs)!=len(a.candidate_runs) or len(a.base_runs)<2:raise ValueError('base/candidate run lists must align and contain >=2 seeds')
tr,_,_=load_competition(a.data_dir);lookup=tr.set_index('id');bases=[];cands=[];foldsets=[];seed_detail=[]
def load(run,key,prefix):return np.load(Path(run)/(f'{prefix}_blend.npy' if key=='blend' else f'{prefix}_{key}.npy'))
def fr(x,folds):
 z=np.empty(len(x))
 for f in np.unique(folds):
  m=folds==f;z[m]=rankdata(np.asarray(x)[m],method='average')/m.sum()
 return z
for br,cr in zip(a.base_runs,a.candidate_runs):
 fb=pd.read_csv(Path(br)/'folds.csv');fc=pd.read_csv(Path(cr)/'folds.csv')
 if not np.array_equal(fb.id.to_numpy(),fc.id.to_numpy()) or not np.array_equal(fb.fold.to_numpy(),fc.fold.to_numpy()):raise ValueError('seed run alignment failed')
 y=lookup.loc[fb.id.to_numpy(),TARGET].astype(int).to_numpy();folds=fb.fold.to_numpy();B=fr(load(br,a.baseline,'oof'),folds);C=fr(load(cr,a.candidate,'oof'),folds);bases.append(B);cands.append(C);foldsets.append(folds);seed_detail.append({'base_run':br,'candidate_run':cr,'candidate_auc':float(roc_auc_score(y,C)),'rank_corr':float(np.corrcoef(B,C)[0,1])})
y=lookup.loc[pd.read_csv(Path(a.base_runs[0])/'folds.csv').id.to_numpy(),TARGET].astype(int).to_numpy();Bavg=np.mean(bases,axis=0);Cavg=np.mean(cands,axis=0);results=[]
for w in a.weights:
 per_seed=[];all_fold=[]
 for i,(B,C,F) in enumerate(zip(bases,cands,foldsets)):
  cmp=paired_compare(y,B,(1-w)*B+w*C,F,n_boot=max(300,a.bootstrap//3),seed=20260816+i+int(w*1000));per_seed.append({'seed_index':i,**cmp});all_fold.extend(cmp['delta_per_fold'])
 pooled=paired_compare(y,Bavg,(1-w)*Bavg+w*Cavg,foldsets[0],n_boot=a.bootstrap,seed=20260901+int(w*1000));results.append({'weight':w,'independent_seed_deltas':[x['delta_auc'] for x in per_seed],'independent_seeds_positive':int(sum(x['delta_auc']>0 for x in per_seed)),'fold_seed_deltas':all_fold,'fold_seed_positive':int(sum(x>0 for x in all_fold)),'pooled_seedbag':pooled})
res={'version':'nomophobia','tier':'S3_diversity_retrial','baseline':a.baseline,'candidate':a.candidate,'n_independent_seed_runs':len(bases),'candidate_seed_detail':seed_detail,'forced_weight_results':results,'admission_note':'Do not kill on optimizer weight=0. Inspect forced-weight paired deltas, mature candidate AUC, and rank correlation (<0.985 is desirable).'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
