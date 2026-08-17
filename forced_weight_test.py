#!/usr/bin/env python
"""Measure diversity contribution at fixed weights; never kill on selector weight=0."""
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
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--base-run',required=True);p.add_argument('--candidate-run',required=True);p.add_argument('--baseline',default='blend');p.add_argument('--candidate',required=True);p.add_argument('--weights',nargs='+',type=float,default=[.05,.10,.15]);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--out',default='artifacts/forced_weight.json');a=p.parse_args()
tr,_,_=load_competition(a.data_dir);br=Path(a.base_run);cr=Path(a.candidate_run);fb=pd.read_csv(br/'folds.csv');fc=pd.read_csv(cr/'folds.csv')
if not np.array_equal(fb.id.to_numpy(),fc.id.to_numpy()) or not np.array_equal(fb.fold.to_numpy(),fc.fold.to_numpy()):raise ValueError('candidate/base ID or fold alignment failed')
lookup=tr.set_index('id');y=lookup.loc[fb.id.to_numpy(),TARGET].astype(int).to_numpy();folds=fb.fold.to_numpy()
def load(run,key,prefix):return np.load(run/(f'{prefix}_blend.npy' if key=='blend' else f'{prefix}_{key}.npy'))
base=load(br,a.baseline,'oof');cand=load(cr,a.candidate,'oof')
def fr(x):
 z=np.empty(len(x))
 for f in np.unique(folds):
  m=folds==f;z[m]=rankdata(x[m],method='average')/m.sum()
 return z
B=fr(base);C=fr(cand);rows=[]
for w in a.weights:
 pred=(1-w)*B+w*C;cmp=paired_compare(y,B,pred,folds,n_boot=a.bootstrap,seed=20260816+int(w*1000));rows.append({'weight':w,**cmp})
res={'version':'nomophobia','baseline':a.baseline,'candidate':a.candidate,'candidate_oof_auc':float(roc_auc_score(y,cand)),'rank_correlation_to_baseline':float(np.corrcoef(B,C)[0,1]),'forced_weight_results':rows,'estimator_count':'see candidate metrics artifact'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
