#!/usr/bin/env python
"""Reproducible empirical-Bayes Evidence Expert retrial against a mature dual-view base."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
from s6e8.evidence import EmpiricalBayesEvidence
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--base-estimators',type=int,default=1000);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--out',default='artifacts/evidence_screen.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir);df=(tr.groupby(TARGET,group_keys=False).sample(frac=min(1,a.rows/len(tr)),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True));y=df[TARGET].astype(int).reset_index(drop=True);Xraw=df.drop(columns=[TARGET]);idx=np.arange(len(df));train,rest=train_test_split(idx,test_size=.25,stratify=y,random_state=a.seed);sel,ev=train_test_split(rest,test_size=.5,stratify=y.iloc[rest],random_state=a.seed+1);views=build_feature_views(Xraw,te.iloc[:1],use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te))
def pred(view,profile):
 X,_=views[view];_,_,Xn,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());m=make_lgb(a.seed,a.base_estimators,profile,device=a.device);m.fit(Xn.iloc[train],y.iloc[train],categorical_feature=cats);return m.predict_proba(Xn.iloc[sel])[:,1],m.predict_proba(Xn.iloc[ev])[:,1]
cs,ce=pred('combined','combined63');rs,re=pred('raw','raw63');r=lambda x:rankdata(x,method='average')/len(x);grid=np.linspace(0,1,41);w0=max(grid,key=lambda w:roc_auc_score(y.iloc[sel],(1-w)*r(cs)+w*r(rs)));base=(1-w0)*r(ce)+w0*r(re);eb=EmpiricalBayesEvidence().fit(Xraw.iloc[train].reset_index(drop=True),y.iloc[train].to_numpy());ee=eb.score(Xraw.iloc[ev]);ER=r(ee);folds=frozen_folds(y.iloc[ev].reset_index(drop=True),5,a.seed+500);forced=[]
for w in [.05,.10,.15]:forced.append({'weight':w,**paired_compare(y.iloc[ev].to_numpy(),base,(1-w)*base+w*ER,folds,n_boot=1000,seed=a.seed+int(w*1000))})
res={'version':'nomophobia','tier':'S1' if len(df)>=120000 else 'S0','rows':len(df),'seed':a.seed,'base_estimator_count':a.base_estimators,'candidate':'empirical_bayes_evidence','candidate_estimator_count':'nonparametric','base_raw_weight':float(w0),'base_blend_eval_auc':float(roc_auc_score(y.iloc[ev],base)),'candidate_eval_auc':float(roc_auc_score(y.iloc[ev],ee)),'candidate_rank_corr_to_base':float(np.corrcoef(ER,base)[0,1]),'forced_weight_results':forced};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
