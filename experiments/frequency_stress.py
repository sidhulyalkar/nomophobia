#!/usr/bin/env python
"""Stress test the load-bearing transductive frequency feature family."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedKFold
from lightgbm import LGBMClassifier
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--source-rows',type=int,default=60000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--out',default='artifacts/frequency_stress.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir)
if len(tr)>a.rows:df=(tr.groupby(TARGET,group_keys=False).sample(frac=a.rows/len(tr),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True))
else:df=tr.reset_index(drop=True)
y=df[TARGET].astype(int).reset_index(drop=True);ti,vi=train_test_split(np.arange(len(df)),test_size=.25,stratify=y,random_state=a.seed+17);full_train=tr.drop(columns=[TARGET]).reset_index(drop=True);empty_test=te.iloc[:0].copy();refs={'train_plus_test':(full_train,te),'train_only':(full_train,empty_test)};preds={};meta={}
for name,ref in refs.items():
 views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=ref);X,_=views['combined'];_,_,A,B,cats=prepare_tree_frames(X.iloc[ti].reset_index(drop=True),X.iloc[vi].reset_index(drop=True));m=make_lgb(a.seed,a.estimators,'combined63',device=a.device);m.fit(A,y.iloc[ti].reset_index(drop=True),categorical_feature=cats);pv=m.predict_proba(B)[:,1];preds[name]=pv;meta[name]={'auc':float(roc_auc_score(y.iloc[vi],pv)),'n_features':int(X.shape[1])}
folds=frozen_folds(y.iloc[vi].reset_index(drop=True),5,a.seed+500);cmp=paired_compare(y.iloc[vi].to_numpy(),preds['train_only'],preds['train_plus_test'],folds,n_boot=1200,seed=a.seed)
n=min(a.source_rows,len(tr),len(te));rs=np.random.default_rng(a.seed);tri=rs.choice(len(tr),n,replace=False);tei=rs.choice(len(te),n,replace=False);source=pd.concat([tr.drop(columns=[TARGET]).iloc[tri].reset_index(drop=True),te.iloc[tei].reset_index(drop=True)],ignore_index=True);source_y=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)];sv,_=build_feature_views(source,te.iloc[:1],use_frequency=True,frequency_reference=(full_train,te))['combined'];freq_cols=[c for c in sv if '__freq' in c or '__logfreq' in c or '_freq' in c];Xsrc=sv[freq_cols].replace([np.inf,-np.inf],np.nan).fillna(-1);skf=StratifiedKFold(3,shuffle=True,random_state=a.seed);po=np.zeros(len(source_y))
for tridx,vidx in skf.split(Xsrc,source_y):
 adv=LGBMClassifier(n_estimators=180,num_leaves=15,learning_rate=.04,min_child_samples=200,verbosity=-1,n_jobs=-1,random_state=a.seed);adv.fit(Xsrc.iloc[tridx],source_y[tridx]);po[vidx]=adv.predict_proba(Xsrc.iloc[vidx])[:,1]
adv_auc=float(roc_auc_score(source_y,po));res={'version':'nomophobia','tier':'S1' if a.rows>=120000 else 'S0','hypothesis':'train+test frequency reference adds target signal without material train/test source asymmetry','rows':len(df),'estimators':a.estimators,'estimator_count':a.estimators,'seed':a.seed,'models':meta,'transductive_minus_train_only':cmp,'frequency_only_train_vs_test_adversarial_auc':adv_auc,'stress_gate':{'warn_source_auc_above':0.60,'stop_source_auc_above':0.70,'source_asymmetry_warning':bool(adv_auc>0.60)}};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
