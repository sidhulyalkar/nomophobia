#!/usr/bin/env python
"""Leak-clean coarse behavioral posterior features, reproduced at mature capacity."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--out',default='artifacts/binned_target_evidence.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir);df=(tr.groupby(TARGET,group_keys=False).sample(frac=min(1,a.rows/len(tr)),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True));y=df[TARGET].astype(int).reset_index(drop=True);idx=np.arange(len(df));ti,vi=train_test_split(idx,test_size=.25,stratify=y,random_state=a.seed);views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te));X,_=views['combined'];raw=df.drop(columns=[TARGET]).reset_index(drop=True);pairs=[('daily_screen_time_hours','social_media_hours'),('daily_screen_time_hours','weekend_screen_time'),('social_media_hours','weekend_screen_time'),('gaming_hours','work_study_hours')]
def make_post(train_ix,apply_ix):
 out=pd.DataFrame(index=np.arange(len(apply_ix)));global_p=float(y.iloc[train_ix].mean());alpha=40.0
 for c in ['daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours','weekend_screen_time']:
  v=pd.to_numeric(raw[c],errors='coerce');edges=np.unique(np.nanquantile(v.iloc[train_ix],np.linspace(0,1,17)));edges[0],edges[-1]=-np.inf,np.inf;kt=pd.cut(v.iloc[train_ix],edges,labels=False,include_lowest=True).astype('Int64').astype(str);ka=pd.cut(v.iloc[apply_ix],edges,labels=False,include_lowest=True).astype('Int64').astype(str);d=pd.DataFrame({'k':kt.to_numpy(),'y':y.iloc[train_ix].to_numpy()}).groupby('k').y.agg(['sum','count']);mp=((d['sum']+alpha*global_p)/(d['count']+alpha)).to_dict();out[f'posterior__{c}']=ka.map(mp).fillna(global_p).to_numpy()
 for a1,b1 in pairs:
  va=pd.to_numeric(raw[a1],errors='coerce');vb=pd.to_numeric(raw[b1],errors='coerce');ea=np.unique(np.nanquantile(va.iloc[train_ix],np.linspace(0,1,9)));eb=np.unique(np.nanquantile(vb.iloc[train_ix],np.linspace(0,1,9)));ea[0],ea[-1]=-np.inf,np.inf;eb[0],eb[-1]=-np.inf,np.inf;kta=pd.cut(va.iloc[train_ix],ea,labels=False).astype('Int64').astype(str)+'|'+pd.cut(vb.iloc[train_ix],eb,labels=False).astype('Int64').astype(str);kaa=pd.cut(va.iloc[apply_ix],ea,labels=False).astype('Int64').astype(str)+'|'+pd.cut(vb.iloc[apply_ix],eb,labels=False).astype('Int64').astype(str);d=pd.DataFrame({'k':kta.to_numpy(),'y':y.iloc[train_ix].to_numpy()}).groupby('k').y.agg(['sum','count']);mp=((d['sum']+alpha*global_p)/(d['count']+alpha)).to_dict();out[f'posterior__{a1}__{b1}']=kaa.map(mp).fillna(global_p).to_numpy()
 return out
Pt=make_post(ti,ti);Pv=make_post(ti,vi);base_tr=X.iloc[ti].reset_index(drop=True);base_va=X.iloc[vi].reset_index(drop=True);cand_tr=pd.concat([base_tr,Pt],axis=1);cand_va=pd.concat([base_va,Pv],axis=1)
def fit(A0,B0):
 _,_,A,B,cats=prepare_tree_frames(A0,B0);m=make_lgb(a.seed,a.estimators,'combined63',device=a.device);m.fit(A,y.iloc[ti].reset_index(drop=True),categorical_feature=cats);return m.predict_proba(B)[:,1]
pb=fit(base_tr,base_va);pc=fit(cand_tr,cand_va);folds=frozen_folds(y.iloc[vi].reset_index(drop=True),5,a.seed+500);cmp=paired_compare(y.iloc[vi].to_numpy(),pb,pc,folds,n_boot=1200,seed=a.seed);res={'version':'nomophobia','tier':'S1' if len(df)>=120000 else 'S0','rows':len(df),'seed':a.seed,'estimator_count':a.estimators,'posterior_features':int(Pt.shape[1]),'base_auc':float(roc_auc_score(y.iloc[vi],pb)),'candidate_auc':float(roc_auc_score(y.iloc[vi],pc)),'candidate_vs_base':cmp};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
