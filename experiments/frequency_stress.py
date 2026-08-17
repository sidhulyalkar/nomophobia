#!/usr/bin/env python
"""Mature frequency stress: target gain plus source-safety decomposition.

The decisive safety comparison is complete-row frequency source AUC, because overall
source separability is already known to be largely explained by missingness shift.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split,StratifiedShuffleSplit
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.config import RAW_COLS,TARGET
from s6e8.cv import frozen_folds
from s6e8.evaluate import paired_compare
from s6e8.features import build_feature_views
from s6e8.io import load_competition
from s6e8.manifest import ExperimentRecorder
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames

def sample_frame(df,n,target,seed):
    if n>=len(df):return df.reset_index(drop=True)
    if target is None:return df.sample(n=n,random_state=seed).reset_index(drop=True)
    s=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed);i,_=next(s.split(np.zeros(len(df)),df[target]));return df.iloc[np.sort(i)].reset_index(drop=True)
def source_auc(X,y,seed):
    sk=StratifiedKFold(3,shuffle=True,random_state=seed);o=np.zeros(len(y))
    for ti,vi in sk.split(X,y):
        m=LGBMClassifier(n_estimators=180,num_leaves=15,learning_rate=.04,min_child_samples=200,verbosity=-1,n_jobs=-1,random_state=seed);m.fit(X.iloc[ti],y[ti]);o[vi]=m.predict_proba(X.iloc[vi])[:,1]
    return float(roc_auc_score(y,o))
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--source-rows',type=int,default=100000);p.add_argument('--bootstrap',type=int,default=1200);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--out',default='artifacts/frequency_stress.json');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with ExperimentRecorder(out.with_name(out.stem+'_manifest.json'),'FREQUENCY-STRESS','S1','Train+test frequency reference adds ranking signal without learning a hidden source fingerprint.','Transductive target gain disappears or complete-row frequency source AUC materially exceeds chance.','Advance transductive frequency only with positive paired CI and complete-row source AUC <=0.55.','Stop density expansion if complete-row source AUC >0.60.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as man:
        tr,te,_=load_competition(a.data_dir);df=sample_frame(tr,min(a.rows,len(tr)),TARGET,a.seed);y=df[TARGET].astype(int).reset_index(drop=True);ti,vi=train_test_split(np.arange(len(df)),test_size=.25,stratify=y,random_state=a.seed+17);full=tr.drop(columns=[TARGET]).reset_index(drop=True);refs={'train_plus_test':(full,te),'train_only':(full,te.iloc[:0].copy())};pred={};models={}
        for name,ref in refs.items():
            X,_=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=ref)['combined'];_,_,A,B,cats=prepare_tree_frames(X.iloc[ti].reset_index(drop=True),X.iloc[vi].reset_index(drop=True));m=make_lgb(a.seed,a.estimators,'combined63',device=a.device);m.fit(A,y.iloc[ti].reset_index(drop=True),categorical_feature=cats);pv=m.predict_proba(B)[:,1];pred[name]=pv;models[name]={'auc':float(roc_auc_score(y.iloc[vi],pv)),'features':int(X.shape[1])}
        yf=y.iloc[vi].to_numpy();folds=frozen_folds(yf,5,a.seed+500);cmp=paired_compare(yf,pred['train_only'],pred['train_plus_test'],folds,n_boot=a.bootstrap,seed=a.seed)
        n=min(a.source_rows,len(tr),len(te));atr=tr.drop(columns=[TARGET]).sample(n=n,random_state=a.seed).reset_index(drop=True);ate=te.sample(n=n,random_state=a.seed+1).reset_index(drop=True);source=pd.concat([atr,ate],ignore_index=True);sy=np.r_[np.zeros(n,int),np.ones(n,int)];F,_=build_feature_views(source,te.iloc[:1],use_frequency=True,frequency_reference=(full,te))['combined'];freq=[c for c in F if '__freq' in c or '__logfreq' in c or '_freq' in c];FX=F[freq].replace([np.inf,-np.inf],np.nan).fillna(-1);miss=pd.DataFrame({'missing_count':source[RAW_COLS].isna().sum(axis=1),'missing_pattern':source[RAW_COLS].isna().astype(np.int8).dot(1<<np.arange(len(RAW_COLS)))})
        overall_freq=source_auc(FX,sy,a.seed);missing_auc=source_auc(miss,sy,a.seed+2);complete=source[RAW_COLS].notna().all(axis=1).to_numpy();complete_freq=source_auc(FX.loc[complete].reset_index(drop=True),sy[complete],a.seed+3) if complete.sum()>=3000 and len(np.unique(sy[complete]))==2 else None
        stop=bool(complete_freq is not None and complete_freq>.60);warn=bool(complete_freq is not None and complete_freq>.55);advance=bool(cmp['delta_ci_95'][0]>0 and not warn);payload={'version':'nomophobia-v0.3','tier':'S1','rows':len(df),'estimators':a.estimators,'models':models,'transductive_minus_train_only':cmp,'source_safety':{'frequency_source_auc_overall':overall_freq,'missingness_only_source_auc':missing_auc,'frequency_source_auc_complete_rows':complete_freq,'warn':warn,'stop':stop,'warn_complete_auc_above':.55,'stop_complete_auc_above':.60},'decision':{'advance_transductive_frequency':advance,'stop_density_expansion':stop}};atomic_write_json(out,payload);man.add_output(out);man.add_metrics(transductive_delta=cmp['delta_auc'],complete_row_source_auc=complete_freq,advance=advance);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
