#!/usr/bin/env python
"""Paired S1 screen for target-free population-density geometry.

Arms extend the current marginal frequency representation with selected pair/triple
counts, value-with-missing-regime density, and unsigned train/test support stability.
Nothing is promoted from this script; candidates only advance to S2.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from lightgbm import LGBMClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.config import TARGET
from s6e8.cv import frozen_folds
from s6e8.evaluate import paired_compare
from s6e8.features import build_features
from s6e8.frequency_geometry import add_joint_density_features,add_regime_conditioned_density_features,add_source_stability_features
from s6e8.io import load_competition
from s6e8.manifest import ExperimentRecorder
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames

def sample_idx(y,n,seed):
    if n>=len(y): return np.arange(len(y))
    s=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed);i,_=next(s.split(np.zeros(len(y)),y));return np.sort(i)
def run_cv(X,y,folds,estimators,seed,device):
    _,_,A,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());o=np.empty(len(y),float);scores=[]
    for f in np.unique(folds):
        ti=np.flatnonzero(folds!=f);vi=np.flatnonzero(folds==f);m=make_lgb(seed+1009*int(f),estimators,'combined63',device=device)
        m.fit(A.iloc[ti],y[ti],categorical_feature=[c for c in cats if c in A.columns]);p=m.predict_proba(A.iloc[vi])[:,1];o[vi]=p;scores.append(float(roc_auc_score(y[vi],p)))
    return o,{'oof_auc':float(roc_auc_score(y,o)),'fold_auc':scores,'fold_std':float(np.std(scores)),'features':int(X.shape[1])}
def complete_source_auc(tr,te,cols,seed):
    raw=[c for c in tr.columns if c in te.columns];mt=tr[raw].notna().all(axis=1);me=te[raw].notna().all(axis=1);n=min(int(mt.sum()),int(me.sum()),50000)
    if n<3000:return None
    rng=np.random.default_rng(seed);a=tr.loc[mt].sample(n=n,random_state=seed).reset_index(drop=True);b=te.loc[me].sample(n=n,random_state=seed+1).reset_index(drop=True);fa,fb,_=add_source_stability_features(a,b,reference_train=tr,reference_test=te);X=pd.concat([fa[cols],fb[cols]],ignore_index=True);y=np.r_[np.zeros(n,int),np.ones(n,int)];fold=frozen_folds(y,3,seed+77);o=np.zeros(len(y))
    for f in np.unique(fold):
        ti=fold!=f;vi=fold==f;m=LGBMClassifier(n_estimators=160,num_leaves=15,learning_rate=.04,min_child_samples=180,verbosity=-1,n_jobs=-1,random_state=seed+int(f));m.fit(X.loc[ti],y[ti]);o[vi]=m.predict_proba(X.loc[vi])[:,1]
    return float(roc_auc_score(y,o))
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--out-dir',default='artifacts/frequency_geometry');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--folds',type=int,default=5);p.add_argument('--bootstrap',type=int,default=1200);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    with ExperimentRecorder(out/'manifest.json','FREQ-GEOMETRY-S1','S1','Higher-order target-free density adds ranking information beyond marginal frequency.','Every arm has nonpositive paired CI or source-stability features create complete-row source separability.','Advance only arms with CI lower >0, >=4/5 positive folds, and no >.003 leak-sized jump.','Kill an arm only at its registered scale; do not infer from selector weight.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as man:
        tr,te,_=load_competition(a.data_dir);idx=sample_idx(tr[TARGET].astype(int).to_numpy(),min(a.rows,len(tr)),a.seed);df=tr.iloc[idx].reset_index(drop=True);y=df[TARGET].astype(int).to_numpy();raw=df.drop(columns=[TARGET]);ref=(tr.drop(columns=[TARGET]),te);base,_=build_features(raw,te.iloc[:1].copy(),use_frequency=True,frequency_reference=ref)
        joint,_,jm=add_joint_density_features(raw,te.iloc[:1].copy(),reference_train=ref[0],reference_test=ref[1]);reg,_,rm=add_regime_conditioned_density_features(raw,te.iloc[:1].copy(),reference_train=ref[0],reference_test=ref[1]);stab,_,sm=add_source_stability_features(raw,te.iloc[:1].copy(),reference_train=ref[0],reference_test=ref[1])
        count_cols=[c for c in joint if c.endswith('__freq') or c.endswith('__logfreq')];arms={'baseline_marginal_frequency':base,'joint_counts':pd.concat([base,joint[count_cols]],axis=1),'joint_full':pd.concat([base,joint],axis=1),'regime_density':pd.concat([base,reg],axis=1),'source_stability':pd.concat([base,stab],axis=1),'joint_plus_regime':pd.concat([base,joint,reg],axis=1),'all_safe_geometry':pd.concat([base,joint,reg,stab],axis=1)}
        folds=frozen_folds(y,a.folds,a.seed);pred={};metrics={}
        for name,X in arms.items():pred[name],metrics[name]=run_cv(X,y,folds,a.estimators,a.seed,a.device);print(f'{name:28s} {metrics[name]["oof_auc"]:.7f}',flush=True)
        baseline=pred['baseline_marginal_frequency'];comparisons={}
        for name in arms:
            if name=='baseline_marginal_frequency':continue
            cmp=paired_compare(y,baseline,pred[name],folds,n_boot=a.bootstrap,seed=a.seed+31);cmp['rank_correlation']=float(np.corrcoef(pd.Series(baseline).rank(),pd.Series(pred[name]).rank())[0,1]);cmp['advance_s1']=bool(cmp['delta_ci_95'][0]>0 and cmp['folds_positive']>=max(4,a.folds-1) and cmp['delta_auc']<=.003);cmp['leak_alert']=bool(cmp['delta_auc']>.003);comparisons[name]=cmp
        source_auc=complete_source_auc(ref[0],ref[1],list(stab.columns),a.seed) if len(stab.columns) else None;source_stop=bool(source_auc is not None and source_auc>.65);source_warn=bool(source_auc is not None and source_auc>.58)
        if source_stop:
            for n in ('source_stability','all_safe_geometry'):
                if n in comparisons:comparisons[n]['advance_s1']=False
        payload={'version':'nomophobia-v0.3','tier':'S1','rows':len(df),'estimators':a.estimators,'feature_metadata':{'joint':jm,'regime':rm,'source_stability':sm},'metrics':metrics,'paired_vs_baseline':comparisons,'source_safety':{'complete_row_source_auc':source_auc,'warn':source_warn,'stop':source_stop,'warn_threshold':.58,'stop_threshold':.65},'advanced_arms':[n for n,r in comparisons.items() if r['advance_s1']]};atomic_write_json(out/'frequency_geometry.json',payload);man.add_output(out/'frequency_geometry.json');man.add_metrics(advanced_arms=payload['advanced_arms'],complete_row_source_auc=source_auc);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
