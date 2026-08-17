#!/usr/bin/env python
"""Paired mature-capacity decomposition of the load-bearing frequency family."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.config import TARGET
from s6e8.cv import frozen_folds
from s6e8.evaluate import paired_compare
from s6e8.features import build_features
from s6e8.frequency import FREQUENCY_ARMS,frequency_columns_for_arm,frequency_feature_groups
from s6e8.io import load_competition
from s6e8.manifest import ExperimentRecorder
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames

def sample_idx(y,n,seed):
    if n>=len(y):return np.arange(len(y))
    s=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed);i,_=next(s.split(np.zeros(len(y)),y));return np.sort(i)
def evaluate(X,y,folds,estimators,seed,device):
    _,_,A,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());o=np.empty(len(y),float);fs=[]
    for f in np.unique(folds):
        ti=np.flatnonzero(folds!=f);vi=np.flatnonzero(folds==f);m=make_lgb(seed+1009*int(f),estimators,'combined63',device=device);m.fit(A.iloc[ti],y[ti],categorical_feature=[c for c in cats if c in A.columns]);p=m.predict_proba(A.iloc[vi])[:,1];o[vi]=p;fs.append(float(roc_auc_score(y[vi],p)))
    return o,{'oof_auc':float(roc_auc_score(y,o)),'fold_auc':fs,'fold_auc_std':float(np.std(fs)),'feature_count':int(X.shape[1])}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--out',default='artifacts/frequency_family_ablation.json');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--folds',type=int,default=5);p.add_argument('--bootstrap',type=int,default=1200);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with ExperimentRecorder(out.with_name(out.stem+'_manifest.json'),'FREQ-FAMILY-ABLATION','S1','A small frequency sub-family carries most of the marginal-frequency gain.','No reduced arm remains statistically indistinguishable from full at mature capacity.','Prune only when paired evidence supports equivalence and downstream robustness.','Do not choose an arm from point AUC alone.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as man:
        tr,te,_=load_competition(a.data_dir);idx=sample_idx(tr[TARGET].astype(int).to_numpy(),min(a.rows,len(tr)),a.seed);df=tr.iloc[idx].reset_index(drop=True);y=df[TARGET].astype(int).to_numpy();X,_=build_features(df.drop(columns=[TARGET]),te.iloc[:1].copy(),use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te));groups=frequency_feature_groups(X.columns);freq={c for v in groups.values() for c in v};backbone=[c for c in X if c not in freq];folds=frozen_folds(y,a.folds,a.seed);pred={};results={}
        for arm in FREQUENCY_ARMS:
            cols=backbone+frequency_columns_for_arm(X.columns,arm);pred[arm],results[arm]=evaluate(X[cols],y,folds,a.estimators,a.seed,a.device);print(f'{arm:20s} {results[arm]["oof_auc"]:.7f} {len(cols)} features',flush=True)
        full=pred['full'];paired={}
        for arm in FREQUENCY_ARMS:
            if arm=='full':continue
            cmp=paired_compare(y,full,pred[arm],folds,n_boot=a.bootstrap,seed=a.seed+41);cmp['rank_correlation_to_full']=float(np.corrcoef(pd.Series(full).rank(),pd.Series(pred[arm]).rank())[0,1]);paired[arm]=cmp;results[arm]['delta_vs_full']=cmp['delta_auc']
        results['full']['delta_vs_full']=0.0;payload={'version':'nomophobia-v0.3','tier':'S1','rows':len(df),'estimators':a.estimators,'frequency_groups':groups,'results':results,'paired_reduced_minus_full':paired};atomic_write_json(out,payload);man.add_output(out);man.add_metrics(full_auc=results['full']['oof_auc'],paired=paired);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
