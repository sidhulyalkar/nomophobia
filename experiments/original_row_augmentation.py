#!/usr/bin/env python
"""Test direct low-weight augmentation from the labeled 7,500-row source dataset.

Prior source experiments used the source signal as a feature, blend member, or boosting
prior. This experiment asks a different question: do the source rows improve the
competition learner when injected as low-weight labeled observations? Validation is
always on untouched competition rows, and exact predictor-overlap source rows are removed.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.config import RAW_COLS,TARGET
from s6e8.cv import frozen_folds
from s6e8.evaluate import paired_compare
from s6e8.features import build_features
from s6e8.io import load_competition
from s6e8.manifest import ExperimentRecorder
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames

def sample_idx(y,n,seed):
    if n>=len(y):return np.arange(len(y))
    s=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=seed);i,_=next(s.split(np.zeros(len(y)),y));return np.sort(i)
def canonical_hash(df):
    x=df[RAW_COLS].copy()
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):x[c]=pd.to_numeric(x[c],errors='coerce').round(8)
        else:x[c]=x[c].astype('string').fillna('__MISSING__')
    return pd.util.hash_pandas_object(x,index=False).to_numpy(dtype=np.uint64)
def clean_source(source,competition):
    missing=[c for c in RAW_COLS+[TARGET] if c not in source]
    if missing:raise ValueError(f'original source missing required columns: {missing}')
    y=pd.to_numeric(source[TARGET],errors='coerce')
    if y.isna().any() or not set(y.astype(int).unique()).issubset({0,1}):raise ValueError('source addicted_label must be binary 0/1')
    source=source.copy();source[TARGET]=y.astype(int);before=len(source);source=source.loc[~pd.Series(canonical_hash(source)).isin(set(canonical_hash(competition))).to_numpy()].copy();overlap=before-len(source);before_dedup=len(source);source=source.loc[~pd.Series(canonical_hash(source)).duplicated().to_numpy()].reset_index(drop=True)
    return source,{'source_rows_input':before,'exact_predictor_overlap_removed':int(overlap),'source_predictor_duplicates_removed':int(before_dedup-len(source)),'source_rows_usable':int(len(source))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--original-csv',required=True);p.add_argument('--out-dir',default='artifacts/original_row_augmentation');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--weights',nargs='+',type=float,default=[.05,.10,.25,.50,1.0]);p.add_argument('--folds',type=int,default=5);p.add_argument('--bootstrap',type=int,default=1200);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--view',choices=['combined','raw'],default='combined');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    if any(w<=0 for w in a.weights):p.error('--weights must be positive')
    with ExperimentRecorder(out/'manifest.json','SOURCE-ROW-AUGMENTATION','S1','Direct low-weight labeled source rows improve competition ranking after predictor-overlap removal.','Every registered source weight has a nonpositive paired interval on competition OOF.','Advance only a broad positive region; do not tune a fourth decimal of source weight.','Never evaluate source rows as validation evidence.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as man:
        tr,te,_=load_competition(a.data_dir);source=pd.read_csv(a.original_csv);source.columns=[str(c).strip().lower() for c in source.columns];source,sm=clean_source(source,tr);idx=sample_idx(tr[TARGET].astype(int).to_numpy(),min(a.rows,len(tr)),a.seed);df=tr.iloc[idx].reset_index(drop=True);y=df[TARGET].astype(int).to_numpy();folds=frozen_folds(y,a.folds,a.seed);freq_ref=(tr.drop(columns=[TARGET]),te)
        if a.view=='combined':
            X,_=build_features(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=freq_ref);SX,_=build_features(source.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=freq_ref)
        else:X=df[RAW_COLS].copy();SX=source[RAW_COLS].copy()
        # Force one shared categorical vocabulary across competition and source frames.
        joined=pd.concat([X,SX],ignore_index=True);_,_,native,_,cats=prepare_tree_frames(joined,joined.iloc[:1].copy());A=native.iloc[:len(X)].reset_index(drop=True);S=native.iloc[len(X):].reset_index(drop=True);sy=source[TARGET].to_numpy(dtype=int)
        predictions={};metrics={}
        for weight in [0.0]+sorted(set(float(w) for w in a.weights)):
            o=np.empty(len(y),float);fs=[]
            for f in np.unique(folds):
                ti=np.flatnonzero(folds!=f);vi=np.flatnonzero(folds==f);m=make_lgb(a.seed+1009*int(f),a.estimators,'combined63' if a.view=='combined' else 'raw63',device=a.device)
                if weight==0:fitX=A.iloc[ti];fity=y[ti];sw=None
                else:fitX=pd.concat([A.iloc[ti],S],ignore_index=True);fity=np.r_[y[ti],sy];sw=np.r_[np.ones(len(ti)),np.full(len(S),weight)]
                m.fit(fitX,fity,sample_weight=sw,categorical_feature=[c for c in cats if c in fitX.columns]);pv=m.predict_proba(A.iloc[vi])[:,1];o[vi]=pv;fs.append(float(roc_auc_score(y[vi],pv)))
            key=f'w{weight:g}';predictions[key]=o;metrics[key]={'source_weight':weight,'oof_auc':float(roc_auc_score(y,o)),'fold_auc':fs,'fold_std':float(np.std(fs))};print(key,metrics[key],flush=True)
        base=predictions['w0'];comparisons={}
        for key,o in predictions.items():
            if key=='w0':continue
            cmp=paired_compare(y,base,o,folds,n_boot=a.bootstrap,seed=a.seed+23);cmp['advance_s1']=bool(cmp['delta_ci_95'][0]>0 and cmp['folds_positive']>=max(4,a.folds-1) and cmp['delta_auc']<=.003);cmp['leak_alert']=bool(cmp['delta_auc']>.003);comparisons[key]=cmp
        positive=[(k,r) for k,r in comparisons.items() if r['advance_s1']];best=max(positive,key=lambda z:z[1]['delta_auc'])[0] if positive else None;payload={'version':'nomophobia-v0.3','tier':'S1','rows':len(df),'estimators':a.estimators,'view':a.view,'source':sm,'metrics':metrics,'paired_vs_no_source':comparisons,'advanced_weights':[k for k,r in comparisons.items() if r['advance_s1']],'best_advanced_weight':best};atomic_write_json(out/'original_row_augmentation.json',payload);man.add_output(out/'original_row_augmentation.json');man.add_metrics(advanced_weights=payload['advanced_weights'],source_rows_usable=sm['source_rows_usable']);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
