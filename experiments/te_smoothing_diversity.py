#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from live_frontier_candidate import TARGET, CAT, fold_te, feature_frame, rank01

def fit(x,y,seed,n):
    m=lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=n,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=100,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_alpha=.1,reg_lambda=2,max_bin=255,random_state=seed,n_jobs=-1,verbosity=-1)
    m.fit(x,y,categorical_feature=CAT); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',default='te_smoothing_diversity.json'); ap.add_argument('--rows',type=int,default=120000); ap.add_argument('--estimators',type=int,default=1200); a=ap.parse_args()
    df=pd.read_csv(Path(a.data_dir)/'train.csv'); yall=df[TARGET].to_numpy(np.int8)
    for c in CAT: df[c]=pd.Categorical(df[c],categories=pd.Index(df[c].dropna().unique()))
    rows=[]
    for seed in [20260816,20260817,20260818]:
        idx,_=train_test_split(np.arange(len(df)),train_size=a.rows,stratify=yall,random_state=seed); y=yall[idx]
        ti,vi=train_test_split(np.arange(len(idx)),test_size=.25,stratify=y,random_state=seed+91)
        tr=df.iloc[idx[ti]].reset_index(drop=True); va=df.iloc[idx[vi]].reset_index(drop=True); yt=y[ti]; yv=y[vi]
        e10t,e10v,_=fold_te(tr,yt,va,va.iloc[:1].copy(),5,10.0,seed+777)
        e20t,e20v,_=fold_te(tr,yt,va,va.iloc[:1].copy(),5,20.0,seed+777)
        m10=fit(feature_frame(tr,e10t,False),yt,seed,a.estimators); m20=fit(feature_frame(tr,e20t,False),yt,seed+1000,a.estimators)
        p10=m10.predict_proba(feature_frame(va,e10v,False))[:,1]; p20=m20.predict_proba(feature_frame(va,e20v,False))[:,1]
        r10,r20=rank01(p10),rank01(p20); pb=.5*r10+.5*r20
        a10=float(roc_auc_score(yv,p10)); a20=float(roc_auc_score(yv,p20)); ab=float(roc_auc_score(yv,pb)); best=max(a10,a20)
        row={'seed':seed,'s10_auc':a10,'s20_auc':a20,'blend_auc':ab,'blend_gain_over_best':ab-best,'rank_corr':float(pd.Series(p10).rank().corr(pd.Series(p20).rank()))}; rows.append(row); print(json.dumps(row),flush=True)
    d=np.array([r['blend_gain_over_best'] for r in rows]); out={'results':rows,'mean_blend_gain':float(d.mean()),'median_blend_gain':float(np.median(d)),'positive_seeds':int((d>0).sum()),'decision':'ADVANCE_SMOOTHING_BLEND' if (d>0).sum()>=2 and d.mean()>0 else 'KILL_SMOOTHING_BLEND'}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
