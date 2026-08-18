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
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',default='te_encoder_bagging.json'); ap.add_argument('--rows',type=int,default=120000); ap.add_argument('--estimators',type=int,default=600); a=ap.parse_args()
    df=pd.read_csv(Path(a.data_dir)/'train.csv'); yall=df[TARGET].to_numpy(np.int8)
    for c in CAT: df[c]=pd.Categorical(df[c],categories=pd.Index(df[c].dropna().unique()))
    rows=[]
    for seed in [20260816,20260817,20260818]:
        idx,_=train_test_split(np.arange(len(df)),train_size=a.rows,stratify=yall,random_state=seed); y=yall[idx]
        ti,vi=train_test_split(np.arange(len(idx)),test_size=.25,stratify=y,random_state=seed+91)
        tr=df.iloc[idx[ti]].reset_index(drop=True); va=df.iloc[idx[vi]].reset_index(drop=True); yt=y[ti]; yv=y[vi]
        e1t,e1v,_=fold_te(tr,yt,va,va.iloc[:1].copy(),5,10.0,seed+777)
        e2t,e2v,_=fold_te(tr,yt,va,va.iloc[:1].copy(),5,10.0,seed+1777)
        m1=fit(feature_frame(tr,e1t,False),yt,seed,a.estimators); m2=fit(feature_frame(tr,e2t,False),yt,seed,a.estimators)
        p1=m1.predict_proba(feature_frame(va,e1v,False))[:,1]; p2=m2.predict_proba(feature_frame(va,e2v,False))[:,1]
        a1=float(roc_auc_score(yv,p1)); a2=float(roc_auc_score(yv,p2)); pb=.5*rank01(p1)+.5*rank01(p2); ab=float(roc_auc_score(yv,pb)); best=max(a1,a2)
        r={'seed':seed,'encoder_a_auc':a1,'encoder_b_auc':a2,'blend_auc':ab,'blend_gain_over_best':ab-best,'rank_corr':float(pd.Series(p1).rank().corr(pd.Series(p2).rank()))}; rows.append(r); print(json.dumps(r),flush=True)
    d=np.array([r['blend_gain_over_best'] for r in rows]); out={'results':rows,'mean_blend_gain':float(d.mean()),'median_blend_gain':float(np.median(d)),'positive_seeds':int((d>0).sum()),'decision':'ADVANCE_ENCODER_BAGGING' if (d>0).sum()>=2 and d.mean()>0 else 'KILL_ENCODER_BAGGING'}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
