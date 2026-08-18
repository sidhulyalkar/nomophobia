#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from live_frontier_candidate import TARGET, CAT, fold_te, feature_frame


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',default='te_capacity_screen.json'); ap.add_argument('--rows',type=int,default=120000); ap.add_argument('--max-estimators',type=int,default=2000); a=ap.parse_args()
    df=pd.read_csv(Path(a.data_dir)/'train.csv'); yall=df[TARGET].to_numpy(np.int8)
    for c in CAT:
        df[c]=pd.Categorical(df[c],categories=pd.Index(df[c].dropna().unique()))
    checkpoints=[600,1200,2000]; results=[]
    for seed in [20260816,20260817,20260818]:
        idx,_=train_test_split(np.arange(len(df)),train_size=a.rows,stratify=yall,random_state=seed); y=yall[idx]
        ti,vi=train_test_split(np.arange(len(idx)),test_size=.25,stratify=y,random_state=seed+91)
        tr=df.iloc[idx[ti]].reset_index(drop=True); va=df.iloc[idx[vi]].reset_index(drop=True); yt=y[ti]; yv=y[vi]
        etr,eva,_=fold_te(tr,yt,va,va.iloc[:1].copy(),5,10.0,seed+777)
        xt=feature_frame(tr,etr,False); xv=feature_frame(va,eva,False)
        m=lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=a.max_estimators,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=100,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_alpha=.1,reg_lambda=2,max_bin=255,random_state=seed,n_jobs=-1,verbosity=-1)
        m.fit(xt,yt,categorical_feature=CAT)
        aucs={str(n):float(roc_auc_score(yv,m.predict_proba(xv,num_iteration=n)[:,1])) for n in checkpoints}
        row={'seed':seed,'auc':aucs,'delta_1200_vs_600':aucs['1200']-aucs['600'],'delta_2000_vs_1200':aucs['2000']-aucs['1200']}; results.append(row); print(json.dumps(row),flush=True)
    d12=np.array([r['delta_1200_vs_600'] for r in results]); d20=np.array([r['delta_2000_vs_1200'] for r in results])
    summary={'results':results,'mean_delta_1200_vs_600':float(d12.mean()),'mean_delta_2000_vs_1200':float(d20.mean()),'positive_2000_vs_1200':int((d20>0).sum()),'decision':'CAPACITY_STILL_RISING' if (d20>0).sum()>=2 and d20.mean()>0 else 'PLATEAU_BY_1200'}
    Path(a.out).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
