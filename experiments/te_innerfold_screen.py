#!/usr/bin/env python
from __future__ import annotations

import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET='addicted_label'; ID='id'; MISS=-1_000_000.0
CAT=['gender','stress_level','academic_work_impact']
NUM=['age','daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours','sleep_hours','notifications_per_day','app_opens_per_day','weekend_screen_time']
RAW=NUM+CAT

def mapping(v,y,s=10.0):
    p=float(y.mean()); k=v.astype('float64').fillna(MISS)
    st=pd.DataFrame({'k':k.to_numpy(),'y':y}).groupby('k',sort=False)['y'].agg(['sum','count'])
    return (st['sum']+s*p)/(st['count']+s)

def apply(v,m,p):
    return v.astype('float64').fillna(MISS).map(m).fillna(p).to_numpy(np.float32)

def encode(tr,y,va,n,seed):
    p=float(y.mean()); a=pd.DataFrame(index=range(len(tr))); b=pd.DataFrame(index=range(len(va)))
    cv=StratifiedKFold(n,shuffle=True,random_state=seed)
    for c in NUM:
        z=np.empty(len(tr),np.float32)
        for ti,vi in cv.split(tr,y):
            z[vi]=apply(tr.iloc[vi][c],mapping(tr.iloc[ti][c],y[ti]),float(y[ti].mean()))
        m=mapping(tr[c],y); a['te__'+c]=z; b['te__'+c]=apply(va[c],m,p)
    return a,b

def frame(raw,te):
    o=raw[RAW].copy().reset_index(drop=True); d=raw['daily_screen_time_hours'].reset_index(drop=True).replace(0,np.nan)
    o['parts_sum']=raw[['social_media_hours','gaming_hours','work_study_hours']].reset_index(drop=True).sum(axis=1)
    o['social_media_share']=raw['social_media_hours'].reset_index(drop=True)/d
    o['gaming_share']=raw['gaming_hours'].reset_index(drop=True)/d
    o['work_study_share']=raw['work_study_hours'].reset_index(drop=True)/d
    o['weekend_minus_daily']=raw['weekend_screen_time'].reset_index(drop=True)-d
    for c in te: o[c]=te[c].to_numpy()
    return o

def fit(x,y,seed,n):
    m=lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=n,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=100,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_alpha=.1,reg_lambda=2,max_bin=255,random_state=seed,n_jobs=-1,verbosity=-1)
    m.fit(x,y,categorical_feature=CAT); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',default='innerfold_screen.json'); ap.add_argument('--rows',type=int,default=120000); ap.add_argument('--estimators',type=int,default=1200); a=ap.parse_args()
    df=pd.read_csv(Path(a.data_dir)/'train.csv'); yall=df[TARGET].to_numpy(np.int8)
    for c in CAT: df[c]=pd.Categorical(df[c],categories=pd.Index(df[c].dropna().unique()))
    rows=[]
    for seed in [20260816,20260817,20260818]:
        idx,_=train_test_split(np.arange(len(df)),train_size=a.rows,stratify=yall,random_state=seed); y=yall[idx]
        ti,vi=train_test_split(np.arange(len(idx)),test_size=.25,stratify=y,random_state=seed+91)
        tr=df.iloc[idx[ti]].reset_index(drop=True); va=df.iloc[idx[vi]].reset_index(drop=True); yt=y[ti]; yv=y[vi]
        e5t,e5v=encode(tr,yt,va,5,seed+777); e10t,e10v=encode(tr,yt,va,10,seed+777)
        m5=fit(frame(tr,e5t),yt,seed,a.estimators); m10=fit(frame(tr,e10t),yt,seed,a.estimators)
        p5=m5.predict_proba(frame(va,e5v))[:,1]; p10=m10.predict_proba(frame(va,e10v))[:,1]
        a5=float(roc_auc_score(yv,p5)); a10=float(roc_auc_score(yv,p10))
        r={'seed':seed,'inner5_auc':a5,'inner10_auc':a10,'delta':a10-a5,'rank_corr':float(pd.Series(p5).rank().corr(pd.Series(p10).rank()))}; rows.append(r); print(json.dumps(r),flush=True)
    d=np.array([r['delta'] for r in rows]); out={'results':rows,'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'positive_seeds':int((d>0).sum()),'decision':'ADVANCE_INNER10' if (d>0).sum()>=2 and d.mean()>0 else 'KEEP_INNER5'}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
