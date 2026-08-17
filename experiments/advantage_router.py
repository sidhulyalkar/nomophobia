#!/usr/bin/env python
"""Mature-capacity retrial of a constrained raw-vs-combined advantage router."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--out',default='artifacts/advantage_router.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir);df=(tr.groupby(TARGET,group_keys=False).sample(frac=min(1,a.rows/len(tr)),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True));y=df[TARGET].astype(int).reset_index(drop=True);idx=np.arange(len(df));base_tr,hold=train_test_split(idx,test_size=.25,stratify=y,random_state=a.seed);route_pool,ev=train_test_split(hold,test_size=.5,stratify=y.iloc[hold],random_state=a.seed+1);rt,rs=train_test_split(route_pool,test_size=.5,stratify=y.iloc[route_pool],random_state=a.seed+2);views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te))
def preds(view,profile):
 X,_=views[view];_,_,Xn,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());m=make_lgb(a.seed,a.estimators,profile,device=a.device);m.fit(Xn.iloc[base_tr],y.iloc[base_tr],categorical_feature=cats);return m.predict_proba(Xn.iloc[rt])[:,1],m.predict_proba(Xn.iloc[rs])[:,1],m.predict_proba(Xn.iloc[ev])[:,1]
crt,crs,cev=preds('combined','combined63');rrt,rrs,rev=preds('raw','raw63');rank=lambda z:rankdata(z,method='average')/len(z);grid=np.linspace(0,1,41);w0=max(grid,key=lambda w:roc_auc_score(y.iloc[rt],(1-w)*rank(crt)+w*rank(rrt)));adv=(2*y.iloc[rt].to_numpy()-1)*(rank(rrt)-rank(crt));raw_cols=['daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours','sleep_hours','weekend_screen_time','notifications_per_day','app_opens_per_day']
def ctx(ix,c,r):
 z=df.iloc[ix][raw_cols].copy().reset_index(drop=True);z=z.fillna(z.median(numeric_only=True));z['combined_rank']=rank(c);z['raw_rank']=rank(r);z['rank_gap']=z['raw_rank']-z['combined_rank'];z['abs_rank_gap']=np.abs(z['rank_gap']);z['missing_count']=df.iloc[ix].isna().sum(axis=1).to_numpy();return z
R=LGBMRegressor(n_estimators=180,num_leaves=7,max_depth=3,learning_rate=.035,min_child_samples=400,reg_lambda=5.0,verbosity=-1,n_jobs=-1,random_state=a.seed);R.fit(ctx(rt,crt,rrt),adv);pred_rs=R.predict(ctx(rs,crs,rrs));pred_ev=R.predict(ctx(ev,cev,rev));scale0=np.std(pred_rs)+1e-12;base_rs=(1-w0)*rank(crs)+w0*rank(rrs);base_ev=(1-w0)*rank(cev)+w0*rank(rev);scales=[0,.03,.05,.08,.10,.15]
def routed(c,r,pred,lam):
 wc=np.clip(w0+lam*np.tanh(pred/scale0),0,1);return (1-wc)*rank(c)+wc*rank(r),wc
best=max(scales,key=lambda lam:roc_auc_score(y.iloc[rs],routed(crs,rrs,pred_rs,lam)[0]));cand,wv=routed(cev,rev,pred_ev,best);folds=frozen_folds(y.iloc[ev].reset_index(drop=True),5,a.seed+500);cmp=paired_compare(y.iloc[ev].to_numpy(),base_ev,cand,folds,n_boot=1200,seed=a.seed)
res={'version':'nomophobia','tier':'S1' if len(df)>=120000 else 'S0','rows':len(df),'seed':a.seed,'estimator_count':a.estimators,'base_raw_weight':float(w0),'router_scale':float(best),'router_train_rows':len(rt),'router_select_rows':len(rs),'eval_rows':len(ev),'mean_eval_raw_weight':float(np.mean(wv)),'std_eval_raw_weight':float(np.std(wv)),'base_eval_auc':float(roc_auc_score(y.iloc[ev],base_ev)),'router_eval_auc':float(roc_auc_score(y.iloc[ev],cand)),'router_vs_fixed':cmp};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
