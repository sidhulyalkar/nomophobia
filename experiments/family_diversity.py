#!/usr/bin/env python
"""Reproducible mature-capacity family diversity trial with forced weights."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb,make_xgb,make_cat
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--base-estimators',type=int,default=1000);p.add_argument('--candidate',choices=['xgb_raw','cat_raw'],default='xgb_raw');p.add_argument('--candidate-estimators',type=int,default=1000);p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--out',default='artifacts/family_diversity.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir);df=(tr.groupby(TARGET,group_keys=False).sample(frac=min(1,a.rows/len(tr)),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True));y=df[TARGET].astype(int).reset_index(drop=True);idx=np.arange(len(df));train,rest=train_test_split(idx,test_size=.25,stratify=y,random_state=a.seed);sel,ev=train_test_split(rest,test_size=.5,stratify=y.iloc[rest],random_state=a.seed+1);views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te))
def lgb_pred(view,profile):
 X,_=views[view];Xc,_,Xn,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());m=make_lgb(a.seed,a.base_estimators,profile,device=a.device);m.fit(Xn.iloc[train],y.iloc[train],categorical_feature=cats);return m.predict_proba(Xn.iloc[sel])[:,1],m.predict_proba(Xn.iloc[ev])[:,1]
cs,ce=lgb_pred('combined','combined63');rs,re=lgb_pred('raw','raw63');r=lambda x:rankdata(x,method='average')/len(x);C,R=r(cs),r(rs);grid=np.linspace(0,1,41);w0=max(grid,key=lambda w:roc_auc_score(y.iloc[sel],(1-w)*C+w*R));base_ev=(1-w0)*r(ce)+w0*r(re);Xraw,_=views['raw'];Xc,_,Xn,_,cats=prepare_tree_frames(Xraw,Xraw.iloc[:1].copy());Ac=Xc.iloc[train];An=Xn.iloc[train];Sc=Xc.iloc[sel];Ec=Xc.iloc[ev];Sn=Xn.iloc[sel];En=Xn.iloc[ev]
if a.candidate=='xgb_raw':m=make_xgb(a.seed,a.candidate_estimators,'raw',device=a.device);m.fit(An,y.iloc[train],verbose=False);pe=m.predict_proba(En)[:,1]
else:m=make_cat(a.seed,a.candidate_estimators,'raw',device=a.device);m.fit(Ac,y.iloc[train],cat_features=cats,verbose=False);pe=m.predict_proba(Ec)[:,1]
PE=r(pe);folds=frozen_folds(y.iloc[ev].reset_index(drop=True),5,a.seed+500);forced=[]
for w in [.05,.10,.15]:forced.append({'weight':w,**paired_compare(y.iloc[ev].to_numpy(),base_ev,(1-w)*base_ev+w*PE,folds,n_boot=1000,seed=a.seed+int(w*1000))})
res={'version':'nomophobia','tier':'S1' if len(df)>=120000 else 'S0','rows':len(df),'seed':a.seed,'base_estimator_count':a.base_estimators,'candidate':a.candidate,'candidate_estimator_count':a.candidate_estimators,'combined_eval_auc':float(roc_auc_score(y.iloc[ev],ce)),'raw_eval_auc':float(roc_auc_score(y.iloc[ev],re)),'base_raw_weight':float(w0),'base_blend_eval_auc':float(roc_auc_score(y.iloc[ev],base_ev)),'candidate_eval_auc':float(roc_auc_score(y.iloc[ev],pe)),'candidate_rank_corr_to_base':float(np.corrcoef(PE,base_ev)[0,1]),'forced_weight_results':forced};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
