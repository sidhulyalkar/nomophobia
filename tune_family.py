#!/usr/bin/env python
"""Tune CatBoost or XGBoost diversity experts on a production-scale inner split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET,EXPERT_SPECS
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_cat,make_xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--expert',choices=['cat_raw','xgb_raw'],required=True);p.add_argument('--rows',type=int,default=628000);p.add_argument('--max-estimators',type=int,default=4000);p.add_argument('--patience',type=int,default=200);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--seed',type=int,default=20260816);p.add_argument('--out',default='artifacts/family_tuning.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir)
if len(tr)>a.rows:df=(tr.groupby(TARGET,group_keys=False).sample(frac=a.rows/len(tr),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True))
else:df=tr.reset_index(drop=True)
y=df[TARGET].astype(int).reset_index(drop=True);spec=EXPERT_SPECS[a.expert];X,_=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=False)[spec['view']];ti,vi=train_test_split(np.arange(len(df)),test_size=.12,stratify=y,random_state=20260815);Ac,Bc,An,Bn,cats=prepare_tree_frames(X.iloc[ti].reset_index(drop=True),X.iloc[vi].reset_index(drop=True))
if a.expert=='cat_raw':
 m=make_cat(a.seed,a.max_estimators,'raw',device=a.device);m.set_params(od_type='Iter',od_wait=a.patience,use_best_model=True);m.fit(Ac,y.iloc[ti],cat_features=cats,eval_set=(Bc,y.iloc[vi]),verbose=False);best=int(m.get_best_iteration()+1);pred=m.predict_proba(Bc)[:,1]
else:
 m=make_xgb(a.seed,a.max_estimators,'raw',device=a.device);m.set_params(early_stopping_rounds=a.patience);m.fit(An,y.iloc[ti],eval_set=[(Bn,y.iloc[vi])],verbose=False);best=int(getattr(m,'best_iteration',a.max_estimators-1)+1);pred=m.predict_proba(Bn,iteration_range=(0,best))[:,1]
res={'version':'nomophobia','expert':a.expert,'rows_total':len(df),'training_rows':int(len(ti)),'validation_rows':int(len(vi)),'device':a.device,'max_estimators':a.max_estimators,'best_iteration':best,'best_score':float(roc_auc_score(y.iloc[vi],pred)),'ceiling_90pct_hit':bool(best>=.9*a.max_estimators),'estimator_count':best};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
