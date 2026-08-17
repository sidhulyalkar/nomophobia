#!/usr/bin/env python
"""Tune a LightGBM expert's frozen iteration count on a production-scale inner holdout."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,lightgbm as lgb
from sklearn.model_selection import train_test_split
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET,EXPERT_SPECS
from s6e8.io import load_competition
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--expert',choices=['lgb_raw63','lgb_combined63'],default='lgb_combined63');p.add_argument('--rows',type=int,default=628000);p.add_argument('--max-estimators',type=int,default=4000);p.add_argument('--patience',type=int,default=200);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--seed',type=int,default=20260816);p.add_argument('--out',default='artifacts/iteration_tuning.json');a=p.parse_args()
tr,te,_=load_competition(a.data_dir);freq=(tr.drop(columns=[TARGET]),te)
if len(tr)>a.rows:df=(tr.groupby(TARGET,group_keys=False).sample(frac=a.rows/len(tr),random_state=a.seed).sample(frac=1,random_state=a.seed).head(a.rows).reset_index(drop=True))
else:df=tr.reset_index(drop=True)
y=df[TARGET].astype(int);spec=EXPERT_SPECS[a.expert];views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=freq);X,_=views[spec['view']];ti,vi=train_test_split(np.arange(len(df)),test_size=.12,stratify=y,random_state=20260815);_,_,A,B,cats=prepare_tree_frames(X.iloc[ti].reset_index(drop=True),X.iloc[vi].reset_index(drop=True));m=make_lgb(a.seed,a.max_estimators,spec['profile'],device=a.device);m.fit(A,y.iloc[ti].reset_index(drop=True),eval_set=[(B,y.iloc[vi].reset_index(drop=True))],eval_metric='auc',categorical_feature=cats,callbacks=[lgb.early_stopping(a.patience,verbose=False)]);best=int(m.best_iteration_);ceiling=bool(best>=int(np.ceil(.90*a.max_estimators)))
res={'version':'nomophobia','expert':a.expert,'rows_total':len(df),'training_rows':int(len(ti)),'validation_rows':int(len(vi)),'device':a.device,'max_estimators':a.max_estimators,'patience':a.patience,'best_iteration':best,'best_score':float(m.best_score_['valid_0']['auc']),'ceiling_90pct_hit':ceiling,'estimator_count':best,'note':'Freeze this count in OOF runs. Inner tuning score is never reported as OOF evidence.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
