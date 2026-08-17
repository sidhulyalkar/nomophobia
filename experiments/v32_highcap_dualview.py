#!/usr/bin/env python
"""Preregistered high-capacity dual-view S1/S2 screen with separate train/select/eval rows."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from s6e8.io import load_competition
from s6e8.config import TARGET
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.models import make_lgb
from s6e8.evaluate import paired_compare
from s6e8.cv import frozen_folds
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--rows',type=int,default=120000);p.add_argument('--estimators',type=int,default=1000);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--out',default='artifacts/highcap_dualview.json');a=p.parse_args();N=a.rows;S=a.seed
full,te,_=load_competition(a.data_dir);ref=(full.drop(columns=[TARGET]),te)
if len(full)>N:tr=(full.groupby(TARGET,group_keys=False).sample(frac=N/len(full),random_state=S).sample(frac=1,random_state=S).head(N).reset_index(drop=True))
else:tr=full.reset_index(drop=True)
y=tr[TARGET].astype(int).to_numpy();idx=np.arange(len(tr));base,val=train_test_split(idx,test_size=.25,stratify=y,random_state=S);sel,ev=train_test_split(val,test_size=.5,stratify=y[val],random_state=S+1);pos={v:i for i,v in enumerate(val)};sp=np.array([pos[int(v)] for v in sel]);ep=np.array([pos[int(v)] for v in ev]);views=build_feature_views(tr.drop(columns=[TARGET]),te,use_frequency=True,frequency_reference=ref);P={}
for name,view,profile in [('combined','combined','combined63'),('raw','raw','raw63')]:
 X,_=views[view];_,_,Xn,_,cats=prepare_tree_frames(X,X.iloc[:1].copy());m=make_lgb(S,a.estimators,profile);m.fit(Xn.iloc[base],y[base],categorical_feature=[c for c in cats if c in Xn.columns]);P[name]=m.predict_proba(Xn.iloc[val])[:,1]
def r(x):return rankdata(x,method='average')/len(x)
best=(-1,None)
for wraw in np.linspace(0,.50,21):
 score=(1-wraw)*r(P['combined'][sp])+wraw*r(P['raw'][sp]);z=roc_auc_score(y[sel],score)
 if z>best[0]:best=(z,float(wraw))
w=best[1];pb=r(P['combined'][ep]);pc=(1-w)*pb+w*r(P['raw'][ep]);folds=frozen_folds(y[ev],5,S+500);cmp=paired_compare(y[ev],pb,pc,folds,n_boot=a.bootstrap,seed=S+3)
res={'tier':'S1' if len(tr)>=120000 and len(tr)<350000 else ('S2' if len(tr)>=350000 and len(tr)<len(full) else 'S0'),'rows':len(tr),'estimators':a.estimators,'seed':S,'combined_validation_auc':float(roc_auc_score(y[val],P['combined'])),'raw_validation_auc':float(roc_auc_score(y[val],P['raw'])),'selected_raw_weight':w,'selection_auc':best[0],'combined_eval_auc':float(roc_auc_score(y[ev],pb)),'blend_eval_auc':float(roc_auc_score(y[ev],pc)),'rank_corr':float(np.corrcoef(r(P['combined']),r(P['raw']))[0,1]),'paired':cmp};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
