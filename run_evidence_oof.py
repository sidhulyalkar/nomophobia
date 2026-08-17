#!/usr/bin/env python
"""Generate full OOF/test predictions for the nonparametric Evidence Expert on frozen folds."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.evidence import EmpiricalBayesEvidence
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--folds',required=True);p.add_argument('--out-dir',default='artifacts/evidence_oof');p.add_argument('--alpha',type=float,default=20);p.add_argument('--n-bins',type=int,default=24);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
tr,te,_=load_competition(a.data_dir);fd=pd.read_csv(a.folds)
if len(fd)!=len(tr) or not np.array_equal(fd.id.to_numpy(),tr.id.to_numpy()):raise ValueError('fold alignment failed')
y=tr[TARGET].astype(int).to_numpy();X=tr.drop(columns=[TARGET]).reset_index(drop=True);folds=fd.fold.to_numpy();oof=np.zeros(len(tr));tp=[];scores=[]
for f in np.unique(folds):
 ti=np.where(folds!=f)[0];vi=np.where(folds==f)[0];m=EmpiricalBayesEvidence(n_bins=a.n_bins,alpha=a.alpha).fit(X.iloc[ti].reset_index(drop=True),y[ti]);oof[vi]=m.score(X.iloc[vi].reset_index(drop=True));tp.append(m.score(te.reset_index(drop=True)));scores.append(float(roc_auc_score(y[vi],oof[vi])))
np.save(out/'oof_empirical_bayes_evidence.npy',oof);np.save(out/'test_empirical_bayes_evidence.npy',np.mean(tp,axis=0));fd.to_csv(out/'folds.csv',index=False);res={'version':'nomophobia','model':'empirical_bayes_evidence','family':'empirical_bayes','fixed_iterations':None,'estimator_count':'nonparametric','fold_auc':scores,'fold_auc_mean':float(np.mean(scores)),'fold_auc_std':float(np.std(scores)),'oof_auc':float(roc_auc_score(y,oof))};(out/'metrics_empirical_bayes_evidence.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
