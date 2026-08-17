#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.config import TARGET
from s6e8.cv import frozen_folds
from s6e8.evaluate import paired_compare
from s6e8.features import build_feature_views
from s6e8.io import load_competition
from s6e8.manifest import ExperimentRecorder
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames
from s6e8.submission import unit_rank

def sample(y,n,s):
 if n>=len(y):return np.arange(len(y))
 z=StratifiedShuffleSplit(n_splits=1,train_size=n,random_state=s);i,_=next(z.split(np.zeros(len(y)),y));return np.sort(i)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--out',default='artifacts/capacity_diversity_curve.json');p.add_argument('--rows',type=int,default=180000);p.add_argument('--estimators',nargs='+',type=int,default=[400,700,1000,1400,2000]);p.add_argument('--eval-fraction',type=float,default=.20);p.add_argument('--raw-weight',type=float,default=.375);p.add_argument('--bootstrap',type=int,default=1200);p.add_argument('--seed',type=int,default=20260816);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();counts=sorted(set(a.estimators));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with ExperimentRecorder(out.with_name(out.stem+'_manifest.json'),'CAPACITY-DIVERSITY-CURVE','S1','Raw retains useful alternative orderings at production-like capacity.','High-capacity correlation >0.988 and fixed blend benefit disappears.','Route S3 compute only; do not select final count here.','If diversity collapses, replace the diversity view.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as m:
  tr,te,_=load_competition(a.data_dir);idx=sample(tr[TARGET].astype(int).to_numpy(),min(a.rows,len(tr)),a.seed);df=tr.iloc[idx].reset_index(drop=True);y=df[TARGET].astype(int).reset_index(drop=True);ti,ei=train_test_split(np.arange(len(df)),test_size=a.eval_fraction,stratify=y,random_state=a.seed+99);views=build_feature_views(df.drop(columns=[TARGET]),te.iloc[:1],use_frequency=True,frequency_reference=(tr.drop(columns=[TARGET]),te));prep={}
  for n in ('combined','raw'):
   X,_=views[n];_,_,A,B,cats=prepare_tree_frames(X.iloc[ti].reset_index(drop=True),X.iloc[ei].reset_index(drop=True));prep[n]=(A,B,cats)
  yt=y.iloc[ti].reset_index(drop=True);ye=y.iloc[ei].to_numpy();folds=frozen_folds(ye,5,a.seed+500);rows=[]
  for n in counts:
   preds={}
   for view,profile in [('combined','combined63'),('raw','raw63')]:
    A,B,cats=prep[view];model=make_lgb(a.seed,n,profile,device=a.device);model.fit(A,yt,categorical_feature=cats);preds[view]=model.predict_proba(B)[:,1]
   blend=(1-a.raw_weight)*unit_rank(preds['combined'])+a.raw_weight*unit_rank(preds['raw']);cmp=paired_compare(ye,preds['combined'],blend,folds,n_boot=a.bootstrap,seed=a.seed+n);corr=float(np.corrcoef(rankdata(preds['combined']),rankdata(preds['raw']))[0,1]);rows.append({'estimators':n,'combined_auc':float(roc_auc_score(ye,preds['combined'])),'raw_auc':float(roc_auc_score(ye,preds['raw'])),'fixed_blend_auc':float(roc_auc_score(ye,blend)),'rank_correlation':corr,'fixed_blend_vs_combined':cmp,'diversity_stop_threshold_breached':bool(corr>.988)});print(n,rows[-1],flush=True)
  h=rows[-1];payload={'version':'nomophobia-v0.3','tier':'S1 directional capacity routing','rows':len(df),'training_rows':len(ti),'evaluation_rows':len(ei),'fixed_raw_weight':a.raw_weight,'results':rows,'high_capacity_diagnosis':{'estimators':h['estimators'],'rank_correlation':h['rank_correlation'],'delta_auc':h['fixed_blend_vs_combined']['delta_auc'],'ci95':h['fixed_blend_vs_combined']['delta_ci_95'],'route':'DUAL_VIEW_STILL_PLAUSIBLE' if h['rank_correlation']<=.988 and h['fixed_blend_vs_combined']['delta_ci_95'][0]>0 else 'PRIORITIZE_NEW_DIVERSITY_BEFORE_OR_ALONGSIDE_S3'}};atomic_write_json(out,payload);m.add_output(out);m.add_metrics(**payload['high_capacity_diagnosis']);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
