#!/usr/bin/env python
"""Aggregate three 5-fold seed runs and enforce the promotion contract."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.evaluate import paired_compare
from s6e8.frontier import weighted_test_like_auc
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--baseline',default='lgb_raw63');p.add_argument('--candidate',default='lgb_combined63');p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--out-dir',default='artifacts/promotion');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
tr,te,sample=load_competition(a.data_dir);y=tr[TARGET].astype(int).to_numpy();ids=tr['id'].to_numpy()
def files(run,key):return (run/'oof_blend.npy',run/'test_blend.npy') if key=='blend' else (run/f'oof_{key}.npy',run/f'test_{key}.npy')
def rank01(z):return rankdata(np.asarray(z),method='average')/len(z)
def readj(p):
 try:return json.loads(Path(p).read_text())
 except Exception:return {}
def iteration_of(run,key):return None if key=='blend' else readj(run/f'metrics_{key}.json').get('fixed_iterations')
all_fold_deltas=[];seed_rows=[];base_o=[];cand_o=[];cand_t=[];base_stds=[];cand_stds=[];seen_seeds=[];canonical_folds=None
for i,r0 in enumerate(a.runs):
 r=Path(r0);fd=pd.read_csv(r/'folds.csv')
 if len(fd)!=len(tr) or not np.array_equal(fd['id'].to_numpy(),ids):raise ValueError(f'ID/fold alignment failed for {r}')
 folds=fd.fold.to_numpy();canonical_folds=folds.copy() if canonical_folds is None else canonical_folds;bo,_=files(r,a.baseline);co,ct=files(r,a.candidate);pb=np.load(bo);pc=np.load(co);pt=np.load(ct)
 cmp=paired_compare(y,pb,pc,folds,n_boot=max(300,a.bootstrap//3),seed=20260816+i);all_fold_deltas.extend(cmp['delta_per_fold']);bfold=[];cfold=[]
 for f in np.unique(folds):
  m=folds==f;bfold.append(roc_auc_score(y[m],pb[m]));cfold.append(roc_auc_score(y[m],pc[m]))
 base_stds.append(float(np.std(bfold)));cand_stds.append(float(np.std(cfold)));wb=weighted_test_like_auc(tr,te,y,pb);wc=weighted_test_like_auc(tr,te,y,pc);run_summary=readj(r/'run_summary.json');seed_value=int(run_summary.get('fold_seed',-1));seen_seeds.append(seed_value);blend=readj(r/'blend.json');corr=None
 if a.baseline!='blend' and a.candidate!='blend' and (r/'expert_rank_correlation.csv').exists():
  cc=pd.read_csv(r/'expert_rank_correlation.csv',index_col=0)
  if a.baseline in cc.index and a.candidate in cc.columns:corr=float(cc.loc[a.baseline,a.candidate])
 seed_rows.append({'run':str(r),'fold_seed':seed_value,'baseline':a.baseline,'candidate':a.candidate,'baseline_estimator_count':iteration_of(r,a.baseline),'candidate_estimator_count':iteration_of(r,a.candidate),'baseline_auc':float(roc_auc_score(y,pb)),'candidate_auc':float(roc_auc_score(y,pc)),'delta_auc':cmp['delta_auc'],'delta_ci_95':cmp['delta_ci_95'],'delong_p':cmp['delong_p'],'n_effective':cmp['n_effective'],'folds_positive':cmp['folds_positive'],'weighted_delta':wc-wb,'baseline_fold_std':base_stds[-1],'candidate_fold_std':cand_stds[-1],'expert_rank_correlation':corr,'blend_auc':run_summary.get('blend_auc_honest'),'blend_selection_auc':run_summary.get('blend_selection_auc'),'blend_selection_optimism':run_summary.get('blend_selection_optimism'),'rotation_weights':blend.get('rotation_weights'),'deploy_weights':blend.get('weights')});base_o.append(rank01(pb));cand_o.append(rank01(pc));cand_t.append(rank01(pt))
pb=np.mean(base_o,axis=0);pc=np.mean(cand_o,axis=0);pooled=paired_compare(y,pb,pc,canonical_folds,n_boot=a.bootstrap,seed=20260831);wb=weighted_test_like_auc(tr,te,y,pb);wc=weighted_test_like_auc(tr,te,y,pc);folds_positive=int(sum(d>0 for d in all_fold_deltas));required=len(a.runs)==3 and len(all_fold_deltas)==15 and len(set(seen_seeds))==3;ci_excludes_zero=pooled['delta_ci_95'][0]>0;weighted_positive=(wc-wb)>0;std_ok=float(np.mean(cand_stds))<=1.20*max(float(np.mean(base_stds)),1e-12);promoted=bool(required and folds_positive>=13 and ci_excludes_zero and weighted_positive and std_ok)
result={'version':'nomophobia','tier':'S3','baseline':a.baseline,'candidate':a.candidate,'n_runs':len(a.runs),'independent_replications':len(a.runs),'seed_results':seed_rows,'fold_seed_deltas':all_fold_deltas,'folds_positive':folds_positive,'folds_total':len(all_fold_deltas),'pooled_rank_ensemble':pooled,'weighted_baseline_auc':wb,'weighted_candidate_auc':wc,'weighted_delta':wc-wb,'mean_baseline_fold_std':float(np.mean(base_stds)),'mean_candidate_fold_std':float(np.mean(cand_stds)),'fold_seeds':seen_seeds,'promotion_rule':{'requires_3_distinct_5fold_runs':required,'13_of_15_positive':folds_positive>=13,'pooled_ci_excludes_zero':ci_excludes_zero,'test_missingness_weighted_positive':weighted_positive,'fold_std_within_20pct':std_ok},'verdict':'PROMOTED' if promoted else 'PARKED'}
(out/'promotion.json').write_text(json.dumps(result,indent=2));pd.DataFrame(seed_rows).drop(columns=['rotation_weights','deploy_weights']).to_csv(out/'seed_results.csv',index=False);ptest=np.mean(cand_t,axis=0);sub=sample.copy();sub[TARGET]=ptest;sub.to_csv(out/'submission_candidate_seedbag.csv',index=False);print(json.dumps(result,indent=2))
