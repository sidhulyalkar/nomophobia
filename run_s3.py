#!/usr/bin/env python
"""Orchestrate the authoritative 3-seed x 5-fold S3 promotion campaign."""
from __future__ import annotations
import argparse,subprocess,sys,json
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--out-root',default='artifacts/s3')
p.add_argument('--seeds',nargs=3,type=int,default=[20260816,20260817,20260818]);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--experts',nargs='+',default=['lgb_combined63','lgb_raw63']);p.add_argument('--lgb-estimators',type=int,default=2200);p.add_argument('--expert-iterations',nargs='*',default=[]);p.add_argument('--baseline',default='lgb_raw63');p.add_argument('--candidate',default='lgb_combined63');p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True);runs=[]
for seed in a.seeds:
 out=root/f'seed_{seed}';runs.append(str(out));cmd=[sys.executable,'train.py','--data-dir',a.data_dir,'--out-dir',str(out),'--preset','full','--device',a.device,'--fold-seed',str(seed),'--n-splits','5','--lgb-estimators',str(a.lgb_estimators),'--experts',*a.experts]
 if a.expert_iterations:cmd+=['--expert-iterations',*a.expert_iterations]
 print(' '.join(cmd),flush=True)
 if not a.dry_run:subprocess.run(cmd,check=True)
for baseline,candidate in [(a.baseline,a.candidate),(a.candidate,'blend')]:
 agg=[sys.executable,'aggregate_promotion.py','--data-dir',a.data_dir,'--runs',*runs,'--baseline',baseline,'--candidate',candidate,'--bootstrap',str(a.bootstrap),'--out-dir',str(root/f'promotion__{baseline}__to__{candidate}')];print(' '.join(agg),flush=True)
 if not a.dry_run:subprocess.run(agg,check=True)
status={'version':'nomophobia','seeds':a.seeds,'runs':runs,'device':a.device,'experts':a.experts,'lgb_estimators':a.lgb_estimators,'expert_iterations':a.expert_iterations,'status':'DRY_RUN' if a.dry_run else 'COMPLETE'}
if not a.dry_run:
 aucs=[];corr=[]
 for rr in map(Path,runs):
  sm=json.loads((rr/'run_summary.json').read_text());aucs.append(float(sm['blend_auc_honest']));cp=rr/'expert_rank_correlation.csv'
  if cp.exists() and 'lgb_combined63' in a.experts and 'lgb_raw63' in a.experts:
   cc=pd.read_csv(cp,index_col=0);corr.append(float(cc.loc['lgb_combined63','lgb_raw63']))
 status['blend_auc_by_seed']=aucs;status['blend_auc_seed_spread']=float(max(aucs)-min(aucs));status['rank_correlation_by_seed']=corr;status['mean_rank_correlation']=float(np.mean(corr)) if corr else None;flags=[]
 if status['blend_auc_seed_spread']>.002:flags.append('STOP_SEED_BLEND_AUC_SPREAD_GT_0.002')
 if corr and max(corr)>.988:flags.append('STOP_DUAL_VIEW_RANK_CORRELATION_GT_0.988')
 status['escalation_flags']=flags;status['status']='STOP_AND_REPORT' if flags else 'S3_COMPLETE_READY_FOR_DIVERSITY_RETRIAL'
(root/'campaign.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
