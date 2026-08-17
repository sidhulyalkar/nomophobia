#!/usr/bin/env python
"""One-command campaign: production-scale iteration tuning, ceiling guard, then S3."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--data-dir',default='data');p.add_argument('--out-root',default='artifacts/winner')
p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--tune-rows',type=int,default=628000)
p.add_argument('--max-estimators',type=int,default=4000);p.add_argument('--patience',type=int,default=200)
p.add_argument('--combined-iterations',type=int);p.add_argument('--raw-iterations',type=int)
p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--dry-run',action='store_true')
p.add_argument('--ceiling-fraction',type=float,default=.90);p.add_argument('--allow-ceiling',action='store_true')
a=p.parse_args();root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True)
train_path=Path(a.data_dir)/'train.csv';n_train=sum(1 for _ in open(train_path,'rb'))-1 if train_path.exists() else None;s3_fold_train=int(n_train*4//5) if n_train else None
vals={};tuning={}
for expert,label,provided in [('lgb_combined63','combined',a.combined_iterations),('lgb_raw63','raw',a.raw_iterations)]:
    if provided is not None:
        vals[label]=int(provided);tuning[label]={'provided':True,'best_iteration':int(provided),'ceiling_90pct_hit':False};continue
    out=root/f'tune_{label}.json';cmd=[sys.executable,'tune_iterations.py','--data-dir',a.data_dir,'--expert',expert,'--rows',str(a.tune_rows),'--max-estimators',str(a.max_estimators),'--patience',str(a.patience),'--device',a.device,'--out',str(out)]
    print(' '.join(cmd),flush=True)
    if a.dry_run:
        vals[label]=1000;tuning[label]={'dry_run':True,'best_iteration':1000,'ceiling_90pct_hit':False};continue
    subprocess.run(cmd,check=True);meta=json.loads(out.read_text());tuning[label]=meta;vals[label]=int(meta['best_iteration'])
    if vals[label] >= a.ceiling_fraction*a.max_estimators and not a.allow_ceiling:
        status={'version':'nomophobia','status':'STOP_MAX_ESTIMATOR_CEILING','expert':expert,'best_iteration':vals[label],'max_estimators':a.max_estimators,'ceiling_fraction':a.ceiling_fraction,'action':'Raise --max-estimators and rerun tuning before S3.','tuning':tuning}
        (root/'winner_campaign.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));raise SystemExit(3)
cmd=[sys.executable,'run_s3.py','--data-dir',a.data_dir,'--out-root',str(root/'s3'),'--device',a.device,'--bootstrap',str(a.bootstrap),'--expert-iterations',f"lgb_combined63={vals['combined']}",f"lgb_raw63={vals['raw']}"]
print(' '.join(cmd),flush=True)
if not a.dry_run:subprocess.run(cmd,check=True)
summary={'version':'nomophobia','status':'DRY_RUN' if a.dry_run else 'S3_COMPLETE','combined_iterations':vals['combined'],'raw_iterations':vals['raw'],'device':a.device,'tune_rows':a.tune_rows,'expected_s3_fold_training_rows':s3_fold_train,'max_estimators':a.max_estimators,'patience':a.patience,'tuning':tuning}
(root/'winner_campaign.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
