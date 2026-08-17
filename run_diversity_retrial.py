#!/usr/bin/env python
"""Retrial mature CatBoost/XGBoost/Evidence diversity against an established S3 base."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--base-s3-root',required=True);p.add_argument('--out-root',default='artifacts/diversity_retrial');p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--tune-rows',type=int,default=628000);p.add_argument('--max-estimators',type=int,default=4000);p.add_argument('--patience',type=int,default=200);p.add_argument('--cat-iterations',type=int);p.add_argument('--xgb-estimators',type=int);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=Path(a.out_root);root.mkdir(parents=True,exist_ok=True);base=Path(a.base_s3_root)
counts={'cat_raw':a.cat_iterations,'xgb_raw':a.xgb_estimators};tuning={}
for expert in ['cat_raw','xgb_raw']:
 if counts[expert] is not None:continue
 out=root/f'tune_{expert}.json';cmd=[sys.executable,'tune_family.py','--data-dir',a.data_dir,'--expert',expert,'--rows',str(a.tune_rows),'--max-estimators',str(a.max_estimators),'--patience',str(a.patience),'--device',a.device,'--out',str(out)];print(' '.join(cmd),flush=True)
 if a.dry_run:counts[expert]=1000;continue
 subprocess.run(cmd,check=True);tuning[expert]=json.loads(out.read_text());counts[expert]=int(tuning[expert]['best_iteration'])
 if tuning[expert].get('ceiling_90pct_hit'):raise SystemExit(f'{expert} tuning hit >=90% estimator ceiling; raise --max-estimators')
reports=[]
for br in sorted(base.glob('seed_*')):
 if not (br/'folds.csv').exists():continue
 seed=int(br.name.split('_')[-1]);cr=root/br.name;cmd=[sys.executable,'train.py','--data-dir',a.data_dir,'--out-dir',str(cr),'--preset','full','--device',a.device,'--fold-seed',str(seed),'--n-splits','5','--experts','cat_raw','xgb_raw','--expert-iterations',f"cat_raw={counts['cat_raw']}",f"xgb_raw={counts['xgb_raw']}"];print(' '.join(cmd),flush=True)
 if not a.dry_run:subprocess.run(cmd,check=True)
 er=root/f'{br.name}__evidence';cmd=[sys.executable,'run_evidence_oof.py','--data-dir',a.data_dir,'--folds',str(br/'folds.csv'),'--out-dir',str(er)];print(' '.join(cmd),flush=True)
 if not a.dry_run:subprocess.run(cmd,check=True)
 for cand,rr in [('cat_raw',cr),('xgb_raw',cr),('empirical_bayes_evidence',er)]:
  fo=root/f'forced__{br.name}__{cand}.json';cmd=[sys.executable,'forced_weight_test.py','--data-dir',a.data_dir,'--base-run',str(br),'--candidate-run',str(rr),'--baseline','blend','--candidate',cand,'--bootstrap',str(a.bootstrap),'--out',str(fo)];print(' '.join(cmd),flush=True)
  if not a.dry_run:subprocess.run(cmd,check=True);reports.append(str(fo))
aggregates={}
if not a.dry_run:
 base_runs=[str(x) for x in sorted(base.glob('seed_*')) if (x/'folds.csv').exists()];family_runs=[str(root/x.name) for x in sorted(base.glob('seed_*')) if (x/'folds.csv').exists()];evidence_runs=[str(root/f'{x.name}__evidence') for x in sorted(base.glob('seed_*')) if (x/'folds.csv').exists()]
 for cand,runs in [('cat_raw',family_runs),('xgb_raw',family_runs),('empirical_bayes_evidence',evidence_runs)]:
  if len(base_runs)>=2:
   fo=root/f'aggregate_forced__{cand}.json';cmd=[sys.executable,'aggregate_forced_diversity.py','--data-dir',a.data_dir,'--base-runs',*base_runs,'--candidate-runs',*runs,'--candidate',cand,'--baseline','blend','--bootstrap',str(a.bootstrap),'--out',str(fo)];print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True);aggregates[cand]=str(fo)
(root/'retrial_campaign.json').write_text(json.dumps({'version':'nomophobia','counts':counts,'tuning':tuning,'reports':reports,'aggregates':aggregates,'dry_run':a.dry_run},indent=2))
