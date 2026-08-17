#!/usr/bin/env python
"""Compatibility CLI for repeated production-scale LightGBM iteration tuning."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from s6e8.artifacts import atomic_write_json
from s6e8.manifest import ExperimentRecorder
from s6e8.tuning import IterationTuningConfig,tune_lightgbm_iterations

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--expert',choices=['lgb_raw63','lgb_combined63'],default='lgb_combined63');p.add_argument('--rows',type=int,default=628000);p.add_argument('--max-estimators',type=int,default=4000);p.add_argument('--patience',type=int,default=200);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');p.add_argument('--seed',type=int,default=20260816);p.add_argument('--repeats',type=int,default=3);p.add_argument('--validation-fraction',type=float,default=.12);p.add_argument('--ceiling-fraction',type=float,default=.90);p.add_argument('--out',default='artifacts/iteration_tuning.json');p.add_argument('--no-hash-inputs',action='store_true');a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    c=IterationTuningConfig(a.data_dir,a.expert,a.rows,a.max_estimators,a.patience,a.device,a.seed,a.repeats,a.validation_fraction,a.ceiling_fraction)
    with ExperimentRecorder(out.with_name(out.stem+'_manifest.json'),f'TUNE-{a.expert}','engineering','Repeated production-scale holdouts yield a stable frozen estimator count.','Any repeat hits the ceiling or iteration range exceeds 30% of median.','Freeze the median best iteration for S3.','Raise estimator ceiling or diagnose tuning instability.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs,repo_root=ROOT) as man:
        r=tune_lightgbm_iterations(c);atomic_write_json(out,r);man.add_output(out);man.add_metrics(best_iteration=r['best_iteration'],iteration_spread_fraction=r['iteration_spread_fraction'],ceiling_hit_any_repeat=r['ceiling_hit_any_repeat']);
        if r['ceiling_hit_any_repeat'] or r['tuning_instability_warning']:man.status='STOP_AND_REPORT'
        print(json.dumps(r,indent=2))
if __name__=='__main__':main()
