from __future__ import annotations
import argparse,json
from pathlib import Path
from .campaign import S3CampaignConfig,WinnerCampaignConfig,run_s3_campaign,run_winner_campaign
from .io import load_competition
from .manifest import ExperimentRecorder,data_provenance
from .router import recommend_next_actions
from .tuning import IterationTuningConfig,tune_lightgbm_iterations

def _hashflag(p):p.add_argument('--no-hash-inputs',action='store_true')
def build_parser():
 p=argparse.ArgumentParser(prog='nomophobia');s=p.add_subparsers(dest='command',required=True)
 v=s.add_parser('validate');v.add_argument('--data-dir',default='data');_hashflag(v)
 r=s.add_parser('route');r.add_argument('--artifact-root',default='artifacts');r.add_argument('--data-dir',default='/kaggle/input/playground-series-s6e8');r.add_argument('--original-csv',default='/kaggle/input/smart-phone/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')
 t=s.add_parser('tune');t.add_argument('--data-dir',default='data');t.add_argument('--expert',choices=['lgb_combined63','lgb_raw63'],default='lgb_combined63');t.add_argument('--rows',type=int,default=628000);t.add_argument('--max-estimators',type=int,default=4000);t.add_argument('--patience',type=int,default=200);t.add_argument('--device',choices=['cpu','gpu'],default='gpu');t.add_argument('--seed',type=int,default=20260816);t.add_argument('--repeats',type=int,default=3);t.add_argument('--validation-fraction',type=float,default=.12);t.add_argument('--ceiling-fraction',type=float,default=.90);t.add_argument('--out',default='artifacts/iteration_tuning.json');_hashflag(t)
 q=s.add_parser('s3');q.add_argument('--data-dir',default='data');q.add_argument('--out-root',default='artifacts/s3');q.add_argument('--seeds',nargs=3,type=int,default=[20260816,20260817,20260818]);q.add_argument('--device',choices=['cpu','gpu'],default='gpu');q.add_argument('--experts',nargs='+',default=['lgb_combined63','lgb_raw63']);q.add_argument('--lgb-estimators',type=int,default=2200);q.add_argument('--expert-iterations',nargs='*',default=[]);q.add_argument('--baseline',default='lgb_raw63');q.add_argument('--candidate',default='lgb_combined63');q.add_argument('--bootstrap',type=int,default=2000);q.add_argument('--dry-run',action='store_true');_hashflag(q)
 w=s.add_parser('winner');w.add_argument('--data-dir',default='data');w.add_argument('--out-root',default='artifacts/winner');w.add_argument('--device',choices=['cpu','gpu'],default='gpu');w.add_argument('--tune-rows',type=int,default=628000);w.add_argument('--max-estimators',type=int,default=4000);w.add_argument('--patience',type=int,default=200);w.add_argument('--tune-repeats',type=int,default=3);w.add_argument('--combined-iterations',type=int);w.add_argument('--raw-iterations',type=int);w.add_argument('--bootstrap',type=int,default=2000);w.add_argument('--dry-run',action='store_true');w.add_argument('--ceiling-fraction',type=float,default=.90);w.add_argument('--allow-ceiling',action='store_true');w.add_argument('--seeds',nargs=3,type=int,default=[20260816,20260817,20260818]);_hashflag(w);return p

def main(argv=None):
 a=build_parser().parse_args(argv)
 if a.command=='validate':
  tr,te,sm=load_competition(a.data_dir);print(json.dumps({'train_rows':len(tr),'test_rows':len(te),'sample_rows':len(sm),'data':data_provenance(a.data_dir,hash_files=not a.no_hash_inputs)},indent=2));return 0
 if a.command=='route':print(json.dumps(recommend_next_actions(a.artifact_root,data_dir=a.data_dir,original_csv=a.original_csv),indent=2));return 0
 if a.command=='tune':
  c=IterationTuningConfig(a.data_dir,a.expert,a.rows,a.max_estimators,a.patience,a.device,a.seed,a.repeats,a.validation_fraction,a.ceiling_fraction);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
  with ExperimentRecorder(out.with_name(out.stem+'_manifest.json'),f'TUNE-{a.expert}','engineering','Repeated production holdouts yield a stable frozen count.','Ceiling hit or >30% iteration spread.','Freeze the median count.','Raise ceiling or diagnose instability.',config=vars(a),data_dir=a.data_dir,hash_inputs=not a.no_hash_inputs) as m:
   r=tune_lightgbm_iterations(c);out.write_text(json.dumps(r,indent=2));m.add_output(out);m.add_metrics(best_iteration=r['best_iteration'],iteration_spread_fraction=r['iteration_spread_fraction'],ceiling_hit_any_repeat=r['ceiling_hit_any_repeat'])
   if r['ceiling_hit_any_repeat'] or r['tuning_instability_warning']:m.status='STOP_AND_REPORT'
  print(json.dumps(r,indent=2));return 0
 if a.command=='s3':print(json.dumps(run_s3_campaign(S3CampaignConfig(a.data_dir,a.out_root,tuple(a.seeds),a.device,tuple(a.experts),a.lgb_estimators,tuple(a.expert_iterations),a.baseline,a.candidate,a.bootstrap,a.dry_run,not a.no_hash_inputs)),indent=2));return 0
 if a.command=='winner':print(json.dumps(run_winner_campaign(WinnerCampaignConfig(a.data_dir,a.out_root,a.device,a.tune_rows,a.max_estimators,a.patience,a.tune_repeats,a.combined_iterations,a.raw_iterations,a.bootstrap,a.dry_run,a.ceiling_fraction,a.allow_ceiling,not a.no_hash_inputs,tuple(a.seeds))),indent=2));return 0
 return 2
