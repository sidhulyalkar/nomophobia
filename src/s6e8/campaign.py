from __future__ import annotations
import json,shutil,subprocess,sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .artifacts import atomic_write_json,sha256_file
from .io import load_competition
from .manifest import ExperimentRecorder
from .s3 import resolution_from_promotions,summarize_s3_runs
from .validation import validate_submission

@dataclass(frozen=True)
class WinnerCampaignConfig:
 data_dir:str='data';out_root:str='artifacts/winner';device:str='gpu';tune_rows:int=628000;max_estimators:int=4000;patience:int=200;tune_repeats:int=3;combined_iterations:int|None=None;raw_iterations:int|None=None;bootstrap:int=2000;dry_run:bool=False;ceiling_fraction:float=.90;allow_ceiling:bool=False;hash_inputs:bool=True;seeds:tuple[int,int,int]=(20260816,20260817,20260818)
@dataclass(frozen=True)
class S3CampaignConfig:
 data_dir:str='data';out_root:str='artifacts/s3';seeds:tuple[int,int,int]=(20260816,20260817,20260818);device:str='gpu';experts:tuple[str,...]=('lgb_combined63','lgb_raw63');lgb_estimators:int=2200;expert_iterations:tuple[str,...]=();baseline:str='lgb_raw63';candidate:str='lgb_combined63';bootstrap:int=2000;dry_run:bool=False;hash_inputs:bool=True

def _run(cmd,dry_run=False):
 print(' '.join(map(str,cmd)),flush=True)
 if not dry_run:subprocess.run(cmd,check=True)
def _read(path):
 p=Path(path);return json.loads(p.read_text()) if p.exists() else {}
def _script(name):
 root=Path(__file__).resolve().parents[2]
 for p in (root/name,Path.cwd()/name):
  if p.exists():return str(p)
 return name

def run_s3_campaign(c:S3CampaignConfig):
 if len(c.seeds)!=3 or len(set(c.seeds))!=3:raise ValueError('authoritative S3 requires exactly three distinct seeds')
 root=Path(c.out_root);root.mkdir(parents=True,exist_ok=True);runs=[root/f'seed_{s}' for s in c.seeds]
 with ExperimentRecorder(root/'s3_manifest.json','S3-DUALVIEW','S3','Combined beats raw and raw retains useful orthogonal ordering.','Combined→blend fails the mechanical gate or production correlation breaches the stop threshold.','Promote only under aggregate_promotion.py.','Never kill raw from optimizer weight alone.',config=c.__dict__,data_dir=c.data_dir,hash_inputs=c.hash_inputs and not c.dry_run) as m:
  for seed,out in zip(c.seeds,runs):
   cmd=[sys.executable,_script('train.py'),'--data-dir',c.data_dir,'--out-dir',str(out),'--preset','full','--device',c.device,'--fold-seed',str(seed),'--n-splits','5','--lgb-estimators',str(c.lgb_estimators),'--experts',*c.experts]
   if c.expert_iterations:cmd+=['--expert-iterations',*c.expert_iterations]
   _run(cmd,c.dry_run)
  pd={}
  for b,k in ((c.baseline,c.candidate),(c.candidate,'blend')):
   out=root/f'promotion__{b}__to__{k}';pd[(b,k)]=out;_run([sys.executable,_script('aggregate_promotion.py'),'--data-dir',c.data_dir,'--runs',*map(str,runs),'--baseline',b,'--candidate',k,'--bootstrap',str(c.bootstrap),'--out-dir',str(out)],c.dry_run)
  if c.dry_run:r={'version':'nomophobia-v0.3','status':'DRY_RUN','runs':list(map(str,runs))}
  else:
   d=summarize_s3_runs(list(map(str,runs)));r1=pd[(c.baseline,c.candidate)]/'promotion.json';r2=pd[(c.candidate,'blend')]/'promotion.json';resolution=resolution_from_promotions(d,combined_to_blend_promotion=r2,raw_to_combined_promotion=r1);r={'version':'nomophobia-v0.3','status':d['status'],'runs':list(map(str,runs)),'diagnostics':d,'resolution':resolution};atomic_write_json(root/'s3_diagnostics.json',r);m.add_output(r1,r2,root/'s3_diagnostics.json');m.add_metrics(resolution=resolution,blend_auc_seed_spread=d.get('blend_auc_seed_spread'),max_rank_correlation=d.get('max_rank_correlation'))
   if d['status']=='STOP_AND_REPORT':m.status='STOP_AND_REPORT'
  atomic_write_json(root/'campaign.json',r);m.add_output(root/'campaign.json');return r

def _materialize(root,s3,data_dir):
 route=(s3.get('resolution') or {}).get('route')
 if route=='FREEZE_DUAL_VIEW_BACKBONE_AND_RETRY_DIVERSITY':src=root/'s3'/'promotion__lgb_combined63__to__blend'/'submission_candidate_seedbag.csv';label='s3_dualview_seedbag'
 elif route=='FREEZE_COMBINED_BACKBONE_RAW_HEDGE_NOT_PROMOTED':src=root/'s3'/'promotion__lgb_raw63__to__lgb_combined63'/'submission_candidate_seedbag.csv';label='s3_combined_seedbag'
 else:return None
 if not src.exists():return None
 dst=root/'submission_s3.csv';shutil.copyfile(src,dst);_,te,_=load_competition(data_dir);import pandas as pd;stats=validate_submission(pd.read_csv(dst),te);rec={'label':label,'source':str(src),'file':str(dst),'sha256':sha256_file(dst),**stats};atomic_write_json(root/'submission_s3.json',rec);return rec

def run_winner_campaign(c:WinnerCampaignConfig):
 root=Path(c.out_root);root.mkdir(parents=True,exist_ok=True)
 with ExperimentRecorder(root/'winner_manifest.json','WINNER-CAMPAIGN-V03','S3','Production-scale tuning followed by S3 resolves backbone and raw diversity.','Tuning hits ceiling/is unstable or S3 fails gates.','Use median repeated-holdout counts then mechanical S3 promotion.','If only raw fails, freeze combined and redirect to new diversity.',config=c.__dict__,data_dir=c.data_dir,hash_inputs=c.hash_inputs and not c.dry_run) as m:
  sel={};tuning={}
  for expert,label,provided in [('lgb_combined63','combined',c.combined_iterations),('lgb_raw63','raw',c.raw_iterations)]:
   if provided is not None:sel[label]=int(provided);tuning[label]={'provided':True,'best_iteration':int(provided),'ceiling_hit_any_repeat':False};continue
   out=root/f'tune_{label}.json';cmd=[sys.executable,_script('tune_iterations.py'),'--data-dir',c.data_dir,'--expert',expert,'--rows',str(c.tune_rows),'--max-estimators',str(c.max_estimators),'--patience',str(c.patience),'--repeats',str(c.tune_repeats),'--ceiling-fraction',str(c.ceiling_fraction),'--device',c.device,'--out',str(out)];_run(cmd,c.dry_run)
   if c.dry_run:sel[label]=1000;tuning[label]={'dry_run':True,'best_iteration':1000,'ceiling_hit_any_repeat':False};continue
   meta=_read(out);tuning[label]=meta;sel[label]=int(meta['best_iteration']);m.add_output(out)
   if meta.get('ceiling_hit_any_repeat') and not c.allow_ceiling:
    r={'version':'nomophobia-v0.3','status':'STOP_MAX_ESTIMATOR_CEILING','expert':expert,'selected_iterations':sel,'tuning':tuning,'action':'Raise --max-estimators and rerun before S3.'};atomic_write_json(root/'winner_campaign.json',r);m.add_output(root/'winner_campaign.json');m.status='STOP_AND_REPORT';return r
  s3=run_s3_campaign(S3CampaignConfig(data_dir=c.data_dir,out_root=str(root/'s3'),seeds=c.seeds,device=c.device,expert_iterations=(f'lgb_combined63={sel["combined"]}',f'lgb_raw63={sel["raw"]}'),bootstrap=c.bootstrap,dry_run=c.dry_run,hash_inputs=False));sub=None if c.dry_run else _materialize(root,s3,c.data_dir);r={'version':'nomophobia-v0.3','status':'DRY_RUN' if c.dry_run else (s3.get('resolution') or {}).get('route',s3.get('status')),'selected_iterations':sel,'tuning':tuning,'s3':s3,'submission':sub};atomic_write_json(root/'winner_campaign.json',r);m.add_output(root/'winner_campaign.json',root/'s3'/'campaign.json');m.add_metrics(selected_iterations=sel,s3_status=s3.get('status'),s3_resolution=s3.get('resolution'))
  if sub:m.add_output(root/'submission_s3.csv',root/'submission_s3.json')
  if not c.dry_run and s3.get('status')=='STOP_AND_REPORT':m.status='STOP_AND_REPORT'
  return r
