from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def _read(path: Path) -> dict[str, Any]:
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}

def _best_advanced(comparisons):
    advanced=[(n,r) for n,r in comparisons.items() if isinstance(r,dict) and r.get('advance_s1')]
    return max(advanced,key=lambda x:float(x[1].get('delta_auc',-1e9))) if advanced else (None,None)

def recommend_next_actions(artifact_root='artifacts',*,data_dir='/kaggle/input/playground-series-s6e8',original_csv='/kaggle/input/smart-phone/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'):
    root=Path(artifact_root);actions=[];evidence={}
    fs=_read(root/'frequency_stress.json');evidence['frequency_stress']=bool(fs)
    if not fs:actions.append({'priority':1,'id':'frequency_safety','why':'Resolve the mature transduction/source-safety gate.','command':f'python experiments/frequency_stress.py --data-dir {data_dir} --rows 120000 --estimators 1000 --device gpu --out {root / "frequency_stress.json"}'})
    elif (fs.get('source_safety') or {}).get('stop'):actions.append({'priority':0,'id':'stop_source_fingerprint','why':'Complete-row source separability breached the stop gate.','command':'Do not expand transductive density features until diagnosed.'})
    fam=_read(root/'frequency_family_ablation.json');evidence['frequency_family_ablation']=bool(fam)
    if not fam:actions.append({'priority':2,'id':'frequency_family_ablation','why':'Identify which marginal frequency family carries the gain.','command':f'python experiments/frequency_family_ablation.py --data-dir {data_dir} --rows 120000 --estimators 1000 --device gpu --out {root / "frequency_family_ablation.json"}'})
    cap=_read(root/'capacity_diversity_curve.json');evidence['capacity_diversity_curve']=bool(cap)
    if not cap:actions.append({'priority':3,'id':'capacity_diversity_curve','why':'Resolve whether raw-view diversity collapses as capacity rises.','command':f'python experiments/capacity_diversity_curve.py --data-dir {data_dir} --rows 180000 --estimators 400 700 1000 1400 2000 --device gpu --out {root / "capacity_diversity_curve.json"}'})
    gp=root/'frequency_geometry'/'frequency_geometry.json';geo=_read(gp);evidence['frequency_geometry']=bool(geo)
    if not geo:actions.append({'priority':4,'id':'frequency_geometry','why':'Probe joint, regime-conditioned, and source-stable population density.','command':f'python experiments/frequency_geometry.py --data-dir {data_dir} --rows 120000 --estimators 1000 --device gpu --out-dir {gp.parent}'})
    else:
        n,b=_best_advanced(geo.get('paired_vs_baseline') or {})
        if n:actions.append({'priority':2,'id':f'geometry_s2__{n}','why':f'{n} advanced at S1 with delta {float(b.get("delta_auc",0)):+.6f}.','command':f'python experiments/frequency_geometry.py --data-dir {data_dir} --rows 350000 --estimators 1600 --device gpu --out-dir {root / "frequency_geometry_s2"}'})
    sp=root/'original_row_augmentation'/'original_row_augmentation.json';src=_read(sp);evidence['original_row_augmentation']=bool(src)
    if not src:actions.append({'priority':5,'id':'original_row_augmentation','why':'Direct low-weight source supervision has not been tested after overlap removal.','command':f'python experiments/original_row_augmentation.py --data-dir {data_dir} --original-csv {original_csv} --rows 120000 --estimators 1000 --device gpu --out-dir {sp.parent}'})
    wp=root/'winner'/'winner_campaign.json';winner=_read(wp);evidence['winner_campaign']=bool(winner)
    if not winner:actions.append({'priority':6,'id':'authoritative_s3','why':'Only S3 can resolve the production backbone and raw hedge.','command':f'python -m s6e8 winner --data-dir {data_dir} --out-root {root / "winner"} --device gpu --tune-rows 628000 --max-estimators 4000 --patience 200 --tune-repeats 3'})
    else:
        route=((winner.get('s3') or {}).get('resolution') or {}).get('route');evidence['s3_route']=route
        if route=='FREEZE_DUAL_VIEW_BACKBONE_AND_RETRY_DIVERSITY':actions.append({'priority':1,'id':'mature_diversity_retrial','why':'S3 promoted dual-view; next gains should come from orthogonal families.','command':f'python run_diversity_retrial.py --data-dir {data_dir} --base-s3-root {root / "winner" / "s3"} --out-root {root / "diversity"} --device gpu'})
        elif route=='FREEZE_COMBINED_BACKBONE_RAW_HEDGE_NOT_PROMOTED':actions.append({'priority':1,'id':'replace_raw_diversity','why':'Combined survived but raw did not. Stop weight probing.','command':'Prioritize geometry, source augmentation, CatBoost/XGBoost/Evidence survivors.'})
    actions.sort(key=lambda x:(x['priority'],x['id']))
    return {'version':'nomophobia-v0.3','artifact_root':str(root),'evidence_present':evidence,'actions':actions}
