#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.io import load_competition
from s6e8.config import TARGET,PRESETS,EXPERT_SPECS,DEFAULT_EXPERTS
from s6e8.features import build_feature_views
from s6e8.preprocess import prepare_tree_frames
from s6e8.cv import frozen_folds,run_gbdt_cv
from s6e8.blend import search_blend,save_blend,model_correlation
from s6e8.diagnostics import regime_report
from s6e8.residuals import add_generator_residuals
from s6e8.frontier import missing_shift_table,missing_pattern_shift,weighted_test_like_auc,residual_forensics
from s6e8.calibration import crossfit_regime_isotonic

p=argparse.ArgumentParser(description='S6E8 Frontier: audited diversity + regime diagnostics')
p.add_argument('--data-dir',default='data');p.add_argument('--out-dir',default='artifacts/run');p.add_argument('--preset',choices=PRESETS,default='quick')
p.add_argument('--experts',nargs='+',choices=list(EXPERT_SPECS),default=None);p.add_argument('--with-residuals',action='store_true');p.add_argument('--with-transformer',action='store_true')
p.add_argument('--with-regime-calibration',action='store_true');p.add_argument('--with-components',action='store_true')
p.add_argument('--components-method',choices=['kmeans','bgmm'],default='kmeans');p.add_argument('--components-k',type=int,default=8);p.add_argument('--components-fit-rows',type=int,default=120000)
p.add_argument('--no-frequency',action='store_true');p.add_argument('--max-rows',type=int,default=None);p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
p.add_argument('--fold-seed',type=int,default=None);p.add_argument('--n-splits',type=int,default=None);p.add_argument('--lgb-estimators',type=int,default=None);p.add_argument('--cat-iterations',type=int,default=None);p.add_argument('--xgb-estimators',type=int,default=None)
p.add_argument('--expert-iterations',nargs='*',default=[],metavar='MODEL=N')
a=p.parse_args();preset=PRESETS[a.preset]
expert_iter={}
for item in a.expert_iterations:
    if '=' not in item: raise ValueError(f'Bad --expert-iterations value: {item}; expected MODEL=N')
    k,v=item.split('=',1)
    if k not in EXPERT_SPECS: raise ValueError(f'Unknown expert in --expert-iterations: {k}')
    expert_iter[k]=int(v)
preset=replace(preset,seed=a.fold_seed if a.fold_seed is not None else preset.seed,n_splits=a.n_splits if a.n_splits is not None else preset.n_splits,lgb_estimators=a.lgb_estimators if a.lgb_estimators is not None else preset.lgb_estimators,cat_iterations=a.cat_iterations if a.cat_iterations is not None else preset.cat_iterations,xgb_estimators=a.xgb_estimators if a.xgb_estimators is not None else preset.xgb_estimators)
out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);experts=a.experts or DEFAULT_EXPERTS[a.preset]
tr_full,te_raw,sample=load_competition(a.data_dir);full_train_for_shift=tr_full.copy();freq_ref=(tr_full.drop(columns=[TARGET]),te_raw);tr_raw=tr_full
max_rows=a.max_rows if a.max_rows is not None else preset.max_rows
if max_rows and len(tr_raw)>max_rows:
    frac=max_rows/len(tr_raw);tr_raw=(tr_raw.groupby(TARGET,group_keys=False).sample(frac=frac,random_state=preset.seed).sample(frac=1,random_state=preset.seed).head(max_rows).reset_index(drop=True))
y=tr_raw[TARGET].astype(int).reset_index(drop=True)
views=build_feature_views(tr_raw.drop(columns=[TARGET]),te_raw,use_frequency=not a.no_frequency,frequency_reference=freq_ref)
component_meta=None
if a.with_components:
    from s6e8.components import add_latent_component_features
    C,CT,component_meta=add_latent_component_features(tr_raw.drop(columns=[TARGET]),te_raw,method=a.components_method,n_components=a.components_k,seed=preset.seed,max_fit_rows=a.components_fit_rows)
    X,T=views['combined'];views['combined']=(pd.concat([X.reset_index(drop=True),C],axis=1),pd.concat([T.reset_index(drop=True),CT],axis=1))
if a.with_residuals:
    X,T=views['combined'];views['combined']=add_generator_residuals(tr_raw.drop(columns=[TARGET]),te_raw,X,T,n_splits=3,seed=preset.seed,n_estimators=100 if a.preset=='smoke' else (180 if a.preset in ('quick','highcap') else 350))
prepared={}
for view_name,(X,T) in views.items():prepared[view_name]=(*prepare_tree_frames(X,T),X.shape[1])
folds=frozen_folds(y,preset.n_splits,preset.seed);pd.DataFrame({'id':tr_raw.get('id',pd.Series(np.arange(len(tr_raw)))),'fold':folds,'target':y}).to_csv(out/'folds.csv',index=False)
oofs={};tests={};metrics=[]
for key in experts:
    spec=EXPERT_SPECS[key];family,view,profile=spec['family'],spec['view'],spec['profile'];Xc,Tc,Xn,Tn,cats,nfeat=prepared[view];local_preset=preset
    if key in expert_iter:
        n=expert_iter[key];local_preset=replace(preset,lgb_estimators=n) if family=='lgb' else (replace(preset,cat_iterations=n) if family=='cat' else replace(preset,xgb_estimators=n))
    o,t,m=run_gbdt_cv(key,family,profile,Xc,Xn,y,Tc,Tn,cats,folds,local_preset,out,device=a.device);m['view']=view;m['n_features']=nfeat;oofs[key]=o;tests[key]=t;metrics.append(m)
if a.with_transformer:
    from s6e8.transformer import run_transformer_cv
    X,T=views['combined'];o,t,s=run_transformer_cv(X,T,y,n_splits=preset.n_splits,seed=preset.seed,epochs=3 if a.preset=='smoke' else (5 if a.preset=='quick' else 9));oofs['transformer_combined']=o;tests['transformer_combined']=t;metrics.append({'model':'transformer_combined','family':'transformer','view':'combined','n_features':X.shape[1],'oof_auc':s});np.save(out/'oof_transformer_combined.npy',o);np.save(out/'test_transformer_combined.npy',t)
blend=search_blend(y,oofs,tests,trials=300 if a.preset=='smoke' else (700 if a.preset in ('quick','highcap') else 1200),seed=preset.seed,folds=folds);save_blend(blend,out)
final_oof,final_test=blend['oof'],blend['test'];cal_meta=None
if a.with_regime_calibration:
    co,ct,cal_meta=crossfit_regime_isotonic(tr_raw.drop(columns=[TARGET]),te_raw,y,final_oof,final_test,folds,top_n=10);np.save(out/'oof_regime_calibrated.npy',co);np.save(out/'test_regime_calibrated.npy',ct);final_oof,final_test=co,ct;(out/'regime_calibration.json').write_text(json.dumps(cal_meta,indent=2))
regime_report(tr_raw,y,final_oof).to_csv(out/'regime_auc.csv',index=False);missing_shift_table(full_train_for_shift,te_raw).to_csv(out/'missing_shift.csv',index=False);missing_pattern_shift(full_train_for_shift,te_raw).head(512).to_csv(out/'missing_pattern_shift.csv',index=False)
names,corr=model_correlation(oofs,folds=folds);pd.DataFrame(corr,index=names,columns=names).to_csv(out/'expert_rank_correlation.csv')
_,bands,hard_rows,scan=residual_forensics(tr_raw,y,oofs,final_oof);bands.to_csv(out/'residual_bands.csv',index=False);scan.to_csv(out/'consensus_hard_feature_scan.csv',index=False);hard_rows.head(2000).to_csv(out/'consensus_hard_cases.csv',index=False)
sub=sample.copy();sub[TARGET]=final_test;sub.to_csv(out/'submission.csv',index=False);pd.DataFrame(metrics).to_json(out/'model_metrics.json',orient='records',indent=2)
try:test_like_auc=weighted_test_like_auc(tr_raw,te_raw,y,final_oof)
except Exception:test_like_auc=None
run={'version':'nomophobia-frontier','tier':{'smoke':'S0','quick':'S1','highcap':'S1','full':'S3'}[a.preset],'rows_train':len(tr_raw),'rows_test':len(te_raw),'views':{k:v[0].shape[1] for k,v in views.items()},'blend_auc_honest':blend['auc'],'blend_selection_auc':blend.get('selection_auc'),'blend_selection_optimism':blend.get('selection_optimism'),'blend_mode':blend['mode'],'blend_fold_auc_mean':blend.get('fold_auc_mean'),'blend_fold_auc_std':blend.get('fold_auc_std'),'regime_calibration':cal_meta,'latent_components':component_meta,'final_oof_auc':float(__import__('sklearn.metrics').metrics.roc_auc_score(y,final_oof)),'test_missingness_weighted_oof_auc':test_like_auc,'experts':list(oofs),'device':a.device,'fold_seed':preset.seed,'n_splits':preset.n_splits,'lgb_estimators':preset.lgb_estimators,'cat_iterations':preset.cat_iterations,'xgb_estimators':preset.xgb_estimators,'expert_iteration_overrides':expert_iter}
(out/'run_summary.json').write_text(json.dumps(run,indent=2));print(json.dumps(run,indent=2))
