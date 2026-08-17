#!/usr/bin/env python
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.library import load_oof_library,deduplicate_library,inspect_library_alignment
from s6e8.blend import search_blend,bagged_caruana_blend,save_blend,model_correlation
p=argparse.ArgumentParser(description='Price Frontier against an aligned public OOF universe');p.add_argument('--data-dir',default='data');p.add_argument('--base-run',required=True);p.add_argument('--library-dir',required=True);p.add_argument('--out-dir',default='artifacts/external_stack');p.add_argument('--corr-threshold',type=float,default=.9995);p.add_argument('--max-models',type=int,default=80);p.add_argument('--with-meta',action='store_true');p.add_argument('--blend-method',choices=['rotating','caruana'],default='caruana');p.add_argument('--caruana-bags',type=int,default=20);p.add_argument('--caruana-size',type=int,default=30);p.add_argument('--caruana-pairs',type=int,default=10000);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
tr,te,sample=load_competition(a.data_dir);y=tr[TARGET].astype(int).to_numpy();base=Path(a.base_run);fold_df=pd.read_csv(base/'folds.csv')
if len(fold_df)!=len(tr) or not np.array_equal(fold_df.id.to_numpy(),tr.id.to_numpy()):raise ValueError('base-run must be full-row and ID-aligned')
folds=fold_df.fold.to_numpy();bo={};bt={}
for f in base.glob('oof_*.npy'):
 name='frontier__'+f.stem[4:];tf=base/f.name.replace('oof_','test_',1)
 if tf.exists() and len(np.load(f,mmap_mode='r'))==len(tr):bo[name]=np.load(f);bt[name]=np.load(tf)
align=inspect_library_alignment(a.library_dir,tr.id.to_numpy(),te.id.to_numpy())
if align.get('folds') is not None and not np.array_equal(np.asarray(align['folds']),folds):raise ValueError('External library has a different fold scheme; rerun Frontier on that fold column before pricing.')
if a.with_meta and align.get('folds') is None:raise ValueError('--with-meta requires aligned external fold provenance; use blend-only mode otherwise.')
lo,lt,_=load_oof_library(a.library_dir,len(tr),len(te));lo={'public__'+k:v for k,v in lo.items()};lt={'public__'+k:v for k,v in lt.items()};all_o={**lo,**bo};all_t={**lt,**bt};o,t,kept=deduplicate_library(y,all_o,all_t,a.corr_threshold,a.max_models);kept['source']=kept.model.str.split('__').str[0];kept.to_csv(out/'kept_models.csv',index=False);rank_mode='fold_rank' if align.get('folds') is not None else 'rank'
blend=(bagged_caruana_blend(y,o,t,folds,seed=20260816,n_bags=a.caruana_bags,ensemble_size=a.caruana_size,n_pairs=a.caruana_pairs,mode=rank_mode) if a.blend_method=='caruana' else search_blend(y,o,t,trials=4000,seed=20260816,calibration_rows=120000,folds=folds));save_blend(blend,out);sub=sample.copy();sub[TARGET]=blend['test'];sub.to_csv(out/'submission.csv',index=False);names,corr=model_correlation(o,folds if rank_mode=='fold_rank' else None);pd.DataFrame(corr,index=names,columns=names).to_csv(out/'rank_correlation.csv')
weights={n:float(w) for n,w in zip(blend['names'],blend['weights'])};fw=float(sum(w for n,w in weights.items() if n.startswith('frontier__')));pw=float(sum(w for n,w in weights.items() if n.startswith('public__')));action=('BUILD_PORTFOLIO_REAL_ORTHOGONAL_SIGNAL' if fw>=.15 else ('MARGINAL_PRIORITIZE_DIVERSITY_RETRIAL' if fw>=.05 else 'STOP_FRONTIER_INSIDE_PUBLIC_BASIN'))
summary={'version':'nomophobia','n_public_input':len(lo),'n_frontier_input':len(bo),'n_models_after_joint_pruning':len(o),'rank_mode':rank_mode,'external_fold_provenance':align.get('folds') is not None,'blend_auc':blend['auc'],'blend_selection_auc':blend.get('selection_auc'),'blend_mode':blend['mode'],'blend_method':blend.get('method',a.blend_method),'frontier_total_weight':fw,'public_total_weight':pw,'frontier_weight_action':action,'weights':weights};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
