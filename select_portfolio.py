#!/usr/bin/env python
"""Select two genuinely different final submissions: best OOF and robust hedge."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent/'src'))
from s6e8.config import TARGET
from s6e8.io import load_competition
from s6e8.frontier import weighted_test_like_auc
p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--run-dir',required=True);p.add_argument('--out-dir',default='artifacts/portfolio');p.add_argument('--max-corr',type=float,default=.999);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);run=Path(a.run_dir);tr,te,sample=load_competition(a.data_dir);y=tr[TARGET].astype(int).to_numpy();folds=pd.read_csv(run/'folds.csv').fold.to_numpy()
def rank01(x):return rankdata(np.asarray(x),method='average')/len(x)
def fold_stats(p):
 vals=[]
 for f in np.unique(folds):
  m=folds==f;vals.append(float(roc_auc_score(y[m],p[m])))
 return vals
cands={}
for op in sorted(run.glob('oof_*.npy')):
 name=op.stem[4:];tp=run/f'test_{name}.npy'
 if tp.exists() and len(np.load(op,mmap_mode='r'))==len(tr) and len(np.load(tp,mmap_mode='r'))==len(te):cands[name]=(np.load(op),np.load(tp))
if (run/'oof_blend.npy').exists() and (run/'test_blend.npy').exists():cands['blend']=(np.load(run/'oof_blend.npy'),np.load(run/'test_blend.npy'))
if len(cands)<2:raise ValueError('Need at least two aligned candidates')
rows=[]
for name,(po,_) in cands.items():
 fs=fold_stats(po);rows.append({'name':name,'auc':float(roc_auc_score(y,po)),'weighted_auc':weighted_test_like_auc(tr,te,y,po),'fold_min':float(min(fs)),'fold_mean':float(np.mean(fs)),'fold_std':float(np.std(fs))})
tab=pd.DataFrame(rows).sort_values('auc',ascending=False).reset_index(drop=True);best=tab.iloc[0]['name'];rb=rank01(cands[best][0]);tab['robust_score']=tab['fold_min']+0.35*(tab['weighted_auc']-tab['weighted_auc'].mean())-0.20*tab['fold_std'];choices=[]
for _,r in tab.sort_values('robust_score',ascending=False).iterrows():
 n=r['name']
 if n==best:continue
 corr=float(np.corrcoef(rb,rank01(cands[n][0]))[0,1])
 if corr<=a.max_corr:choices.append((r['robust_score'],n,corr))
if not choices:raise RuntimeError(f'No second candidate has rank correlation <= {a.max_corr} with best={best}. Add a genuinely different hypothesis.')
_,robust,corr=max(choices);tab.to_csv(out/'portfolio_candidates.csv',index=False)
for slot,name in [('slot1_best',best),('slot2_robust',robust)]:
 q=sample.copy();q[TARGET]=rank01(cands[name][1]);q.to_csv(out/f'{slot}__{name}.csv',index=False)
res={'slot1_best':best,'slot2_robust':robust,'slot_rank_correlation':corr,'max_allowed_correlation':a.max_corr,'candidates':tab.to_dict('records')};(out/'portfolio.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
