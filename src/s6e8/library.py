from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _numeric_prediction_columns(df: pd.DataFrame, n_expected: int):
    cols=[]
    for c in df.columns:
        if c.lower() in {'id','target','addicted_label','fold','index'}: continue
        if pd.api.types.is_numeric_dtype(df[c]) and len(df[c])==n_expected:
            v=df[c].to_numpy(float)
            if np.isfinite(v).all() and np.nanstd(v)>1e-9: cols.append(c)
    return cols


def inspect_library_alignment(path, train_ids=None, test_ids=None):
    """Audit ID and fold provenance before an external stack is trusted (A8)."""
    path=Path(path); info={'has_csv':False,'train_id_aligned':None,'test_id_aligned':None,'folds':None}
    a_path,b_path=path/'train.csv',path/'test.csv'
    if not (a_path.exists() and b_path.exists()): return info
    a=pd.read_csv(a_path,nrows=None); b=pd.read_csv(b_path,nrows=None); info['has_csv']=True
    if train_ids is not None:
        if 'id' not in a.columns: raise ValueError('External train.csv has no id column; cannot assert row alignment.')
        ok=np.array_equal(a['id'].to_numpy(),np.asarray(train_ids)); info['train_id_aligned']=bool(ok)
        if not ok: raise ValueError('External OOF train IDs are not aligned with competition train IDs.')
    if test_ids is not None:
        if 'id' not in b.columns: raise ValueError('External test.csv has no id column; cannot assert row alignment.')
        ok=np.array_equal(b['id'].to_numpy(),np.asarray(test_ids)); info['test_id_aligned']=bool(ok)
        if not ok: raise ValueError('External OOF test IDs are not aligned with competition test IDs.')
    if 'fold' in a.columns: info['folds']=a['fold'].to_numpy()
    return info


def load_oof_library(path,n_train:int,n_test:int):
    path=Path(path);oofs={};tests={};meta=None
    if (path/'manifest.csv').exists():
        try:meta=pd.read_csv(path/'manifest.csv')
        except Exception:meta=None
    train_csv=path/'train.csv';test_csv=path/'test.csv'
    if train_csv.exists() and test_csv.exists():
        a=pd.read_csv(train_csv);b=pd.read_csv(test_csv)
        if len(a)!=n_train or len(b)!=n_test: raise ValueError(f'Library row mismatch: train {len(a)} != {n_train} or test {len(b)} != {n_test}')
        common=[c for c in _numeric_prediction_columns(a,n_train) if c in b.columns]
        for c in common:oofs[c]=a[c].to_numpy(float);tests[c]=b[c].to_numpy(float)
    else:
        for f in sorted(path.glob('oof_*.npy')):
            name=f.stem[4:];tf=path/f'test_{name}.npy'
            if not tf.exists():continue
            o=np.load(f);t=np.load(tf)
            if len(o)==n_train and len(t)==n_test:oofs[name]=np.asarray(o,float);tests[name]=np.asarray(t,float)
    if not oofs:raise FileNotFoundError('No aligned prediction streams found. Expected train.csv/test.csv or oof_*.npy + test_*.npy pairs.')
    return oofs,tests,meta


def _rank(x):return rankdata(x,method='average')/len(x)

def deduplicate_library(y,oofs,tests,corr_threshold=.9995,max_models=80):
    names=list(oofs);auc={n:float(roc_auc_score(y,oofs[n])) for n in names};names.sort(key=lambda n:auc[n],reverse=True);kept=[];ranks={}
    for n in names:
        r=_rank(oofs[n]);ranks[n]=r
        if any(float(np.corrcoef(r,ranks[k])[0,1])>=corr_threshold for k in kept):continue
        kept.append(n)
        if len(kept)>=max_models:break
    return ({n:oofs[n] for n in kept},{n:tests[n] for n in kept},pd.DataFrame({'model':kept,'oof_auc':[auc[n] for n in kept]}))
