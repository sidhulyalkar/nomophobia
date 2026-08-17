from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from .config import NUM_COLS, CAT_COLS


def missing_pattern(df: pd.DataFrame) -> np.ndarray:
    mask=np.zeros(len(df),dtype=np.uint16)
    for i,c in enumerate(NUM_COLS+CAT_COLS):
        mask |= (df[c].isna().to_numpy(dtype=np.uint16) << i)
    return mask.astype(str)


def _bucket_patterns(tr_pat,te_pat,top_n=10):
    allp=pd.Series(np.r_[tr_pat,te_pat]); top=set(allp.value_counts().head(top_n).index.astype(str))
    return np.array([p if p in top else 'OTHER' for p in tr_pat],dtype=object), np.array([p if p in top else 'OTHER' for p in te_pat],dtype=object), sorted(top)


def _fit_iso(score,y):
    if len(score)<80 or len(np.unique(y))<2: return None
    return IsotonicRegression(out_of_bounds='clip').fit(score,y)


def crossfit_regime_isotonic(train_raw,test_raw,y,oof_score,test_score,folds,top_n=10):
    y=np.asarray(y); s=np.asarray(oof_score,float); ts=np.asarray(test_score,float); folds=np.asarray(folds)
    trp,tep,top=_bucket_patterns(missing_pattern(train_raw),missing_pattern(test_raw),top_n)
    out=np.zeros(len(y)); regimes=sorted(set(trp))
    for f in np.unique(folds):
        tr=folds!=f; va=folds==f; global_iso=_fit_iso(s[tr],y[tr])
        for r in regimes:
            a=tr&(trp==r); b=va&(trp==r)
            if not b.any(): continue
            iso=_fit_iso(s[a],y[a]); model=iso if iso is not None else global_iso
            out[b]=model.predict(s[b]) if model is not None else s[b]
    tout=np.zeros(len(ts)); global_iso=_fit_iso(s,y)
    for r in sorted(set(tep)):
        a=trp==r; b=tep==r; iso=_fit_iso(s[a],y[a]); model=iso if iso is not None else global_iso
        tout[b]=model.predict(ts[b]) if model is not None else ts[b]
    return out,tout,{'raw_auc':float(roc_auc_score(y,s)),'calibrated_auc':float(roc_auc_score(y,out)),'top_patterns':top,'n_regimes':len(regimes)}
