from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from .models import make_lgb, make_cat, make_xgb
from .config import MONOTONE_POSITIVE
from .utils import stable_seed


def frozen_folds(y,n_splits=5,seed=20260816):
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed); fold=np.full(len(y),-1,dtype=np.int8); dummy=np.zeros(len(y))
    for f,(_,va) in enumerate(skf.split(dummy,y)): fold[va]=f
    return fold


def _monotone_vector(columns,profile:str):
    if profile!="monotone31": return None
    return [1 if c in MONOTONE_POSITIVE else 0 for c in columns]


def run_gbdt_cv(model_key,family,profile,X_cat,X_native,y,T_cat,T_native,cat_cols,folds,preset,out_dir,device="cpu"):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); oof=np.zeros(len(y),dtype=np.float64); test_preds=[]; scores=[]
    for f in np.unique(folds):
        tr_idx=np.where(folds!=f)[0]; va_idx=np.where(folds==f)[0]; seed=stable_seed(preset.seed,model_key,int(f))
        if family=="cat":
            model=make_cat(seed,preset.cat_iterations,profile,device=device); model.fit(X_cat.iloc[tr_idx],y.iloc[tr_idx],cat_features=cat_cols,verbose=False); pv=model.predict_proba(X_cat.iloc[va_idx])[:,1]; pt=model.predict_proba(T_cat)[:,1]; used_iter=preset.cat_iterations
        elif family=="lgb":
            mono=_monotone_vector(X_native.columns,profile); model=make_lgb(seed,preset.lgb_estimators,profile,mono,device=device); model.fit(X_native.iloc[tr_idx],y.iloc[tr_idx],categorical_feature=[c for c in cat_cols if c in X_native.columns]); pv=model.predict_proba(X_native.iloc[va_idx])[:,1]; pt=model.predict_proba(T_native)[:,1]; used_iter=preset.lgb_estimators
        elif family=="xgb":
            model=make_xgb(seed,preset.xgb_estimators,profile,device=device); model.fit(X_native.iloc[tr_idx],y.iloc[tr_idx],verbose=False); pv=model.predict_proba(X_native.iloc[va_idx])[:,1]; pt=model.predict_proba(T_native)[:,1]; used_iter=preset.xgb_estimators
        else: raise ValueError(family)
        oof[va_idx]=pv; test_preds.append(pt); score=float(roc_auc_score(y.iloc[va_idx],pv)); scores.append(score); print(f"{model_key} fold {f}: {score:.7f} (fixed_iter={used_iter})",flush=True)
    test_pred=np.mean(test_preds,axis=0); overall=float(roc_auc_score(y,oof)); np.save(out_dir/f"oof_{model_key}.npy",oof); np.save(out_dir/f"test_{model_key}.npy",test_pred)
    metric={"model":model_key,"family":family,"profile":profile,"fold_auc":scores,"fold_auc_mean":float(np.mean(scores)),"fold_auc_std":float(np.std(scores)),"fixed_iterations":used_iter,"oof_auc":overall}
    with open(out_dir/f"metrics_{model_key}.json","w") as fh: json.dump(metric,fh,indent=2)
    print(f"{model_key} OOF: {overall:.7f}",flush=True); return oof,test_pred,metric
