from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from .config import NUM_COLS, CAT_COLS


def _simple_encode(train,test):
    tr=train[NUM_COLS+CAT_COLS].copy(); te=test[NUM_COLS+CAT_COLS].copy()
    for c in CAT_COLS:
        allv=pd.concat([tr[c].astype('string'),te[c].astype('string')]).fillna('MISSING'); cats=sorted(allv.unique()); mapping={v:i for i,v in enumerate(cats)}; tr[c]=tr[c].astype('string').fillna('MISSING').map(mapping); te[c]=te[c].astype('string').fillna('MISSING').map(mapping)
    for c in NUM_COLS:
        med=pd.concat([tr[c],te[c]]).median(); tr[c]=tr[c].fillna(med); te[c]=te[c].fillna(med)
    return tr.astype(float),te.astype(float)


def add_generator_residuals(train_raw,test_raw,train_feat,test_feat,n_splits=3,seed=20260816,targets=None,n_estimators=250):
    targets=targets or ['daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours','sleep_hours','weekend_screen_time']; base_tr,base_te=_simple_encode(train_raw,test_raw); out_tr,out_te=train_feat.copy(),test_feat.copy(); kf=KFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for target in targets:
        feats=[c for c in base_tr.columns if c!=target]; obs=train_raw[target].notna().to_numpy(); idx=np.where(obs)[0]; miss=np.where(~obs)[0]; oof=np.full(len(train_raw),np.nan,dtype=np.float32); test_parts=[]; missing_parts=[]
        for f,(a,b) in enumerate(kf.split(idx)):
            ti,vi=idx[a],idx[b]; m=LGBMRegressor(n_estimators=n_estimators,learning_rate=.04,num_leaves=31,min_child_samples=100,subsample=.85,colsample_bytree=.85,random_state=seed+f,n_jobs=-1,verbosity=-1); m.fit(base_tr.iloc[ti][feats],train_raw.iloc[ti][target]); oof[vi]=m.predict(base_tr.iloc[vi][feats]);
            if len(miss): missing_parts.append(m.predict(base_tr.iloc[miss][feats]))
            test_parts.append(m.predict(base_te[feats]))
        if len(miss): oof[miss]=np.mean(missing_parts,axis=0)
        tp=np.mean(test_parts,axis=0); out_tr[f'recon_{target}__pred']=oof; out_tr[f'recon_{target}__resid']=train_raw[target]-oof; out_tr[f'recon_{target}__absresid']=np.abs(out_tr[f'recon_{target}__resid']); out_te[f'recon_{target}__pred']=tp; out_te[f'recon_{target}__resid']=test_raw[target]-tp; out_te[f'recon_{target}__absresid']=np.abs(out_te[f'recon_{target}__resid'])
    return out_tr,out_te
