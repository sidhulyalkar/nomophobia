from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .config import NUM_COLS, CAT_COLS, CORE_USAGE, TARGET


def safe_auc(y,p):
    if len(np.unique(y))<2:return np.nan
    return roc_auc_score(y,p)


def data_report(train: pd.DataFrame)->pd.DataFrame:
    rows=[]; y=train[TARGET]
    for c in NUM_COLS:
        s=train[c]; valid=s.notna(); raw_auc=safe_auc(y[valid],s[valid]) if valid.sum() else np.nan; raw_auc=max(raw_auc,1-raw_auc) if pd.notna(raw_auc) else raw_auc
        rows.append({"feature":c,"dtype":"numeric","missing_pct":100*s.isna().mean(),"n_unique":s.nunique(dropna=True),"single_feature_auc_abs":raw_auc,"target_if_missing":y[s.isna()].mean() if s.isna().any() else np.nan,"target_if_present":y[valid].mean() if valid.any() else np.nan})
    for c in CAT_COLS:
        s=train[c]; rows.append({"feature":c,"dtype":"categorical","missing_pct":100*s.isna().mean(),"n_unique":s.nunique(dropna=True),"single_feature_auc_abs":np.nan,"target_if_missing":y[s.isna()].mean() if s.isna().any() else np.nan,"target_if_present":y[s.notna()].mean() if s.notna().any() else np.nan})
    return pd.DataFrame(rows)


def regime_report(train_raw: pd.DataFrame,y,pred)->pd.DataFrame:
    regimes={"all":np.ones(len(y),dtype=bool),"core_complete":train_raw[CORE_USAGE].notna().all(axis=1).to_numpy(),"daily_missing":train_raw["daily_screen_time_hours"].isna().to_numpy(),"social_missing":train_raw["social_media_hours"].isna().to_numpy(),"weekend_missing":train_raw["weekend_screen_time"].isna().to_numpy(),"multi_core_missing":(train_raw[CORE_USAGE].isna().sum(axis=1)>=2).to_numpy()}
    rows=[]; ya=np.asarray(y); pa=np.asarray(pred)
    for name,m in regimes.items(): rows.append({"regime":name,"n":int(m.sum()),"prevalence":float(ya[m].mean()) if m.any() else np.nan,"auc":safe_auc(ya[m],pa[m]) if m.sum()>5 else np.nan})
    return pd.DataFrame(rows)
