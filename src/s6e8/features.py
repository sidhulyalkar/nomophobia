from __future__ import annotations
import numpy as np
import pandas as pd
from .config import CAT_COLS, NUM_COLS, CORE_USAGE, DECIMAL_COLS, ID_COL

EPS = 1e-3


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.abs() + EPS)


def _row_sum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].sum(axis=1, min_count=1)


def add_natural_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "stress_level" in x:
        x["stress_level__ord"] = x["stress_level"].map({"Low": 0, "Medium": 1, "High": 2}).astype("float32")
    if "academic_work_impact" in x:
        x["academic_work_impact__ord"] = x["academic_work_impact"].map({"No": 0, "Yes": 1}).astype("float32")
    return x


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_natural_ordinals(df)
    d=x["daily_screen_time_hours"]; s=x["social_media_hours"]; g=x["gaming_hours"]; w=x["work_study_hours"]
    sl=x["sleep_hours"]; wk=x["weekend_screen_time"]; n=x["notifications_per_day"]; o=x["app_opens_per_day"]
    x["leisure_hours"]=_row_sum(x,["social_media_hours","gaming_hours"])
    x["accounted_screen_hours"]=_row_sum(x,["social_media_hours","gaming_hours","work_study_hours"])
    x["unaccounted_screen_hours"]=d-x["accounted_screen_hours"]
    x["social_share"]=safe_div(s,d); x["gaming_share"]=safe_div(g,d); x["work_share"]=safe_div(w,d)
    x["leisure_share"]=safe_div(x["leisure_hours"],d); x["weekend_delta"]=wk-d; x["weekend_ratio"]=safe_div(wk,d)
    x["social_weekend_ratio"]=safe_div(s,wk); x["screen_sleep_ratio"]=safe_div(d,sl); x["screen_plus_sleep"]=d+sl
    x["awake_hours"]=24.0-sl; x["screen_awake_share"]=safe_div(d,x["awake_hours"])
    x["notifications_per_open"]=safe_div(n,o); x["opens_per_screen_hour"]=safe_div(o,d); x["notifications_per_screen_hour"]=safe_div(n,d)
    x["social_to_leisure"]=safe_div(s,x["leisure_hours"]); x["gaming_to_leisure"]=safe_div(g,x["leisure_hours"])
    x["severity_exposure"]=0.45*d+0.30*wk+0.25*s; x["severity_centered"]=0.45*(d-7.5)+0.30*(wk-9.0)+0.25*(s-3.0)
    return x


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy(); x["missing_count"]=x[NUM_COLS+CAT_COLS].isna().sum(axis=1).astype(np.int8); x["core_missing_count"]=x[CORE_USAGE].isna().sum(axis=1).astype(np.int8)
    mask=np.zeros(len(x),dtype=np.uint16)
    for i,c in enumerate(NUM_COLS+CAT_COLS): mask|=(x[c].isna().to_numpy(dtype=np.uint16)<<i)
    x["missing_pattern"]=pd.Series(mask,index=x.index).astype(str); return x


def add_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    for c in DECIMAL_COLS:
        v=x[c]; scaled=np.rint(v*100); tenths=((scaled//10)%10); hundredths=(scaled%10)
        x[f"{c}__tenths_digit"]=pd.Series(tenths,index=x.index).astype("Int64").astype(str).replace("<NA>","MISSING")
        x[f"{c}__hundredths_digit"]=pd.Series(hundredths,index=x.index).astype("Int64").astype(str).replace("<NA>","MISSING")
        x[f"{c}__tenths_ord"]=pd.Series(tenths,index=x.index).fillna(-1).astype("int8"); x[f"{c}__hundredths_ord"]=pd.Series(hundredths,index=x.index).fillna(-1).astype("int8")
        x[f"{c}__dist_integer"]=np.abs(v-np.rint(v)); x[f"{c}__dist_half"]=np.abs(v*2-np.rint(v*2))/2; x[f"{c}__dist_tenth"]=np.abs(v*10-np.rint(v*10))/10
    return x


def add_frequency_features(train: pd.DataFrame,test: pd.DataFrame,*,reference_train: pd.DataFrame|None=None,reference_test: pd.DataFrame|None=None)->tuple[pd.DataFrame,pd.DataFrame]:
    tr,te=train.copy(),test.copy(); rr=reference_train if reference_train is not None else train; rt=reference_test if reference_test is not None else test
    both=pd.concat([rr,rt],axis=0,ignore_index=True)
    for c in NUM_COLS+CAT_COLS:
        if c in NUM_COLS:
            sentinel=-9.87654321e15; key=both[c].fillna(sentinel); vc=key.value_counts(dropna=False); tr_key=tr[c].fillna(sentinel); te_key=te[c].fillna(sentinel)
        else:
            key=both[c].astype("string").fillna("__MISSING__"); vc=key.value_counts(dropna=False); tr_key=tr[c].astype("string").fillna("__MISSING__"); te_key=te[c].astype("string").fillna("__MISSING__")
        tr[f"{c}__freq"]=tr_key.map(vc).astype(np.float32).to_numpy(); te[f"{c}__freq"]=te_key.map(vc).astype(np.float32).to_numpy()
        tr[f"{c}__logfreq"]=np.log1p(tr[f"{c}__freq"]); te[f"{c}__logfreq"]=np.log1p(te[f"{c}__freq"])
    for c in DECIMAL_COLS:
        for dec in (0,1):
            sentinel=-9.87654321e15; br=both[c].round(dec).fillna(sentinel); vc=br.value_counts(dropna=False); tr_key=tr[c].round(dec).fillna(sentinel); te_key=te[c].round(dec).fillna(sentinel); name=f"{c}__round{dec}_freq"
            tr[name]=tr_key.map(vc).astype(np.float32).to_numpy(); te[name]=te_key.map(vc).astype(np.float32).to_numpy()
    return tr,te


def build_features(train: pd.DataFrame,test: pd.DataFrame,*,use_frequency: bool=True,frequency_reference: tuple[pd.DataFrame,pd.DataFrame]|None=None):
    tr=train.drop(columns=[ID_COL],errors="ignore").copy(); te=test.drop(columns=[ID_COL],errors="ignore").copy()
    tr=add_behavior_features(add_missingness_features(add_digit_features(tr))); te=add_behavior_features(add_missingness_features(add_digit_features(te)))
    if use_frequency:
        rr,rt=frequency_reference if frequency_reference is not None else (tr,te); tr,te=add_frequency_features(tr,te,reference_train=rr,reference_test=rt)
    return tr,te


def categorical_columns(df: pd.DataFrame)->list[str]:
    return [c for c in df.columns if df[c].dtype=="object" or str(df[c].dtype).startswith(("string","category"))]


def build_feature_views(train: pd.DataFrame,test: pd.DataFrame,*,use_frequency: bool=True,frequency_reference: tuple[pd.DataFrame,pd.DataFrame]|None=None):
    tr0=train.drop(columns=[ID_COL],errors="ignore").copy(); te0=test.drop(columns=[ID_COL],errors="ignore").copy(); rr,rt=frequency_reference if frequency_reference is not None else (tr0,te0); rr=rr.drop(columns=[ID_COL],errors="ignore"); rt=rt.drop(columns=[ID_COL],errors="ignore")
    semantic_tr=add_behavior_features(add_missingness_features(tr0)); semantic_te=add_behavior_features(add_missingness_features(te0))
    generator_tr=add_natural_ordinals(add_missingness_features(add_digit_features(tr0))); generator_te=add_natural_ordinals(add_missingness_features(add_digit_features(te0)))
    combined_tr=add_behavior_features(add_missingness_features(add_digit_features(tr0))); combined_te=add_behavior_features(add_missingness_features(add_digit_features(te0)))
    if use_frequency:
        ftr,fte=add_frequency_features(tr0,te0,reference_train=rr,reference_test=rt); freq_cols=[c for c in ftr.columns if c not in tr0.columns]
        generator_tr=pd.concat([generator_tr.reset_index(drop=True),ftr[freq_cols].reset_index(drop=True)],axis=1); generator_te=pd.concat([generator_te.reset_index(drop=True),fte[freq_cols].reset_index(drop=True)],axis=1)
        combined_tr=pd.concat([combined_tr.reset_index(drop=True),ftr[freq_cols].reset_index(drop=True)],axis=1); combined_te=pd.concat([combined_te.reset_index(drop=True),fte[freq_cols].reset_index(drop=True)],axis=1)
    return {"raw":(tr0.copy(),te0.copy()),"semantic":(semantic_tr,semantic_te),"generator":(generator_tr,generator_te),"combined":(combined_tr,combined_te)}
