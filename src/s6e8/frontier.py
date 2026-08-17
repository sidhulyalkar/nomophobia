from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, log_loss
from .config import RAW_COLS, NUM_COLS, CAT_COLS, CORE_USAGE


def missing_pattern_code(df: pd.DataFrame, cols=None) -> np.ndarray:
    cols = cols or RAW_COLS
    b = df[cols].isna().to_numpy(dtype=np.uint16)
    powers = (1 << np.arange(len(cols), dtype=np.uint16))
    return b @ powers


def missing_shift_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in RAW_COLS:
        a = float(train[c].isna().mean()); b = float(test[c].isna().mean())
        rows.append({"feature": c, "train_missing": a, "test_missing": b, "delta": b-a})
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def missing_pattern_shift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    a = pd.Series(missing_pattern_code(train)).value_counts().rename("train_n")
    b = pd.Series(missing_pattern_code(test)).value_counts().rename("test_n")
    z = pd.concat([a, b], axis=1).fillna(0)
    z["train_freq"] = z.train_n / len(train); z["test_freq"] = z.test_n / len(test)
    z["test_train_ratio"] = (z.test_freq / z.train_freq.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    z.index.name = "pattern"; return z.reset_index().sort_values("test_freq", ascending=False)


def test_pattern_weights(train: pd.DataFrame, test: pd.DataFrame, clip=(0.25, 4.0)) -> np.ndarray:
    tab = missing_pattern_shift(train, test).set_index("pattern")
    ratio = tab.test_train_ratio.fillna(1.0).clip(*clip)
    w = pd.Series(missing_pattern_code(train)).map(ratio).fillna(1.0).to_numpy(float)
    return w / w.mean()


def weighted_test_like_auc(train: pd.DataFrame, test: pd.DataFrame, y, pred) -> float:
    w = test_pattern_weights(train, test)
    return float(roc_auc_score(y, pred, sample_weight=w))


def _rank01(x):
    return rankdata(np.asarray(x), method="average") / len(x)


def disagreement_frame(oof_dict: dict[str, np.ndarray], blend: np.ndarray) -> pd.DataFrame:
    names = list(oof_dict)
    R = np.column_stack([_rank01(oof_dict[n]) for n in names])
    out = pd.DataFrame({
        "blend": blend,
        "expert_mean_rank": R.mean(axis=1),
        "expert_std_rank": R.std(axis=1),
        "expert_spread_rank": R.max(axis=1) - R.min(axis=1),
        "expert_min_rank": R.min(axis=1),
        "expert_max_rank": R.max(axis=1),
    })
    for j, n in enumerate(names): out[f"rank__{n}"] = R[:, j]
    return out


def residual_forensics(train_raw: pd.DataFrame, y, oof_dict: dict[str, np.ndarray], blend: np.ndarray):
    y = np.asarray(y); p = np.clip(np.asarray(blend), 1e-7, 1-1e-7)
    d = disagreement_frame(oof_dict, p)
    d["target"] = y
    d["abs_error"] = np.abs(y-p)
    d["logloss"] = -(y*np.log(p)+(1-y)*np.log(1-p))
    d["missing_count"] = train_raw[RAW_COLS].isna().sum(axis=1).to_numpy()
    d["core_missing_count"] = train_raw[CORE_USAGE].isna().sum(axis=1).to_numpy()
    d["confidence"] = np.abs(_rank01(p)-0.5)*2
    d["pred_decile"] = pd.qcut(_rank01(p), 10, labels=False, duplicates="drop")
    if d.expert_std_rank.nunique(dropna=True) <= 1:
        d["disagree_quintile"] = 0
    else:
        d["disagree_quintile"] = pd.qcut(d.expert_std_rank.rank(method="average"), 5, labels=False, duplicates="drop").fillna(0).astype(int)

    bands = []
    for keys, g in d.groupby(["pred_decile", "disagree_quintile"], observed=True):
        yy=g.target.to_numpy(); pp=g.blend.to_numpy()
        auc = roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 and len(g)>20 else np.nan
        bands.append({
            "pred_decile": int(keys[0]), "disagree_quintile": int(keys[1]), "n": len(g),
            "prevalence": float(yy.mean()), "mean_pred": float(pp.mean()),
            "auc": float(auc) if np.isfinite(auc) else np.nan,
            "mean_logloss": float(g.logloss.mean()), "mean_abs_error": float(g.abs_error.mean()),
            "mean_disagreement": float(g.expert_std_rank.mean()),
        })
    bands = pd.DataFrame(bands)
    if len(bands):
        bands = bands.sort_values("mean_logloss", ascending=False)
    else:
        bands = pd.DataFrame(columns=["pred_decile","disagree_quintile","n","prevalence","mean_pred","auc","mean_logloss","mean_abs_error","mean_disagreement"])

    consensus = d.expert_std_rank.to_numpy() <= np.quantile(d.expert_std_rank, 0.35)
    hard = d.logloss.to_numpy() >= np.quantile(d.logloss, 0.99)
    hm = consensus & hard
    idx = np.where(hm)[0]
    hard_rows = train_raw.iloc[idx].copy().reset_index().rename(columns={"index": "row_index"})
    hard_rows = pd.concat([hard_rows.reset_index(drop=True), d.iloc[idx].reset_index(drop=True)], axis=1)

    scans=[]
    for c in NUM_COLS:
        allv=train_raw[c]; h=train_raw.loc[hm,c]
        sd=float(allv.std(skipna=True)) or 1.0
        scans.append({
            "feature":c,"type":"numeric","hard_mean":float(h.mean()),"all_mean":float(allv.mean()),
            "std_mean_diff":float((h.mean()-allv.mean())/sd),
            "hard_missing":float(h.isna().mean()),"all_missing":float(allv.isna().mean()),
        })
    for c in CAT_COLS:
        hmiss=float(train_raw.loc[hm,c].isna().mean()); amiss=float(train_raw[c].isna().mean())
        scans.append({"feature":c,"type":"categorical","hard_mean":np.nan,"all_mean":np.nan,
                      "std_mean_diff":np.nan,"hard_missing":hmiss,"all_missing":amiss})
    scan=pd.DataFrame(scans)
    return d, bands, hard_rows, scan
