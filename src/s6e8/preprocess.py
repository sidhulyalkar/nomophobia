from __future__ import annotations
import pandas as pd


def prepare_tree_frames(train: pd.DataFrame, test: pd.DataFrame):
    """Prepare CatBoost-string and native-categorical frames for LGBM/XGBoost."""
    tr_cat, te_cat = train.copy(), test.copy()
    cat_cols = [
        c for c in tr_cat
        if tr_cat[c].dtype == "object"
        or str(tr_cat[c].dtype).startswith("string")
        or str(tr_cat[c].dtype).startswith("category")
    ]

    tr_native, te_native = train.copy(), test.copy()
    for c in cat_cols:
        tr_s = tr_cat[c].astype("string").fillna("MISSING").astype(str)
        te_s = te_cat[c].astype("string").fillna("MISSING").astype(str)
        tr_cat[c], te_cat[c] = tr_s, te_s
        cats = sorted(set(tr_s.unique()).union(te_s.unique()))
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        tr_native[c] = tr_s.astype(dtype)
        te_native[c] = te_s.astype(dtype)

    return tr_cat, te_cat, tr_native, te_native, cat_cols
