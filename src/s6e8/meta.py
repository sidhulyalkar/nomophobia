from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from .features import build_feature_views
from .preprocess import prepare_tree_frames


def _rank01(x):
    return rankdata(np.asarray(x), method='average') / len(x)


def _fold_rank01(x, folds):
    x=np.asarray(x); folds=np.asarray(folds); out=np.empty(len(x),float)
    for f in np.unique(folds):
        m=folds==f; out[m]=rankdata(x[m],method='average')/m.sum()
    return out


def build_meta_features(train_raw, test_raw, oof_dict, test_dict, folds,
                        context_view='combined', use_frequency=False,
                        frequency_reference=None, rank_only=True):
    """Leak-resistant level-2 representation.

    Default v3 contract is *rank-only* base information (audit A3): OOF ranks are
    normalized within fold, test ranks globally. Probability/logit columns are omitted
    because single-fold OOF and 5-model-averaged test predictions have different scale.
    """
    views=build_feature_views(
        train_raw,test_raw,use_frequency=use_frequency,
        frequency_reference=frequency_reference,
    )
    X,T=views[context_view]
    _,_,Xe,Te,_=prepare_tree_frames(X,T)
    Xe=Xe.reset_index(drop=True);Te=Te.reset_index(drop=True)
    names=list(oof_dict)
    probs=np.column_stack([np.asarray(oof_dict[n]) for n in names])
    tprobs=np.column_stack([np.asarray(test_dict[n]) for n in names])
    ranks=np.column_stack([_fold_rank01(oof_dict[n],folds) for n in names])
    tranks=np.column_stack([_rank01(test_dict[n]) for n in names])
    for j,n in enumerate(names):
        Xe[f'base_rank__{n}']=ranks[:,j];Te[f'base_rank__{n}']=tranks[:,j]
        if not rank_only:
            Xe[f'base_prob__{n}']=probs[:,j];Te[f'base_prob__{n}']=tprobs[:,j]
            Xe[f'base_logit__{n}']=np.log(np.clip(probs[:,j],1e-6,1-1e-6)/np.clip(1-probs[:,j],1e-6,1))
            Te[f'base_logit__{n}']=np.log(np.clip(tprobs[:,j],1e-6,1-1e-6)/np.clip(1-tprobs[:,j],1e-6,1))

    A,B=ranks,tranks
    Xe['base_rank_mean']=A.mean(1);Te['base_rank_mean']=B.mean(1)
    Xe['base_rank_std']=A.std(1);Te['base_rank_std']=B.std(1)
    Xe['base_rank_spread']=A.max(1)-A.min(1);Te['base_rank_spread']=B.max(1)-B.min(1)
    Xe['base_rank_min']=A.min(1);Te['base_rank_min']=B.min(1)
    Xe['base_rank_max']=A.max(1);Te['base_rank_max']=B.max(1)
    return Xe,Te,names


def make_meta_model(seed=20260816,n_estimators=500):
    return LGBMClassifier(
        objective='binary', n_estimators=n_estimators, learning_rate=.025,
        num_leaves=31, min_child_samples=150, subsample=.90, colsample_bytree=.80,
        reg_alpha=.30, reg_lambda=3.0, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def crossfit_meta(X, y, T, folds, seed=20260816, n_estimators=500):
    """Fixed-iteration cross-fit; outer fold is never used for model selection."""
    y=pd.Series(np.asarray(y)).reset_index(drop=True); folds=np.asarray(folds)
    oof=np.zeros(len(y)); tests=[]; scores=[]
    for f in np.unique(folds):
        a=np.where(folds!=f)[0];b=np.where(folds==f)[0]
        m=make_meta_model(seed+int(f)*131,n_estimators)
        m.fit(X.iloc[a],y.iloc[a],categorical_feature=[c for c in X.columns if str(X[c].dtype)=='category'])
        oof[b]=m.predict_proba(X.iloc[b])[:,1];tests.append(m.predict_proba(T)[:,1])
        scores.append(float(roc_auc_score(y.iloc[b],oof[b])))
    return oof,np.mean(tests,axis=0),{'fold_auc':scores,'oof_auc':float(roc_auc_score(y,oof)),'fixed_iterations':n_estimators}


def fit_full_meta(X,y,T,seed=20260816,n_estimators=500):
    m=make_meta_model(seed,n_estimators)
    m.fit(X,np.asarray(y),categorical_feature=[c for c in X.columns if str(X[c].dtype)=='category'])
    return m.predict_proba(T)[:,1],m
