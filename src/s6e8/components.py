from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import QuantileTransformer
from .config import NUM_COLS, CAT_COLS
from .utils import stable_seed


def _latent_matrix(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    both = pd.concat([train, test], ignore_index=True)
    blocks = []
    num = both[NUM_COLS].copy()
    miss = num.isna().astype(np.float32).to_numpy()
    med = num.median(axis=0)
    num = num.fillna(med)
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(num)), output_distribution="normal",
        subsample=min(200_000, len(num)), random_state=stable_seed(seed, "qt"),
    )
    z = qt.fit_transform(num)
    blocks.extend([z.astype(np.float32), miss])
    cat = both[CAT_COLS].astype("string").fillna("MISSING")
    d = pd.get_dummies(cat, prefix=CAT_COLS, dtype=np.float32)
    blocks.append(d.to_numpy(np.float32))
    A = np.concatenate(blocks, axis=1)
    return A[:len(train)], A[len(train):]


def _entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def add_latent_component_features(train: pd.DataFrame,test: pd.DataFrame,*,method: str="kmeans",n_components: int=8,seed: int=20260816,max_fit_rows: int=120_000):
    X,T=_latent_matrix(train,test,seed); A=np.vstack([X,T]); rng=np.random.default_rng(stable_seed(seed,method,n_components,"fit")); fit_idx=np.arange(len(A))
    if len(fit_idx)>max_fit_rows: fit_idx=np.sort(rng.choice(fit_idx,max_fit_rows,replace=False))
    fit=A[fit_idx]; prefix=f"latent_{method}{n_components}"
    if method=="kmeans":
        model=MiniBatchKMeans(n_clusters=n_components,batch_size=4096,n_init=5,random_state=stable_seed(seed,"kmeans",n_components)).fit(fit)
        dist=model.transform(A); tau=float(np.median(np.min(dist,axis=1))+1e-6); logits=-dist/tau; logits-=logits.max(axis=1,keepdims=True); p=np.exp(logits); p/=p.sum(axis=1,keepdims=True); loglik=-np.min(dist,axis=1)
    elif method=="bgmm":
        model=BayesianGaussianMixture(n_components=n_components,covariance_type="diag",max_iter=180,reg_covar=1e-5,weight_concentration_prior_type="dirichlet_process",random_state=stable_seed(seed,"bgmm",n_components)).fit(fit)
        p=model.predict_proba(A); loglik=model.score_samples(A)
    else: raise ValueError("method must be 'kmeans' or 'bgmm'")
    arg=p.argmax(axis=1); ent=_entropy(p); feats={f"{prefix}__p{i:02d}":p[:,i].astype(np.float32) for i in range(n_components)}; feats[f"{prefix}__argmax"]=pd.Series(arg).astype(str).to_numpy(); feats[f"{prefix}__entropy"]=ent.astype(np.float32); feats[f"{prefix}__loglik"]=np.asarray(loglik,np.float32); feats[f"{prefix}__maxprob"]=p.max(axis=1).astype(np.float32); Z=pd.DataFrame(feats)
    return Z.iloc[:len(train)].reset_index(drop=True),Z.iloc[len(train):].reset_index(drop=True),{"method":method,"n_components":int(n_components),"fit_rows":int(len(fit_idx)),"component_counts":np.bincount(arg,minlength=n_components).astype(int).tolist(),"mean_entropy":float(ent.mean())}
