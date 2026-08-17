from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def _rank01(x): return rankdata(x,method='average')/len(x)
def _fold_rank01(x,folds):
    out=np.empty(len(x),dtype=float); folds=np.asarray(folds); x=np.asarray(x)
    for f in np.unique(folds):
        m=folds==f; out[m]=rankdata(x[m],method='average')/m.sum()
    return out

def _matrix_for_mode(O,T,mode,folds=None):
    if mode=='probability': return O,T
    if mode=='rank': return np.column_stack([_rank01(O[:,i]) for i in range(O.shape[1])]),np.column_stack([_rank01(T[:,i]) for i in range(T.shape[1])])
    if mode=='fold_rank':
        if folds is None: raise ValueError('fold_rank requires folds')
        return np.column_stack([_fold_rank01(O[:,i],folds) for i in range(O.shape[1])]),np.column_stack([_rank01(T[:,i]) for i in range(T.shape[1])])
    raise ValueError(mode)
def _candidate_weights(n_models,rng,trials):
    out=[np.eye(n_models)[i] for i in range(n_models)]; out.append(np.ones(n_models)/n_models)
    if n_models==2: out.extend(np.array([a,1-a]) for a in np.linspace(0,1,101))
    else:
        for alpha in (.7,1.5,3.0): out.extend(rng.dirichlet(np.full(n_models,alpha),size=max(40,trials//3)))
    return out
def _stratified_subset(y,idx,n,rng):
    idx=np.asarray(idx); y=np.asarray(y)
    if len(idx)<=n:return idx
    p=idx[y[idx]==1]; q=idx[y[idx]==0]; npick=int(n*len(p)/len(idx)); nnick=n-npick; return np.sort(np.r_[rng.choice(p,npick,replace=False),rng.choice(q,nnick,replace=False)])
def _select_weight(y,M,candidates,idx):
    best=(-1,None)
    for w in candidates:
        s=float(roc_auc_score(y[idx],M[idx]@w))
        if s>best[0]:best=(s,w)
    return best
def _fold_scores(y,p,folds): return [float(roc_auc_score(np.asarray(y)[np.asarray(folds)==f],np.asarray(p)[np.asarray(folds)==f])) for f in np.unique(folds)]

def search_blend(y,oof_dict,test_dict,trials=800,seed=20260816,calibration_rows=90_000,folds=None):
    names=list(oof_dict); O=np.column_stack([oof_dict[n] for n in names]); T=np.column_stack([test_dict[n] for n in names]); y=np.asarray(y); folds=np.asarray(folds) if folds is not None else None
    if O.shape[1]==1:
        p=O[:,0]; fs=_fold_scores(y,p,folds) if folds is not None else []; return {"auc":float(roc_auc_score(y,p)),"selection_auc":float(roc_auc_score(y,p)),"mode":"probability","weights":np.array([1.]),"oof":p,"test":T[:,0],"names":names,"fold_auc":fs,"fold_auc_mean":float(np.mean(fs)) if fs else None,"fold_auc_std":float(np.std(fs)) if fs else None,"rotation_weights":[]}
    rng=np.random.default_rng(seed); candidates=_candidate_weights(O.shape[1],rng,trials); modes=['probability','rank']+(['fold_rank'] if folds is not None else []); best=None
    for mode in modes:
        M,TT=_matrix_for_mode(O,T,mode,folds); idx_all=_stratified_subset(y,np.arange(len(y)),calibration_rows,rng); top=[]
        for w in candidates: top.append((float(roc_auc_score(y[idx_all],M[idx_all]@w)),w))
        top.sort(key=lambda z:z[0],reverse=True); selection_auc=-1; selection_w=None
        for _,w in top[:40]:
            s=float(roc_auc_score(y,M@w))
            if s>selection_auc:selection_auc=s;selection_w=np.asarray(w)
        if folds is None: p=M@selection_w; fs=[]; deploy=selection_w; rotation=[]
        else:
            p=np.zeros(len(y)); rotation=[]
            for held in np.unique(folds):
                tr=np.where(folds!=held)[0]; va=np.where(folds==held)[0]; sel=_stratified_subset(y,tr,calibration_rows,rng); _,w=_select_weight(y,M,candidates,sel); w=np.asarray(w); p[va]=M[va]@w; rotation.append(w)
            deploy=np.mean(rotation,axis=0); fs=_fold_scores(y,p,folds)
        auc=float(roc_auc_score(y,p)); std=float(np.std(fs)) if fs else 0.0; objective=auc-.04*std
        if best is None or objective>best['objective']: best={"objective":objective,"auc":auc,"selection_auc":float(selection_auc),"selection_optimism":float(selection_auc-auc),"mode":mode,"weights":deploy,"oof":p,"test":TT@deploy,"names":names,"fold_auc":fs,"fold_auc_mean":float(np.mean(fs)) if fs else None,"fold_auc_std":std if fs else None,"rotation_weights":[w.tolist() for w in rotation] if folds is not None else []}
    return best

def model_correlation(oof_dict,folds=None):
    names=list(oof_dict); M=np.column_stack([_fold_rank01(oof_dict[n],folds) if folds is not None else _rank01(oof_dict[n]) for n in names]); return names,np.corrcoef(M,rowvar=False)
def save_blend(best,out_dir):
    out_dir=Path(out_dir); np.save(out_dir/'oof_blend.npy',best['oof']); np.save(out_dir/'test_blend.npy',best['test']); meta={k:v for k,v in best.items() if k not in ('oof','test','weights')}; meta['weights']={n:float(w) for n,w in zip(best['names'],best['weights'])}; (out_dir/'blend.json').write_text(json.dumps(meta,indent=2))
def _pairwise_selection_pairs(y,idx,n_pairs,rng):
    idx=np.asarray(idx); y=np.asarray(y); pos=idx[y[idx]==1]; neg=idx[y[idx]==0]
    if not len(pos) or not len(neg):raise ValueError('Both classes required for Caruana selection')
    return rng.choice(pos,n_pairs,replace=True),rng.choice(neg,n_pairs,replace=True)
def _greedy_pairwise_caruana(M,y,train_idx,candidate_pool,ensemble_size,n_pairs,rng):
    pi,ni=_pairwise_selection_pairs(y,train_idx,n_pairs,rng); pool=np.asarray(candidate_pool,dtype=int); D=M[pi][:,pool]-M[ni][:,pool]; cur=np.zeros(n_pairs,dtype=np.float32); counts=np.zeros(M.shape[1],dtype=float)
    for _ in range(int(ensemble_size)):
        Z=cur[:,None]+D; scores=(Z>0).mean(axis=0)+0.5*(Z==0).mean(axis=0); j=int(np.argmax(scores)); mid=int(pool[j]); counts[mid]+=1.0; cur+=D[:,j]
    return counts/max(counts.sum(),1.0)
def bagged_caruana_blend(y,oof_dict,test_dict,folds,*,seed=20260816,n_bags=20,ensemble_size=30,n_pairs=10_000,mode='fold_rank'):
    names=list(oof_dict); O=np.column_stack([oof_dict[n] for n in names]); T=np.column_stack([test_dict[n] for n in names]); y=np.asarray(y); folds=np.asarray(folds); M,TT=_matrix_for_mode(O,T,mode,folds)
    if M.shape[1]==1:
        p=M[:,0]; fs=_fold_scores(y,p,folds); return {"auc":float(roc_auc_score(y,p)),"mode":mode,"weights":np.array([1.]),"oof":p,"test":TT[:,0],"names":names,"fold_auc":fs,"fold_auc_std":float(np.std(fs)),"rotation_weights":[],"method":"bagged_caruana"}
    rng=np.random.default_rng(seed); helds=np.unique(folds); oof=np.zeros(len(y)); all_w=[]; rot=[]; bag_counts=np.full(len(helds),n_bags//len(helds),dtype=int); bag_counts[:n_bags%len(helds)]+=1
    for h,nb in zip(helds,bag_counts):
        tr=np.where(folds!=h)[0]; va=np.where(folds==h)[0]; ws=[]
        for _ in range(max(int(nb),1)):
            pool=rng.integers(0,M.shape[1],size=M.shape[1]); w=_greedy_pairwise_caruana(M,y,tr,pool,ensemble_size,n_pairs,rng); ws.append(w); all_w.append(w)
        wh=np.mean(ws,axis=0); oof[va]=M[va]@wh; rot.append(wh)
    deploy=np.mean(all_w,axis=0); test=TT@deploy; fs=_fold_scores(y,oof,folds); return {"auc":float(roc_auc_score(y,oof)),"selection_auc":None,"selection_optimism":None,"mode":mode,"weights":deploy,"oof":oof,"test":test,"names":names,"fold_auc":fs,"fold_auc_mean":float(np.mean(fs)),"fold_auc_std":float(np.std(fs)),"rotation_weights":[w.tolist() for w in rot],"method":"bagged_caruana","n_bags":int(n_bags),"ensemble_size":int(ensemble_size),"n_pairs":int(n_pairs)}
