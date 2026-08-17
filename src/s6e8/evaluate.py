from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1); i=j
    T2=np.empty(N,float); T2[J]=T+1; return T2


def _fast_delong(preds_sorted_transposed,label_1_count):
    m=label_1_count; n=preds_sorted_transposed.shape[1]-m; k=preds_sorted_transposed.shape[0]; positive=preds_sorted_transposed[:,:m]; negative=preds_sorted_transposed[:,m:]
    tx=np.empty((k,m)); ty=np.empty((k,n)); tz=np.empty((k,m+n))
    for r in range(k): tx[r]=_compute_midrank(positive[r]); ty[r]=_compute_midrank(negative[r]); tz[r]=_compute_midrank(preds_sorted_transposed[r])
    aucs=tz[:,:m].sum(axis=1)/(m*n)-(m+1.0)/(2.0*n); v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m; sx=np.cov(v01); sy=np.cov(v10)
    if k==1: sx=np.array([[sx]]); sy=np.array([[sy]])
    return aucs,sx/m+sy/n


def delong_test(y_true,pred_a,pred_b):
    y=np.asarray(y_true,dtype=int); a=np.asarray(pred_a,float); b=np.asarray(pred_b,float); order=np.argsort(-y); m=int(y.sum()); aucs,cov=_fast_delong(np.vstack([a,b])[:,order],m); l=np.array([[1.,-1.]]); var=float((l@cov@l.T).item())
    if var<=0:return 1.0
    z=abs(float(aucs[0]-aucs[1]))/math.sqrt(var); return float(2*norm.sf(z))


def _prep_weighted_auc(y,score):
    order=np.argsort(np.asarray(score),kind='mergesort'); s=np.asarray(score)[order]; yy=np.asarray(y,dtype=np.int8)[order]; gid=np.cumsum(np.r_[0,(s[1:]!=s[:-1]).astype(np.int32)]); return order,yy,gid,int(gid[-1]+1)


def _weighted_auc_from_counts(prep,w):
    order,yy,gid,ng=prep; sw=np.asarray(w,float)[order]; neg=np.bincount(gid,weights=sw*(yy==0),minlength=ng); pos=np.bincount(gid,weights=sw*(yy==1),minlength=ng); before=np.cumsum(neg)-neg; den=pos.sum()*neg.sum()
    if den<=0:return np.nan
    return float(np.sum(pos*(before+0.5*neg))/den)


def _exact_bootstrap(y,a,b,n_boot,rng):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; out=np.empty(n_boot,float)
    for i in range(n_boot):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]; out[i]=roc_auc_score(y[idx],b[idx])-roc_auc_score(y[idx],a[idx])
    return out,{"bootstrap_population":"full","bootstrap_rows":int(len(y)),"bootstrap_bags":1}


def _blb_bootstrap(y,a,b,n_boot,rng,rows=20_000,bags=4):
    y=np.asarray(y,dtype=np.int8); a=np.asarray(a,float); b=np.asarray(b,float); P=np.where(y==1)[0]; N=np.where(y==0)[0]; rows=min(int(rows),len(y)); bp=max(2,int(round(rows*len(P)/len(y)))); bn=max(2,rows-bp); bags=max(1,int(bags)); per=np.full(bags,n_boot//bags,dtype=int); per[:n_boot%bags]+=1; outs=[]
    for nb in per:
        sp=rng.choice(P,min(bp,len(P)),replace=False); sn=rng.choice(N,min(bn,len(N)),replace=False); idx=np.r_[sp,sn]; yy=y[idx]; aa=a[idx]; bb=b[idx]; pa=_prep_weighted_auc(yy,aa); pb=_prep_weighted_auc(yy,bb); lp=np.where(yy==1)[0]; ln=np.where(yy==0)[0]; pp=np.full(len(lp),1/len(lp)); pn=np.full(len(ln),1/len(ln))
        for _ in range(int(nb)):
            w=np.zeros(len(idx),dtype=np.int32); w[lp]=rng.multinomial(len(P),pp); w[ln]=rng.multinomial(len(N),pn); outs.append(_weighted_auc_from_counts(pb,w)-_weighted_auc_from_counts(pa,w))
    return np.asarray(outs,float),{"bootstrap_population":"BLB_stratified_subsamples","bootstrap_rows":int(rows),"bootstrap_bags":int(bags)}


def paired_compare(y,pred_a,pred_b,folds,n_boot=2000,seed=0,bootstrap_exact_max_rows=50_000,blb_rows=20_000,blb_bags=4)->dict:
    y=np.asarray(y,dtype=np.int8); a=np.asarray(pred_a,float); b=np.asarray(pred_b,float); f=np.asarray(folds); good=np.isfinite(a)&np.isfinite(b)&np.isfinite(y); y,a,b,f=y[good],a[good],b[good],f[good]; auc_a=float(roc_auc_score(y,a)); auc_b=float(roc_auc_score(y,b)); delta=auc_b-auc_a; per=[]
    for k in np.unique(f):
        m=f==k
        if len(np.unique(y[m]))==2: per.append(float(roc_auc_score(y[m],b[m])-roc_auc_score(y[m],a[m])))
    rng=np.random.default_rng(seed)
    if n_boot<=0: boots=np.array([delta]); bmeta={"bootstrap_population":"disabled","bootstrap_rows":0,"bootstrap_bags":0}
    elif len(y)<=bootstrap_exact_max_rows: boots,bmeta=_exact_bootstrap(y,a,b,n_boot,rng)
    else: boots,bmeta=_blb_bootstrap(y,a,b,n_boot,rng,rows=blb_rows,bags=blb_bags)
    ci=np.quantile(boots,[.025,.975]); return {"auc_a":auc_a,"auc_b":auc_b,"delta_auc":float(delta),"delta_ci_95":[float(ci[0]),float(ci[1])],"delta_per_fold":per,"folds_positive":int(sum(x>0 for x in per)),"delong_p":delong_test(y,a,b),"n_effective":int(len(y)),"n_boot":int(n_boot),"seed":int(seed),**bmeta}
