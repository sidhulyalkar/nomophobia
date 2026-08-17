from __future__ import annotations
import numpy as np, pandas as pd
from .config import NUM_COLS, CAT_COLS

class EmpiricalBayesEvidence:
    """Naive empirical-Bayes class-conditional evidence model for diversity trials."""
    def __init__(self,n_bins=24,pair_bins=10,alpha=20.0):
        self.n_bins=int(n_bins);self.pair_bins=int(pair_bins);self.alpha=float(alpha);self.num_edges={};self.tables={};self.pair_tables={};self.prior=0.0;self.pairs=[('daily_screen_time_hours','social_media_hours'),('daily_screen_time_hours','weekend_screen_time'),('social_media_hours','weekend_screen_time'),('gaming_hours','work_study_hours')]
    @staticmethod
    def _cat_key(s):return s.astype('string').fillna('__MISSING__').astype(str)
    def _fit_table(self,key,y):
        d=pd.DataFrame({'k':key,'y':np.asarray(y,int)});g=d.groupby('k',dropna=False).y.agg(['sum','count']);pos=g['sum'].astype(float);neg=g['count']-pos;P=float(np.sum(y));N=float(len(y)-P);K=max(len(g),1);a=self.alpha;lr=np.log((pos+a)/(P+a*K))-np.log((neg+a)/(N+a*K));return lr.to_dict()
    def fit(self,X,y):
        X=X.reset_index(drop=True);y=np.asarray(y,int);self.prior=float(np.log((y.mean()+1e-6)/(1-y.mean()+1e-6)))
        for c in NUM_COLS:
            v=pd.to_numeric(X[c],errors='coerce');e=np.unique(np.nanquantile(v,np.linspace(0,1,self.n_bins+1)));e=np.array([-np.inf,np.inf]) if len(e)<3 else e; e[0],e[-1]=-np.inf,np.inf;self.num_edges[c]=e;key=pd.cut(v,e,labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','MISSING');self.tables[c]=self._fit_table(key,y)
        for c in CAT_COLS:self.tables[c]=self._fit_table(self._cat_key(X[c]),y)
        for a,b in self.pairs:
            va=pd.to_numeric(X[a],errors='coerce');vb=pd.to_numeric(X[b],errors='coerce');ea=np.unique(np.nanquantile(va,np.linspace(0,1,self.pair_bins+1)));eb=np.unique(np.nanquantile(vb,np.linspace(0,1,self.pair_bins+1)));ea=np.array([-np.inf,np.inf]) if len(ea)<3 else ea;eb=np.array([-np.inf,np.inf]) if len(eb)<3 else eb;ea[0],ea[-1]=-np.inf,np.inf;eb[0],eb[-1]=-np.inf,np.inf;ka=pd.cut(va,ea,labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','M');kb=pd.cut(vb,eb,labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','M');self.pair_tables[(a,b)]=(ea,eb,self._fit_table(ka+'|'+kb,y))
        return self
    def score(self,X):
        X=X.reset_index(drop=True);s=np.full(len(X),self.prior,dtype=float);n=np.ones(len(X),dtype=float)
        for c in NUM_COLS:
            v=pd.to_numeric(X[c],errors='coerce');key=pd.cut(v,self.num_edges[c],labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','MISSING');s+=key.map(self.tables[c]).fillna(0).to_numpy();n+=1
        for c in CAT_COLS:s+=self._cat_key(X[c]).map(self.tables[c]).fillna(0).to_numpy();n+=1
        for (a,b),(ea,eb,t) in self.pair_tables.items():
            ka=pd.cut(pd.to_numeric(X[a],errors='coerce'),ea,labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','M');kb=pd.cut(pd.to_numeric(X[b],errors='coerce'),eb,labels=False,include_lowest=True).astype('Int64').astype(str).replace('<NA>','M');s+=(ka+'|'+kb).map(t).fillna(0).to_numpy();n+=1
        return s/n
