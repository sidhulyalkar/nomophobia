from __future__ import annotations
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
from scipy.stats import rankdata

class TokenTransformer(nn.Module):
    def __init__(self,n_features,d_model=32,nhead=4,layers=2,dropout=.1):
        super().__init__();self.feature_emb=nn.Embedding(n_features,d_model);self.value_w=nn.Parameter(torch.randn(n_features,d_model)*.02);self.value_b=nn.Parameter(torch.zeros(n_features,d_model));self.missing_emb=nn.Parameter(torch.randn(n_features,d_model)*.02)
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=d_model*3,dropout=dropout,batch_first=True,norm_first=True,activation='gelu');self.encoder=nn.TransformerEncoder(enc,layers);self.norm=nn.LayerNorm(d_model);self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,1))
    def forward(self,x,missing):
        _,f=x.shape;ids=torch.arange(f,device=x.device);tok=self.feature_emb(ids)[None,:,:]+x[:,:,None]*self.value_w[None,:,:]+self.value_b[None,:,:];tok=tok+missing[:,:,None]*self.missing_emb[None,:,:];z=self.encoder(tok).mean(1);return self.head(self.norm(z)).squeeze(1)

def _matrix(train,test):
    allx=pd.concat([train,test],ignore_index=True).copy()
    for c in allx.columns:
        if allx[c].dtype=='object' or str(allx[c].dtype).startswith(('string','category')):
            vals=allx[c].astype('string').fillna('MISSING');cats=sorted(vals.unique());mp={v:i for i,v in enumerate(cats)};allx[c]=vals.map(mp).astype(float)
    A=allx.to_numpy(dtype=np.float32);miss=np.isnan(A).astype(np.float32);med=np.nanmedian(A,axis=0);med=np.where(np.isfinite(med),med,0);A=np.where(np.isnan(A),med,A);n=len(train);tr=A[:n];te=A[n:];mtr=miss[:n];mte=miss[n:];mu=tr.mean(0);sd=tr.std(0);sd=np.where(sd<1e-6,1,sd);return np.clip((tr-mu)/sd,-8,8),mtr,np.clip((te-mu)/sd,-8,8),mte

def _fit_epochs(model,opt,lossfn,X,M,y,idx,epochs,batch_size,device):
    for _ in range(epochs):
        model.train();order=np.random.permutation(idx)
        for st in range(0,len(order),batch_size):
            ix=order[st:st+batch_size];xb=torch.from_numpy(X[ix]).to(device);mb=torch.from_numpy(M[ix]).to(device);yb=torch.from_numpy(y[ix]).to(device);opt.zero_grad(set_to_none=True);loss=lossfn(model(xb,mb),yb);loss.backward();opt.step()

def _predict(model,X,M,idx,batch_size,device):
    model.eval();out=[]
    with torch.no_grad():
        for st in range(0,len(idx),batch_size):
            ix=idx[st:st+batch_size];out.append(torch.sigmoid(model(torch.from_numpy(X[ix]).to(device),torch.from_numpy(M[ix]).to(device))).cpu().numpy())
    return np.concatenate(out)

def run_transformer_cv(train,test,y,n_splits=3,seed=20260816,epochs=6,batch_size=2048):
    """Outer-fold-clean transformer; checkpoint epoch is chosen on an inner split (A9)."""
    torch.manual_seed(seed);np.random.seed(seed);X,M,T,TM=_matrix(train,test);y=np.asarray(y,dtype=np.float32);oof=np.zeros(len(y));tps=[];skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for f,(ti,vi) in enumerate(skf.split(X,y)):
        fit_idx,es_idx=train_test_split(ti,test_size=.10,stratify=y[ti],random_state=seed+f)
        probe=TokenTransformer(X.shape[1]).to(device);opt=torch.optim.AdamW(probe.parameters(),lr=2e-3,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss();best_auc=-1;best_ep=1
        for ep in range(1,epochs+1):
            _fit_epochs(probe,opt,lossfn,X,M,y,fit_idx,1,batch_size,device);p=_predict(probe,X,M,es_idx,batch_size,device);auc=roc_auc_score(y[es_idx],p)
            if auc>best_auc:best_auc=auc;best_ep=ep
        torch.manual_seed(seed+1000+f);model=TokenTransformer(X.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);_fit_epochs(model,opt,lossfn,X,M,y,ti,best_ep,batch_size,device)
        pv=_predict(model,X,M,vi,batch_size,device);pt=_predict(model,T,TM,np.arange(len(T)),batch_size,device);oof[vi]=rankdata(pv,method='average')/len(pv);tps.append(rankdata(pt,method='average')/len(pt));print(f'transformer fold {f}: {roc_auc_score(y[vi],pv):.7f} (inner_best_epoch={best_ep})',flush=True)
    return oof,np.mean(tps,axis=0),float(roc_auc_score(y,oof))
