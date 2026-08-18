#!/usr/bin/env python
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from live_frontier_candidate import TARGET, ID, CAT, fold_te, feature_frame

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out-dir',default='s20_results'); ap.add_argument('--estimators',type=int,default=4500); a=ap.parse_args()
    p=Path(a.data_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(p/'train.csv'); test=pd.read_csv(p/'test.csv'); sample=pd.read_csv(p/'sample_submission.csv')
    if len(train)!=691369 or len(test)!=296302 or not test[ID].equals(sample[ID]): raise ValueError('competition contract mismatch')
    for c in CAT:
        cats=pd.Index(pd.concat([train[c],test[c]],ignore_index=True).dropna().unique())
        train[c]=pd.Categorical(train[c],categories=cats); test[c]=pd.Categorical(test[c],categories=cats)
    y=train[TARGET].to_numpy(np.int8); cv=StratifiedKFold(5,shuffle=True,random_state=20260801)
    oof=np.zeros(len(train)); tp=np.zeros(len(test)); fold=np.zeros(len(train),np.int8); metrics=[]; start=time.time()
    for f,(ti,vi) in enumerate(cv.split(train,y)):
        tr=train.iloc[ti].reset_index(drop=True); va=train.iloc[vi].reset_index(drop=True); yt=y[ti]; yv=y[vi]
        etr,eva,ete=fold_te(tr,yt,va,test.reset_index(drop=True),5,20.0,20260801+f)
        xt=feature_frame(tr,etr,False); xv=feature_frame(va,eva,False); xte=feature_frame(test.reset_index(drop=True),ete,False)
        m=lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=a.estimators,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=100,subsample=.9,subsample_freq=1,colsample_bytree=.9,reg_alpha=.1,reg_lambda=2,max_bin=255,random_state=20260801+f,n_jobs=-1,verbosity=-1)
        t=time.time(); m.fit(xt,yt,eval_set=[(xv,yv)],eval_metric='auc',categorical_feature=CAT,callbacks=[lgb.early_stopping(150,verbose=False)])
        pv=m.predict_proba(xv,num_iteration=m.best_iteration_)[:,1]; pt=m.predict_proba(xte,num_iteration=m.best_iteration_)[:,1]
        oof[vi]=pv; tp+=pt/5; fold[vi]=f
        row={'fold':f,'auc':float(roc_auc_score(yv,pv)),'best_iteration':int(m.best_iteration_),'seconds':round(time.time()-t,2)}; metrics.append(row); print(json.dumps(row),flush=True)
    auc=float(roc_auc_score(y,oof)); np.save(out/'oof_s20.npy',oof); np.save(out/'test_s20.npy',tp); np.save(out/'fold.npy',fold); np.save(out/'target.npy',y)
    pd.DataFrame({ID:test[ID],TARGET:tp}).to_csv(out/'submission_s20.csv',index=False)
    pd.DataFrame({ID:train[ID],TARGET:y,'fold':fold,'prediction':oof}).to_csv(out/'oof_s20.csv',index=False)
    dec={'rows':{'train':len(train),'test':len(test)},'folds':5,'inner_folds':5,'smoothing':20.0,'auc':auc,'fold_metrics':metrics,'elapsed_seconds':round(time.time()-start,2)}
    (out/'decision.json').write_text(json.dumps(dec,indent=2)+'\n'); print(json.dumps(dec,indent=2),flush=True)
if __name__=='__main__': main()
