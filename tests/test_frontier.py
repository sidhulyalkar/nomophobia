import numpy as np
import pandas as pd
from s6e8.frontier import missing_pattern_code,missing_shift_table,residual_forensics


def _df(n=20):
    rng=np.random.default_rng(1)
    return pd.DataFrame({'age':rng.normal(25,4,n),'daily_screen_time_hours':rng.normal(7,2,n),'social_media_hours':rng.normal(3,1,n),'gaming_hours':rng.normal(2,1,n),'work_study_hours':rng.normal(3,1,n),'sleep_hours':rng.normal(7,1,n),'notifications_per_day':rng.integers(1,200,n),'app_opens_per_day':rng.integers(1,150,n),'weekend_screen_time':rng.normal(9,2,n),'gender':['Male']*n,'stress_level':['Medium']*n,'academic_work_impact':['Yes']*n})


def test_missing_pattern_and_shift():
    a=_df();b=_df();a.loc[0,'age']=np.nan;b.loc[:2,'age']=np.nan;assert missing_pattern_code(a)[0]!=missing_pattern_code(a)[1];z=missing_shift_table(a,b);assert float(z.loc[z.feature=='age','delta'].iloc[0])>0


def test_residual_forensics_shapes():
    x=_df(100);y=np.array([0,1]*50);p1=np.linspace(.05,.95,100);p2=np.clip(p1+.02*np.sin(np.arange(100)),0,1);d,b,h,s=residual_forensics(x,y,{'a':p1,'b':p2},(p1+p2)/2);assert len(d)==100 and len(b)>0 and 'expert_std_rank' in d and 'feature' in s


def test_library_dedup_synthetic():
    from s6e8.library import deduplicate_library
    y=np.array([0,1]*50);a=np.linspace(0,1,100);b=a.copy();c=np.sin(np.arange(100)/7)+a;o={'a':a,'b':b,'c':c};t={k:v[:20] for k,v in o.items()};oo,tt,_=deduplicate_library(y,o,t,corr_threshold=.9999,max_models=10);assert len(oo)==2 and set(oo)==set(tt)


def test_meta_feature_builder():
    from s6e8.meta import build_meta_features
    x=_df(30);t=_df(10);folds=np.array([0,1,2]*10);o={'a':np.linspace(.1,.9,30),'b':np.linspace(.12,.88,30)};q={'a':np.linspace(.1,.9,10),'b':np.linspace(.2,.8,10)};X,T,n=build_meta_features(x,t,o,q,folds,context_view='raw',use_frequency=False);assert len(X)==30 and len(T)==10 and n==['a','b'];assert 'base_rank_std' in X and 'base_rank__a' in X and 'base_prob__a' not in X


def test_residual_forensics_single_expert_does_not_crash():
    x=_df(100);y=np.array([0,1]*50);p=np.linspace(.05,.95,100);d,b,h,s=residual_forensics(x,y,{'only':p},p);assert len(d)==100 and 'mean_logloss' in b.columns
