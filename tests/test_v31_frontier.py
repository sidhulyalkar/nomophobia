import numpy as np
import pandas as pd
from s6e8.utils import stable_seed
from s6e8.blend import bagged_caruana_blend
from s6e8.components import add_latent_component_features
from s6e8.models import focal_binary_objective


def _raw(n=80,seed=1):
    r=np.random.default_rng(seed)
    return pd.DataFrame({'id':np.arange(n),'age':r.integers(18,36,n).astype(float),'daily_screen_time_hours':r.normal(7,2,n),'social_media_hours':r.normal(3,1,n),'gaming_hours':r.normal(2,1,n),'work_study_hours':r.normal(3,1,n),'sleep_hours':r.normal(7,1,n),'notifications_per_day':r.integers(5,250,n).astype(float),'app_opens_per_day':r.integers(5,180,n).astype(float),'weekend_screen_time':r.normal(9,2,n),'gender':r.choice(['Male','Female','Other'],n),'stress_level':r.choice(['Low','Medium','High'],n),'academic_work_impact':r.choice(['Yes','No'],n)})


def test_stable_seed_is_repeatable():
    assert stable_seed(1,'abc',2)==stable_seed(1,'abc',2);assert stable_seed(1,'abc',2)!=stable_seed(1,'abc',3)


def test_bagged_caruana_shapes_and_prefers_signal():
    r=np.random.default_rng(3);n=600;y=r.integers(0,2,n);folds=np.arange(n)%5;good=y+r.normal(0,.8,n);weak=r.normal(size=n);bad=-good+r.normal(0,.3,n);o={'good':good,'weak':weak,'bad':bad};t={k:r.normal(size=100) for k in o};z=bagged_caruana_blend(y,o,t,folds,seed=7,n_bags=10,ensemble_size=8,n_pairs=1000);assert len(z['oof'])==n and len(z['test'])==100;assert np.isclose(z['weights'].sum(),1);assert z['weights'][0]>z['weights'][2]


def test_latent_component_features_are_transductive_and_target_free():
    tr=_raw(60);te=_raw(20,2);tr.loc[:5,'daily_screen_time_hours']=np.nan;a,b,m=add_latent_component_features(tr,te,method='kmeans',n_components=4,max_fit_rows=80,seed=4);assert a.shape[0]==60 and b.shape[0]==20;assert 'latent_kmeans4__p00' in a and 'latent_kmeans4__argmax' in a;assert np.allclose(a.filter(like='__p').sum(axis=1),1,atol=1e-5);assert m['n_components']==4


def test_focal_objective_finite_positive_curvature():
    fn=focal_binary_objective(alpha=.35,gamma=1.5);y=np.array([0,1,0,1],float);raw=np.array([-2,-1,1,2],float);g,h=fn(y,raw);assert np.isfinite(g).all() and np.isfinite(h).all() and (h>0).all()


def test_paired_compare_blb_identity():
    from s6e8.evaluate import paired_compare
    y=np.array([0,1]*40);p=np.linspace(0,1,len(y));f=np.arange(len(y))%5;z=paired_compare(y,p,p,f,n_boot=20,seed=2,bootstrap_exact_max_rows=10,blb_rows=30,blb_bags=2);assert z['bootstrap_population']=='BLB_stratified_subsamples';assert z['delta_ci_95']==[0.0,0.0]
