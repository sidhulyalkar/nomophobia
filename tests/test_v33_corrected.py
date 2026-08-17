import numpy as np,pandas as pd
from s6e8.evidence import EmpiricalBayesEvidence
from s6e8.features import add_frequency_features


def _raw(n=30):
 r=np.random.default_rng(5);return pd.DataFrame({'age':r.integers(18,40,n).astype(float),'daily_screen_time_hours':r.normal(7,2,n),'social_media_hours':r.normal(3,1,n),'gaming_hours':r.normal(2,1,n),'work_study_hours':r.normal(3,1,n),'sleep_hours':r.normal(7,1,n),'notifications_per_day':r.integers(1,200,n).astype(float),'app_opens_per_day':r.integers(1,150,n).astype(float),'weekend_screen_time':r.normal(9,2,n),'gender':r.choice(['Male','Female'],n),'stress_level':r.choice(['Low','Medium','High'],n),'academic_work_impact':r.choice(['Yes','No'],n)})


def test_evidence_expert_scores_finite():
 x=_raw(80);y=np.array([0,1]*40);m=EmpiricalBayesEvidence(n_bins=8,pair_bins=5).fit(x,y);s=m.score(x.iloc[:20]);assert len(s)==20 and np.isfinite(s).all()


def test_train_only_frequency_reference_allows_unseen_test_values():
 tr=_raw(10);te=_raw(3);te.loc[0,'age']=999.0;empty=te.iloc[:0];a,b=add_frequency_features(tr,te,reference_train=tr,reference_test=empty);assert np.isnan(float(b.loc[0,'age__freq']));assert float(a.loc[0,'age__freq'])>=1
