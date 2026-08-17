import numpy as np
import pandas as pd
from s6e8.features import build_feature_views


def toy():
    tr=pd.DataFrame({'id':[1,2], 'age':[20.,np.nan], 'daily_screen_time_hours':[8.12,5.0],'social_media_hours':[3.21,1.0], 'gaming_hours':[2.0,np.nan],'work_study_hours':[2.5,4.0], 'sleep_hours':[7.,8.],'notifications_per_day':[100.,80.], 'app_opens_per_day':[50.,40.],'weekend_screen_time':[10.4,6.0], 'gender':['Female',None],'stress_level':['High','Low'], 'academic_work_impact':['Yes','No']})
    te=tr.copy();te['id']=[3,4];return tr,te


def test_views_are_distinct_and_target_free():
    tr,te=toy();views=build_feature_views(tr,te)
    assert set(views)=={'raw','semantic','generator','combined'}
    assert views['raw'][0].shape[1]==12
    assert views['semantic'][0].shape[1]<views['combined'][0].shape[1]
    assert views['generator'][0].shape[1]<views['combined'][0].shape[1]
    assert 'unaccounted_screen_hours' in views['semantic'][0]
    assert 'weekend_screen_time__hundredths_digit' in views['generator'][0]
    assert 'missing_pattern' in views['combined'][0]


def test_unaccounted_screen_math():
    tr,te=toy();X,_=build_feature_views(tr,te)['semantic'];expected=8.12-(3.21+2.0+2.5);assert abs(X.loc[0,'unaccounted_screen_hours']-expected)<1e-9


def test_frequency_reference_is_scale_invariant():
    tr,te=toy();ref_tr=pd.concat([tr,tr],ignore_index=True);ref_te=te.copy();a=build_feature_views(tr.iloc[:1],te,use_frequency=True,frequency_reference=(ref_tr,ref_te))['combined'][0];b=build_feature_views(tr,te,use_frequency=True,frequency_reference=(ref_tr,ref_te))['combined'][0].iloc[:1];cols=[c for c in a if c.endswith('__freq') or 'round0_freq' in c or 'round1_freq' in c];assert cols and np.allclose(a[cols].to_numpy(float),b[cols].to_numpy(float),equal_nan=True)
