import numpy as np
import pandas as pd
from s6e8.preprocess import prepare_tree_frames
from s6e8.features import add_digit_features
from s6e8.evaluate import paired_compare


def test_native_categories_not_scrambled_ordinals():
    tr=pd.DataFrame({'d':['7','2','7','9'],'x':[1.,2.,3.,4.]});te=pd.DataFrame({'d':['2','8'],'x':[5.,6.]});_,_,tn,_,cats=prepare_tree_frames(tr,te)
    assert str(tn['d'].dtype)=='category';assert cats==['d'];assert list(tn['d'].cat.categories)==['2','7','8','9']


def test_digit_ordinal_copy_is_natural():
    x=pd.DataFrame({'daily_screen_time_hours':[1.27,np.nan]})
    for c in ['social_media_hours','gaming_hours','work_study_hours','sleep_hours','weekend_screen_time']:x[c]=x['daily_screen_time_hours']
    z=add_digit_features(x);assert int(z.loc[0,'daily_screen_time_hours__tenths_ord'])==2;assert int(z.loc[0,'daily_screen_time_hours__hundredths_ord'])==7;assert int(z.loc[1,'daily_screen_time_hours__hundredths_ord'])==-1


def test_paired_compare_identity():
    y=np.array([0,1]*30);p=np.linspace(0,1,60);f=np.arange(60)%3;r=paired_compare(y,p,p,f,n_boot=20,seed=1);assert abs(r['delta_auc'])<1e-12 and r['delta_ci_95']==[0.0,0.0]
