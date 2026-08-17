from s6e8.config import PRESETS,DEFAULT_EXPERTS


def test_highcap_preset_is_dual_view_s1():
    p=PRESETS['highcap'];assert p.max_rows==120_000;assert p.lgb_estimators==1000;assert DEFAULT_EXPERTS['highcap']==['lgb_combined63','lgb_raw63']


def test_full_default_remains_audited_dual_view():
    assert DEFAULT_EXPERTS['full']==['lgb_combined63','lgb_raw63']
