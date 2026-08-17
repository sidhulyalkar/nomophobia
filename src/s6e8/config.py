from dataclasses import dataclass

TARGET = "addicted_label"
ID_COL = "id"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
RAW_COLS = NUM_COLS + CAT_COLS
CORE_USAGE = [
    "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "weekend_screen_time",
]
MONOTONE_POSITIVE = [
    "daily_screen_time_hours", "social_media_hours", "weekend_screen_time"
]
DECIMAL_COLS = [
    "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "weekend_screen_time",
]

@dataclass(frozen=True)
class Preset:
    n_splits: int
    seed: int
    lgb_estimators: int
    cat_iterations: int
    xgb_estimators: int
    early_stopping: int
    max_rows: int | None = None

PRESETS = {
    "smoke": Preset(2, 20260816, 120, 80, 160, 20, 8_000),
    "quick": Preset(3, 20260816, 700, 450, 700, 70, 100_000),
    "highcap": Preset(3, 20260816, 1000, 600, 900, 100, 120_000),
    "full": Preset(5, 20260816, 2200, 2200, 2000, 180, None),
}

EXPERT_SPECS = {
    "lgb_raw63": {"family": "lgb", "view": "raw", "profile": "raw63"},
    "lgb_combined63": {"family": "lgb", "view": "combined", "profile": "combined63"},
    "lgb_monotone31": {"family": "lgb", "view": "raw", "profile": "monotone31"},
    "lgb_semantic15": {"family": "lgb", "view": "semantic", "profile": "semantic15"},
    "lgb_generator31": {"family": "lgb", "view": "generator", "profile": "generator31"},
    "cat_raw": {"family": "cat", "view": "raw", "profile": "raw"},
    "xgb_raw": {"family": "xgb", "view": "raw", "profile": "raw"},
}

DEFAULT_EXPERTS = {
    "smoke": ["lgb_combined63", "lgb_raw63"],
    "quick": ["lgb_combined63", "lgb_raw63"],
    "highcap": ["lgb_combined63", "lgb_raw63"],
    "full": ["lgb_combined63", "lgb_raw63"],
}
