from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import CompetitionContractError, validate_competition_frames


EXPECTED_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def load_competition(data_dir: str | Path, *, validate: bool = True):
    data_dir = Path(data_dir)
    missing = [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Competition data directory {data_dir} is missing required files: {missing}"
        )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if validate:
        validate_competition_frames(train, test, sample)
    return train, test, sample


__all__ = ["CompetitionContractError", "EXPECTED_FILES", "load_competition"]
