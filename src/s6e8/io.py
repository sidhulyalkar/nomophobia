from pathlib import Path
import pandas as pd
from .config import TARGET


def load_competition(data_dir: str | Path):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if TARGET not in train:
        raise ValueError(f"Missing target {TARGET} in train.csv")
    if TARGET in test:
        raise ValueError(f"Unexpected target {TARGET} in test.csv")
    return train, test, sample
