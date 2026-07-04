from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_episode_split(
    labels: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    sample_ids = labels["sample_id"].astype(int).drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * train)
    n_val = int(n * val)

    train_ids = sample_ids[:n_train]
    val_ids = sample_ids[n_train : n_train + n_val]
    test_ids = sample_ids[n_train + n_val :]

    rows = [
        *({"sample_id": int(sample_id), "split": "train"} for sample_id in train_ids),
        *({"sample_id": int(sample_id), "split": "val"} for sample_id in val_ids),
        *({"sample_id": int(sample_id), "split": "test"} for sample_id in test_ids),
    ]
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def save_split(split: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)

