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

    sample_ids = labels["sample_id"].astype(int).drop_duplicates().to_numpy(copy=True)
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


def validate_episode_split(split: pd.DataFrame) -> None:
    required_columns = {"sample_id", "split"}
    missing_columns = required_columns - set(split.columns)
    if missing_columns:
        raise ValueError(f"split file missing required columns: {sorted(missing_columns)}")

    duplicated = split["sample_id"][split["sample_id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"sample_id appears in multiple split rows: {duplicated[:10]}")

    split_sets = {
        split_name: set(split.loc[split["split"] == split_name, "sample_id"].astype(int))
        for split_name in split["split"].unique()
    }
    for left_name, left_ids in split_sets.items():
        for right_name, right_ids in split_sets.items():
            if left_name >= right_name:
                continue
            overlap = left_ids & right_ids
            if overlap:
                raise ValueError(
                    f"data leakage: {left_name} and {right_name} share sample_ids "
                    f"{sorted(overlap)[:10]}"
                )


def validate_feature_frame_membership(
    features: pd.DataFrame,
    split: pd.DataFrame,
    split_name: str,
) -> None:
    validate_episode_split(split)
    expected_ids = set(split.loc[split["split"] == split_name, "sample_id"].astype(int))
    actual_ids = set(features["sample_id"].astype(int))
    unexpected_ids = actual_ids - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"data leakage: features_{split_name} contains sample_ids outside {split_name}: "
            f"{sorted(unexpected_ids)[:10]}"
        )
