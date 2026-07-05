import pandas as pd

import pytest

from predictive_maintenance.data.split import (
    create_episode_split,
    validate_episode_split,
    validate_feature_frame_membership,
)


def test_create_episode_split_assigns_each_sample_once():
    labels = pd.DataFrame({"sample_id": list(range(100)), "sc_type": [0] * 100})
    split = create_episode_split(labels, train=0.7, val=0.15, test=0.15, seed=42)

    assert len(split) == 100
    assert split["sample_id"].nunique() == 100
    assert set(split["split"]) == {"train", "val", "test"}
    validate_episode_split(split)


def test_validate_episode_split_rejects_duplicate_sample_ids():
    split = pd.DataFrame(
        {
            "sample_id": [1, 1, 2],
            "split": ["train", "test", "val"],
        }
    )

    with pytest.raises(ValueError, match="sample_id appears"):
        validate_episode_split(split)


def test_validate_feature_frame_membership_rejects_wrong_split_ids():
    split = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "split": ["train", "val", "test"],
        }
    )
    features = pd.DataFrame({"sample_id": [1, 2], "feature": [0.1, 0.2]})

    with pytest.raises(ValueError, match="data leakage"):
        validate_feature_frame_membership(features, split, "train")
