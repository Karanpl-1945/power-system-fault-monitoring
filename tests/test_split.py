import pandas as pd

from predictive_maintenance.data.split import create_episode_split


def test_create_episode_split_assigns_each_sample_once():
    labels = pd.DataFrame({"sample_id": list(range(100)), "sc_type": [0] * 100})
    split = create_episode_split(labels, train=0.7, val=0.15, test=0.15, seed=42)

    assert len(split) == 100
    assert split["sample_id"].nunique() == 100
    assert set(split["split"]) == {"train", "val", "test"}

