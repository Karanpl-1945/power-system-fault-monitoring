from __future__ import annotations

import argparse

import pandas as pd

from predictive_maintenance.data.split import (
    validate_episode_split,
    validate_feature_frame_membership,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check episode-level split leakage.")
    parser.add_argument("--split", default="data/splits/protect90_split.csv")
    parser.add_argument("--feature-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = pd.read_csv(args.split)
    validate_episode_split(split)

    for split_name in ["train", "val", "test"]:
        feature_path = f"{args.feature_dir}/features_{split_name}.csv"
        features = pd.read_csv(feature_path, usecols=["sample_id"])
        validate_feature_frame_membership(features, split, split_name)
        print(f"{split_name}: OK ({features['sample_id'].nunique()} sample_ids)")

    print("No episode-level leakage detected.")


if __name__ == "__main__":
    main()
