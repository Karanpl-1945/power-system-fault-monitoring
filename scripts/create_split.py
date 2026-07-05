from __future__ import annotations

import argparse

from predictive_maintenance.data.protect90 import load_labels
from predictive_maintenance.data.split import create_episode_split, save_split, validate_episode_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split by sample_id.")
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--output", default="data/splits/protect90_split.csv")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    split = create_episode_split(labels, args.train, args.val, args.test, args.seed)
    validate_episode_split(split)
    save_split(split, args.output)
    print(split["split"].value_counts().to_string())
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
