from __future__ import annotations

import argparse

import pandas as pd

from predictive_maintenance.data.protect90 import load_labels
from predictive_maintenance.data.split import validate_episode_split
from predictive_maintenance.features.dataset import write_split_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build statistical window features from PROTECT-90 waveform files."
    )
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--split", default="data/splits/protect90_split.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-samples", type=int, default=640)
    parser.add_argument("--stride-samples", type=int, default=128)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit episodes per split for quick experiments.",
    )
    parser.add_argument(
        "--channel-limit",
        type=int,
        default=6,
        help="Limit waveform channels. Default 6 uses one measurement location; use 48 for all channels.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    split = pd.read_csv(args.split)
    validate_episode_split(split)

    write_split_features(
        labels=labels,
        split=split,
        waveform_dir=args.waveform_dir,
        output_dir=args.output_dir,
        split_names=args.splits,
        window_samples=args.window_samples,
        stride_samples=args.stride_samples,
        max_episodes=args.max_episodes,
        channel_limit=args.channel_limit,
    )


if __name__ == "__main__":
    main()
