from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from predictive_maintenance.data.protect90 import load_labels
from predictive_maintenance.data.raw_windows import RawWindowDataset, build_raw_window_index
from predictive_maintenance.data.split import validate_episode_split
from predictive_maintenance.models.cnn1d import FaultCNN1D
from predictive_maintenance.models.evaluation import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved 1D-CNN checkpoint on a rare-fault raw-window test set."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--split", default="data/splits/protect90_split.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--window-samples", type=int, default=640)
    parser.add_argument("--stride-samples", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fault-ratio", type=float, required=True)
    parser.add_argument("--max-faults", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_rare_fault_index(index: pd.DataFrame, fault_ratio: float, max_faults: int | None, seed: int) -> pd.DataFrame:
    if not 0 < fault_ratio < 1:
        raise ValueError("--fault-ratio must be between 0 and 1")

    fault = index[index["binary_target"] == 1]
    normal = index[index["binary_target"] == 0]

    n_fault = len(fault) if max_faults is None else min(max_faults, len(fault))
    n_normal_needed = int(round(n_fault * (1 - fault_ratio) / fault_ratio))

    if n_normal_needed > len(normal):
        n_normal_needed = len(normal)
        n_fault = int(round(n_normal_needed * fault_ratio / (1 - fault_ratio)))

    sampled_fault = fault.sample(n=n_fault, random_state=seed)
    sampled_normal = normal.sample(n=n_normal_needed, random_state=seed)
    return (
        pd.concat([sampled_fault, sampled_normal], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    channels = checkpoint["channels"]
    threshold = float(checkpoint["threshold"])

    labels = load_labels(args.labels)
    split = pd.read_csv(args.split)
    validate_episode_split(split)

    full_test_index = build_raw_window_index(
        labels,
        split,
        args.waveform_dir,
        "test",
        args.window_samples,
        args.stride_samples,
        max_episodes=args.max_episodes,
        max_windows=None,
        seed=args.seed,
    )

    rare_index = build_rare_fault_index(full_test_index, args.fault_ratio, args.max_faults, args.seed)
    dataset = RawWindowDataset(rare_index, args.waveform_dir, channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultCNN1D(in_channels=len(channels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())

    scores_arr = np.concatenate(scores)
    targets_arr = np.concatenate(targets).astype(int)
    pred = (scores_arr >= threshold).astype(int)

    metrics = binary_metrics(targets_arr, pred)
    metrics["alerts_per_10000_windows"] = float(metrics["false_positive_rate"] * 10000)
    metrics["fault_ratio"] = float(rare_index["binary_target"].mean())
    metrics["rows"] = int(len(rare_index))

    report = {
        "model_path": args.model,
        "fault_ratio_requested": args.fault_ratio,
        "threshold": threshold,
        "channel_count": len(channels),
        "metrics": metrics,
    }

    if args.output is None:
        model_name = Path(args.model).stem
        pct = int(round(args.fault_ratio * 100))
        output = Path("reports") / f"{model_name}_on_realistic_{pct}pct.json"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_report={output}")


if __name__ == "__main__":
    main()
