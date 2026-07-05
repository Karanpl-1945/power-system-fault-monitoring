from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from predictive_maintenance.data.protect90 import load_labels
from predictive_maintenance.data.raw_windows import (
    RawWindowDataset,
    build_raw_window_index,
    infer_channels,
)
from predictive_maintenance.data.split import validate_episode_split
from predictive_maintenance.models.cnn1d import FaultCNN1D
from predictive_maintenance.models.evaluation import binary_metrics, choose_threshold_for_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small 1D-CNN fault detector on raw windows.")
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--split", default="data/splits/protect90_split.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--window-samples", type=int, default=640)
    parser.add_argument("--stride-samples", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-train-windows", type=int, default=3000)
    parser.add_argument("--max-val-windows", type=int, default=1500)
    parser.add_argument("--max-test-windows", type=int, default=1500)
    parser.add_argument("--channel-limit", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", default="models/cnn1d_fault_detector_smoke.pt")
    parser.add_argument("--report-output", default="reports/cnn1d_fault_detector_smoke_report.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset: RawWindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(scores), np.concatenate(targets).astype(int), elapsed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x.to(device))
        loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    labels = load_labels(args.labels)
    split = pd.read_csv(args.split)
    validate_episode_split(split)

    first_train_id = int(split.loc[split["split"] == "train", "sample_id"].iloc[0])
    channels = infer_channels(args.waveform_dir, first_train_id, args.channel_limit)

    train_index = build_raw_window_index(
        labels,
        split,
        args.waveform_dir,
        "train",
        args.window_samples,
        args.stride_samples,
        max_episodes=args.max_episodes,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val_index = build_raw_window_index(
        labels,
        split,
        args.waveform_dir,
        "val",
        args.window_samples,
        args.stride_samples,
        max_episodes=args.max_episodes,
        max_windows=args.max_val_windows,
        seed=args.seed,
    )
    test_index = build_raw_window_index(
        labels,
        split,
        args.waveform_dir,
        "test",
        args.window_samples,
        args.stride_samples,
        max_episodes=args.max_episodes,
        max_windows=args.max_test_windows,
        seed=args.seed,
    )

    train_dataset = RawWindowDataset(train_index, args.waveform_dir, channels)
    val_dataset = RawWindowDataset(val_index, args.waveform_dir, channels)
    test_dataset = RawWindowDataset(test_index, args.waveform_dir, channels)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultCNN1D(in_channels=len(channels)).to(device)

    positives = float(train_index["binary_target"].sum())
    negatives = float(len(train_index) - positives)
    pos_weight = torch.tensor([negatives / positives if positives else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_losses.append(loss)
        print(f"epoch={epoch} loss={loss:.6f}", flush=True)

    val_scores, val_targets, val_elapsed = collect_scores(model, val_loader, device)
    threshold, val_metrics = choose_threshold_for_recall(
        val_targets,
        val_scores,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
    )
    test_scores, test_targets, test_elapsed = collect_scores(model, test_loader, device)
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = binary_metrics(test_targets, test_pred)

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "channels": channels,
            "args": vars(args),
        },
        model_output,
    )

    report = {
        "model": "FaultCNN1D",
        "target": "binary_target",
        "threshold": threshold,
        "feature_count": None,
        "channel_count": len(channels),
        "train_rows": len(train_index),
        "val_rows": len(val_index),
        "test_rows": len(test_index),
        "device": str(device),
        "epoch_losses": epoch_losses,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency": {
            "val_seconds": val_elapsed,
            "test_seconds": test_elapsed,
            "test_ms_per_window": (test_elapsed / len(test_index)) * 1000 if len(test_index) else None,
        },
    }

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_model={model_output}")
    print(f"saved_report={report_output}")


if __name__ == "__main__":
    main()
