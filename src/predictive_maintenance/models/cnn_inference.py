from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from predictive_maintenance.models.cnn1d import FaultCNN1D


def load_cnn_checkpoint(model_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = FaultCNN1D(in_channels=len(checkpoint["channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    checkpoint["model"] = model
    return checkpoint


def normalize_window(window: np.ndarray) -> np.ndarray:
    """window: (channels, time). Per-channel z-score, matching RawWindowDataset's training-time normalization."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-6
    return (window - mean) / std


def channels_to_array(
    channels: dict[str, list[float]],
    channel_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    sample_length = len(next(iter(channels.values())))
    missing_channels = [channel for channel in channel_order if channel not in channels]
    rows = [channels.get(channel, [0.0] * sample_length) for channel in channel_order]
    return np.asarray(rows, dtype=np.float32), missing_channels


def predict_window(
    checkpoint: dict[str, Any],
    channels: dict[str, list[float]],
    explain_limit: int = 5,
) -> dict[str, Any]:
    channel_order = checkpoint["channels"]
    raw_array, missing_channels = channels_to_array(channels, channel_order)
    normalized = normalize_window(raw_array)

    model = checkpoint["model"]
    x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    x.requires_grad_(True)

    logit = model(x)
    probability = torch.sigmoid(logit)
    grad = torch.autograd.grad(outputs=probability, inputs=x)[0]

    saliency = grad.detach().abs().sum(dim=2).squeeze(0).numpy()
    peak_to_peak = raw_array.max(axis=1) - raw_array.min(axis=1)

    ranked = sorted(
        zip(channel_order, saliency.tolist(), peak_to_peak.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    threshold = float(checkpoint["threshold"])
    prob_value = float(probability.detach().item())

    return {
        "probability": prob_value,
        "threshold": threshold,
        "is_fault": prob_value >= threshold,
        "missing_channels": missing_channels,
        "channel_count": len(channel_order),
        "top_channels": [
            {"channel": channel, "importance": importance, "value": value}
            for channel, importance, value in ranked[:explain_limit]
            if importance > 0
        ],
    }
