from __future__ import annotations

import numpy as np
import pandas as pd


def rms(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def extract_basic_features(window: pd.DataFrame, channels: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for channel in channels:
        values = window[channel]
        features[f"{channel}_mean"] = float(values.mean())
        features[f"{channel}_std"] = float(values.std())
        features[f"{channel}_min"] = float(values.min())
        features[f"{channel}_max"] = float(values.max())
        features[f"{channel}_rms"] = rms(values)
        features[f"{channel}_ptp"] = float(values.max() - values.min())
    return features

