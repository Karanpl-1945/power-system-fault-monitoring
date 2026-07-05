from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from predictive_maintenance.data.protect90 import load_waveform
from predictive_maintenance.features.dataset import waveform_channels
from predictive_maintenance.features.windowing import iter_windows, window_fault_label


def build_raw_window_index(
    labels: pd.DataFrame,
    split: pd.DataFrame,
    waveform_dir: str | Path,
    split_name: str,
    window_samples: int,
    stride_samples: int,
    max_episodes: int | None = None,
    max_windows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    split_ids = split.loc[split["split"] == split_name, "sample_id"].astype(int).tolist()
    if max_episodes is not None:
        split_ids = split_ids[:max_episodes]

    label_by_id = labels.set_index("sample_id")
    rows: list[dict[str, object]] = []
    for sample_id in split_ids:
        label_row = label_by_id.loc[sample_id]
        waveform = load_waveform(waveform_dir, sample_id)
        for start_idx, end_idx, window in iter_windows(waveform, window_samples, stride_samples):
            start_time = float(window.iloc[0]["time_s"])
            end_time = float(window.iloc[-1]["time_s"])
            label = window_fault_label(
                start_time,
                end_time,
                float(label_row["t_evnt_start"]),
                float(label_row["t_evnt_end"]),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "binary_target": int(label == "fault"),
                    "window_label": label,
                }
            )

    index = pd.DataFrame(rows)
    if max_windows is not None and len(index) > max_windows:
        sampled_indices = []
        for label in sorted(index["binary_target"].unique()):
            label_indices = index[index["binary_target"] == label].index.to_series()
            n_label = max(1, round(max_windows * len(label_indices) / len(index)))
            sampled_indices.append(label_indices.sample(n=n_label, random_state=seed))
        selected_indices = (
            pd.concat(sampled_indices)
            .sample(frac=1.0, random_state=seed)
            .to_list()
        )
        index = index.loc[selected_indices].reset_index(drop=True)
    return index


class RawWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        window_index: pd.DataFrame,
        waveform_dir: str | Path,
        channels: Sequence[str],
    ) -> None:
        self.window_index = window_index.reset_index(drop=True)
        self.waveform_dir = waveform_dir
        self.channels = list(channels)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.window_index)

    def _waveform_array(self, sample_id: int) -> np.ndarray:
        if sample_id not in self._cache:
            waveform = load_waveform(self.waveform_dir, sample_id)
            self._cache[sample_id] = waveform[self.channels].to_numpy(dtype=np.float32)
        return self._cache[sample_id]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.window_index.iloc[index]
        sample_id = int(row["sample_id"])
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])
        window = self._waveform_array(sample_id)[start_idx:end_idx]

        window = window.T
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True) + 1e-6
        window = (window - mean) / std

        x = torch.from_numpy(window.astype(np.float32))
        y = torch.tensor(float(row["binary_target"]), dtype=torch.float32)
        return x, y


def infer_channels(
    waveform_dir: str | Path,
    sample_id: int,
    channel_limit: int | None = None,
) -> list[str]:
    channels = waveform_channels(load_waveform(waveform_dir, sample_id))
    if channel_limit is not None:
        return channels[:channel_limit]
    return channels
