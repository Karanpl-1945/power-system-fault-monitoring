from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


WAVEFORM_RE = re.compile(r"(?P<sample_id>\d+)_sample_hv_double_line_90kv\.pkl$")


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if "sample_id" not in labels.columns:
        raise ValueError("labels file must contain a sample_id column")
    return labels


def waveform_path(waveform_dir: str | Path, sample_id: int) -> Path:
    return Path(waveform_dir) / f"{sample_id}_sample_hv_double_line_90kv.pkl"


def load_waveform(waveform_dir: str | Path, sample_id: int) -> pd.DataFrame:
    path = waveform_path(waveform_dir, sample_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_pickle(path)


def available_waveform_ids(waveform_dir: str | Path) -> set[int]:
    ids: set[int] = set()
    for path in Path(waveform_dir).glob("*_sample_hv_double_line_90kv.pkl"):
        match = WAVEFORM_RE.match(path.name)
        if match:
            ids.add(int(match.group("sample_id")))
    return ids


def missing_waveforms(labels: pd.DataFrame, waveform_dir: str | Path) -> list[int]:
    available = available_waveform_ids(waveform_dir)
    expected = set(labels["sample_id"].astype(int).tolist())
    return sorted(expected - available)

