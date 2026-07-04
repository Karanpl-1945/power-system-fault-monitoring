from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def iter_windows(
    waveform: pd.DataFrame,
    window_samples: int,
    stride_samples: int,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield sliding windows as (start_idx, end_idx, frame)."""
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive")

    for start in range(0, len(waveform) - window_samples + 1, stride_samples):
        end = start + window_samples
        yield start, end, waveform.iloc[start:end]


def window_fault_label(
    start_time: float,
    end_time: float,
    fault_start: float,
    fault_end: float,
) -> str:
    if end_time < fault_start:
        return "normal"
    if start_time <= fault_end and end_time >= fault_start:
        return "fault"
    return "post_fault"

