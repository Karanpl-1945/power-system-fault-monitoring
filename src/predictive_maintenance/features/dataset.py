from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from predictive_maintenance.data.protect90 import load_waveform
from predictive_maintenance.features.statistical import extract_basic_features
from predictive_maintenance.features.windowing import iter_windows, window_fault_label


def waveform_channels(waveform: pd.DataFrame) -> list[str]:
    return [column for column in waveform.columns if column != "time_s"]


def build_episode_feature_rows(
    label_row: pd.Series,
    waveform_dir: str | Path,
    window_samples: int,
    stride_samples: int,
    channels: list[str] | None = None,
) -> list[dict[str, object]]:
    sample_id = int(label_row.name if label_row.name is not None else label_row["sample_id"])
    waveform = load_waveform(waveform_dir, sample_id)
    selected_channels = channels or waveform_channels(waveform)

    rows: list[dict[str, object]] = []
    for start_idx, end_idx, window in iter_windows(waveform, window_samples, stride_samples):
        start_time = float(window.iloc[0]["time_s"])
        end_time = float(window.iloc[-1]["time_s"])
        window_label = window_fault_label(
            start_time,
            end_time,
            float(label_row["t_evnt_start"]),
            float(label_row["t_evnt_end"]),
        )

        row: dict[str, object] = {
            "sample_id": sample_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": start_time,
            "end_time": end_time,
            "window_label": window_label,
            "binary_target": int(window_label == "fault"),
            "sc_type": int(label_row["sc_type"]) if window_label == "fault" else -1,
            "fault_target": str(label_row["fault_target"]) if window_label == "fault" else "none",
            "phase_select": int(label_row["phase_select"]),
            "fault_resistance": float(label_row["fault_resistance"]),
            "sc_location": float(label_row["sc_location"]),
        }
        row.update(extract_basic_features(window, selected_channels))
        rows.append(row)

    return rows


def build_feature_frame(
    labels: pd.DataFrame,
    split: pd.DataFrame,
    waveform_dir: str | Path,
    split_name: str,
    window_samples: int,
    stride_samples: int,
    max_episodes: int | None = None,
    channel_limit: int | None = None,
) -> pd.DataFrame:
    split_ids = split.loc[split["split"] == split_name, "sample_id"].astype(int).tolist()
    if max_episodes is not None:
        split_ids = split_ids[:max_episodes]

    label_by_id = labels.set_index("sample_id")
    rows: list[dict[str, object]] = []
    channels: list[str] | None = None

    for index, sample_id in enumerate(split_ids, start=1):
        label_row = label_by_id.loc[sample_id]
        if channels is None:
            first_waveform = load_waveform(waveform_dir, sample_id)
            channels = waveform_channels(first_waveform)
            if channel_limit is not None:
                channels = channels[:channel_limit]

        rows.extend(
            build_episode_feature_rows(
                label_row=label_row,
                waveform_dir=waveform_dir,
                window_samples=window_samples,
                stride_samples=stride_samples,
                channels=channels,
            )
        )

        if index % 25 == 0:
            print(f"{split_name}: processed {index}/{len(split_ids)} episodes", flush=True)

    return pd.DataFrame(rows)


def write_split_features(
    labels: pd.DataFrame,
    split: pd.DataFrame,
    waveform_dir: str | Path,
    output_dir: str | Path,
    split_names: Iterable[str],
    window_samples: int,
    stride_samples: int,
    max_episodes: int | None = None,
    channel_limit: int | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name in split_names:
        frame = build_feature_frame(
            labels=labels,
            split=split,
            waveform_dir=waveform_dir,
            split_name=split_name,
            window_samples=window_samples,
            stride_samples=stride_samples,
            max_episodes=max_episodes,
            channel_limit=channel_limit,
        )
        destination = output_path / f"features_{split_name}.csv"
        frame.to_csv(destination, index=False)
        print(f"saved={destination} rows={len(frame)} cols={len(frame.columns)}", flush=True)
