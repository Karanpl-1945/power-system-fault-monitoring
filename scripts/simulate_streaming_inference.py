from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from predictive_maintenance.data.protect90 import load_labels, load_waveform
from predictive_maintenance.features.dataset import waveform_channels
from predictive_maintenance.features.statistical import extract_basic_features
from predictive_maintenance.features.windowing import iter_windows, window_fault_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay waveform windows locally and run streaming XGBoost inference."
    )
    parser.add_argument("--labels", default="hv_double_line_90kv_labels.csv")
    parser.add_argument("--waveform-dir", default="hv_double_line_90kv_preprocessed_data")
    parser.add_argument("--model", default="models/xgboost_fault_detector_48ch_tuned_recall97.joblib")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--window-samples", type=int, default=640)
    parser.add_argument("--stride-samples", type=int, default=128)
    parser.add_argument("--channel-limit", type=int, default=48)
    parser.add_argument("--max-windows", type=int, default=80)
    parser.add_argument("--output-report", default="reports/streaming_simulation_report.json")
    parser.add_argument("--output-alerts", default="reports/streaming_alerts.jsonl")
    return parser.parse_args()


def build_features(
    window: pd.DataFrame,
    channels: list[str],
    label_row: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    features: dict[str, float] = {
        "phase_select": float(label_row["phase_select"]),
        "fault_resistance": float(label_row["fault_resistance"]),
        "sc_location": float(label_row["sc_location"]),
    }
    features.update(extract_basic_features(window, channels))
    missing_features = [column for column in feature_columns if column not in features]
    frame = pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])
    return frame, missing_features, features


def top_feature_messages(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    limit: int = 5,
) -> list[dict[str, float | str]]:
    model = artifact["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    rows = []
    for feature, importance in zip(artifact["feature_columns"], importances, strict=True):
        if importance <= 0:
            continue
        rows.append(
            {
                "feature": feature,
                "importance": float(importance),
                "value": float(frame.iloc[0][feature]),
            }
        )
    rows.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rows[:limit]


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    label_row = labels.set_index("sample_id").loc[args.sample_id]
    waveform = load_waveform(args.waveform_dir, args.sample_id)
    channels = waveform_channels(waveform)[: args.channel_limit]

    artifact = joblib.load(args.model)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    events = []
    alerts = []
    latencies_ms = []

    for window_index, (start_idx, end_idx, window) in enumerate(
        iter_windows(waveform, args.window_samples, args.stride_samples),
        start=1,
    ):
        if args.max_windows is not None and window_index > args.max_windows:
            break

        start_time = float(window.iloc[0]["time_s"])
        end_time = float(window.iloc[-1]["time_s"])
        true_window_label = window_fault_label(
            start_time,
            end_time,
            float(label_row["t_evnt_start"]),
            float(label_row["t_evnt_end"]),
        )

        inference_start = time.perf_counter()
        frame, missing_features, _ = build_features(window, channels, label_row, feature_columns)
        probability = float(model.predict_proba(frame[feature_columns])[:, 1][0])
        latency_ms = (time.perf_counter() - inference_start) * 1000
        latencies_ms.append(latency_ms)
        is_alert = probability >= threshold

        event = {
            "sample_id": args.sample_id,
            "window_index": window_index,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "window_start_time": start_time,
            "window_end_time": end_time,
            "true_window_label": true_window_label,
            "probability": probability,
            "threshold": threshold,
            "prediction": "fault" if is_alert else "normal",
            "alert": is_alert,
            "latency_ms": latency_ms,
            "missing_feature_count": len(missing_features),
            "top_features": top_feature_messages(artifact, frame, limit=5),
        }
        events.append(event)
        if is_alert:
            alerts.append(event)

    alert_path = Path(args.output_alerts)
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with alert_path.open("w", encoding="utf-8") as file:
        for alert in alerts:
            file.write(json.dumps(alert) + "\n")

    first_alert = alerts[0] if alerts else None
    report = {
        "mode": "local_streaming_simulation",
        "model_path": args.model,
        "sample_id": args.sample_id,
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "channel_count": len(channels),
        "threshold": threshold,
        "windows_processed": len(events),
        "alerts": len(alerts),
        "first_alert": first_alert,
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        "max_latency_ms": max(latencies_ms) if latencies_ms else None,
        "true_fault_windows": sum(1 for event in events if event["true_window_label"] == "fault"),
        "predicted_fault_windows": sum(1 for event in events if event["prediction"] == "fault"),
        "events_preview": events[:5],
        "alerts_path": str(alert_path),
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"saved_report={report_path}")
    print(f"saved_alerts={alert_path}")


if __name__ == "__main__":
    main()
