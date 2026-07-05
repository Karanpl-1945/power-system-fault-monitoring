from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from confluent_kafka import Consumer, Producer

from predictive_maintenance.explainability.messages import feature_message
from predictive_maintenance.features.statistical import extract_basic_features
from predictive_maintenance.streaming.schemas import PredictionEvent, WaveformWindowEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume waveform windows and publish XGBoost predictions.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--input-topic", default="power.waveform.windows")
    parser.add_argument("--prediction-topic", default="power.fault.predictions")
    parser.add_argument("--alert-topic", default="power.fault.alerts")
    parser.add_argument("--group-id", default="fault-inference-consumer")
    parser.add_argument(
        "--offset-reset",
        default="earliest",
        choices=["earliest", "latest"],
        help="Use 'latest' for a persistent live consumer that should ignore backlog.",
    )
    parser.add_argument("--model", default="models/xgboost_fault_detector_48ch_tuned_recall97.joblib")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Stop after this many idle seconds. Use 0 to run indefinitely.",
    )
    parser.add_argument("--output-report", default="reports/kafka_inference_report.json")
    parser.add_argument("--output-alerts", default="reports/kafka_alerts.jsonl")
    return parser.parse_args()


def build_feature_frame(
    event: WaveformWindowEvent,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    window = pd.DataFrame(event.channels)
    features: dict[str, float] = {
        "phase_select": float(event.context.get("phase_select", -1)),
        "fault_resistance": float(event.context.get("fault_resistance", 0.0)),
        "sc_location": float(event.context.get("sc_location", -1.0)),
    }
    features.update(extract_basic_features(window, list(event.channels.keys())))
    missing_features = [column for column in feature_columns if column not in features]
    frame = pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])
    return frame, missing_features


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
                "message": feature_message(feature),
            }
        )
    rows.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rows[:limit]


def publish_json(producer: Producer, topic: str, key: str, payload: dict[str, Any]) -> None:
    producer.produce(topic, key=key, value=json.dumps(payload))
    producer.poll(0)


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap_servers,
            "group.id": args.group_id,
            "auto.offset.reset": args.offset_reset,
            "enable.auto.commit": True,
        }
    )
    producer = Producer({"bootstrap.servers": args.bootstrap_servers})
    consumer.subscribe([args.input_topic])

    processed = 0
    alerts = []
    latencies_ms = []
    start_time = time.perf_counter()

    try:
        while True:
            if args.max_messages is not None and processed >= args.max_messages:
                break
            if args.timeout_seconds > 0 and time.perf_counter() - start_time > args.timeout_seconds:
                break

            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                print(f"consumer_error={message.error()}", flush=True)
                continue

            event = WaveformWindowEvent.model_validate_json(message.value())
            inference_start = time.perf_counter()
            frame, missing_features = build_feature_frame(event, feature_columns)
            probability = float(model.predict_proba(frame[feature_columns])[:, 1][0])
            latency_ms = (time.perf_counter() - inference_start) * 1000
            is_alert = probability >= threshold
            latencies_ms.append(latency_ms)

            prediction = PredictionEvent(
                sample_id=event.sample_id,
                window_index=event.window_index,
                window_start_time=event.window_start_time,
                window_end_time=event.window_end_time,
                prediction="fault" if is_alert else "normal",
                probability=probability,
                threshold=threshold,
                alert=is_alert,
                latency_ms=latency_ms,
                true_window_label=event.true_window_label,
                top_features=top_feature_messages(artifact, frame),
            )
            payload = prediction.model_dump()
            payload["missing_feature_count"] = len(missing_features)

            key = f"{event.sample_id}:{event.window_index}"
            publish_json(producer, args.prediction_topic, key, payload)
            if is_alert:
                publish_json(producer, args.alert_topic, key, payload)
                alerts.append(payload)
            processed += 1
    finally:
        consumer.close()
        producer.flush()

    alert_path = Path(args.output_alerts)
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    with alert_path.open("w", encoding="utf-8") as file:
        for alert in alerts:
            file.write(json.dumps(alert) + "\n")

    report = {
        "mode": "kafka_inference_consumer",
        "model_path": args.model,
        "input_topic": args.input_topic,
        "prediction_topic": args.prediction_topic,
        "alert_topic": args.alert_topic,
        "messages_processed": processed,
        "alerts": len(alerts),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        "max_latency_ms": max(latencies_ms) if latencies_ms else None,
        "first_alert": alerts[0] if alerts else None,
        "alerts_path": str(alert_path),
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved_report={report_path}", flush=True)
    print(f"saved_alerts={alert_path}", flush=True)


if __name__ == "__main__":
    main()
