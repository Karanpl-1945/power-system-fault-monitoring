from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from predictive_maintenance.data.protect90 import load_labels, load_waveform
from predictive_maintenance.explainability.messages import feature_message
from predictive_maintenance.features.dataset import waveform_channels
from predictive_maintenance.features.statistical import extract_basic_features
from predictive_maintenance.features.windowing import window_fault_label
from predictive_maintenance.serving.kafka_bridge import ConnectionManager, KafkaAlertBridge


MODEL_PATH = Path("models/xgboost_fault_detector_48ch_tuned_recall97.joblib")
REPORTS_DIR = Path("reports")
STATIC_DIR = Path(__file__).resolve().parent / "static"
LABELS_PATH = Path("hv_double_line_90kv_labels.csv")
WAVEFORM_DIR = Path("hv_double_line_90kv_preprocessed_data")


class WindowContext(BaseModel):
    phase_select: int = -1
    fault_resistance: float = 0.0
    sc_location: float = -1.0


class WindowPredictionRequest(BaseModel):
    channels: dict[str, list[float]] = Field(
        ...,
        description="Mapping of channel name to equal-length waveform samples.",
    )
    context: WindowContext = Field(default_factory=WindowContext)

    @model_validator(mode="after")
    def validate_channels(self) -> "WindowPredictionRequest":
        if not self.channels:
            raise ValueError("channels must not be empty")
        lengths = {len(values) for values in self.channels.values()}
        if len(lengths) != 1:
            raise ValueError("all channels must have the same number of samples")
        sample_count = lengths.pop()
        if sample_count < 2:
            raise ValueError("each channel must have at least two samples")
        return self


class ExplanationItem(BaseModel):
    feature: str
    importance: float
    value: float
    message: str


class WindowPredictionResponse(BaseModel):
    model_path: str
    probability: float
    threshold: float
    is_fault: bool
    label: str
    latency_ms: float
    feature_count: int
    missing_features: list[str]
    top_features: list[ExplanationItem]


connection_manager = ConnectionManager()
kafka_bridge: KafkaAlertBridge | None = None
live_consumer_process: subprocess.Popen | None = None


def start_live_consumer_process() -> subprocess.Popen:
    """Spawn the inference consumer as a sibling process so the dashboard's
    "Replay Episode" button has something turning waveform windows into
    predictions, without embedding model inference inside the API's event loop.

    This is a demo convenience: a real deployment would run the consumer as
    its own service, not something the API process spawns and supervises.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "scripts/kafka_inference_consumer.py",
            "--group-id",
            "dashboard-live-consumer",
            "--offset-reset",
            "latest",
            "--timeout-seconds",
            "0",
            "--output-report",
            "reports/dashboard_live_inference_report.json",
            "--output-alerts",
            "reports/dashboard_live_alerts.jsonl",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kafka_bridge, live_consumer_process
    loop = asyncio.get_running_loop()
    kafka_bridge = KafkaAlertBridge(connection_manager, loop)
    kafka_bridge.start()
    try:
        live_consumer_process = start_live_consumer_process()
    except FileNotFoundError:
        live_consumer_process = None
    try:
        yield
    finally:
        if kafka_bridge is not None:
            kafka_bridge.stop()
        if live_consumer_process is not None:
            live_consumer_process.terminate()
            try:
                live_consumer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                live_consumer_process.kill()


app = FastAPI(title="Predictive Maintenance Power Supply API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def cached_labels() -> pd.DataFrame:
    return load_labels(LABELS_PATH)


@lru_cache(maxsize=1)
def cached_labels_by_sample_id() -> pd.DataFrame:
    return cached_labels().set_index("sample_id")


@lru_cache(maxsize=64)
def cached_waveform(sample_id: int) -> pd.DataFrame:
    return load_waveform(WAVEFORM_DIR, sample_id)


def build_feature_frame(request: WindowPredictionRequest, feature_columns: list[str]) -> pd.DataFrame:
    window = pd.DataFrame(request.channels)
    features: dict[str, float] = {
        "phase_select": float(request.context.phase_select),
        "fault_resistance": float(request.context.fault_resistance),
        "sc_location": float(request.context.sc_location),
    }
    features.update(extract_basic_features(window, list(request.channels.keys())))

    missing_features = [column for column in feature_columns if column not in features]
    frame = pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])
    frame.attrs["missing_features"] = missing_features
    frame.attrs["all_features"] = features
    return frame


def top_explanations(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    limit: int = 5,
) -> list[ExplanationItem]:
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    rows = []
    for feature, importance in zip(feature_columns, importances, strict=True):
        if importance <= 0:
            continue
        value = float(frame.iloc[0][feature])
        rows.append(
            ExplanationItem(
                feature=feature,
                importance=float(importance),
                value=value,
                message=feature_message(feature),
            )
        )
    rows.sort(key=lambda item: item.importance, reverse=True)
    return rows[:limit]


def predict_frame(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[float, bool, list[ExplanationItem]]:
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]
    probability = float(model.predict_proba(frame[feature_columns])[:, 1][0])
    return probability, probability >= threshold, top_explanations(artifact, frame)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int = 25) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()]


def collapse_alert_incidents(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for alert in sorted(alerts, key=lambda item: (item["sample_id"], item["window_index"])):
        if (
            incidents
            and incidents[-1]["sample_id"] == alert["sample_id"]
            and alert["window_index"] <= incidents[-1]["end_window"] + 1
        ):
            incident = incidents[-1]
            incident["end_window"] = alert["window_index"]
            incident["end_time"] = alert["window_end_time"]
            incident["count"] += 1
            incident["max_probability"] = max(incident["max_probability"], alert["probability"])
            continue

        incidents.append(
            {
                "sample_id": alert["sample_id"],
                "start_window": alert["window_index"],
                "end_window": alert["window_index"],
                "start_time": alert["window_start_time"],
                "end_time": alert["window_end_time"],
                "count": 1,
                "max_probability": alert["probability"],
                "true_window_label": alert.get("true_window_label"),
                "top_feature": (alert.get("top_features") or [{}])[0].get("feature"),
            }
        )
    return incidents


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def metric_percent(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100, 2)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ready")
def ready() -> dict[str, str | int | float]:
    try:
        artifact = load_model_artifact()
    except FileNotFoundError:
        return {"status": "not_ready", "reason": f"missing model: {MODEL_PATH}"}
    return {
        "status": "ready",
        "model_path": str(MODEL_PATH),
        "threshold": float(artifact["threshold"]),
        "feature_count": len(artifact["feature_columns"]),
        "kafka_bridge_status": kafka_bridge.status if kafka_bridge else "stopped",
    }


@app.get("/status/stream")
def stream_status() -> dict[str, Any]:
    if kafka_bridge is None:
        return {"status": "stopped", "connected_clients": 0, "messages_relayed": 0}
    return {
        "status": kafka_bridge.status,
        "last_error": kafka_bridge.last_error,
        "messages_relayed": kafka_bridge.messages_relayed,
        "connected_clients": connection_manager.connection_count,
    }


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    await connection_manager.connect(websocket)
    try:
        while True:
            # Dashboard clients don't send anything; this just detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect(websocket)


@app.get("/model/info")
def model_info() -> dict[str, str | int | float | list[str]]:
    try:
        artifact = load_model_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "model_path": str(MODEL_PATH),
        "threshold": float(artifact["threshold"]),
        "feature_count": len(artifact["feature_columns"]),
        "required_features": artifact["feature_columns"],
    }


@app.get("/reports/summary")
def report_summary() -> dict[str, Any]:
    model_report = read_json(REPORTS_DIR / "xgboost_fault_detector_48ch_tuned_recall97_report.json")
    kafka_report = read_json(REPORTS_DIR / "kafka_inference_report.json")
    rare_05 = read_json(
        REPORTS_DIR / "xgboost_fault_detector_48ch_tuned_recall97_on_realistic_0_5pct.json"
    )
    rare_025 = read_json(
        REPORTS_DIR / "xgboost_fault_detector_48ch_tuned_recall97_on_realistic_0_25pct.json"
    )
    test_metrics = model_report["test_metrics"]
    return {
        "best_model": {
            "name": "48-channel XGBoost tuned threshold",
            "path": "models/xgboost_fault_detector_48ch_tuned_recall97.joblib",
            "threshold": model_report["threshold"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_fpr": test_metrics["false_positive_rate"],
            "test_fn": test_metrics["fn"],
            "test_fp": test_metrics["fp"],
        },
        "rare_fault": {
            "fault_0_5pct": rare_05["metrics"],
            "fault_0_25pct": rare_025["metrics"],
        },
        "kafka": kafka_report,
        "display": {
            "test_recall_pct": metric_percent(test_metrics["recall"]),
            "test_precision_pct": metric_percent(test_metrics["precision"]),
            "test_fpr_pct": metric_percent(test_metrics["false_positive_rate"]),
            "rare_0_5_recall_pct": metric_percent(rare_05["metrics"]["recall"]),
            "rare_0_25_recall_pct": metric_percent(rare_025["metrics"]["recall"]),
        },
    }


@app.get("/reports/model-comparison")
def model_comparison() -> list[dict[str, Any]]:
    return read_csv_records(REPORTS_DIR / "model_comparison.csv")


@app.get("/reports/realistic-evaluation")
def realistic_evaluation() -> list[dict[str, Any]]:
    return read_csv_records(REPORTS_DIR / "realistic_eval_comparison.csv")


@app.get("/reports/kafka")
def kafka_report() -> dict[str, Any]:
    return read_json(REPORTS_DIR / "kafka_inference_report.json")


@app.get("/alerts/latest")
def latest_alerts(limit: int = 25) -> list[dict[str, Any]]:
    return read_jsonl(REPORTS_DIR / "kafka_alerts.jsonl", limit=limit)


@app.get("/alerts/incidents")
def alert_incidents(limit: int = 25) -> list[dict[str, Any]]:
    alerts = read_jsonl(REPORTS_DIR / "kafka_alerts.jsonl", limit=500)
    return collapse_alert_incidents(alerts)[-limit:]


@app.get("/waveforms/sample/{sample_id}")
def waveform_sample(sample_id: int = 0, max_points: int = 640) -> dict[str, Any]:
    waveform = cached_waveform(sample_id)
    channels = waveform_channels(waveform)[:48]
    points = min(max_points, len(waveform))
    return {
        "sample_id": sample_id,
        "points": points,
        "time_s": waveform["time_s"].iloc[:points].astype(float).tolist(),
        "channels": {
            channel: waveform[channel].iloc[:points].astype(float).tolist()
            for channel in channels
        },
    }


class KafkaReplayRequest(BaseModel):
    sample_id: int = 0
    max_windows: int | None = 46
    sleep_seconds: float = 0.05


@app.get("/demo/episodes")
def demo_episodes(limit: int = 12) -> list[dict[str, Any]]:
    labels = cached_labels()
    per_type = max(1, limit // max(labels["sc_type"].nunique(), 1))
    picked = labels.groupby("sc_type", group_keys=False).head(per_type).head(limit)
    return [
        {
            "sample_id": int(row["sample_id"]),
            "sc_type": str(row["sc_type"]),
            "fault_target": str(row.get("fault_target", "")),
        }
        for _, row in picked.iterrows()
    ]


@app.post("/demo/kafka-replay")
def trigger_kafka_replay(request: KafkaReplayRequest) -> dict[str, Any]:
    if live_consumer_process is None or live_consumer_process.poll() is not None:
        raise HTTPException(
            status_code=503,
            detail="live inference consumer is not running; replayed windows would not be scored",
        )

    cmd = [
        sys.executable,
        "scripts/kafka_replay_producer.py",
        "--sample-id",
        str(request.sample_id),
        "--sleep-seconds",
        str(request.sleep_seconds),
    ]
    if request.max_windows is not None:
        cmd += ["--max-windows", str(request.max_windows)]

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"failed to start replay: {exc}") from exc

    return {"status": "started", "sample_id": request.sample_id, "max_windows": request.max_windows}


@app.get("/demo/replay/window")
def replay_window(
    sample_id: int = 0,
    window_index: int = 1,
    window_samples: int = 640,
    stride_samples: int = 128,
) -> dict[str, Any]:
    if window_index < 1:
        raise HTTPException(status_code=400, detail="window_index must be >= 1")

    label_row = cached_labels_by_sample_id().loc[sample_id]
    waveform = cached_waveform(sample_id)
    channels = waveform_channels(waveform)[:48]
    max_windows = ((len(waveform) - window_samples) // stride_samples) + 1
    if window_index > max_windows:
        raise HTTPException(status_code=404, detail="window_index beyond available windows")

    start_idx = (window_index - 1) * stride_samples
    end_idx = start_idx + window_samples
    window = waveform.iloc[start_idx:end_idx]
    start_time = float(window.iloc[0]["time_s"])
    end_time = float(window.iloc[-1]["time_s"])
    true_label = window_fault_label(
        start_time,
        end_time,
        float(label_row["t_evnt_start"]),
        float(label_row["t_evnt_end"]),
    )

    request = WindowPredictionRequest(
        channels={
            channel: window[channel].astype(float).tolist()
            for channel in channels
        },
        context=WindowContext(
            phase_select=int(label_row["phase_select"]),
            fault_resistance=float(label_row["fault_resistance"]),
            sc_location=float(label_row["sc_location"]),
        ),
    )
    artifact = load_model_artifact()
    start = time.perf_counter()
    frame = build_feature_frame(request, artifact["feature_columns"])
    probability, is_fault, explanations = predict_frame(artifact, frame)
    latency_ms = (time.perf_counter() - start) * 1000

    display_channels = channels[:8]
    return {
        "sample_id": sample_id,
        "window_index": window_index,
        "max_windows": max_windows,
        "window_start_time": start_time,
        "window_end_time": end_time,
        "true_window_label": true_label,
        "probability": probability,
        "threshold": float(artifact["threshold"]),
        "prediction": "fault" if is_fault else "normal",
        "alert": is_fault,
        "latency_ms": latency_ms,
        "top_features": [item.model_dump() for item in explanations],
        "waveform": {
            "time_s": window["time_s"].astype(float).tolist(),
            "channels": {
                channel: window[channel].astype(float).tolist()
                for channel in display_channels
            },
        },
    }


@app.post("/predict/window", response_model=WindowPredictionResponse)
def predict_window(request: WindowPredictionRequest) -> WindowPredictionResponse:
    try:
        artifact = load_model_artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    start = time.perf_counter()
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]
    frame = build_feature_frame(request, feature_columns)
    probability, is_fault, explanations = predict_frame(artifact, frame)
    latency_ms = (time.perf_counter() - start) * 1000

    return WindowPredictionResponse(
        model_path=str(MODEL_PATH),
        probability=probability,
        threshold=threshold,
        is_fault=is_fault,
        label="fault" if is_fault else "normal",
        latency_ms=latency_ms,
        feature_count=len(feature_columns),
        missing_features=frame.attrs["missing_features"],
        top_features=explanations,
    )
