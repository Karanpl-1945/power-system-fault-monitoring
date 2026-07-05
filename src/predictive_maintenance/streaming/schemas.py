from __future__ import annotations

from pydantic import BaseModel, Field


class TelemetrySample(BaseModel):
    sample_id: int
    timestamp: float
    values: dict[str, float]


class WaveformWindowEvent(BaseModel):
    sample_id: int
    window_index: int
    start_idx: int
    end_idx: int
    window_start_time: float
    window_end_time: float
    true_window_label: str | None = None
    channels: dict[str, list[float]]
    context: dict[str, float | int] = Field(default_factory=dict)


class PredictionEvent(BaseModel):
    sample_id: int
    window_index: int
    window_start_time: float
    window_end_time: float
    prediction: str
    probability: float
    threshold: float
    alert: bool
    latency_ms: float
    true_window_label: str | None = None
    top_features: list[dict[str, float | str]] = Field(default_factory=list)
