from __future__ import annotations

from pydantic import BaseModel


class TelemetrySample(BaseModel):
    sample_id: int
    timestamp: float
    values: dict[str, float]


class PredictionEvent(BaseModel):
    sample_id: int
    window_start: float
    window_end: float
    prediction: str
    confidence: float
    latency_ms: float

