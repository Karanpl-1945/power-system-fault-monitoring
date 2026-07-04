from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="Predictive Maintenance Power Supply API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}

