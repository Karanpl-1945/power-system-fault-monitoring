# API Notes

Implemented endpoints:

```text
GET  /
GET  /health
GET  /ready
GET  /model/info
POST /predict/window
GET  /reports/summary
GET  /reports/model-comparison
GET  /reports/realistic-evaluation
GET  /reports/kafka
GET  /alerts/latest
GET  /waveforms/sample/{sample_id}
```

Current production candidate:

```text
models/xgboost_fault_detector_48ch_tuned_recall97.joblib
```

`POST /predict/window` accepts one rolling waveform window as JSON:

```json
{
  "channels": {
    "Bus_1_Line_01_02A_cur_L1_A": [1.0, 2.0, 3.0],
    "Bus_1_Line_01_02A_cur_L2_A": [1.0, 2.0, 3.0]
  },
  "context": {
    "phase_select": -1,
    "fault_resistance": 0.0,
    "sc_location": -1.0
  }
}
```

All channel arrays must have the same length. For the current 48-channel model, send the same 48 waveform channels used during feature building.

Prediction responses include:

- fault probability
- configured threshold
- predicted label
- latency
- missing feature list
- top feature-importance messages

Run locally:

```bash
uvicorn predictive_maintenance.serving.app:app --reload
```

Dashboard:

```text
http://localhost:8000/
```

The dashboard uses HTTP requests from the browser to load model reports, Kafka alert files, and waveform samples.

Important deployment note:

The current trained XGBoost artifact expects three context features:

- `phase_select`
- `fault_resistance`
- `sc_location`

For demos, these can use safe default values. For a real industrial deployment, retraining without non-live context fields may be better unless those values are available from the system.
