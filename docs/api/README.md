# API Notes

Planned endpoints:

```text
GET  /health
GET  /ready
POST /predict
POST /predict_batch
GET  /metrics
```

Prediction responses should include:

- prediction label
- confidence
- latency
- top explanation features

