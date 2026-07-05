# Streaming Demo

This project uses Kafka as the transport layer, not as the data generator.

Current demo flow:

```text
PROTECT-90 waveform replay
-> Kafka topic: power.waveform.windows
-> XGBoost inference consumer
-> Kafka topics: power.fault.predictions, power.fault.alerts
-> FastAPI/UI can display predictions and alerts
```

Current model:

```text
models/xgboost_fault_detector_48ch_tuned_recall97.joblib
```

## Local Simulation First

Run this without Kafka to validate inference logic:

```bash
python scripts/simulate_streaming_inference.py \
  --sample-id 0 \
  --max-windows 80 \
  --output-report reports/streaming_simulation_report.json \
  --output-alerts reports/streaming_alerts.jsonl
```

## Kafka Demo

Start Kafka:

```bash
docker compose up kafka
```

Run consumer:

```bash
python scripts/kafka_inference_consumer.py \
  --max-messages 46 \
  --timeout-seconds 60 \
  --output-report reports/kafka_inference_report.json \
  --output-alerts reports/kafka_alerts.jsonl
```

Run producer:

```bash
python scripts/kafka_replay_producer.py \
  --sample-id 0 \
  --max-windows 46 \
  --sleep-seconds 0.02
```

Topics:

```text
power.waveform.windows
power.fault.predictions
power.fault.alerts
```

Future option:

After larger GPU training, the Kafka consumer can swap XGBoost for the 1D-CNN raw-window model.
