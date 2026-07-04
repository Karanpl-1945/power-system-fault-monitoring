# Predictive Maintenance for Power-Supply Monitoring

ML pipeline for power-supply / grid-fault monitoring using voltage and current waveforms. The first benchmark dataset is **PROTECT-90**, a labelled high-voltage EMT fault waveform dataset.

The project goal is to train fault-detection and fault-classification models offline, then replay test waveforms through Kafka/MQTT-style streaming to validate real-time prediction latency.

## What This Repository Contains

```text
architecture.md                 # system design
data&model_training_plan.md     # data and model plan
project_workflow.md             # build workflow
project_description.md          # client-style requirement brief
hv_double_line_90kv_labels.csv  # PROTECT-90 metadata/labels
src/                            # Python package
scripts/                        # runnable utilities
configs/                        # project configs
docs/                           # dataset/API notes
tests/                          # tests
```

The large waveform dataset is intentionally ignored by Git:

```text
hv_double_line_90kv_preprocessed_data.zip
hv_double_line_90kv_preprocessed_data/
```

## Dataset

Download and extract the PROTECT-90 waveform archive so the local layout is:

```text
hv_double_line_90kv_labels.csv
hv_double_line_90kv_preprocessed_data/
  0_sample_hv_double_line_90kv.pkl
  1_sample_hv_double_line_90kv.pkl
  ...
```

The original dataset README is preserved at:

```text
docs/datasets/PROTECT90_README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify Dataset

```bash
python -m predictive_maintenance.data.inspect_dataset \
  --labels hv_double_line_90kv_labels.csv \
  --waveform-dir hv_double_line_90kv_preprocessed_data \
  --sample-id 0
```

## Create Train/Validation/Test Split

```bash
python scripts/create_split.py \
  --labels hv_double_line_90kv_labels.csv \
  --output data/splits/protect90_split.csv \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

Split by `sample_id`, never by waveform row/window. This avoids data leakage between training and testing.

## Model Plan

Recommended order:

1. XGBoost / RandomForest baseline on engineered window features.
2. 1D-CNN on raw waveform windows.
3. Temporal CNN / LSTM early-warning model using pre-fault windows.
4. Autoencoder anomaly detector trained on normal/pre-fault windows.

Primary metrics:

```text
Recall >= 95%
False-positive rate < 3%
Streaming latency < 5 seconds
```

## Real-Time Simulation

The trained model will be evaluated in a production-like streaming path:

```text
PROTECT-90 test waveform files
        -> Kafka replay producer
        -> Kafka topic: power.telemetry
        -> streaming consumer
        -> rolling window
        -> model prediction
        -> alert/API response
```

## Important Note

PROTECT-90 is simulated EMT data. It is excellent for pipeline development and reproducible benchmarking, but final production performance on client equipment would require real client logs.

