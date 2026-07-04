# Project Workflow — Predictive Maintenance / Fault Detection

This workflow explains how to use the PROTECT-90 dataset for model training, validation, testing, and real-time prediction simulation using Kafka.

## 1. Dataset Status

We currently have:

```text
README.md
hv_double_line_90kv_labels.csv
```

The CSV file contains the labels and metadata for each fault episode.

To train the actual voltage/current model, we also need to download and extract:

```text
hv_double_line_90kv_preprocessed_data.zip
```

After extraction, the folder should look like:

```text
preprocessed_data/
  0_sample_hv_double_line_90kv.pkl
  1_sample_hv_double_line_90kv.pkl
  2_sample_hv_double_line_90kv.pkl
  ...
```

Each `.pkl` file contains one 1-second waveform episode with:

```text
time_s
V_A, V_B, V_C, I_A, I_B, I_C at multiple locations
```

The label file tells us:

```text
sample_id
sc_type
fault_target
sc_location
fault_resistance
t_evnt_start
t_evnt_end
```

## 2. What We Can Train

Because the dataset is labelled, we should start with a supervised model.

Recommended first task:

```text
Input: voltage/current waveform window
Output: fault type
Target column: sc_type
```

Other useful tasks:

| Task | Input | Target |
| --- | --- | --- |
| Fault type classification | Voltage/current waveform | `sc_type` |
| Faulted line classification | Voltage/current waveform | `fault_target` |
| Fault location prediction | Voltage/current waveform | `sc_location` |
| Fault detection | Window before/during fault | normal/fault label from `t_evnt_start`, `t_evnt_end` |

The first model should focus on:

```text
fault detection + fault type classification
```

## 3. Train / Validation / Test Split

Split by `sample_id`, not by waveform rows.

Correct split:

```text
70% sample_ids -> train
15% sample_ids -> validation
15% sample_ids -> test
```

Why this matters:

Each `sample_id` is one complete fault episode. If we split rows or windows randomly, the model may see part of the same fault during training and another part during testing. That causes data leakage and fake-high accuracy.

Recommended split steps:

1. Load `hv_double_line_90kv_labels.csv`.
2. Shuffle `sample_id` values with a fixed random seed.
3. Assign each `sample_id` to train, validation, or test.
4. Save the split file:

```text
data/splits/protect90_split.csv
```

Example split file:

```text
sample_id,split
0,train
1,train
2,val
3,test
```

## 4. Window Creation

Each waveform episode has 1 second of data sampled at 6.4 kHz:

```text
6400 rows per episode
```

Use sliding windows to create model inputs.

Example window sizes:

```text
1 cycle at 50 Hz  = 20 ms  = 128 samples
5 cycles          = 100 ms = 640 samples
10 cycles         = 200 ms = 1280 samples
```

Recommended first choice:

```text
5-cycle window = 640 samples
```

For each window:

- Read voltage/current waveform values.
- Assign label using `t_evnt_start` and `t_evnt_end`.

Label logic:

```text
window before t_evnt_start        -> normal / pre-fault
window overlaps fault interval    -> fault
window after t_evnt_end           -> post-fault
```

For fault type classification:

```text
if window overlaps fault interval:
    label = sc_type
else:
    label = normal
```

## 5. Feature Strategy

Start simple, then move to deep learning.

### Option A — Feature-Based Baseline

Extract features from each window:

```text
mean
standard deviation
min
max
RMS
peak-to-peak
zero-crossing rate
phase imbalance
current spike ratio
voltage sag ratio
```

Train:

```text
XGBoost / Random Forest / LightGBM
```

Why:

- Fast to train.
- Good first benchmark.
- Easier to explain.
- Works well for tabular features.

### Option B — Deep Learning Model

Use raw waveform windows directly.

Input shape:

```text
window_samples x channels
```

Example:

```text
640 x 6
```

or if using all 8 locations:

```text
640 x 48
```

Recommended deep model:

```text
1D-CNN
```

Why 1D-CNN first:

- Good for waveform pattern detection.
- Faster than LSTM.
- Works well on voltage/current time-series.
- Easier to deploy for low-latency inference.

Later model:

```text
LSTM / GRU / Transformer
```

Use these only if the 1D-CNN baseline is not enough.

## 6. Recommended Model Order

Build models in this order:

### Model 1 — XGBoost Baseline

```text
Input: engineered window features
Output: normal / fault or fault type
```

Purpose:

- Quick sanity check.
- Gives first validation report.
- Helps understand important features.

### Model 2 — 1D-CNN

```text
Input: raw voltage/current waveform window
Output: normal / fault or fault type
```

Purpose:

- Main waveform model.
- Good fit for high-frequency voltage/current signals.
- Suitable for real-time inference.

### Model 3 — Anomaly Detector

```text
Input: normal/pre-fault windows only
Output: anomaly score
```

Options:

```text
Isolation Forest
LSTM Autoencoder
CNN Autoencoder
```

Purpose:

- Detect unknown fault patterns.
- Useful when real labels are missing.

## 7. Training Workflow

Training should be offline.

Full training path:

```text
Download dataset
        ↓
Extract waveform .pkl files
        ↓
Load labels CSV
        ↓
Create train/val/test split by sample_id
        ↓
Create waveform windows
        ↓
Extract features or prepare raw window tensors
        ↓
Train model on train split
        ↓
Tune threshold/hyperparameters on validation split
        ↓
Evaluate once on test split
        ↓
Save model artifact
```

Artifacts to save:

```text
models/fault_classifier.pkl
models/fault_classifier.pt
models/scaler.pkl
reports/validation_report.md
data/splits/protect90_split.csv
```

## 8. Evaluation Metrics

Report these metrics:

```text
accuracy
precision
recall
F1-score
false-positive rate
confusion matrix
latency
```

For the project acceptance criteria, the most important metrics are:

```text
recall >= 95%
false-positive rate < 3%
stream latency < 5 seconds
```

Important note:

These metrics are valid for the PROTECT-90 simulated dataset. They should not be presented as guaranteed production performance on real client equipment.

## 9. Real-Time Prediction with Kafka

After offline training, use Kafka to simulate live industrial data.

Kafka pipeline:

```text
PROTECT-90 test waveform files
        ↓
Kafka replay producer
        ↓
Kafka topic: power.telemetry
        ↓
Streaming consumer
        ↓
Window buffer
        ↓
Feature extraction
        ↓
Model prediction
        ↓
Alert / REST API / logs
```

## 10. Kafka Components

### 10.1 Replay Producer

Reads `.pkl` waveform files and sends each row as a Kafka message.

Example message:

```json
{
  "sample_id": 123,
  "timestamp": 0.125,
  "V_A": 12.4,
  "V_B": -6.1,
  "V_C": -6.3,
  "I_A": 0.52,
  "I_B": -0.21,
  "I_C": -0.31
}
```

This simulates real sensor streaming.

### 10.2 Kafka Topic

Recommended topic:

```text
power.telemetry
```

This topic carries live voltage/current readings.

### 10.3 Streaming Consumer

The consumer:

1. Reads messages from Kafka.
2. Groups messages by `sample_id` or `device_id`.
3. Maintains a rolling window, for example 640 samples.
4. Runs feature extraction.
5. Sends the window to the trained model.
6. Emits prediction.

### 10.4 Prediction Output

Example output:

```json
{
  "sample_id": 123,
  "window_start": 0.120,
  "window_end": 0.220,
  "prediction": "fault",
  "fault_type": 2,
  "confidence": 0.97,
  "latency_ms": 42
}
```

## 11. REST / gRPC Service

The trained model should be wrapped in a service.

Recommended endpoints:

```text
GET  /health
GET  /ready
POST /predict
POST /predict_batch
GET  /metrics
```

For Kafka real-time mode:

```text
Kafka consumer calls model directly
or
Kafka consumer calls POST /predict
```

For client integration:

```text
Client control software can call REST/gRPC API
```

## 12. Final Build Order

Build the project in this order:

1. Download and extract full PROTECT-90 waveform data.
2. Verify that `.pkl` files match `sample_id` values in the label CSV.
3. Create train/validation/test split by `sample_id`.
4. Create windowing code.
5. Train XGBoost baseline using engineered features.
6. Train 1D-CNN using raw waveform windows.
7. Compare validation/test metrics.
8. Save the best model.
9. Build REST prediction service.
10. Build Kafka replay producer.
11. Build Kafka streaming consumer.
12. Run test episodes through Kafka.
13. Measure real-time latency.
14. Generate validation and latency report.
15. Package with Docker.

## 13. What This Proves

This workflow proves:

- The model can learn from voltage/current fault waveforms.
- The pipeline can process both batch and live-style data.
- Kafka can simulate real-time industrial telemetry.
- The service can produce predictions within the required latency.
- The project can be demonstrated without client data.

## 14. What This Does Not Prove

This does not fully prove:

- Final production recall on real client equipment.
- Final false-positive rate in a real plant.
- Long-term degradation prediction before a fault develops over days/weeks.

For that, real client logs or additional synthetic degradation data are needed.

## 15. Short Summary

Use the labelled PROTECT-90 dataset like this:

```text
Train offline using train/val/test split.
Validate model accuracy and false positives.
Replay test waveforms through Kafka.
Run real-time window prediction.
Expose prediction through REST/gRPC.
Report model metrics and latency.
```

Best first model:

```text
XGBoost baseline -> 1D-CNN main model
```

Best split:

```text
70% train / 15% validation / 15% test by sample_id
```

Best real-time approach:

```text
Kafka replay producer + streaming consumer + trained model service
```
