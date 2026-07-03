# Predictive Maintenance for Power-Supply Monitoring — Architecture & Delivery Plan

**Status:** Draft v1  
**Owner:** Madhwansh  
**Purpose:** Single source of truth for the system design and the ordered steps to take it from prototype to production.

## 1. Problem Statement

Build an ML module that monitors incoming power-supply telemetry (voltage, current, frequency, power, etc.), learns normal vs. failing behaviour from historical logs, and flags incipient and critical faults in advance through a clean, documented service that the client's existing control software can integrate directly.

The phrase **in advance** is load-bearing. This is not only "classify the current sample as faulty/healthy"; it is "detect the degradation trajectory before the hard failure." That distinction drives the data strategy (Section 8) and the modelling approach (Section 5).

## 2. Goals & Acceptance Criteria

Every criterion below maps to a test we will ship. Do not consider a phase "done" until its criteria are green in CI.

| ID | Requirement | Target | Verified by |
| --- | --- | --- | --- |
| AC-1 | End-to-end pipeline processes both historical (batch) and live (stream) data | Works for both paths | Integration test |
| AC-2 | Real-time stream latency (ingest → prediction available) | < 5 s | Latency benchmark (p50/p95/p99) |
| AC-3 | Fault-detection recall on held-out validation set | ≥ 95% | Validation report |
| AC-4 | False-positive rate | < 3% | Validation report |
| AC-5 | Container cold-start | < 30 s | Startup benchmark |
| AC-6 | All included tests pass on a clean machine | 100% pass | CI on fresh runner + `docker compose up` |
| AC-7 | Predictions explainable to operators | Per-alert feature attributions | Explainability test + demo |

**Note on AC-3/AC-4:** recall and false-positive rate trade off against each other and both depend entirely on the decision threshold. We will report the full precision-recall curve and pick the operating threshold that satisfies both simultaneously, then lock it. If the data cannot hit both at once, that is a data/quality finding to raise with the client early — not something to hide.

## 3. Guiding Principles

- **Data-source agnostic core.** The model and serving layers must not know or care whether a sample came from Kafka, MQTT, a CSV upload, or a replay simulator. Everything enters through one normalized schema (Section 4.2). This is what lets us start on public/synthetic data and swap in the client's real logs later with zero model rewrites.
- **The ML is not the hard part; the system around it is.** A well-tuned anomaly detector is a solved problem. The differentiation, and most of the acceptance criteria, live in ingestion, latency, packaging, health/failover, and explainability. Weight effort accordingly.
- **Reproducibility over cleverness.** Every number we claim (recall, latency, startup time) must come from a script anyone can re-run. Pin dependencies, seed randomness, version the data.
- **Fail loud, degrade gracefully.** A monitoring system that silently stops monitoring is worse than useless. Health checks, heartbeats, and a defined behaviour when a dependency is down are first-class requirements, not afterthoughts.

## 4. High-Level Architecture

```text
                        ┌──────────────────────────────────────────────┐
   Live feed            │                  INGESTION LAYER             │
   (Kafka / MQTT) ─────▶│  - Kafka consumer / MQTT subscriber          │
                        │  - Schema validation + normalization         │
   Batch uploads  ─────▶│  - Batch loader (CSV/Parquet historical logs)│
   (historical logs)    │  - Dead-letter queue for bad records         │
                        └───────────────────┬──────────────────────────┘
                                            │ normalized samples
                                            ▼
                        ┌──────────────────────────────────────────────┐
                        │               FEATURE LAYER                  │
                        │  - Rolling-window feature extraction         │
                        │  - Online (stream) + offline (batch) parity  │
                        │  - Feature store (fast read: Redis/Feast)    │
                        │  - Raw + features persisted (Parquet/TSDB)   │
                        └───────────┬─────────────────────┬────────────┘
                                    │                     │
                     (training)     ▼                     ▼   (inference)
              ┌───────────────────────────┐   ┌───────────────────────────┐
              │      TRAINING PIPELINE    │   │       SERVING LAYER       │
              │  - Preprocess + split     │   │  - Model loaded in memory │
              │  - Train classifier /     │   │  - gRPC + REST endpoints  │
              │    anomaly detector       │   │  - Explainability (SHAP)  │
              │  - Threshold selection    │   │  - /health, /ready,       │
              │  - Validation report      │   │    /metrics               │
              │  - Model registry +       │──▶│  - Graceful failover      │
              │    version                │   │  - Loads model from       │
              │                           │   │    registry               │
              └───────────────────────────┘   └─────────────┬─────────────┘
                                                            │ predictions +
                                                            │ explanations
                                                            ▼
                                              Client control software / alert bus
                        ┌──────────────────────────────────────────────┐
                        │        OBSERVABILITY (cross-cutting)         │
                        │  Prometheus metrics · structured logs ·      │
                        │  latency histograms · drift monitors         │
                        └──────────────────────────────────────────────┘
```

### 4.1 Component Responsibilities

- **Ingestion layer:** one adapter per source (Kafka consumer, MQTT subscriber, batch file loader). Each adapter's only job is to pull raw records, validate them against the schema, normalize units, and push valid samples onward. Malformed records go to a dead-letter queue; they never crash the pipeline.
- **Feature layer:** computes features (statistical rolling windows, rate-of-change, frequency-domain features via FFT, etc.). Critically, the same feature code must run in both the batch (training) and streaming (inference) paths to avoid train/serve skew. A feature store (Redis or Feast) provides low-latency reads for online inference.
- **Training pipeline:** offline. Consumes historical features, trains the model, selects the operating threshold against `AC-3`/`AC-4`, produces the validation report, and writes a versioned artifact to the model registry.
- **Serving layer:** loads the current model at startup, exposes gRPC + REST, returns a prediction plus a per-alert explanation, and reports health. This is the thing the client integrates against.
- **Observability:** Prometheus-style metrics, structured JSON logs, latency histograms, and data-drift monitors run across every layer.

### 4.2 The Schema Contract

Define one canonical sample schema early and treat it as a contract. Everything upstream normalizes into it; everything downstream reads from it.

```yaml
NormalizedSample:
  device_id: str
  timestamp: int64   # epoch ms, UTC
  voltage: float     # volts
  current: float     # amperes
  frequency: float   # Hz
  power: float       # kW (derived if absent)
  extra_sensor_channels: ...
  meta:
    source: "kafka|mqtt|batch|replay"
    ingest_ts: int64
```

When the client's real data arrives, the only new work is a mapping from their field names/units into this schema. The model never changes. This single decision is what makes the "we have no data yet" problem tractable.

## 5. Modelling Approach

Model in two complementary layers so we satisfy both "flags incipient faults" and "flags critical faults":

- **Anomaly detector (unsupervised / semi-supervised):** learns the shape of normal operation and scores how far each windowed sample deviates. Good for catching novel/incipient degradation we have few or no labelled examples of. Candidates: Autoencoder / LSTM-Autoencoder reconstruction error, Isolation Forest as a fast baseline.
- **Supervised classifier (when labels exist):** for known fault categories (line-to-ground, line-to-line, overload, short circuit, etc.). Candidates: gradient-boosted trees (strong tabular baseline, fast, naturally explainable) and a 1D-CNN or LSTM for the time-series/deep-learning requirement.

**Framework:** PyTorch or TensorFlow for the deep models (spec allows either; I would lean PyTorch for iteration speed). Keep a scikit-learn / XGBoost baseline — it is often within a few points of the deep model, trains in seconds, and gives you a credible floor and a sanity check.

Class imbalance is the central ML challenge here. Failures are rare, which is exactly why recall (`AC-3`) matters and why plain accuracy is a trap. Plan for: stratified splits, class weighting / focal loss, threshold tuning on the PR curve (not 0.5), and possibly SMOTE-style resampling on the training set only. Never resample the validation/test set — it must reflect the real failure rarity.

Evaluation must be leakage-free. For time-series, split by time or by device, never by shuffling rows; otherwise windows from the same event leak across train/test and your 95% recall is a lie. Report precision, recall, F1, PR-AUC, and a confusion matrix per fault class in the validation report.

### 5.1 Explainability (AC-7)

Operators must trust the alert, so every prediction ships with why. Use SHAP for the tree/tabular models (fast, exact for trees) and SHAP or Integrated Gradients for the deep models. The serving response includes the top-N contributing features for that specific sample (e.g. "voltage sag on phase B + rising current rate"). This is a genuine differentiator — most portfolio/prototype systems skip it — and it is explicitly required.

## 6. Serving & Deployment

- **API:** expose both gRPC (low-latency, for the control software) and REST (easy integration/testing). Define the gRPC contract in a `.proto` file. Include single prediction and batch prediction endpoints.
- **Health & readiness:** `/health` (liveness), `/ready` (model loaded + dependencies reachable), `/metrics` (Prometheus). Kubernetes/Docker uses these for restart and traffic decisions.
- **Graceful failover:** define explicit behaviour when a dependency (feature store, broker) is unavailable; for example, fall back to computing features from the raw sample in-request, serve last-known-good model, and surface a degraded-mode flag rather than 500-ing. Never drop silently.
- **Packaging:** multi-stage Dockerfile, model baked into the image or pulled at startup from the registry. Target cold-start < 30 s (`AC-5`) — this constrains image size and lazy-vs-eager model loading, so benchmark it.
- **Orchestration:** `docker compose` for local/on-prem single-node; Kubernetes manifests (Deployment + Service + probes + HPA) for cloud. Both must be provided per the spec.

## 7. Repository Structure

```text
predictive-maintenance/
├── architecture.md              # this file
├── README.md                    # setup guide (AC docs requirement)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml               # pinned deps (uv-friendly)
├── proto/
│   └── prediction.proto
├── src/
│   ├── ingestion/               # kafka, mqtt, batch adapters + schema
│   ├── features/                # shared feature code (batch + stream)
│   ├── training/                # train, threshold-select, validation report
│   ├── serving/                 # gRPC + REST app, health, explainability
│   └── common/                  # schema, config, logging
├── data/
│   ├── raw/ processed/ synthetic/  # gitignored; DVC or pointers
│   └── replay/                  # dataset→stream simulator
├── models/                      # registry / versioned artifacts
├── notebooks/                   # EDA only, not the source of truth
├── tests/
│   ├── unit/ integration/ benchmarks/
└── k8s/                         # deployment manifests + probes
```

## 8. Data Strategy

This is phased so we are never blocked. See the companion notes for dataset links and the exact acquisition steps.

### Phase A — Public Benchmark Data (start here, day one)

Use real, published power-telemetry datasets to build and validate the entire pipeline:

- **Electrical Fault Detection and Classification:** line voltages & currents labelled by fault type (line-to-ground, line-to-line, etc.). Directly matches the "voltage/current → fault class" supervised task.
- **Smart Grid Monitoring:** time-series voltage, current, frequency, power + fault indicator; includes FFT features. Good for the streaming/time-series path.
- **NASA C-MAPSS (turbofan run-to-failure):** the canonical degradation-over-time dataset. Not power-supply per se, but it is how we prove the "predict failure in advance" capability (remaining-useful-life / early-warning), which the pure fault-classification sets do not cover.

### Phase B — Synthetic Generation

Write a generator that simulates power-supply behaviour and injects faults with controllable timing, severity, and rarity. This gives us:

- Control over class balance to stress-test `AC-3`/`AC-4`.
- Labelled degradation trajectories for the "in advance" requirement.
- Unlimited volume for the latency benchmark.

### Phase C — Live-Stream Simulation

The client has no live Kafka/MQTT feed yet, so build a replay producer that reads a dataset (public or synthetic) and publishes it into Kafka/MQTT at realistic rates, with configurable timestamps. This exercises the exact real-time code path and is how we measure the < 5 s latency. When the real feed appears, we point the consumer at it and delete nothing.

### Phase D — Client's Real Historical Logs

Because everything normalizes into the schema contract (4.2), onboarding the client's real data is just: write a field/unit mapping → validate → retrain → re-run the validation report. No architectural change.

**Framing for the client:** we can deliver a fully working, benchmarked prototype on public + synthetic data immediately, then re-validate on their historical logs the moment they are shared. This de-risks the timeline and lets us commit to milestones now.

## 9. Testing & Metrics

- **Unit tests:** schema validation, feature functions (batch/stream parity), adapters, threshold logic.
- **Integration tests:** full path: replay producer → ingest → features → model → API response (covers `AC-1`).
- **Benchmarks:** latency (p50/p95/p99, must show p95 < 5 s for `AC-2`), throughput, and container cold-start (`AC-5`).
- **Validation report:** generated artifact (Markdown/HTML) with PR curves, per-class confusion matrices, the chosen threshold, and the final recall/FPR numbers (`AC-3`/`AC-4`).
- **CI:** runs the full suite on a clean runner and does a `docker compose up` smoke test so `AC-6` cannot silently regress.

## 10. Milestones

| # | Milestone | Exit criteria |
| --- | --- | --- |
| 0 | Repo scaffold, schema contract, CI skeleton, pinned deps | `docker compose up` runs an empty service; CI green |
| 1 | Ingestion + batch loader + dead-letter + tests | Public dataset loads & validates end-to-end |
| 2 | Feature layer (batch+stream parity) + feature store | Same features from both paths on identical input |
| 3 | Baseline model (XGBoost) + validation report | Report generated; baseline recall/FPR recorded |
| 4 | Deep model (LSTM-AE / 1D-CNN) + threshold tuning | Meets `AC-3` (≥95% recall) & `AC-4` (<3% FP) on held-out set |
| 5 | Explainability wired into predictions | Per-alert top-N features returned (`AC-7`) |
| 6 | Serving layer: gRPC+REST, health, graceful failover | Endpoints live; failover behaviour tested |
| 7 | Replay producer → live path + latency benchmark | p95 latency < 5 s (`AC-2`); `AC-1` integration test green |
| 8 | Containerization + K8s manifests + startup benchmark | Cold-start < 30 s (`AC-5`); `AC-6` all tests pass clean |
| 9 | Docs: README setup guide + API reference | A stranger can integrate from docs alone |
| 10 | Re-validate on client's real logs (when provided) | Validation report re-run on real data |

## 11. Risks & Mitigations

- **Public data will not perfectly match the client's failure modes.** Treat Phase A/B numbers as pipeline-validation, not final performance; re-baseline in Milestone 10.
- **95% recall + <3% FP may be infeasible on some data.** Surface the PR curve to the client early; it may be a data-quality/labelling conversation, not a modelling failure.
- **Train/serve skew from duplicated feature code.** Enforce shared feature module (Milestone 2 exit criterion).
- **Time-series leakage inflating metrics.** Split by time/device; code-review the split logic specifically.
- **Latency budget blown by feature computation or model size.** Benchmark from Milestone 7; keep the XGBoost baseline as a fast fallback path.

This document is the plan of record. Update the status line and milestone table as phases complete.
