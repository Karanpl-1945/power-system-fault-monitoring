# Data & Model-Training Plan — Predictive Maintenance for Power-Supply Monitoring

**Status:** Draft v1  
**Owner:** Madhwansh  
**Companion to:** [architecture.md](architecture.md)  
**Purpose:** Define exactly what data feeds the system, in what order, and how each model is trained, split, and validated against the acceptance criteria.

## 0. The One Idea That Makes This Work

Every data source normalizes into a single schema contract (`NormalizedSample`: `device_id`, `timestamp`, `voltage`, `current`, `frequency`, `power`, ...). The models never know whether a sample came from a public dataset, the synthetic generator, or the client's real logs.

This decoupling is why "we have no client data yet" is not a blocker. We build and validate the whole pipeline on public + synthetic data now, and swap in the client's real logs at the end with zero model rewrites.

**Consequence to keep repeating to the client:** public/synthetic data validates the pipeline; only the client's real logs can close the performance criteria (`AC-3` recall, `AC-4` false-positive rate) for production.

## Part 1 — Data Plan

### 1.1 Acceptance-Criteria Coverage by Data Source

Which criteria each data source can actually verify:

| Criterion | Public data | Synthetic data | Client real logs |
| --- | --- | --- | --- |
| AC-1 — batch + live pipeline | ✅ full | ✅ full | ✅ final |
| AC-2 — < 5 s stream latency | ✅ full (via replay) | ✅ full | ✅ final |
| AC-3 — ≥ 95% recall | ⚠️ pipeline only | ⚠️ stress-test | ✅ closes it |
| AC-4 — < 3% false positives | ⚠️ pipeline only | ⚠️ stress-test | ✅ closes it |
| AC-5 — cold-start < 30 s | ✅ full | ✅ full | ✅ final |
| AC-6 — tests pass clean | ✅ full | ✅ full | ✅ final |
| AC-7 — explainability | ✅ full | ✅ full | ✅ final |
| "Predict faults in advance" | ⚠️ weak (only C-MAPSS) | ✅ primary source | ✅ final |

**Legend:** ✅ genuinely satisfied · ⚠️ demonstrable but not final.

Read this table honestly: public data gets almost everything working and demonstrable. The two things it cannot do are:

- Prove the recall/FP numbers on the client's real problem.
- Teach genuine early-warning on power-supply data.

Synthetic data covers early warning and stress-tests recall/FP. The client's real logs close final production performance.

### 1.2 The Five Data Stages

#### Stage 1 — Electrical Fault Detection & Classification (public, start here)

**What it is:** line voltages and currents labelled by fault condition. Binary detection set (~12k rows) + multi-class set (~7.8k rows) across six categories: no-fault, line-to-ground, line-to-line, line-to-line-to-ground, three-phase, three-phase-to-ground.

**Why first:** most direct match for "voltage/current → fault type"; lets us stand up the supervised path and produce the first validation report fast.

**Feeds:** the supervised classifier (Part 2, Model A).

**Caveat:** current-state labels only — it classifies the present sample, it does not predict a future failure. Published results are near-trivially high (99%+), so treat its numbers as pipeline validation, not achievement.

#### Stage 2 — Smart Grid Monitoring (public)

**What it is:** continuous time-series of voltage, current, frequency, and power with a fault indicator, plus 128 FFT frequency-domain features.

**Why:** exercises the streaming/time-series feature path and the healthy-vs-faulty separation over time.

**Feeds:** the anomaly detector's normal training data (Model B) and the streaming feature layer.

**Caveat:** fault classes are near-balanced by construction — unrealistic rarity; do not tune final thresholds on it.

#### Stage 3 — NASA C-MAPSS (turbofan run-to-failure) (public)

**What it is:** the canonical run-to-failure degradation dataset — multivariate sensor trajectories from healthy through to failure, with remaining-useful-life ground truth.

**Why:** it is the only public source with genuine degradation-over-time, which is how we demonstrate the spec's core "predict failure in advance" requirement.

**Feeds:** the degradation / early-warning model (Model C) as a technique proof.

**Caveat:** turbofan engines, not power supplies. The method transfers; the data does not. The power-supply-shaped version of this is trained on synthetic trajectories (Stage 4).

#### Stage 4 — Synthetic Generator (runs alongside Stages 1–3; not optional)

**What it is:** a generator we write that simulates voltage/current/frequency signals and injects faults with controllable timing, severity, and rarity.

**Why it is essential — three things public data cannot give:**

- Controllable rarity. Real failures are rare; that is why recall matters. We set the failure rate to realistic low values to stress-test whether `AC-3` (≥95% recall) and `AC-4` (<3% FP) can hold simultaneously.
- Labelled degradation trajectories on power-supply-shaped signals — the "in advance" capability, without waiting on the client.
- Unlimited volume for the latency and throughput benchmarks.

**Feeds:** all three models:

- Labelled faults → classifier.
- Normal signals → anomaly detector.
- Trajectories → degradation model.

#### Stage 5 — Replay Producer + Client's Real Logs

**Replay producer:** a small service that reads any dataset from Stages 1–4 and publishes rows into Kafka/MQTT at realistic timestamps. This exercises the exact real-time code path and is how `AC-1` (batch + live) and `AC-2` (<5 s latency) are proven without a real feed.

**Client real logs:** when provided, the only new work is a field/unit mapping into the schema contract → retrain → re-run the validation report. No architectural change. This is the stage that closes `AC-3` and `AC-4` for production.

### 1.3 Data Handling Rules

Apply these rules to every stage:

- Normalize every source into `NormalizedSample` before it touches features or models.
- Persist raw + computed features (Parquet / time-series store) so training is reproducible.
- Malformed records → dead-letter queue, never crash the pipeline.
- Version datasets (DVC or content-hash pointers); pin the exact snapshot used for each reported number.
- Never resample or "clean" the validation/test data in a way that changes the failure rate.

### 1.4 Access Note

Kaggle datasets require a free account and the Kaggle API/CLI to download. Store credentials outside the repo; scripts read them from environment/config, never hard-coded.

## Part 2 — Model Training Plan

### 2.1 This Is Three Models, Not One

The spec asks for three distinct capabilities. Each trains differently on different data. "Which data" only has an answer per-model.

| Model | Type | Trained on | Purpose | Milestone |
| --- | --- | --- | --- | --- |
| A. Fault classifier | Supervised (XGBoost → 1D-CNN/LSTM) | Electrical Fault (labelled) + synthetic labelled | Known fault categories | 3 → 4 |
| B. Anomaly detector | Semi-supervised (LSTM-autoencoder; Isolation Forest baseline) | Healthy data only (Smart Grid normal + synthetic normal) | Incipient / novel faults | 4 |
| C. Degradation model | Sequence / RUL (LSTM/GRU regression or trajectory) | C-MAPSS (technique) + synthetic trajectories | Predict failure in advance | 4 |

### 2.2 Model A — Supervised Fault Classifier

**Data:** Stage 1 labelled voltage/current, augmented with synthetic labelled faults.

**Approach:** start with XGBoost — trains in seconds, strong tabular baseline, natively explainable via SHAP, gives an honest performance floor. Then a 1D-CNN or LSTM for the deep-learning requirement and to capture temporal structure.

**Why first:** fastest route to a runnable end-to-end result and the first validation report (Milestone 3). Never skip the baseline — a deep model that cannot beat XGBoost is a signal something is wrong.

### 2.3 Model B — Anomaly Detector (the one that matters most)

**Data:** normal / healthy data ONLY. This is the critical, easily-missed point.

**How it works:** an LSTM-autoencoder learns to reconstruct normal operation. On a degrading or novel signal it reconstructs poorly, reconstruction error spikes, and that error is the anomaly score. Faults are never shown during training.

**Why it is the priority:** the client will supply few labelled failures. Semi-supervised training on abundant normal data sidesteps that scarcity and catches incipient/novel faults a supervised classifier would miss.

**Baseline:** Isolation Forest — fast sanity check before the deep model.

### 2.4 Model C — Degradation / Early-Warning Model

**Data:** C-MAPSS to prove the technique; synthetic power-supply trajectories for the real target signal.

**Approach:** sequence model predicting remaining-useful-life or a "time-to-failure below threshold" early-warning flag.

**Why separate:** Models A and B judge the current window; only C answers "is a failure coming," which is the literal wording of the spec.

### 2.5 Training Mechanics — Shared Across All Three

Get these right or the metrics lie.

#### Splitting — Do This Before Touching Any Model

- Three splits: train (fits the model), validation (tunes threshold, hyperparameters, early stopping), and test (touched exactly once, produces reported numbers).
- Split by time or by device — NEVER shuffle rows. Time-series windows from the same fault event leaking across train/test is the #1 cause of fake high recall. Code-review the splitter specifically for this.

#### Class Imbalance — Handle in Training Only

- Use class weights or focal loss; optionally SMOTE-style resampling.
- Apply resampling to the training split only. Validation and test keep the true failure rarity, or the recall/FP numbers become meaningless.

#### Threshold Selection — Separate from Fitting Weights

- After training, pick the decision threshold on the validation set's precision-recall curve. This is the knob that trades `AC-3` (recall) against `AC-4` (false positives).
- Lock the threshold once both targets are satisfied simultaneously, then evaluate once on the untouched test set.
- If both cannot be satisfied at once on the data, that is a data-quality finding to raise with the client early — not something to bury.

#### Reproducibility

- Seed all randomness, pin dependencies, version the exact data snapshot.
- Every reported number must come from a re-runnable script.

### 2.6 Explainability (AC-7) — Trained/Attached per Model

- SHAP for XGBoost/tabular (exact and fast for trees).
- SHAP or Integrated Gradients for the deep models.
- Serving returns the top-N contributing features per alert (e.g. "voltage sag phase B + rising current rate"), so operators trust and can act on the flag.
