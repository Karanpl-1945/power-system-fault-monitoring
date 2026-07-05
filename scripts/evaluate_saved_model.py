from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from predictive_maintenance.models.evaluation import binary_metrics


NON_FEATURE_COLUMNS = {
    "sample_id",
    "start_idx",
    "end_idx",
    "start_time",
    "end_time",
    "window_label",
    "binary_target",
    "sc_type",
    "fault_target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model artifact on feature CSV data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    data = pd.read_csv(args.data)
    x = data[feature_columns]
    y = data["binary_target"].astype(int).to_numpy()

    scores = model.predict_proba(x)[:, 1]
    pred = (scores >= threshold).astype(int)
    metrics = binary_metrics(y, pred)
    metrics["alerts_per_10000_windows"] = float(metrics["false_positive_rate"] * 10000)
    metrics["fault_ratio"] = float(data["binary_target"].mean())
    metrics["rows"] = int(len(data))

    report = {
        "model_path": args.model,
        "data_path": args.data,
        "threshold": threshold,
        "metrics": metrics,
    }

    if args.output is None:
        model_name = Path(args.model).stem
        data_name = Path(args.data).stem
        output = Path("reports") / f"{model_name}_on_{data_name}.json"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_report={output}")


if __name__ == "__main__":
    main()
