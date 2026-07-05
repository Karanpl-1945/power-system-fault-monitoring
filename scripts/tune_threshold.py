from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictive_maintenance.models.evaluation import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a saved classifier threshold using validation data only."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--min-recall", type=float, default=0.96)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument(
        "--model-output",
        default="models/xgboost_fault_detector_48ch_tuned_threshold.joblib",
    )
    parser.add_argument(
        "--report-output",
        default="reports/xgboost_fault_detector_48ch_tuned_threshold_report.json",
    )
    parser.add_argument(
        "--sweep-output",
        default="reports/xgboost_fault_detector_48ch_threshold_sweep.csv",
    )
    return parser.parse_args()


def load_xy(path: str | Path, feature_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    return frame[feature_columns], frame["binary_target"].astype(int).to_numpy()


def score_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(y_score)
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        metrics = binary_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": float(threshold),
                "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "precision": metrics["precision"],
                "fn": metrics["fn"],
                "fp": metrics["fp"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    sweep: pd.DataFrame,
    min_recall: float,
    max_fpr: float,
) -> pd.Series:
    valid = sweep[
        (sweep["recall"] >= min_recall)
        & (sweep["false_positive_rate"] <= max_fpr)
    ]
    if valid.empty:
        valid = sweep[sweep["recall"] >= min_recall]
    if valid.empty:
        return sweep.sort_values(
            ["recall", "false_positive_rate", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]
    return valid.sort_values(
        ["false_positive_rate", "precision", "threshold"],
        ascending=[True, False, False],
    ).iloc[0]


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    original_threshold = float(artifact["threshold"])

    x_val, y_val = load_xy(args.val, feature_columns)
    val_scores = model.predict_proba(x_val)[:, 1]
    sweep = score_thresholds(y_val, val_scores)
    selected = choose_threshold(sweep, args.min_recall, args.max_fpr)
    tuned_threshold = float(selected["threshold"])

    x_test, y_test = load_xy(args.test, feature_columns)
    test_scores = model.predict_proba(x_test)[:, 1]
    original_test_metrics = binary_metrics(
        y_test,
        (test_scores >= original_threshold).astype(int),
    )
    tuned_test_metrics = binary_metrics(
        y_test,
        (test_scores >= tuned_threshold).astype(int),
    )

    tuned_artifact = dict(artifact)
    tuned_artifact["threshold"] = tuned_threshold
    tuned_artifact["threshold_tuning"] = {
        "source": "validation",
        "min_recall": args.min_recall,
        "max_fpr": args.max_fpr,
        "original_threshold": original_threshold,
    }

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_artifact, model_output)

    sweep_output = Path(args.sweep_output)
    sweep_output.parent.mkdir(parents=True, exist_ok=True)
    sweep.sort_values("threshold", ascending=False).to_csv(sweep_output, index=False)

    report = {
        "model": "XGBClassifierThresholdTuned",
        "base_model_path": args.model,
        "tuned_model_path": str(model_output),
        "target": "binary_target",
        "threshold": tuned_threshold,
        "original_threshold": original_threshold,
        "feature_count": len(feature_columns),
        "train_rows": None,
        "val_rows": len(x_val),
        "test_rows": len(x_test),
        "tuning_data": args.val,
        "test_data": args.test,
        "tuning_goal": {
            "min_validation_recall": args.min_recall,
            "max_validation_fpr": args.max_fpr,
        },
        "validation_metrics": {
            "tn": int(selected["tn"]),
            "fp": int(selected["fp"]),
            "fn": int(selected["fn"]),
            "tp": int(selected["tp"]),
            "precision": float(selected["precision"]),
            "recall": float(selected["recall"]),
            "false_positive_rate": float(selected["false_positive_rate"]),
        },
        "test_metrics": tuned_test_metrics,
        "original_test_metrics": original_test_metrics,
    }

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_model={model_output}")
    print(f"saved_report={report_output}")
    print(f"saved_sweep={sweep_output}")


if __name__ == "__main__":
    main()
