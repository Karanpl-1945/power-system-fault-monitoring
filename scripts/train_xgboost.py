from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from predictive_maintenance.models.evaluation import binary_metrics, choose_threshold_for_recall


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
    parser = argparse.ArgumentParser(description="Train XGBoost binary fault detector.")
    parser.add_argument("--train", default="data/processed/features_train.csv")
    parser.add_argument("--val", default="data/processed/features_val.csv")
    parser.add_argument("--test", default="data/processed/features_test.csv")
    parser.add_argument("--model-output", default="models/xgboost_fault_detector.joblib")
    parser.add_argument("--report-output", default="reports/xgboost_fault_detector_report.json")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_xy(path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    frame = pd.read_csv(path)
    feature_columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return frame[feature_columns], frame["binary_target"].astype(int), feature_columns


def main() -> None:
    args = parse_args()

    x_train, y_train, feature_columns = load_xy(args.train)
    x_val, y_val, _ = load_xy(args.val)
    x_test, y_test, _ = load_xy(args.test)

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold, val_metrics = choose_threshold_for_recall(
        y_true=y_val.to_numpy(),
        y_score=val_scores,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
    )

    test_scores = model.predict_proba(x_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = binary_metrics(y_test.to_numpy(), test_pred)

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": feature_columns,
        },
        model_output,
    )

    report = {
        "model": "XGBClassifier",
        "target": "binary_target",
        "threshold": threshold,
        "feature_count": len(feature_columns),
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "test_rows": len(x_test),
        "scale_pos_weight": scale_pos_weight,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_model={model_output}")
    print(f"saved_report={report_output}")


if __name__ == "__main__":
    main()
