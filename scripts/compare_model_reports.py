from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model JSON reports.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[
            "reports/baseline_fault_detector_report.json",
            "reports/xgboost_fault_detector_report.json",
            "reports/xgboost_fault_detector_48ch_report.json",
            "reports/xgboost_fault_detector_48ch_tuned_threshold_report.json",
            "reports/xgboost_fault_detector_48ch_tuned_recall97_report.json",
            "reports/xgboost_fault_detector_48ch_optuna_report.json",
            "reports/cnn1d_fault_detector_smoke_report.json",
        ],
    )
    parser.add_argument("--output", default="reports/model_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for report_path in args.reports:
        path = Path(report_path)
        if not path.exists():
            print(f"missing={path}; skipping")
            continue
        report = json.loads(path.read_text())
        row = {
            "report": report_path,
            "model": report["model"],
            "feature_count": report["feature_count"],
            "threshold": report["threshold"],
            "val_recall": report["validation_metrics"]["recall"],
            "val_fpr": report["validation_metrics"]["false_positive_rate"],
            "test_recall": report["test_metrics"]["recall"],
            "test_fpr": report["test_metrics"]["false_positive_rate"],
            "test_precision": report["test_metrics"]["precision"],
            "test_fn": report["test_metrics"]["fn"],
            "test_fp": report["test_metrics"]["fp"],
        }
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        by=["test_recall", "test_fpr"],
        ascending=[False, True],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
