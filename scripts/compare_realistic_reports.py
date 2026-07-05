from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved-model realistic evaluation reports.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[
            "reports/xgboost_fault_detector_on_features_test_realistic_5pct.json",
            "reports/xgboost_fault_detector_on_features_test_realistic_1pct.json",
            "reports/xgboost_fault_detector_on_features_test_realistic_0_5pct.json",
            "reports/xgboost_fault_detector_on_features_test_realistic_0_25pct.json",
            "reports/xgboost_fault_detector_48ch_tuned_recall97_on_realistic_0_5pct.json",
            "reports/xgboost_fault_detector_48ch_tuned_recall97_on_realistic_0_25pct.json",
            "reports/baseline_fault_detector_on_features_test_realistic_5pct.json",
            "reports/baseline_fault_detector_on_features_test_realistic_1pct.json",
            "reports/baseline_fault_detector_on_features_test_realistic_0_5pct.json",
            "reports/baseline_fault_detector_on_features_test_realistic_0_25pct.json",
        ],
    )
    parser.add_argument("--output", default="reports/realistic_eval_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for report_path in args.reports:
        report = json.loads(Path(report_path).read_text())
        metrics = report["metrics"]
        rows.append(
            {
                "report": report_path,
                "model": Path(report["model_path"]).stem,
                "data": Path(report["data_path"]).stem,
                "fault_ratio": metrics["fault_ratio"],
                "recall": metrics["recall"],
                "fpr": metrics["false_positive_rate"],
                "precision": metrics["precision"],
                "fn": metrics["fn"],
                "fp": metrics["fp"],
                "alerts_per_10000_windows": metrics["alerts_per_10000_windows"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values(["data", "recall"], ascending=[True, False])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
