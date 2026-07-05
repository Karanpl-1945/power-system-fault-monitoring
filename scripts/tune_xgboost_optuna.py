from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import optuna
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
    parser = argparse.ArgumentParser(description="Tune XGBoost with Optuna.")
    parser.add_argument("--train", default="data/processed_48ch/features_train.csv")
    parser.add_argument("--val", default="data/processed_48ch/features_val.csv")
    parser.add_argument("--test", default="data/processed_48ch/features_test.csv")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional stratified train-row sample for faster Optuna search. Best params are retrained on full train data.",
    )
    parser.add_argument("--min-recall", type=float, default=0.97)
    parser.add_argument("--max-fpr", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", default="models/xgboost_fault_detector_48ch_optuna.joblib")
    parser.add_argument("--report-output", default="reports/xgboost_fault_detector_48ch_optuna_report.json")
    parser.add_argument("--trials-output", default="reports/xgboost_fault_detector_48ch_optuna_trials.csv")
    return parser.parse_args()


def load_xy(path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    frame = pd.read_csv(path)
    feature_columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return frame[feature_columns], frame["binary_target"].astype(int), feature_columns


def stratified_sample(
    x: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(x) <= max_rows:
        return x, y
    sampled_indices = []
    for label in sorted(y.unique()):
        label_indices = y[y == label].index.to_series()
        n_label = max(1, round(max_rows * len(label_indices) / len(y)))
        sampled_indices.append(label_indices.sample(n=n_label, random_state=random_state))

    indices = (
        pd.concat(sampled_indices)
        .sample(frac=1.0, random_state=random_state)
        .to_list()
    )
    return x.loc[indices], y.loc[indices]


def objective_score(metrics: dict[str, float | int], min_recall: float, max_fpr: float) -> float:
    recall = float(metrics["recall"])
    precision = float(metrics["precision"])
    fpr = float(metrics["false_positive_rate"])

    if recall < min_recall:
        return recall - min_recall - fpr
    if fpr > max_fpr:
        return recall - min_recall - fpr
    return precision + recall - (5.0 * fpr)


def suggest_params(
    trial: optuna.Trial,
    scale_pos_weight_base: float,
    random_state: int,
) -> dict[str, float | int | str]:
    scale_multiplier = trial.suggest_float("scale_pos_weight_multiplier", 0.8, 2.5)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "scale_pos_weight": scale_pos_weight_base * scale_multiplier,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def main() -> None:
    args = parse_args()
    x_train, y_train, feature_columns = load_xy(args.train)
    x_val, y_val, _ = load_xy(args.val)
    x_test, y_test, _ = load_xy(args.test)
    x_train_search, y_train_search = stratified_sample(
        x_train,
        y_train,
        args.max_train_rows,
        args.random_state,
    )

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight_base = negative / positive if positive else 1.0

    trial_rows: list[dict[str, float | int | str]] = []
    best: dict[str, object] = {}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, scale_pos_weight_base, args.random_state)
        model = XGBClassifier(**params)
        model.fit(x_train_search, y_train_search)

        val_scores = model.predict_proba(x_val)[:, 1]
        threshold, val_metrics = choose_threshold_for_recall(
            y_true=y_val.to_numpy(),
            y_score=val_scores,
            min_recall=args.min_recall,
            max_fpr=args.max_fpr,
        )
        score = objective_score(val_metrics, args.min_recall, args.max_fpr)

        trial_row = {
            "trial": trial.number,
            "score": score,
            "threshold": threshold,
            "val_recall": val_metrics["recall"],
            "val_fpr": val_metrics["false_positive_rate"],
            "val_precision": val_metrics["precision"],
            "val_fn": val_metrics["fn"],
            "val_fp": val_metrics["fp"],
            **params,
        }
        trial_rows.append(trial_row)

        if not best or score > float(best["score"]):
            best.clear()
            best.update(
                {
                    "score": score,
                    "model": model,
                    "threshold": threshold,
                    "validation_metrics": val_metrics,
                    "params": params,
                }
            )

        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("val_recall", val_metrics["recall"])
        trial.set_user_attr("val_fpr", val_metrics["false_positive_rate"])
        trial.set_user_attr("val_precision", val_metrics["precision"])
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, n_jobs=1)

    best_params = best["params"]
    best_model = XGBClassifier(**best_params)
    best_model.fit(x_train, y_train)

    val_scores = best_model.predict_proba(x_val)[:, 1]
    best_threshold, validation_metrics = choose_threshold_for_recall(
        y_true=y_val.to_numpy(),
        y_score=val_scores,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
    )

    test_scores = best_model.predict_proba(x_test)[:, 1]
    test_pred = (test_scores >= best_threshold).astype(int)
    test_metrics = binary_metrics(y_test.to_numpy(), test_pred)

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "feature_columns": feature_columns,
            "optuna": {
                "score": best["score"],
                "params": best_params,
                "min_recall": args.min_recall,
                "max_fpr": args.max_fpr,
                "trials": args.trials,
                "max_train_rows": args.max_train_rows,
            },
        },
        model_output,
    )

    trials_output = Path(args.trials_output)
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).sort_values("score", ascending=False).to_csv(
        trials_output,
        index=False,
    )

    report = {
        "model": "XGBClassifierOptuna",
        "target": "binary_target",
        "threshold": best_threshold,
        "feature_count": len(feature_columns),
        "train_rows": len(x_train),
        "search_train_rows": len(x_train_search),
        "val_rows": len(x_val),
        "test_rows": len(x_test),
        "scale_pos_weight_base": scale_pos_weight_base,
        "best_score": best["score"],
        "best_params": best_params,
        "validation_metrics": validation_metrics,
        "search_validation_metrics": best["validation_metrics"],
        "test_metrics": test_metrics,
    }

    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"saved_model={model_output}")
    print(f"saved_report={report_output}")
    print(f"saved_trials={trials_output}")


if __name__ == "__main__":
    main()
