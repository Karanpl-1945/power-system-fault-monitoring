from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


def choose_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float = 0.95,
    max_fpr: float | None = 0.03,
) -> tuple[float, dict[str, float | int]]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    sorted_score = y_score[order]
    sorted_true = y_true[order]

    unique_indices = np.where(np.diff(sorted_score))[0]
    threshold_indices = np.r_[unique_indices, len(sorted_score) - 1]

    total_positive = int(sorted_true.sum())
    total_negative = int(len(sorted_true) - total_positive)
    cumulative_positive = np.cumsum(sorted_true)[threshold_indices]
    predicted_positive = threshold_indices + 1

    candidates: list[tuple[float, dict[str, float | int]]] = []
    recall_candidates: list[tuple[float, dict[str, float | int]]] = []
    all_candidates: list[tuple[float, dict[str, float | int]]] = []

    for index, tp_value, pred_pos_value in zip(
        threshold_indices,
        cumulative_positive,
        predicted_positive,
        strict=True,
    ):
        tp = int(tp_value)
        fp = int(pred_pos_value - tp)
        fn = int(total_positive - tp)
        tn = int(total_negative - fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        accuracy = (tp + tn) / len(sorted_true) if len(sorted_true) else 0.0
        threshold = float(sorted_score[index])
        metrics = {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        }
        all_candidates.append((threshold, metrics))
        if metrics["recall"] >= min_recall:
            recall_candidates.append((threshold, metrics))
            if max_fpr is None or metrics["false_positive_rate"] <= max_fpr:
                candidates.append((threshold, metrics))

    if candidates:
        return max(candidates, key=lambda item: (item[1]["precision"], item[0]))

    if recall_candidates:
        return max(recall_candidates, key=lambda item: (item[1]["precision"], item[0]))

    return max(all_candidates, key=lambda item: item[1]["recall"])
