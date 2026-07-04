from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def make_random_forest_baseline(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

