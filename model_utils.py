"""Shared validation and metric helpers for the trained classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Column order used when showing the results table in the app
METRIC_COLUMNS = (
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
)


def validate_test_data(
    data: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    valid_targets: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate an uploaded labeled CSV and return features in training order."""
    if data.empty:
        raise ValueError("The uploaded CSV has no rows.")

    # The file must have every training feature plus the target column
    required_columns = [*feature_names, target_column]
    missing_columns = [name for name in required_columns if name not in data.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    # Select columns in the same order the models were trained on, and turn
    # anything non-numeric into NaN so we can catch it below
    features = data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        invalid_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Feature values must be numeric and non-empty. Check: "
            + ", ".join(invalid_columns)
        )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must all be finite numbers.")

    # Labels must be whole numbers (0 or 1 for this dataset)
    target = pd.to_numeric(data[target_column], errors="coerce")
    if target.isna().any() or not np.equal(target, np.floor(target)).all():
        raise ValueError(f"{target_column!r} must contain integer class labels.")
    target = target.astype(int)
    unexpected_targets = sorted(set(target.unique()) - set(valid_targets))
    if unexpected_targets:
        raise ValueError(
            f"Unexpected target labels: {unexpected_targets}. "
            f"Expected labels: {valid_targets}."
        )

    return features, target


def evaluate_classifier(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    positive_label: int,
) -> dict[str, float]:
    """Calculate the six assignment metrics for a fitted binary classifier."""
    predictions = model.predict(features)
    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f"Positive class {positive_label} was not learned by the model.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not provide probabilities required for AUC.")

    # AUC needs the predicted probability of the positive class, so find
    # which probability column belongs to it
    positive_index = classes.index(positive_label)
    probabilities = model.predict_proba(features)[:, positive_index]
    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(target, probabilities)),
        "Precision": float(
            precision_score(target, predictions, pos_label=positive_label, zero_division=0)
        ),
        "Recall": float(
            recall_score(target, predictions, pos_label=positive_label, zero_division=0)
        ),
        "F1": float(
            f1_score(target, predictions, pos_label=positive_label, zero_division=0)
        ),
        # MCC is symmetric, so it doesn't need a positive label
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
