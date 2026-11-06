"""Train, evaluate, and save all assignment classifiers on the UCI dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_utils import evaluate_classifier


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TARGET_COLUMN = "target"
TEST_SIZE = 0.20
RANDOM_STATE = 42  # fixed seed so the split and results are reproducible


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.Series(dataset.target, name=TARGET_COLUMN)

    # Stratify so train and test keep the same malignant/benign ratio
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    # Save the held-out rows as test_data.csv; this is the file we upload in the app
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    # Models that depend on distances or gradients get a StandardScaler inside
    # a pipeline; the tree and Naive Bayes don't need scaling
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        # probability=True is needed so we can compute AUC
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    model_files: dict[str, str] = {}
    # Fit each model on the training split, score it on the test split,
    # then save it so the app can load it without retraining
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics[model_name] = evaluate_classifier(model, X_test, y_test, positive_label=1)
        filename = model_name.lower().replace(" ", "_").replace("-", "") + ".joblib"
        model_files[model_name] = filename
        joblib.dump(model, MODEL_DIR / filename)

    # The app reads this file for feature names, class labels and model paths
    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository, DOI: 10.24432/C5DW2B",
        "target_column": TARGET_COLUMN,
        "feature_names": list(features.columns),
        "target_names": {str(index): name for index, name in enumerate(dataset.target_names)},
        "positive_label": 1,
        "valid_targets": [0, 1],
        "total_rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "metrics": metrics,
        "model_files": model_files,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Shape: {len(features)} rows x {features.shape[1]} features")
    print(f"Test rows: {len(X_test)}")
    print(pd.DataFrame(metrics).T.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
