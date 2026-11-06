"""Streamlit interface for evaluating the saved breast cancer classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from model_utils import METRIC_COLUMNS, evaluate_classifier, validate_test_data


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"

st.set_page_config(
    page_title="Breast Cancer Classifier Lab",
    page_icon="🧬",
    layout="wide",
)


# Cached so the JSON isn't re-read on every interaction
@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Model metadata is missing. Run `python train_models.py` from the project folder."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


# Models are loaded once and reused; nothing here ever calls fit()
@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved model is missing: {path.name}. Run `python train_models.py`."
        )
    return joblib.load(path)


def main() -> None:
    st.title("🧬 Breast Cancer Classifier Lab")
    st.caption(
        "Compare five saved classifiers on a labeled test CSV. Models are loaded from "
        "disk; uploaded data is evaluated only and never used for training."
    )

    try:
        metadata = load_metadata()
    except (OSError, json.JSONDecodeError, FileNotFoundError) as error:
        st.error(str(error))
        st.stop()

    # Sidebar: dataset info, model dropdown and the CSV upload
    with st.sidebar:
        st.header("Experiment setup")
        st.write(f"**Dataset:** {metadata['dataset_name']}")
        st.write(f"**Features:** {metadata['feature_count']}")
        st.write(f"**Held-out test rows:** {metadata['test_rows']}")
        model_names = list(metadata["model_files"])
        selected_model = st.selectbox("Choose a model", ["All models", *model_names])
        uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type="csv")
        st.caption(
            "The CSV must include all 30 feature columns plus `target` "
            "(0 = malignant, 1 = benign)."
        )

    # Nothing uploaded yet: show a preview of the sample test file and stop here
    if uploaded_file is None:
        st.info("Upload a labeled CSV to see model metrics, a confusion matrix, and a report.")
        st.subheader("Included test data")
        test_path = PROJECT_DIR / "test_data.csv"
        if test_path.is_file():
            sample = pd.read_csv(test_path, nrows=5)
            st.dataframe(sample, width="stretch", hide_index=True)
            st.download_button(
                "Download the included test CSV",
                data=test_path.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
            )
        else:
            st.warning("The test CSV is missing. Run `python train_models.py`.")
        st.stop()

    # Read and check the upload before using it, so bad files give a clear message
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        features, target = validate_test_data(
            uploaded_data,
            metadata["feature_names"],
            metadata["target_column"],
            metadata["valid_targets"],
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not use this CSV: {error}")
        st.stop()

    st.success(f"Loaded {len(features):,} labeled rows with all required features.")
    # Evaluate either every model or just the one picked in the dropdown
    selected_names = model_names if selected_model == "All models" else [selected_model]
    results: dict[str, dict[str, float]] = {}

    try:
        for model_name in selected_names:
            model_path = MODEL_DIR / metadata["model_files"][model_name]
            model = load_model(str(model_path))
            results[model_name] = evaluate_classifier(
                model, features, target, metadata["positive_label"]
            )
    except (OSError, ValueError, AttributeError) as error:
        st.error(f"Model evaluation failed: {error}")
        st.stop()

    # Metrics table; the best value in each column is highlighted
    st.subheader("Evaluation metrics")
    result_table = pd.DataFrame.from_dict(results, orient="index")[list(METRIC_COLUMNS)]
    st.dataframe(
        result_table.style.format("{:.4f}").highlight_max(axis=0, color="#d9f2e6"),
        width="stretch",
    )

    # Confusion matrix and report only make sense for a single model
    if selected_model != "All models":
        model = load_model(str(MODEL_DIR / metadata["model_files"][selected_model]))
        predictions = model.predict(features)
        class_labels = [metadata["target_names"][str(label)] for label in metadata["valid_targets"]]
        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            figure, axis = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(
                confusion_matrix(target, predictions, labels=metadata["valid_targets"]),
                display_labels=class_labels,
            ).plot(ax=axis, colorbar=False, cmap="Blues", values_format="d")
            axis.set_title(selected_model)
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)  # free the figure so memory doesn't build up
        with right:
            st.subheader("Classification report")
            report = classification_report(
                target,
                predictions,
                labels=metadata["valid_targets"],
                target_names=class_labels,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                width="stretch",
            )
    else:
        st.caption("Select one model to view its confusion matrix and classification report.")

    with st.expander("About the dataset and experiment"):
        st.write(
            "The Breast Cancer Wisconsin (Diagnostic) dataset contains measurements "
            "computed from digitized images of breast mass fine needle aspirates. "
            "The saved estimators were trained once on a stratified training partition; "
            "this app only loads them and calculates metrics for the uploaded labeled data."
        )
        st.markdown(
            "[UCI dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)"
        )


if __name__ == "__main__":
    main()
