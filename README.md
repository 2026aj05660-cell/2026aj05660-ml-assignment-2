# Breast Cancer Classification — ML Assignment 2

## a. Problem statement

Build and compare five supervised classification models for the Breast Cancer
Wisconsin (Diagnostic) dataset. The Streamlit app evaluates the saved models on
a labeled test CSV, reports the required metrics, and displays a confusion
matrix and classification report. Uploaded files are used only for evaluation;
the app does not retrain models.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), a binary classification dataset.
- **Source:** UCI Machine Learning Repository,
  [dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
  DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).
- **Size:** 569 observations and 30 numeric input features.
- **Target:** `target` — `0` means malignant and `1` means benign.
- **Split:** Stratified 80% training / 20% test split, random seed 42. The included
  `test_data.csv` contains the 114 held-out observations and target labels.
- **Preprocessing:** Standard scaling is part of the Logistic Regression, KNN,
  and SVM model pipelines. Decision Tree and Gaussian Naive Bayes use raw features.

## c. GitHub Repository Link

**Repository:** https://github.com/2026aj05660-cell/2026aj05660-ml-assignment-2

## d. Models used

All five models were fit on the same training partition and evaluated on the same
held-out test data. AUC uses the probability of class `1` (benign); Precision,
Recall, and F1 use class `1` as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint-highest accuracy (98.25%), F1 (0.9861), and MCC (0.9623); it also has the highest AUC (0.9954), making it the strongest overall model on this split. |
| Decision Tree | Lowest accuracy (91.23%), AUC (0.9157), and MCC (0.8174); its high precision (0.9559) but lower recall (0.9028) means it misses more benign cases than the other top models. |
| KNN | Strong accuracy (97.37%) and AUC (0.9884), with 100% recall for the benign class on this split; its precision is lower than Logistic Regression and SVM. |
| Naive Bayes | Moderate accuracy (93.86%) but a strong AUC (0.9878); ranking is comparatively good while its thresholded predictions are less accurate. |
| Support Vector Machine | Ties Logistic Regression for accuracy, precision, recall, F1, and MCC; its AUC (0.9950) is marginally below Logistic Regression. |
| Overall, Winner for your dataset? | Logistic Regression is the overall winner on this stratified holdout: it ties for the best accuracy and has the highest AUC. SVM is a close second; these results are from one fixed split, not a guarantee of future performance. |

## Streamlit Community Cloud

- **Live app:** _Add the deployed Streamlit Community Cloud URL after deployment._
- **App entry point:** `app.py`

The app supports uploading a labeled CSV, choosing one or all five models,
viewing all six evaluation metrics, and (for an individual model) its confusion
matrix and classification report.

## Run locally / on BITS Virtual Lab

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py          # regenerates test_data.csv, model/*.joblib, metadata.json
python -m unittest discover -s tests -v
streamlit run app.py
```

## Project files

```text
app.py
model_utils.py
train_models.py
requirements.txt
README.md
test_data.csv
model/
  metadata.json
  *.joblib            # five fitted models
tests/
  test_model_utils.py
```
