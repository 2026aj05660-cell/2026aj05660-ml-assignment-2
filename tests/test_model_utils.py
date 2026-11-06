import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from model_utils import evaluate_classifier, validate_test_data


class ModelUtilsTests(unittest.TestCase):
    def setUp(self):
        # A tiny dataset and a dummy model, just enough to exercise the helpers
        self.features = pd.DataFrame({"first": [0.0, 1.0, 2.0, 3.0]})
        self.target = pd.Series([0, 0, 1, 1])
        self.model = DummyClassifier(strategy="prior").fit(self.features, self.target)

    def test_validate_test_data_preserves_feature_order_and_integer_labels(self):
        # Columns arrive shuffled; they should come back in training order
        raw_data = pd.DataFrame(
            {"target": [0, 1], "second": [3.0, 4.0], "first": [1.0, 2.0]}
        )
        features, target = validate_test_data(
            raw_data, ["first", "second"], "target", [0, 1]
        )

        self.assertEqual(list(features.columns), ["first", "second"])
        self.assertEqual(target.tolist(), [0, 1])

    def test_validate_test_data_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_test_data(
                pd.DataFrame({"first": [1.0], "target": [0]}),
                ["first", "second"],
                "target",
                [0, 1],
            )

    def test_validate_test_data_rejects_non_numeric_features(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_test_data(
                pd.DataFrame({"first": ["bad"], "target": [0]}),
                ["first"],
                "target",
                [0, 1],
            )

    def test_validate_test_data_rejects_unknown_target_labels(self):
        with self.assertRaisesRegex(ValueError, "Unexpected target labels"):
            validate_test_data(
                pd.DataFrame({"first": [1.0], "target": [2]}),
                ["first"],
                "target",
                [0, 1],
            )

    def test_evaluate_classifier_returns_all_six_metrics(self):
        metrics = evaluate_classifier(self.model, self.features, self.target, 1)

        self.assertEqual(
            set(metrics),
            {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"},
        )
        # Every metric is between 0 and 1, except MCC which can go down to -1
        self.assertTrue(
            all(
                -1.0 <= value <= 1.0 if name == "MCC" else 0.0 <= value <= 1.0
                for name, value in metrics.items()
            )
        )


if __name__ == "__main__":
    unittest.main()
