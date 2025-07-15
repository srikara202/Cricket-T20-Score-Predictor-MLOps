import os
import unittest

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class TestCricketScorePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ─── MLflow / DagsHub auth ──────────────────────────────────────
        token = os.getenv("CAPSTONE_TEST")
        if not token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        mlflow.set_tracking_uri(
            "https://dagshub.com/"
            "srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow"
        )

        # ─── resolve latest version by alias prefix ────────────────────
        cls.model_name = "my_model"
        cls.alias_prefix = "staging"
        cls.model_version = cls.get_latest_model_version(
            cls.model_name,
            cls.alias_prefix
        )

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ─── load holdout data ──────────────────────────────────────────
        df = pd.read_csv("data/processed/test_final.csv")
        cls.X_test = df.drop(columns=["total_runs"])
        cls.y_test = df["total_runs"]

    @staticmethod
    def get_latest_model_version(model_name: str, alias_prefix: str):
        """
        Fetch the most‐recent model version whose aliases list
        contains any entry starting with alias_prefix
        (e.g. "staging123456789").
        """
        client = MlflowClient()

        # 1) Retrieve all versions for this model
        all_versions = client.search_model_versions(f"name = '{model_name}'")

        # 2) Keep only those with an alias that starts with alias_prefix
        candidates = [
            mv for mv in all_versions
            if any(a.startswith(alias_prefix) for a in mv.aliases)
        ]

        if not candidates:
            raise ValueError(
                f"No versions found for model='{model_name}' "
                f"with alias starting '{alias_prefix}'"
            )

        # 3) Sort by last_updated_timestamp descending and pick the newest
        candidates.sort(key=lambda mv: mv.last_updated_timestamp, reverse=True)
        return candidates[0].version

    def test_model_loaded(self):
        """Model should load via alias without error."""
        self.assertIsNotNone(self.model)

    def test_signature_matches(self):
        """If you logged an input signature, it should match the DataFrame columns."""
        metadata = getattr(self.model, "metadata", None)
        if metadata is None or metadata.get_input_schema() is None:
            self.skipTest("No input signature logged with this model")
        schema = metadata.get_input_schema().inputs
        expected_cols = [inp.name for inp in schema]
        self.assertListEqual(expected_cols, list(self.X_test.columns))

    def test_regression_performance(self):
        """Check basic regression metrics against your holdout set."""
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        mae = mean_absolute_error(self.y_test, preds)
        r2  = r2_score(self.y_test, preds)

        # adjust these thresholds as appropriate for your data
        self.assertLessEqual(mse, 300, f"MSE too high: {mse:.1f}")
        self.assertLessEqual(mae, 15,  f"MAE too high: {mae:.1f}")
        self.assertGreaterEqual(r2,  0.50, f"R² too low: {r2:.2f}")


if __name__ == "__main__":
    unittest.main()
