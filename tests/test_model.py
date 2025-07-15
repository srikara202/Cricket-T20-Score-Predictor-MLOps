import os
import unittest

import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow import MlflowClient


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
    def get_latest_model_version(model_name: str, alias_prefix: str) -> str:
        """
        Fetch the most‐recent model version that has ANY alias
        starting with alias_prefix (e.g. "staging12345").
        """
        client = MlflowClient()
        # fetch all versions for this model
        all_versions = client.search_model_versions(f"name = '{model_name}'")

        # filter those whose aliases list contains an alias that starts with the prefix
        candidates = [
            mv for mv in all_versions
            if any(alias.startswith(alias_prefix) for alias in mv.aliases)
        ]
        if not candidates:
            raise ValueError(f"No versions found for model='{model_name}' alias prefix='{alias_prefix}'")

        # pick the one most recently updated
        candidates.sort(key=lambda mv: mv.last_updated_timestamp, reverse=True)
        return candidates[0].version

    def test_model_loaded(self):
        """Model should load via the staging alias without error."""
        self.assertIsNotNone(self.model)

    def test_signature_matches(self):
        """If you logged an input signature, it should match the DataFrame columns."""
        meta = getattr(self.model, "metadata", None)
        sig = meta.get_input_schema() if meta else None
        if sig is None:
            self.skipTest("No input signature logged with this model")
        expected = [inp.name for inp in sig.inputs]
        self.assertListEqual(expected, list(self.X_test.columns))

    def test_regression_performance(self):
        """Check basic regression metrics against your holdout set."""
        preds = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, preds)
        mae = mean_absolute_error(self.y_test, preds)
        r2  = r2_score(self.y_test, preds)

        # adjust thresholds as appropriate
        self.assertLessEqual(mse, 300, f"MSE too high: {mse:.1f}")
        self.assertLessEqual(mae, 15,  f"MAE too high: {mae:.1f}")
        self.assertGreaterEqual(r2,  0.50, f"R² too low: {r2:.2f}")


if __name__ == "__main__":
    unittest.main()
