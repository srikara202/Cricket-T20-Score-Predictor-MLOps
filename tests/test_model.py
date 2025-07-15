import os
import unittest

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


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

        # ─── resolve latest version by staging‐alias prefix ────────────────
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(
            cls.model_name,
            staging_prefix="staging"
        )

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # ─── load holdout data ──────────────────────────────────────────
        df = pd.read_csv("data/processed/test_final.csv")
        cls.X_test = df.drop(columns=["total_runs"])
        cls.y_test = df["total_runs"]

    @staticmethod
    def get_latest_model_version(model_name: str, staging_prefix: str):
        """
        Fetch the most‐recent model version whose aliases list
        contains any alias starting with `staging_prefix`.
        """
        client = mlflow.MlflowClient()
        all_versions = client.search_model_versions(f"name = '{model_name}'")

        # filter for any alias that begins with our staging prefix
        staging_entries = [
            mv for mv in all_versions
            if any(alias.startswith(staging_prefix) for alias in mv.aliases)
        ]
        if not staging_entries:
            raise ValueError(
                f"No versions found for model='{model_name}' "
                f"with an alias starting with '{staging_prefix}'"
            )

        # pick the one most recently updated
        staging_entries.sort(
            key=lambda mv: mv.last_updated_timestamp,
            reverse=True
        )
        return staging_entries[0].version

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
        rmse = root_mean_squared_error(self.y_test, preds)
        mae = mean_absolute_error(self.y_test, preds)
        r2  = r2_score(self.y_test, preds)
        exp_rmse = 10
        exp_mae = 10
        exp_r2 = 0.95
        # adjust these thresholds as needed for your project
        self.assertLessEqual(rmse, exp_rmse, f"MSE obtained: {mse:.1f}, should be less than {exp_rmse}")
        self.assertLessEqual(mae, exp_mae,  f"MAE obtained: {mae:.1f}, should be less than {exp_mae}")
        self.assertGreaterEqual(r2,  exp_r2, f"R² obtained: {r2:.2f}, should be greater than {exp_r2}")


if __name__ == "__main__":
    unittest.main()
