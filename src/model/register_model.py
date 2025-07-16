#!/usr/bin/env python3
"""
Register an MLflow model version and tag it as 'staging'.
"""

import json
import os
import warnings
import logging

import mlflow
from mlflow.tracking import MlflowClient

# suppress harmless warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def load_model_info(file_path: str) -> dict:
    """Load the model info (run_id + artifact path) from JSON."""
    try:
        with open(file_path, "r") as f:
            info = json.load(f)
        logging.debug("Loaded model info from %s", file_path)
        return info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to MLflow Model Registry and tag it 'staging'."""
    client = MlflowClient()
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    # 1) register the new version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logging.debug("Registered model %s version %s", model_name, mv.version)

    # 2) tag it as staging
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="stage",
        value="staging"
    )
    logging.debug("Tagged model %s version %s with stage=staging", model_name, mv.version)
    print(f"✅ Registered '{model_name}' v{mv.version} and tagged stage=staging")


def main():
    # ─── configure MLflow / DagsHub authentication ─────────────────────
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # point at your DagsHub MLflow registry
    uri = "https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow"
    mlflow.set_tracking_uri(uri)
    # -------------------------------------------------------------------------------------
    # Below code block is for local use
    # -------------------------------------------------------------------------------------
    # mlflow.set_tracking_uri('https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow')
    # dagshub.init(repo_owner='srikara202', repo_name='Cricket-T20-Score-Predictor-MLOps', mlflow=True)
    # -------------------------------------------------------------------------------------

    try:
        info = load_model_info("reports/experiment_info.json")
        register_model("my_model", info)
    except Exception as e:
        logging.error("Failed to register model: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
