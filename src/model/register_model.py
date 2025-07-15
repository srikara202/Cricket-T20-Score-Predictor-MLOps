#!/usr/bin/env python3
"""
Register a model run in MLflow and assign it a unique 'staging*' alias.
"""

import json
import os
import random
import warnings
import logging

import mlflow
from mlflow import MlflowClient

# suppress harmless warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
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
    """Register the model to MLflow Model Registry and alias it as staging."""
    # build the MLflow URI for the artifact you saved in your run
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    client = MlflowClient()

    # 1) register the new version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logging.debug("Registered model %s version %s", model_name, mv.version)

    # 2) create a unique staging alias, e.g. staging1234567890
    alias = f"staging{random.randint(10**9, 10**10 - 1)}"
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version,
    )
    logging.debug("Assigned alias '%s' → version %s", alias, mv.version)
    print(f"✅ Registered '{model_name}' v{mv.version} with alias '{alias}'")


def main():
    # ─── configure MLflow / DagsHub authentication ─────────────────────
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    uri = "https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow"
    mlflow.set_tracking_uri(uri)

    try:
        info = load_model_info("reports/experiment_info.json")
        register_model("my_model", info)
    except Exception as e:
        logging.error("Failed to register model: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
