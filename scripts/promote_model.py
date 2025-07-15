#!/usr/bin/env python3
"""
Promote an MLflow model version from any 'staging*' alias to the 'production' alias.
"""

import os
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException

def main():
    # ─── Auth ────────────────────────────────────────────────────────
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ─── Point to your DagsHub registry ──────────────────────────────
    dagshub_url = "https://dagshub.com"
    owner = "srikara202"
    repo  = "Cricket-T20-Score-Predictor-MLOps"
    mlflow.set_tracking_uri(f"{dagshub_url}/{owner}/{repo}.mlflow")

    client = MlflowClient()
    model_name      = "my_model"
    staging_prefix  = "staging"
    production_alias = "production"

    # ─── 1. Find all model versions and pick those with a staging* alias ─
    all_versions = client.search_model_versions(f"name = '{model_name}'")
    staging_versions = [
        mv for mv in all_versions
        if any(alias[0].startswith(staging_prefix) for alias in mv.aliases)
    ]
    if not staging_versions:
        raise ValueError(f"No model version has an alias starting with '{staging_prefix}'")
    # choose the highest version number
    to_promote = max(staging_versions, key=lambda mv: int(mv.version)).version

    # ─── 2. Remove old production alias ───────────────────────────────
    for mv in all_versions:
        if production_alias in mv.aliases:
            try:
                client.delete_registered_model_alias(
                    name = model_name,
                    alias = production_alias,
                )
                print(f"Removed alias '{production_alias}' (was on v{mv.version})")
            except RestException as e:
                print(f"Warning: could not delete alias on v{mv.version}: {e}")

    # ─── 3. Set production on the new candidate ───────────────────────
    client.set_registered_model_alias(
        name    = model_name,
        alias   = production_alias,
        version = to_promote,
    )
    print(f"✅ Model version {to_promote} is now aliased as '{production_alias}'")

if __name__ == "__main__":
    main()
