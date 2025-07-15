#!/usr/bin/env python3
"""
Promote an MLflow model version from any 'staging*' alias to the 'production' alias.
"""

import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import mlflow

def main():
    # ─── Auth ────────────────────────────────────────────────────────
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ─── Point to your DagsHub registry ───────────────────────────────
    dagshub_url = "https://dagshub.com"
    owner      = "srikara202"
    repo       = "Cricket-T20-Score-Predictor-MLOps"
    mlflow.set_tracking_uri(f"{dagshub_url}/{owner}/{repo}.mlflow")

    client            = MlflowClient()
    model_name        = "my_model"
    staging_prefix    = "staging"
    production_alias  = "production"

    # ─── 1. Fetch all versions and filter those with an alias starting with "staging" ───
    all_versions = client.search_model_versions(f"name = '{model_name}'")
    staging_versions = [
        mv for mv in all_versions
        if any(alias.startswith(staging_prefix) for alias in mv.aliases)
    ]
    if not staging_versions:
        raise ValueError(f"No model version has an alias starting with '{staging_prefix}'")
    # pick the highest numeric version
    to_promote = str(max(int(mv.version) for mv in staging_versions))

    # ─── 2. Remove existing 'production' alias from any version ────────────────
    for mv in all_versions:
        if production_alias in mv.aliases:
            try:
                client.delete_model_version_alias(
                    name=model_name,
                    version=mv.version,
                    alias=production_alias
                )
                print(f"Removed alias '{production_alias}' from version {mv.version}")
            except RestException as e:
                print(f"Warning: could not remove alias '{production_alias}' from v{mv.version}: {e}")

    # ─── 3. Add 'production' alias to the selected version ─────────────────────
    client.create_model_version_alias(
        name=model_name,
        version=to_promote,
        alias=production_alias
    )
    print(f"✅ Model version {to_promote} is now aliased as '{production_alias}'")

if __name__ == "__main__":
    main()
