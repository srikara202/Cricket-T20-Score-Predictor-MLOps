#!/usr/bin/env python3
"""
Promote an MLflow model version from the 'staging*' alias to the 'production' alias.
"""

import os
import mlflow
from mlflow.exceptions import RestException

def main():
    # 1. Read your DagsHub token from env
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # 2. Point MLflow at your DagsHub registry
    dagshub_url = "https://dagshub.com"
    repo_owner = "srikara202"
    repo_name = "Cricket-T20-Score-Predictor-MLOps"
    tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    model_name = "my_model"
    staging_prefix = "staging"
    production_alias = "production"

    # 3. Find all versions whose aliases start with "staging"
    all_versions = client.search_model_versions(f"name='{model_name}'")
    staging_candidates = [
        mv for mv in all_versions
        if any(alias.startswith(staging_prefix) for alias in mv.aliases)
    ]
    if not staging_candidates:
        raise ValueError(f"No model version has an alias starting with '{staging_prefix}'")

    # pick the highest version number among them
    to_promote = max(staging_candidates, key=lambda mv: int(mv.version)).version

    # 4. Strip the "production" alias off any version that has it
    for mv in all_versions:
        if production_alias in mv.aliases:
            try:
                client.delete_model_version_alias(
                    name=model_name,
                    version=mv.version,
                    alias=production_alias,
                )
                print(f"Removed alias '{production_alias}' from version {mv.version}")
            except RestException as e:
                print(f"Warning: could not remove alias '{production_alias}' from v{mv.version}: {e}")

    # 5. Create the new "production" alias on your chosen staging candidate
    client.create_model_version_alias(
        name=model_name,
        version=to_promote,
        alias=production_alias,
    )
    print(f"✅ Model version {to_promote} is now aliased '{production_alias}'")

if __name__ == "__main__":
    main()
