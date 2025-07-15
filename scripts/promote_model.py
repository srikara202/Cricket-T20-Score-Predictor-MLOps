#!/usr/bin/env python3
"""
Promote an MLflow model version tagged 'staging' → 'production' via version TAGS.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

def main():
    # ─── Auth ────────────────────────────────────────────────────────
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ─── Point to DagsHub registry ────────────────────────────────
    dagshub_url = "https://dagshub.com"
    owner      = "srikara202"
    repo       = "Cricket-T20-Score-Predictor-MLOps"
    mlflow.set_tracking_uri(f"{dagshub_url}/{owner}/{repo}.mlflow")

    client               = MlflowClient()
    model_name           = "my_model"
    staging_tag_value    = "staging"
    production_tag_value = "production"

    # 1) collect all versions, pick those with stage=staging tag
    all_versions = client.search_model_versions(f"name = '{model_name}'")
    staging_candidates = [
        mv for mv in all_versions
        if mv.tags.get("stage") == staging_tag_value
    ]
    if not staging_candidates:
        raise ValueError(f"No model version found with stage='{staging_tag_value}'")

    # choose the newest by update timestamp
    to_promote = max(staging_candidates, key=lambda mv: mv.last_updated_timestamp).version

    # 2) de‐tag any existing production versions
    for mv in all_versions:
        if mv.tags.get("stage") == production_tag_value:
            client.delete_model_version_tag(
                name=model_name,
                version=mv.version,
                key="stage"
            )

    # 3) tag the chosen version as production
    client.set_model_version_tag(
        name=model_name,
        version=to_promote,
        key="stage",
        value=production_tag_value
    )

    print(f"✅ Promoted model version {to_promote} to production")


if __name__ == "__main__":
    main()
