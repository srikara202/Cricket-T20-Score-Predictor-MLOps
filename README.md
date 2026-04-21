# Cricket T20 Score Predictor - End-to-End MLOps Project

This repository is an end-to-end MLOps project built around one focused machine learning task: predicting the eventual first-innings total in an international men's T20 cricket match from the current innings state. The regression model itself is intentionally conventional. The strength of the project is the lifecycle around it: S3-backed data ingestion, DVC pipeline orchestration, feature engineering, experiment tracking in MLflow via DagsHub, model registration and promotion, a Flask inference layer that resolves the latest production-tagged model at startup, and deployment-oriented packaging with Docker, GitHub Actions, and Kubernetes.

## Table of Contents

- [Key Highlights](#key-highlights)
- [Problem Statement](#problem-statement)
- [What the Model Predicts](#what-the-model-predicts)
- [What This Project Is Really About](#what-this-project-is-really-about)
- [End-to-End Architecture / Pipeline Overview](#end-to-end-architecture--pipeline-overview)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Experiment Tracking and Registry](#experiment-tracking-and-registry)
- [Model Promotion Workflow](#model-promotion-workflow)
- [Serving and Inference API](#serving-and-inference-api)
- [Deployment](#deployment)
- [Monitoring / Observability](#monitoring--observability)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [How to Reproduce the Pipeline](#how-to-reproduce-the-pipeline)
- [Environment Variables / Secrets Needed](#environment-variables--secrets-needed)
- [Example Usage](#example-usage)
- [Testing](#testing)
- [Known Limitations / Current Caveats](#known-limitations--current-caveats)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

## Key Highlights

- End-to-end DVC pipeline from raw YAML scorecards to model registration.
- Supervised learning setup that predicts eventual first-innings total from current match state.
- Experiment tracking, artifact logging, and model registry integration through MLflow on DagsHub.
- Registry-driven model serving: the Flask app loads the latest `production`-tagged model from MLflow at startup.
- Lightweight inference container that packages the serving layer separately from the training environment.
- GitHub Actions workflow that automates pipeline execution, tests, model promotion, container build, and Kubernetes deployment on push.
- Application-side Prometheus instrumentation exposed at `/metrics`.

## Problem Statement

In a T20 match, the projected first-innings total changes ball by ball as score, overs, wickets, venue context, and recent scoring momentum evolve. This project frames that as a supervised regression problem:

> Given the current state of the first innings, predict the eventual final total.

That framing makes the task practical for both modeling and operationalization. Each training row captures a partial innings state, and the target is the final score eventually reached at the end of the innings.

## What the Model Predicts

| In scope | Out of scope |
| --- | --- |
| First-innings total score | Second-innings chase outcome |
| Men's T20 international match context | Win probability |
| Score projection from current innings state | Player-level outcomes |

## What This Project Is Really About

This repository is best understood as an MLOps portfolio project rather than a novel modeling project. The underlying estimator is an `XGBRegressor` wrapped in a scikit-learn pipeline. The more interesting engineering work is around how data is ingested, transformed, versioned, evaluated, tracked, promoted, and served in a deployment-oriented workflow.

The project demonstrates a production-style ML lifecycle:

- data ingestion from S3-compatible storage
- reproducible DVC stages
- feature generation from live match state
- tracked experiments and model artifacts
- registry-based promotion using explicit tags
- serving through a Flask app and JSON API
- deployment via Docker, GitHub Actions, and Kubernetes
- operational visibility through Prometheus metrics

## End-to-End Architecture / Pipeline Overview

```text
S3 bucket and prefix: t20s YAML scorecards
  -> data_ingestion
  -> data_preprocessing
  -> feature_engineering
  -> model_building
  -> model_evaluation
  -> model_registration
  -> MLflow / DagsHub model registry
  -> scripts/promote_model.py
  -> production-tagged model version
  -> Flask app
     -> /predict
     -> /metrics

GitHub Actions on push
  -> pipeline execution and tests
  -> Docker image build
  -> DigitalOcean Container Registry
  -> Kubernetes deployment
  -> Flask app
```

At a high level:

1. Raw match scorecards are read from S3-compatible object storage.
2. DVC orchestrates ingestion, preprocessing, feature engineering, training, evaluation, and model registration.
3. Evaluation logs metrics and artifacts to MLflow on DagsHub.
4. Registered models are tagged with a custom `stage` model-version tag.
5. The latest `stage=production` model is loaded by the Flask app at startup.
6. The serving layer exposes a browser form, a JSON prediction endpoint, and Prometheus metrics.

## Repository Structure

The repository follows a Cookiecutter Data Science-inspired layout, with the project-specific MLOps logic concentrated in the data, model, serving, and deployment paths:

```text
.
|-- .github/
|   `-- workflows/
|       `-- ci.yaml
|-- flask_app/
|   |-- app.py
|   |-- eligible_cities.txt
|   |-- requirements.txt
|   `-- templates/
|       `-- index.html
|-- notebooks/
|   |-- life-cycle.ipynb
|   `-- t20s/
|-- scripts/
|   `-- promote_model.py
|-- src/
|   |-- connections/
|   |-- data/
|   |-- features/
|   `-- model/
|-- tests/
|-- Dockerfile
|-- deployment.yaml
|-- dvc.yaml
|-- params.yaml
`-- README.md
```

Related supporting files such as `docs/`, `Makefile`, `setup.py`, and `test_environment.py` reflect the repo's evolution from a project scaffold into a more complete MLOps portfolio project.

## Data Pipeline

The pipeline is defined in `dvc.yaml` and is organized as sequential stages:

| DVC stage | Script | What it does | Primary outputs |
| --- | --- | --- | --- |
| `data_ingestion` | `src/data/data_ingestion.py` | Fetches raw YAML scorecards from the `t20s` S3-compatible bucket/prefix, flattens scorecards, filters to men's 20-over matches, and extracts first-innings delivery records | `data/raw/data.csv` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | Derives `bowling_team`, drops the original `teams` list, and filters to a curated set of supported international sides | `data/interim/interim_data.csv` |
| `feature_engineering` | `src/features/feature_engineering.py` | Builds model-ready match-state features, filters low-volume cities, shuffles the dataset, and creates the train/test split | `data/processed/train_final.csv`, `data/processed/test_final.csv`, `eligible_cities.txt`, `flask_app/eligible_cities.txt` |
| `model_building` | `src/model/model_building.py` | Trains the scikit-learn pipeline with `XGBRegressor` and serializes the model | `models/model.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | Evaluates the trained model, saves metrics, and logs artifacts to MLflow | `reports/metrics.json`, `reports/experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | Registers the model in MLflow and tags the new version as `stage=staging` | MLflow model version |

### Ingestion specifics

The ingestion code is focused on the exact supervised learning problem this repo solves:

- raw source format: YAML cricket scorecards
- storage location: S3-compatible object storage
- match filter: men's matches with `info.overs == 20`
- innings filter: first innings only
- granularity after extraction: one row per delivery

Examples of fields present in the extracted delivery-level data include:

- `match_id`
- `batting_team`
- `batsman`
- `bowler`
- `runs`
- `player_dismissed`
- `city`
- `venue`

The repo also contains sample YAML files under `notebooks/t20s/`, which are useful for inspection, but the executable ingestion path expects object storage access.

## Feature Engineering

Feature engineering converts raw delivery records into a supervised dataset where each row represents the current innings state and the label is the final total score.

### Engineered features

| Feature | Meaning |
| --- | --- |
| `batting_team` | Team currently batting |
| `bowling_team` | Opponent team derived from the match teams list |
| `city` | Match city, with missing values imputed from venue |
| `current_score` | Runs scored so far in the innings |
| `balls_left` | Deliveries remaining in a 20-over innings |
| `wickets_left` | Remaining wickets after cumulative dismissals |
| `crr` | Current run rate |
| `last_five` | Rolling runs scored over the last 30 balls |

### Target

The target column is `total_runs`, computed as the final first-innings score for the match.

### Why `eligible_cities.txt` exists

The feature engineering stage filters out low-volume cities and writes the supported city list to:

- `eligible_cities.txt`
- `flask_app/eligible_cities.txt`

This file is an important bridge artifact between training and serving. It keeps the Flask UI and containerized app aligned with the cities seen often enough during training to support inference reliably.

In practical terms, the project learns from rows of the form:

```text
current innings state -> eventual first-innings total
```

## Modeling Approach

Model training lives in `src/model/model_building.py`.

### Estimator design

- model type: `XGBRegressor`
- pipeline wrapper: scikit-learn `Pipeline`
- categorical preprocessing: `ColumnTransformer` with `OneHotEncoder`
- categorical features encoded: `batting_team`, `bowling_team`, `city`
- numerical preprocessing: remaining features passed through and then scaled with `StandardScaler`
- hyperparameter source: `params.yaml`

This is a pragmatic modeling setup rather than an attempt at algorithmic novelty. The repo uses a proven gradient-boosting regressor and focuses engineering effort on the lifecycle around it.

## Experiment Tracking and Registry

Experiment tracking is implemented in `src/model/model_evaluation.py` using MLflow with DagsHub as the remote tracking backend.

### What gets tracked

- evaluation metrics: `R2`, `MAE`, `RMSE`
- model parameters
- serialized MLflow model artifacts
- experiment metadata needed for later registration

### Authentication model

The code expects a `CAPSTONE_TEST` environment variable and uses it to set:

- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

That token is required for:

- logging evaluation metrics and artifacts to MLflow
- accessing the registry from the serving app
- running tests that load models from DagsHub/MLflow

### Registry contract

- registered model name: `my_model`
- custom model-version tag key: `stage`
- staging value: `staging`
- production value: `production`

The project uses explicit model-version tags instead of relying on classic MLflow stage transitions.

## Model Promotion Workflow

Model promotion is handled by `scripts/promote_model.py`.

The workflow is:

1. Find the newest model version tagged `stage=staging`.
2. Remove the `stage=production` tag from any currently promoted version.
3. Tag the selected staging version as `stage=production`.

This promotion scheme matters because the serving layer is registry-driven. The app does not depend on a hardcoded local model path for inference. Instead, it resolves the latest production-tagged model version from MLflow at startup.

## Serving and Inference API

The serving layer lives in `flask_app/app.py`, with the HTML interface defined in `flask_app/templates/index.html`.

### Startup behavior

On startup, the Flask app:

- authenticates to DagsHub / MLflow using `CAPSTONE_TEST`
- finds the latest version of `my_model` tagged `stage=production`
- loads that model through MLflow's Python function interface

### Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Render the HTML prediction form |
| `/predict` | `POST` | Serve predictions from either form data or JSON |
| `/metrics` | `GET` | Expose Prometheus metrics |

### Browser form flow

The form-based UI accepts:

- batting team
- bowling team
- city
- current score
- overs done
- wickets out
- runs in last 5 overs

Before inference, the app converts form values into the feature space used during training:

- `overs` -> `balls_left`
- `wickets out` -> `wickets_left`
- `current_score / overs_done` -> `crr`

### JSON API flow

The JSON API is slightly lower level than the browser form. It expects model-ready fields such as:

- `batting_team`
- `bowling_team`
- `city`
- `current_score`
- `balls_left`
- `wickets_left`
- `crr`
- `last_five`

The response is a rounded score prediction.

## Deployment

Deployment is represented in the repo as a DigitalOcean-targeted CI/CD path rather than a complete platform blueprint.

### What is implemented in the repository

- GitHub Actions workflow: `.github/workflows/ci.yaml`
- container build: `Dockerfile`
- Kubernetes manifest: `deployment.yaml`

### CI/CD flow on push

The GitHub Actions workflow currently:

1. checks out the repository
2. installs Python dependencies
3. runs `dvc repro`
4. runs the model tests
5. promotes the latest staging model to production
6. runs Flask app tests
7. builds the Docker image
8. pushes the image to DigitalOcean Container Registry
9. applies the Kubernetes manifest and restarts the deployment

### Containerization details

The Docker image is intentionally slim:

- base image: `python:3.10-slim`
- copied code: only `flask_app/`
- install source: `flask_app/requirements.txt`
- runtime server: `gunicorn`

This keeps serving lighter than the full training environment. The image does not bundle a local production model artifact. Instead, the app pulls the current production-tagged model from MLflow/DagsHub at startup.

### Kubernetes details

`deployment.yaml` defines:

- a deployment with `2` replicas
- a `LoadBalancer` service
- runtime injection of `CAPSTONE_TEST` from a Kubernetes secret

## Monitoring / Observability

Application-side Prometheus instrumentation is implemented directly in the Flask app using `prometheus_client`.

The app exports:

- request count
- request latency

Metrics are exposed at:

```text
GET /metrics
```

The repo's current monitoring story is intentionally described in an honest way:

- implemented in code: Prometheus-compatible application metrics
- documented operationally: Prometheus/Grafana setup and usage notes
- not fully present as code: a complete monitoring stack expressed as repository-managed infrastructure manifests

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data access | `boto3`, YAML scorecards, S3-compatible object storage |
| Pipeline orchestration | DVC |
| Data processing | pandas, NumPy |
| Modeling | scikit-learn, XGBoost |
| Experiment tracking and registry | MLflow, DagsHub |
| Serving | Flask, Jinja2, Gunicorn |
| Observability | `prometheus_client` |
| Containerization | Docker |
| Deployment | GitHub Actions, DigitalOcean Container Registry, Kubernetes |
| Testing | Python `unittest` |

## How to Run Locally

### Prerequisites

- Python 3.10 is the safest baseline because the CI workflow and Docker image both use it.
- Access to the raw YAML scorecards in the S3-compatible `t20s` bucket/prefix.
- A DagsHub / MLflow token stored in `CAPSTONE_TEST`.

### 1. Install dependencies

```bash
git clone https://github.com/srikara202/Cricket-T20-Score-Predictor-MLOps.git
cd Cricket-T20-Score-Predictor-MLOps
pip install -r requirements.txt
```

### 2. Set required environment variables

Bash:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export CAPSTONE_TEST=your_dagshub_token
```

PowerShell:

```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:CAPSTONE_TEST="your_dagshub_token"
```

### 3. Reproduce the pipeline

```bash
dvc repro
```

### 4. Start the Flask app

```bash
cd flask_app
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## How to Reproduce the Pipeline

The main reproducibility entry point is:

```bash
dvc repro
```

What this does in practice:

- pulls raw scorecards from S3-compatible storage during ingestion
- generates intermediate and processed datasets under `data/`
- trains the model and saves `models/model.pkl`
- evaluates the model and writes report artifacts
- registers the model in MLflow as `my_model` with `stage=staging`

Generated artifacts include:

- `data/raw/data.csv`
- `data/interim/interim_data.csv`
- `data/processed/train_final.csv`
- `data/processed/test_final.csv`
- `eligible_cities.txt`
- `flask_app/eligible_cities.txt`
- `models/model.pkl`
- `reports/metrics.json`
- `reports/experiment_info.json`

Important practical note: reproducibility here depends on external services and credentials. This is not a fully self-contained offline pipeline.

## Environment Variables / Secrets Needed

| Variable | Required for | Where it is used |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | Local pipeline runs | Raw data ingestion from S3-compatible storage |
| `AWS_SECRET_ACCESS_KEY` | Local pipeline runs | Raw data ingestion from S3-compatible storage |
| `CAPSTONE_TEST` | Local evaluation, registration, app runtime, tests, CI | DagsHub / MLflow authentication |
| `DO_TOKEN` | CI/CD deployment only | Authenticate to DigitalOcean |
| `DO_REGISTRY` | CI/CD deployment only | Push Docker image to DigitalOcean Container Registry |
| `DO_CLUSTER_NAME` | CI/CD deployment only | Fetch kubeconfig and deploy to Kubernetes |

Runtime note: the serving app only needs `CAPSTONE_TEST` at deployment time because model loading is registry-driven.

## Example Usage

### Browser form

Start the app locally, open `/`, and enter:

- batting team
- bowling team
- city
- current score
- overs done
- wickets out
- runs in the last 5 overs

The app converts those values into training-compatible features and returns a rounded projected first-innings total.

### JSON prediction request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "batting_team": "India",
    "bowling_team": "England",
    "city": "London",
    "current_score": 80,
    "balls_left": 60,
    "wickets_left": 6,
    "crr": 8.0,
    "last_five": 30
  }'
```

Response shape:

```text
{"predicted_score": <integer>}
```

### Metrics endpoint

```bash
curl http://127.0.0.1:5000/metrics
```

## Testing

The repository includes two primary tests:

| Test | Command | Notes |
| --- | --- | --- |
| Model test | `python -m unittest tests/test_model.py` | Loads the latest `stage=staging` model from MLflow/DagsHub and evaluates it against local processed test data |
| Flask app test | `python -m unittest tests/test_flask_app.py` | Imports the Flask app, exercises `/` and `/predict`, and therefore still depends on registry access because the app loads a production-tagged model at import time |

Practical testing notes:

- `CAPSTONE_TEST` is required for both tests.
- The model test expects generated pipeline outputs such as `data/processed/test_final.csv`.
- These tests are integration-leaning rather than fully isolated unit tests.

## Known Limitations / Current Caveats

- The executable pipeline depends on external services: S3-compatible object storage for raw data and DagsHub / MLflow for tracking and registry access.
- Generated outputs such as processed datasets and evaluation reports are not all committed to the repository by default.
- `dvc.lock` reflects a previous successful run and may not match the current `params.yaml` exactly.
- Team coverage is limited to a curated list of international sides defined in preprocessing.
- City coverage is limited to cities that survive the volume threshold in feature engineering.
- The monitoring stack is only partly codified in the repo; application metrics are implemented, but full Prometheus/Grafana infrastructure is described more in documentation and notes than in deployment manifests.
- The serving container assumes a valid production-tagged model already exists in the MLflow registry and that credentials are available at runtime.
- The project focuses only on men's T20 first-innings score prediction and does not extend to second-innings strategy, win prediction, or other formats.

## Future Improvements

- Add stronger schema validation and input checks across ingestion and prediction paths.
- Improve test isolation by introducing local fixtures or mocked registry/model-loading paths.
- Expand feature engineering with richer venue, match-context, or recent-form signals.
- Add more formal monitoring-as-code for Prometheus, Grafana, and alerting configuration.
- Introduce data quality and drift monitoring around incoming inference traffic.
- Decouple training and serving environments further so local setup can be lighter.
- Broaden support for additional teams, cities, and possibly adjacent match formats.

## Conclusion

This repository is a well-scoped, operationally aware MLOps project built around a clear cricket forecasting problem. It is strongest as an example of how a conventional ML model can be turned into a versioned, tracked, promoted, and deployable system with honest attention to reproducibility, serving, and observability.
