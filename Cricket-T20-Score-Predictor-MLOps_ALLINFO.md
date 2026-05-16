# Cricket-T20-Score-Predictor-MLOps ALLINFO

Generated from repository evidence only. Secret values are not copied; credential-bearing variables and config fields are named with values redacted.

## 1. Executive Summary

This repository is an end-to-end MLOps project for predicting the eventual first-innings score in an international men's T20 cricket match from the current innings state. The model itself is a conventional supervised regression pipeline using scikit-learn preprocessing and `XGBRegressor`; the main value is the full lifecycle around that model: YAML scorecard ingestion, DVC orchestration, feature engineering, model training, MLflow/DagsHub tracking, registry tagging and promotion, Flask serving, Prometheus metrics, Docker packaging, GitHub Actions, and Kubernetes deployment manifests.

- **Problem solved:** Given live or partial first-innings match context, estimate the final first-innings total so users can quickly reason about the projected score.
- **Target users:** ML/MLOps learners, interviewers/reviewers, developers maintaining the repo, and cricket analytics users who want a lightweight score projection demo.
- **Core workflow:** Fetch raw T20 YAML scorecards -> extract first-innings delivery rows -> clean/filter teams and cities -> engineer match-state features -> train an XGBoost regression pipeline -> evaluate and log to MLflow -> register and promote model versions -> serve predictions through Flask.
- **Main technical achievement:** The project demonstrates a production-style ML lifecycle more than a novel algorithm: reproducible stages with DVC, external object storage, remote experiment tracking, a custom tag-based promotion flow, registry-driven inference, CI/CD, containerization, Kubernetes deployment, and Prometheus instrumentation.
- **Interview pitch:** "I built an end-to-end MLOps pipeline for T20 cricket score prediction. It transforms raw YAML scorecards into delivery-level training data, engineers match-state features like current score, balls left, wickets left, current run rate, and last-five-over momentum, trains an XGBoost regressor, tracks and registers models in MLflow on DagsHub, promotes versions with explicit staging/production tags, and serves the production model through a Flask app instrumented with Prometheus and deployable via Docker, GitHub Actions, and Kubernetes."

## 2. Project Metadata

| Item | Evidence-based value |
|---|---|
| Inferred project name | `Cricket-T20-Score-Predictor-MLOps` from `Makefile` `PROJECT_NAME`, README title, DagsHub repo names, and repository directory |
| Repository type | Python ML/MLOps application with Flask serving layer, DVC pipeline, docs scaffold, CI/CD, and deployment manifests |
| Main language(s) | Python, YAML, HTML/Jinja2, shell/Makefile, reStructuredText |
| Main ML libraries | pandas, NumPy, scikit-learn, XGBoost, MLflow, DagsHub |
| Data/storage libraries | PyYAML, boto3, DVC with `dvc-s3`; optional/legacy SQL Server access through pyodbc |
| Serving libraries | Flask, Jinja2 templates, MLflow pyfunc loading, prometheus_client |
| Build/package tools | setuptools via `pyproject.toml` and `setup.py`; pip requirements files; Docker |
| Runtime assumptions | Python 3.10 is used in CI and Docker; `model_dir` metadata was generated with Python 3.13.5; external credentials are needed for S3 and MLflow/DagsHub flows |
| External services/integrations | S3-compatible bucket/remote `s3://t20s`; DagsHub MLflow tracking/registry; DigitalOcean Container Registry; DigitalOcean Kubernetes; Prometheus-compatible metrics endpoint |
| Important app entrypoints | `flask_app/app.py`, routes `/`, `/predict`, `/metrics` |
| Important pipeline entrypoints | `dvc.yaml`, `src/data/data_ingestion.py`, `src/data/data_preprocessing.py`, `src/features/feature_engineering.py`, `src/model/model_building.py`, `src/model/model_evaluation.py`, `src/model/register_model.py` |
| Promotion entrypoint | `scripts/promote_model.py` |
| Important config files | `params.yaml`, `dvc.yaml`, `dvc.lock`, `.dvc/config`, `.github/workflows/ci.yaml`, `Dockerfile`, `deployment.yaml`, `flask_app/requirements.txt`, `requirements.txt`, `model_dir/MLmodel`, `tox.ini` |
| Test commands found | `python -m unittest tests/test_model.py`; `python -m unittest tests/test_flask_app.py`; `python test_environment.py`; `make test_environment`; `flake8 src` via `make lint` |
| Run/build commands found | `pip install -r requirements.txt`; `dvc repro`; `cd flask_app` then `python app.py`; `docker build -t flask-app:latest .`; docs `make html`; Kubernetes `kubectl apply -f deployment.yaml` |
| Deployment clues | GitHub Actions workflow builds and pushes Docker image to DigitalOcean registry and deploys to DOKS; `deployment.yaml` runs 2 Flask replicas behind a LoadBalancer; `CAPSTONE_TEST` comes from Kubernetes secret `capstone-secret` |
| Tracked files | 204 files from `git ls-files` |
| Tracked folders | 20 tracked folders from `git ls-files` parent directories |
| Notable untracked files | None observed before creating this ALLINFO file; `git status --short` returned no output |

## 3. Quick Start Guide

### Prerequisites

- Python 3.10 is the safest local target because `.github/workflows/ci.yaml` and `Dockerfile` both use Python 3.10.
- pip.
- DVC with S3 support, installed through `requirements.txt` (`dvc==3.53.0`, `dvc-s3==3.2.0`).
- AWS/S3-compatible credentials for the `t20s` bucket used by `src/data/data_ingestion.py`.
- A DagsHub/MLflow token exposed as `CAPSTONE_TEST`.
- For Docker deployment, Docker and a registry.
- For Kubernetes deployment, `kubectl`, DigitalOcean `doctl`, `DO_TOKEN`, `DO_REGISTRY`, and `DO_CLUSTER_NAME` in CI.

### Install Steps

```bash
pip install -r requirements.txt
```

`requirements.txt` ends with `-e .`, so the local package named `src` from `setup.py` is installed editable.

### Environment Variables

| Variable | Required for | Value handling |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | DVC pipeline ingestion from S3-compatible object storage | Secret; use `[REDACTED]` in docs/logs |
| `AWS_SECRET_ACCESS_KEY` | DVC pipeline ingestion from S3-compatible object storage | Secret; use `[REDACTED]` in docs/logs |
| `CAPSTONE_TEST` | MLflow/DagsHub evaluation, registration, promotion, Flask startup, and tests | Secret token; app maps it to `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` |
| `DO_TOKEN` | GitHub Actions DigitalOcean auth | Secret |
| `DO_REGISTRY` | GitHub Actions image tagging/pushing | Secret or private registry name |
| `DO_CLUSTER_NAME` | GitHub Actions DOKS kubeconfig retrieval | Secret or environment-specific value |

`src/connections/config.json` contains SQL Server config keys `server`, `database`, `table`, `username`, and `pass`. Values are intentionally not copied here.

### Reproduce The Pipeline

```bash
dvc repro
```

This runs the stages in `dvc.yaml`:

1. `data_ingestion`
2. `data_preprocessing`
3. `feature_engineering`
4. `model_building`
5. `model_evaluation`
6. `model_registration`

Important limitation: this is not an offline-only pipeline. It requires object storage credentials for ingestion and `CAPSTONE_TEST` for MLflow/DagsHub evaluation and registry registration.

### Start The Flask App

```bash
cd flask_app
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

Important runtime behavior: importing `flask_app/app.py` immediately requires `CAPSTONE_TEST`, queries the MLflow registry for the newest `my_model` version tagged `stage=production`, and loads that model. The app will not start if the token is absent, the registry is unreachable, or no production-tagged model exists.

### Build And Deploy

```bash
docker build -t flask-app:latest .
```

The Dockerfile copies only `flask_app/` into `/app`, installs `flask_app/requirements.txt`, exposes port 5000, and starts `gunicorn`. Repository evidence shows `gunicorn` is referenced by the Dockerfile but is not listed in `flask_app/requirements.txt`, which is an operational risk.

Kubernetes deployment:

```bash
kubectl apply -f deployment.yaml
```

### Test Commands

```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_flask_app.py
python test_environment.py
```

Test caveats from code:

- `tests/test_model.py` loads the newest `stage=staging` model from MLflow/DagsHub and reads `data/processed/test_final.csv`, which is a generated DVC output not tracked in Git.
- `tests/test_flask_app.py` imports `flask_app.app`, so it also requires `CAPSTONE_TEST` and a production-tagged model at import time.
- There are no isolated unit tests for pure feature engineering, ingestion parsing, validation, or model training functions.

### Common Troubleshooting

- **`CAPSTONE_TEST environment variable is not set`:** Set the DagsHub/MLflow token before running evaluation, registration, promotion, tests, or the Flask app.
- **S3 ingestion fails:** Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; confirm the `t20s` bucket/prefix exists and contains YAML files.
- **Flask app fails at import/startup:** Check MLflow tracking URI, token, registered model name `my_model`, and that a model version has tag `stage=production`.
- **Model tests fail because file is missing:** Run `dvc repro` first so `data/processed/test_final.csv` exists.
- **Docker container cannot start `gunicorn`:** Add `gunicorn` to `flask_app/requirements.txt` or change the Docker command to a server that is installed. This is a repository-visible gap; do not assume it works as-is.
- **Bootstrap CSS/JS not loading in browser:** `flask_app/templates/index.html` uses placeholder `integrity="sha384-..."`, which may cause browser Subresource Integrity checks to block CDN assets.
- **DVC lock mismatch:** `params.yaml` currently says `feature_engineering.test_size: 0.5`, `learning_rate: 0.1`, `max_depth: 10`; `dvc.lock` records older values `test_size: 0.2`, `learning_rate: 0.2`, `max_depth: 12`.

## 4. What The Project Does

### Product/Application Behavior

The application predicts a rounded final first-innings T20 score from current match conditions. It has a browser form and a JSON API. The browser form collects human-friendly values and computes model-ready features; the JSON API expects model-ready fields directly.

### Main User-Facing Features

- Browser form at `/` for team, city, current score, overs done, wickets out, and runs in last five overs.
- Prediction endpoint at `/predict`.
- JSON prediction support when `Content-Type: application/json`.
- HTML prediction result display.
- Validation messages for invalid form inputs.
- Prometheus metrics endpoint at `/metrics`.

### Main Developer-Facing Features

- DVC pipeline for reproducible ML stages.
- Parameter file `params.yaml` for feature split and XGBoost hyperparameters.
- MLflow/DagsHub experiment logging, model artifact saving, and registry registration.
- Tag-based promotion from staging to production.
- CI pipeline that runs DVC, tests, promotion, Docker build, registry push, and Kubernetes deployment.
- Sphinx docs scaffold.
- Cookiecutter-style project skeleton with `src/`, `docs/`, `notebooks/`, `reports/`, `references/`, and `tests/`.

### Inputs And Outputs

| Layer | Inputs | Outputs |
|---|---|---|
| Ingestion | YAML scorecards from S3 bucket/prefix `t20s`; local sample YAMLs in `notebooks/t20s` for inspection | `data/raw/data.csv` generated by DVC |
| Preprocessing | Delivery rows with teams, batting side, ball, runs, dismissals, city, venue | `data/interim/interim_data.csv` |
| Feature engineering | Cleaned delivery rows | `data/processed/train_final.csv`, `data/processed/test_final.csv`, `eligible_cities.txt`, `flask_app/eligible_cities.txt` |
| Training | `train_final.csv`; `params.yaml` model settings | `models/model.pkl` generated by DVC |
| Evaluation | `models/model.pkl`, `test_final.csv` | `reports/metrics.json`, `reports/experiment_info.json`, MLflow metrics/artifacts |
| Registration | `reports/experiment_info.json` | MLflow model version tagged `stage=staging` |
| Promotion | MLflow registry model versions | Newest staging version tagged `stage=production` |
| Serving | Browser form or JSON request | Rounded integer `predicted_score` or rendered HTML result |

### Happy Path

1. Raw YAML scorecards exist in the S3-compatible `t20s` bucket.
2. `dvc repro` ingests, transforms, trains, evaluates, and registers a staging model.
3. `scripts/promote_model.py` promotes a staging model version to production.
4. Flask starts with `CAPSTONE_TEST`, resolves the newest production model, loads it through MLflow pyfunc, renders `/`, accepts `/predict`, and emits `/metrics`.
5. CI/CD builds and deploys the app container to DigitalOcean Kubernetes.

### Important Failure Paths

- Missing AWS credentials stops data ingestion.
- Missing `CAPSTONE_TEST` stops model evaluation at import time, registration, promotion, tests, and app startup.
- No `stage=staging` model stops promotion.
- No `stage=production` model stops Flask startup.
- Missing generated DVC outputs stops downstream stages and tests.
- Unexpected YAML schema can break `extract_delivery_df`, especially around `innings[0]['1st innings']['deliveries']`, required `info.*` columns, or missing `info.city`.
- Form validation catches missing/non-numeric current score, overs outside 5 to 19, wickets outside 0 to 9, and last-five runs greater than current score.
- JSON API catches missing model-ready numeric fields with HTTP 400, but does not deeply validate field ranges.

### Known Limitations Visible From Code

- The app is tightly coupled to an external MLflow registry at import time.
- Local pipeline reproducibility depends on external object storage and secrets.
- Tests are integration-heavy and require external services.
- `dvc.lock` does not match current `params.yaml`.
- `feature_engineering.py` writes tracked text files as a side effect, but `dvc.yaml` does not declare those text files as stage outputs.
- Docker runtime references `gunicorn`, but the Flask requirements file does not list it.
- Kubernetes manifest lacks readiness/liveness probes, resource requests/limits, and autoscaling.
- `Makefile` target `data` references `src/data/make_dataset.py`, which is not tracked.
- Several imports are unused or duplicated.

## 5. High-Level Architecture

```text
Raw T20 YAML scorecards
  stored in S3-compatible bucket/prefix t20s
        |
        v
src/data/data_ingestion.py
  -> pandas-normalized match data
  -> first-innings delivery records
        |
        v
src/data/data_preprocessing.py
  -> bowling_team derivation
  -> curated team filtering
        |
        v
src/features/feature_engineering.py
  -> city filtering
  -> current_score, balls_left, wickets_left, crr, last_five
  -> train/test CSVs
        |
        v
src/model/model_building.py
  -> sklearn Pipeline
  -> OneHotEncoder + StandardScaler + XGBRegressor
        |
        v
src/model/model_evaluation.py
  -> metrics JSON
  -> MLflow model artifacts in DagsHub
        |
        v
src/model/register_model.py
  -> model name my_model
  -> tag stage=staging
        |
        v
scripts/promote_model.py
  -> newest staging tag becomes production
        |
        v
flask_app/app.py
  -> loads newest stage=production model
  -> routes /, /predict, /metrics
```

### Major Modules

| Module/path | Responsibility | Communication |
|---|---|---|
| `.dvc/`, `dvc.yaml`, `dvc.lock` | DVC remote and pipeline orchestration | Stages call Python scripts and track generated outputs/metrics |
| `src/connections/` | S3 and optional SQL Server data access helpers | `data_ingestion.py` imports `s3_connection` |
| `src/data/` | Raw scorecard loading and delivery-level preprocessing | Produces raw and interim CSV outputs |
| `src/features/` | Model feature generation and train/test split | Produces processed CSVs and city list files |
| `src/model/` | Training, evaluation, MLflow logging, and registration | Reads processed/model/report artifacts, writes model and report outputs |
| `scripts/` | Operational model promotion | Calls MLflow registry API |
| `flask_app/` | Runtime inference service and UI | Loads MLflow pyfunc model, handles HTTP requests |
| `.github/workflows/` | CI/CD automation | Runs DVC/tests/promotion/container/deploy commands |
| `deployment.yaml` | Kubernetes runtime definition | Injects secret and exposes LoadBalancer |

### Frontend/Backend Split

There is no separate JavaScript frontend. The UI is server-rendered Flask/Jinja2 in `flask_app/templates/index.html`. The backend is the same Flask app in `flask_app/app.py`.

### Database/Storage Layer

- Primary executable data source: S3-compatible object storage via boto3 in `src/connections/s3_connection.py`.
- DVC remote: `.dvc/config` points to `s3://t20s`.
- Optional/legacy SQL Server helpers exist in `src/connections/ssms_connection.py` and `src/connections/ssms_connection_old.py`, configured by `src/connections/config.json`.
- Generated local storage expected by DVC: `data/raw`, `data/interim`, `data/processed`, `models/model.pkl`, `reports/metrics.json`, `reports/experiment_info.json`; most are ignored or DVC-managed rather than tracked.

### Authentication/Authorization

There is no application user authentication. Operational auth is environment-variable based:

- AWS/S3 credentials for ingestion.
- `CAPSTONE_TEST` for MLflow/DagsHub.
- GitHub Actions secrets for DigitalOcean deployment.
- Kubernetes secret `capstone-secret` supplies `CAPSTONE_TEST` to the app.

### AI/ML Components

- Feature engineering converts delivery state into tabular features.
- The trained model is a scikit-learn `Pipeline`.
- Categorical fields `batting_team`, `bowling_team`, `city` are one-hot encoded.
- Remaining numeric fields pass through and are scaled.
- `XGBRegressor` predicts `total_runs`.
- MLflow pyfunc is used for serving.

### Background Jobs/Workflows

No runtime background workers are implemented. The important background workflows are CI/CD and DVC stages triggered manually or by GitHub Actions on push.

### Error Handling And Observability

- Pipeline scripts use `try`/`except`, log errors, and re-raise for many helper functions.
- Some `main()` functions catch exceptions, log, and print rather than re-raising, which may hide CI failures if a script exits with code 0 after printing an error.
- `src/logger/__init__.py` configures root logging to console and rotating log files under `logs/`.
- Flask app uses `logging.exception` for prediction errors and exposes Prometheus counters/histograms.

## 6. End-to-End Workflows

### Workflow A: DVC Training Pipeline

| Step | Trigger | Files/functions | Inputs | Outputs | Side effects/failure behavior |
|---|---|---|---|---|---|
| 1 | `dvc repro` stage `data_ingestion` | `src/data/data_ingestion.py`, `s3_connection.s3_operations.fetch_yaml_folder_from_s3`, `extract_delivery_df`, `save_data` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, S3 bucket `t20s` | `data/raw/data.csv` | Logs; skips per-file YAML parsing errors in S3 helper; `main()` catches final exceptions and prints |
| 2 | DVC dependency on `data/raw` | `src/data/data_preprocessing.py`, `preprocess_dataframe` | Raw delivery CSV | `data/interim/interim_data.csv` | Derives bowling side by parsing `teams`; filters supported teams |
| 3 | DVC dependency on `data/interim` | `src/features/feature_engineering.py`, `engineer_and_split` | Interim CSV, `params.yaml` `feature_engineering.test_size` | `data/processed/train_final.csv`, `data/processed/test_final.csv` | Also writes `eligible_cities.txt` and `flask_app/eligible_cities.txt` though not declared in `dvc.yaml` |
| 4 | DVC dependency on `data/processed` | `src/model/model_building.py`, `build_and_train_model`, `save_model` | `train_final.csv`, `params.yaml` model settings | `models/model.pkl` | Trains XGBoost; writes pickle |
| 5 | DVC dependency on model | `src/model/model_evaluation.py`, `evaluate_model`, `save_metrics`, `save_model_info` | `models/model.pkl`, `test_final.csv`, `CAPSTONE_TEST` | `reports/metrics.json`, `reports/experiment_info.json`, MLflow artifacts | Raises at module import if `CAPSTONE_TEST` missing |
| 6 | DVC dependency on report info | `src/model/register_model.py` | `reports/experiment_info.json`, `CAPSTONE_TEST` | MLflow model version tagged `stage=staging` | Fails if token, run id, or registry unavailable |

### Workflow B: Model Promotion

- **Trigger:** `python scripts/promote_model.py`, or CI step "Promote model to production".
- **Files/functions:** `scripts/promote_model.py:main`.
- **Flow:**
  1. Reads `CAPSTONE_TEST`.
  2. Sets `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.
  3. Points MLflow at `https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow`.
  4. Searches versions of model `my_model`.
  5. Filters versions whose model-version tag `stage` equals `staging`.
  6. Chooses newest by `last_updated_timestamp`.
  7. Deletes `stage` tag from any production versions.
  8. Tags the chosen version as `stage=production`.
- **Output:** Updated MLflow registry tags.
- **Failure behavior:** Raises `EnvironmentError` if token missing; raises `ValueError` if no staging version exists.

### Workflow C: Flask Startup

- **Trigger:** `python flask_app/app.py`, Gunicorn import `app:app`, or tests importing `flask_app.app`.
- **Files/functions:** `flask_app/app.py`, `get_model_version_by_stage`.
- **Flow:**
  1. Reads `CAPSTONE_TEST`.
  2. Sets MLflow auth environment variables.
  3. Sets tracking URI to DagsHub.
  4. Searches model versions for `my_model`.
  5. Selects latest model version with tag `stage=production`.
  6. Loads `models:/my_model/<version>` through `mlflow.pyfunc.load_model`.
  7. Reads `eligible_cities.txt` from current working directory.
  8. Creates Flask app and Prometheus metrics.
- **Output:** Importable Flask `app` object and loaded global `model`.
- **Failure behavior:** Startup fails before route handling if token/registry/model/city file is unavailable.

### Workflow D: Browser Prediction

- **Trigger:** User posts HTML form to `/predict`.
- **Files/functions:** `flask_app/app.py:predict`, `flask_app/templates/index.html`.
- **Inputs:** `batting_team`, `bowling_team`, `city`, `current_score`, `overs`, `wickets`, `last_five`.
- **Flow:**
  1. Increment Prometheus request counter.
  2. Read form values with Flask type conversion.
  3. Validate current score, overs in 5 to 19, wickets in 0 to 9, and last-five runs.
  4. Compute `balls_bowled = int(overs_done * 6)`.
  5. Compute `balls_left = max(120 - balls_bowled, 0)`.
  6. Compute `wickets_left = max(10 - wickets_out, 0)`.
  7. Compute `crr = round(current_score / overs_done, 2)`.
  8. Build one-row pandas DataFrame with training-compatible feature names.
  9. Call `model.predict`.
  10. Round to int and render template.
  11. Observe latency.
- **Output:** HTML page with result or validation/prediction error.
- **Side effects:** Prometheus metrics update.

### Workflow E: JSON Prediction

- **Trigger:** POST `/predict` with JSON content type.
- **Inputs:** `batting_team`, `bowling_team`, `city`, `current_score`, `balls_left`, `wickets_left`, `crr`, `last_five`.
- **Flow:** Reads payload, supplies defaults only for categorical fields, requires numeric feature fields, builds DataFrame, predicts, rounds, returns JSON.
- **Output:** `{"predicted_score": <int>}`.
- **Errors:** Missing numeric fields return HTTP 400; prediction exceptions return HTTP 500 with error text.

### Workflow F: CI/CD

- **Trigger:** GitHub Actions `on: [push]`.
- **Files:** `.github/workflows/ci.yaml`, `Dockerfile`, `deployment.yaml`, `scripts/promote_model.py`, tests.
- **Flow:** Checkout -> setup Python 3.10 -> cache pip -> install root requirements -> run `dvc repro` with secrets -> run model tests -> promote model -> run Flask tests -> log in to DOCR -> build/tag/push Docker image -> install kubectl and save DOKS kubeconfig -> create/update Kubernetes secret -> apply deployment -> restart deployment.
- **Failure behavior:** Later steps use `if: success()` on promotion and Flask tests, so they require earlier steps to pass.

## 7. Data Model, State, And Configuration

### Raw YAML Scorecard Shape

Tracked sample YAML files under `notebooks/t20s` share top-level keys:

- `meta`
- `info`
- `innings`

A sampled file `211028.yaml` has `info` keys including `balls_per_over`, `city`, `dates`, `gender`, `match_type`, `outcome`, `overs`, `player_of_match`, `players`, `registry`, `teams`, `toss`, `umpires`, and `venue`. `innings` is a list. The ingestion code expects `row['innings'][0]['1st innings']['deliveries']`.

### Delivery DataFrame

`extract_delivery_df` produces delivery-level rows with:

- `match_id`
- `teams`
- `batting_team`
- `ball`
- `batsman`
- `bowler`
- `runs`
- `player_dismissed`
- `city`
- `venue`

### Interim DataFrame

`preprocess_dataframe` returns:

- `match_id`
- `batting_team`
- `bowling_team`
- `ball`
- `runs`
- `player_dismissed`
- `city`
- `venue`

It derives `bowling_team` from `teams` and filters batting/bowling teams to a hardcoded list of supported sides.

### Final Modeling Dataset

`engineer_and_split` returns train/test DataFrames with:

- `batting_team`
- `bowling_team`
- `city`
- `current_score`
- `balls_left`
- `wickets_left`
- `crr`
- `last_five`
- `total_runs`

The target is `total_runs`.

### API Request/Response Shape

JSON request:

```json
{
  "batting_team": "India",
  "bowling_team": "England",
  "city": "London",
  "current_score": 80,
  "balls_left": 60,
  "wickets_left": 6,
  "crr": 8.0,
  "last_five": 30
}
```

JSON response:

```json
{"predicted_score": 160}
```

The score above is shape-only; repository evidence does not guarantee that specific numeric prediction.

### Parameters

`params.yaml`:

| Key | Current value | Used by |
|---|---:|---|
| `feature_engineering.test_size` | `0.5` | `src/features/feature_engineering.py` |
| `model_building.n_estimators` | `1000` | `src/model/model_building.py` |
| `model_building.learning_rate` | `0.1` | `src/model/model_building.py` |
| `model_building.max_depth` | `10` | `src/model/model_building.py` |

`dvc.lock` records older values for some params, so the lockfile should be regenerated after intentional parameter changes.

### Model Registry Contract

- MLflow tracking URI: DagsHub URL for this repository.
- Registered model name: `my_model`.
- Model-version tag key: `stage`.
- Staging tag value: `staging`.
- Production tag value: `production`.

### State Management

- Training state is file-based and DVC-controlled.
- Runtime serving state is process-global: `model`, `MODEL_VERSION`, `TEAMS`, and `CITIES` are loaded at import/startup.
- Metrics state is in a custom Prometheus `CollectorRegistry`.

### Config Files

- `.dvc/config`: default DVC remote `myremote`, URL `s3://t20s`.
- `.github/workflows/ci.yaml`: CI/CD steps and secret injection.
- `deployment.yaml`: Kubernetes replicas, image, port, and secret-backed env.
- `Dockerfile`: serving image build.
- `model_dir/*.yaml` and `model_dir/MLmodel`: MLflow saved model environment and flavor metadata.
- `src/connections/config.json`: SQL Server config keys with sensitive values redacted.

## 8. API, Routes, Commands, And Entrypoints

| Entrypoint | Type | Called by | Calls/uses | Result |
|---|---|---|---|---|
| `flask_app/app.py` module import | App bootstrap | `python app.py`, Gunicorn, tests | MLflow registry, `eligible_cities.txt`, Flask, Prometheus | Loaded app/model |
| `/` | HTTP GET | Browser/test client | `home()` renders `index.html` | Form page |
| `/predict` | HTTP POST | Browser form or JSON client | `predict()`, pandas, MLflow model | HTML result or JSON prediction |
| `/metrics` | HTTP GET | Prometheus/curl | `generate_latest(registry)` | Prometheus text exposition |
| `dvc.yaml:data_ingestion` | DVC stage | `dvc repro` | `python src/data/data_ingestion.py` | Raw CSV output |
| `dvc.yaml:data_preprocessing` | DVC stage | DVC | `python src/data/data_preprocessing.py` | Interim CSV |
| `dvc.yaml:feature_engineering` | DVC stage | DVC | `python src/features/feature_engineering.py` | Processed train/test CSV |
| `dvc.yaml:model_building` | DVC stage | DVC | `python src/model/model_building.py` | Pickled model |
| `dvc.yaml:model_evaluation` | DVC stage | DVC | `python src/model/model_evaluation.py` | Metrics/report info and MLflow artifacts |
| `dvc.yaml:model_registration` | DVC stage | DVC | `python src/model/register_model.py` | MLflow model version tagged staging |
| `scripts/promote_model.py` | Ops script | CI or manual | MLflow registry | Promotes staging to production |
| `tests/test_model.py` | Test command | unittest/CI | MLflow registry and processed test data | Model load/signature/performance checks |
| `tests/test_flask_app.py` | Test command | unittest/CI | Flask test client | Home and prediction endpoint checks |
| `test_environment.py` | Utility script | Makefile/manual | `sys.version_info` | Validates Python major version |
| `Makefile` | Developer commands | `make ...` | pip, flake8, awscli, conda/virtualenv | Environment, lint, sync helper tasks |
| `docs/Makefile`, `docs/make.bat` | Docs commands | `make html` or `make.bat html` | Sphinx | Documentation builds |
| `.github/workflows/ci.yaml` | CI/CD | GitHub push | DVC, tests, Docker, doctl, kubectl | Build/test/deploy pipeline |

## 9. Full Repository Map

### Folder Purposes

| Folder | Purpose |
|---|---|
| `.dvc/` | DVC repository metadata and remote configuration |
| `.github/workflows/` | GitHub Actions CI/CD workflow |
| `docs/` | Sphinx documentation scaffold |
| `flask_app/` | Flask serving app, runtime requirements, HTML template, city list |
| `model_dir/` | Tracked MLflow saved model artifact metadata plus binary model pickle |
| `notebooks/` | Exploratory notebook and tracked sample T20 YAML scorecards |
| `notebooks/t20s/` | 137 tracked YAML match scorecards |
| `references/` | Placeholder for reference material |
| `reports/` | Placeholder and ignore rules for generated evaluation reports |
| `reports/figures/` | Placeholder for generated figures |
| `scripts/` | Operational scripts, currently model promotion |
| `src/` | Installable Python package and pipeline implementation |
| `src/connections/` | S3 and SQL Server connection helpers |
| `src/data/` | Data ingestion and preprocessing stages |
| `src/features/` | Feature engineering stage |
| `src/logger/` | Shared logging setup |
| `src/model/` | Model training, evaluation, and registry stages |
| `src/visualization/` | Placeholder visualization package |
| `tests/` | unittest integration tests |

### Tracked File Map

| Path | One-line purpose |
|---|---|
| `.dvc/.gitignore` | Ignores local DVC config, cache, and temp internals |
| `.dvc/config` | DVC default remote config pointing to S3 URL |
| `.dvcignore` | DVC ignore file scaffold |
| `.github/workflows/ci.yaml` | Push-triggered CI/CD workflow |
| `.gitignore` | Python/data-science ignore patterns; contains NUL bytes near tail |
| `Dockerfile` | Flask serving container definition |
| `LICENSE` | MIT license |
| `Makefile` | Cookiecutter-style developer commands |
| `README.md` | Main project overview and operating instructions |
| `deployment.yaml` | Kubernetes Deployment and LoadBalancer Service |
| `docs/Makefile` | Unix Sphinx build wrapper |
| `docs/commands.rst` | Make command docs |
| `docs/conf.py` | Sphinx configuration |
| `docs/getting-started.rst` | Placeholder setup docs |
| `docs/index.rst` | Sphinx docs index |
| `docs/make.bat` | Windows Sphinx build wrapper |
| `dvc.lock` | DVC pipeline lockfile from a previous run |
| `dvc.yaml` | DVC stage graph |
| `eligible_cities.txt` | Root serving/training bridge artifact listing eligible cities |
| `flask_app/app.py` | Flask app, model loading, prediction routes, metrics |
| `flask_app/eligible_cities.txt` | Container-local eligible city list |
| `flask_app/requirements.txt` | Slim serving dependencies |
| `flask_app/templates/index.html` | Bootstrap/Jinja prediction form |
| `model_dir/MLmodel` | MLflow saved model metadata |
| `model_dir/conda.yaml` | MLflow conda environment metadata |
| `model_dir/model.pkl` | Binary serialized ML model artifact |
| `model_dir/python_env.yaml` | MLflow virtualenv metadata |
| `model_dir/requirements.txt` | MLflow model dependency metadata |
| `notebooks/.gitkeep` | Keeps notebooks folder tracked |
| `notebooks/life-cycle.ipynb` | Exploratory full ML lifecycle notebook |
| `notebooks/t20s/*.yaml` | 137 raw cricket scorecard YAML data files listed in section 18 |
| `params.yaml` | Feature split and model hyperparameters |
| `projectflow.txt` | Manual project build/deployment notes |
| `pyproject.toml` | setuptools build backend declaration |
| `references/.gitkeep` | Keeps references folder tracked |
| `reports/.gitignore` | Ignores generated report JSON artifacts |
| `reports/.gitkeep` | Keeps reports folder tracked |
| `reports/figures/.gitkeep` | Keeps figures folder tracked |
| `requirements.txt` | Full training/pipeline dependency set |
| `scripts/promote_model.py` | MLflow staging-to-production tag promotion |
| `setup.py` | Package metadata for editable install |
| `src/__init__.py` | Package marker |
| `src/connections/__init__.py` | Connections package marker |
| `src/connections/config.json` | SQL Server connection config with sensitive values |
| `src/connections/s3_connection.py` | Current S3 CSV/YAML loader |
| `src/connections/s3_connection_old.py` | Older S3 CSV-only loader |
| `src/connections/ssms_connection.py` | Current SQL Server/YAML blob loader |
| `src/connections/ssms_connection_old.py` | Older SQL Server table loader |
| `src/data/.gitkeep` | Keeps data package folder tracked |
| `src/data/__init__.py` | Data package marker |
| `src/data/data_ingestion.py` | S3/local YAML ingestion to raw delivery CSV |
| `src/data/data_preprocessing.py` | Team cleaning and bowling team derivation |
| `src/features/.gitkeep` | Keeps features folder tracked |
| `src/features/__init__.py` | Features package marker |
| `src/features/feature_engineering.py` | Match-state feature generation and train/test split |
| `src/logger/__init__.py` | Root logging configuration |
| `src/model/.gitkeep` | Keeps model package folder tracked |
| `src/model/__init__.py` | Model package marker |
| `src/model/model_building.py` | XGBoost pipeline training and pickle save |
| `src/model/model_evaluation.py` | Model evaluation and MLflow artifact logging |
| `src/model/register_model.py` | MLflow model registration and staging tag |
| `src/visualization/.gitkeep` | Keeps visualization folder tracked |
| `src/visualization/__init__.py` | Visualization package marker |
| `src/visualization/visualize.py` | Empty visualization placeholder |
| `test_environment.py` | Python-major-version check |
| `tests/test_flask_app.py` | Flask integration tests |
| `tests/test_model.py` | MLflow model integration/performance tests |
| `tox.ini` | flake8 line length and complexity config |

## 10. File-By-File Deep Dive

For repeated raw match YAML files, detailed executable-code fields are consolidated because they are first-party data assets with the same schema, not source modules.

### `.dvc/.gitignore`

**Role:** Ignores local-only DVC internals.

**Why it matters:** Prevents `config.local`, `tmp`, and `cache` from being committed.

**Key dependencies/imports:** None.

**Exports/public surface:** Ignore patterns.

**Used by:** Git.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-3 | `/config.local`, `/tmp`, `/cache` | Keeps machine-local DVC files untracked | Local file paths | Git ignore behavior | None | Correctly protects cache bloat |

**Potential interview talking points:** Shows awareness that DVC metadata and DVC cache are different concerns.

**Possible improvements or risks:** None material.

### `.dvc/config`

**Role:** Configures the DVC remote.

**Why it matters:** DVC pipeline artifacts can be pushed/pulled from an S3-compatible remote.

**Key dependencies/imports:** DVC.

**Exports/public surface:** Remote name `myremote`, URL `s3://t20s`.

**Used by:** DVC commands such as `dvc repro`, `dvc push`, and `dvc pull`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-2 | `[core] remote = myremote` | Sets default DVC remote | DVC config | Default remote selection | None | Requires DVC installed |
| 3-4 | remote URL | Points remote at `s3://t20s` | S3-compatible credentials | Remote object storage address | None | Bucket/access must exist outside repo |

**Potential interview talking points:** DVC separates code tracking from artifact storage.

**Possible improvements or risks:** No endpoint/region config is visible; behavior depends on environment/default S3 settings.

### `.dvcignore`

**Role:** Placeholder DVC ignore file.

**Why it matters:** Can improve DVC performance by excluding files from DVC scanning.

**Key dependencies/imports:** DVC.

**Exports/public surface:** No active custom patterns.

**Used by:** DVC.

**Detailed code/chunk walkthrough:** Comments only; no project-specific ignored paths.

**Potential interview talking points:** Shows standard DVC scaffold.

**Possible improvements or risks:** Could add large local scratch paths if DVC scans become slow.

### `.github/workflows/ci.yaml`

**Role:** Defines push-triggered CI/CD.

**Why it matters:** This is the automation backbone for reproducing the ML pipeline, testing, promoting, building the container, and deploying.

**Key dependencies/imports:** GitHub Actions, actions/checkout, actions/setup-python, actions/cache, digitalocean/action-doctl, Docker, DVC, kubectl.

**Exports/public surface:** Job `project-testing`.

**Used by:** GitHub Actions on push.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-6 | Workflow trigger/job | Runs on every push on `ubuntu-latest` | Git push | CI job | Starts workflow | No branch filter |
| 9-24 | Checkout, Python 3.10, pip cache, install | Prepares runner and installs `requirements.txt` | Repo, pip cache | Python env | Downloads packages | Full requirements are large |
| 26-31 | `dvc repro` | Reproduces pipeline | AWS secrets, `CAPSTONE_TEST` | Generated DVC outputs, MLflow effects | External object storage and registry writes | Requires remote services |
| 33-48 | Model and Flask tests | Runs unittest files | `CAPSTONE_TEST`, generated data | Test pass/fail | Loads registry models | Integration-heavy |
| 38-42 | Promotion | Runs `scripts/promote_model.py` | `CAPSTONE_TEST` | Production tag update | Mutates MLflow registry | Only runs on success |
| 54-68 | DOCR login/build/push | Builds and pushes Docker image | `DO_TOKEN`, `DO_REGISTRY` | Image in registry | Registry write | Dockerfile depends on missing gunicorn dep |
| 74-88 | DOKS deploy | Installs kubectl, saves kubeconfig, creates secret, applies manifest, restarts deployment | `DO_CLUSTER_NAME`, `CAPSTONE_TEST` | Running Kubernetes deployment | Cluster changes | Secret passed through shell command; GitHub masks secrets but caution is warranted |

**Potential interview talking points:** Demonstrates CI/CD across ML pipeline, model promotion, image build, and K8s rollout.

**Possible improvements or risks:** Add branch protections/environments, separate training from deployment, cache DVC artifacts, avoid promotion/deploy on every push, add smoke tests after deploy, and add safer secret creation.

### `.gitignore`

**Role:** Ignores local/generated files.

**Why it matters:** Keeps raw/generated data, models, caches, environments, notebooks outputs, credentials, and editor artifacts out of Git.

**Key dependencies/imports:** Git.

**Exports/public surface:** Ignore patterns.

**Used by:** Git status/add.

**Detailed code/chunk walkthrough:**

| Section | What It Does | Notes |
|---|---|---|
| Project-specific top lines | Ignores `INTERVIEW_PREP_PACK.md`, `creds.txt`, local S3/data/model scratch folders, notebook-generated datasets, `/data/`, `/models/` | Protects generated and potentially sensitive artifacts |
| Python template patterns | Ignores bytecode, build outputs, coverage, virtual envs, caches, Sphinx build, notebooks checkpoints | Standard Python `.gitignore` coverage |
| Secret patterns | Ignores `.env`, `.envrc`, `.pypirc`, and `creds.txt` | Good baseline secret hygiene |
| Tail bytes | Contains NUL bytes around a final `data/` pattern | `rg` treats it as binary; worth cleaning later |

**Potential interview talking points:** Generated data and model artifacts are intentionally not ordinary Git assets.

**Possible improvements or risks:** Remove NUL bytes so tools parse `.gitignore` as text.

### `Dockerfile`

**Role:** Builds the serving container.

**Why it matters:** This is the deployable runtime for the Flask inference layer.

**Key dependencies/imports:** `python:3.10-slim`, pip, `flask_app/requirements.txt`, Gunicorn.

**Exports/public surface:** Container image exposing port 5000.

**Used by:** GitHub Actions Docker build and manual Docker commands.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1 | Base image | Uses Python 3.10 slim | Docker registry | Base runtime | Downloads image | Matches CI Python version |
| 3-7 | Workdir, copy, install | Copies `flask_app/` only and installs its requirements | `flask_app` files | Python env in image | pip install | Training code is intentionally excluded |
| 9 | `EXPOSE 5000` | Documents app port | None | Image metadata | None | Does not publish port by itself |
| 15 | Gunicorn command | Starts `app:app` on 0.0.0.0:5000 | Installed `gunicorn`, `CAPSTONE_TEST` env | Running server | Loads MLflow model at startup | `gunicorn` is not in `flask_app/requirements.txt` |

**Potential interview talking points:** Separate slim serving image from full training environment.

**Possible improvements or risks:** Add `gunicorn` dependency, healthcheck, non-root user, pinned base digest, and startup smoke test.

### `LICENSE`

**Role:** MIT license.

**Why it matters:** Defines reuse rights and warranty disclaimer.

**Key dependencies/imports:** None.

**Exports/public surface:** MIT license terms.

**Used by:** Users and redistributors.

**Detailed code/chunk walkthrough:** Standard MIT license text with copyright `2025 srikara202`.

**Potential interview talking points:** Open-source-friendly license.

**Possible improvements or risks:** None.

### `Makefile`

**Role:** Developer command shortcuts.

**Why it matters:** Captures scaffolded local workflow commands for environment setup, requirements, linting, S3 sync, and help output.

**Key dependencies/imports:** make, conda/virtualenv, pip, flake8, awscli, sed/awk/more.

**Exports/public surface:** `requirements`, `data`, `clean`, `lint`, `sync_data_to_s3`, `sync_data_from_s3`, `create_environment`, `test_environment`, `help`.

**Used by:** Developers.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 7-11 | Globals | Defines project dir, bucket placeholder, profile, project name, interpreter | Make variables | Reusable variables | None | Bucket is placeholder |
| 13-17 | Conda detection | Determines whether conda exists | Shell `which conda` | `HAS_CONDA` | None | Unix-oriented |
| 23-30 | `requirements`, `data` | Installs deps; `data` calls `src/data/make_dataset.py` | Python env | Dependencies/data | pip install | `make_dataset.py` is not tracked |
| 32-39 | `clean`, `lint` | Removes bytecode/cache; runs flake8 | File tree | Cleaner tree/lint result | Deletes generated cache files | Unix commands |
| 41-55 | S3 sync | Syncs local `data/` to/from configured bucket | AWS CLI, bucket/profile | S3/local data sync | External storage changes | Placeholder bucket must be changed |
| 57-77 | Environment/test env | Creates conda/virtualenv and validates Python | conda/virtualenv | Env | Installs virtualenvwrapper if needed | Legacy scaffold |
| 89-144 | Self-doc help | Generates help from comments | sed/awk/tput | Help output | None | Unix-oriented, not native Windows |

**Potential interview talking points:** The repo started from a Cookiecutter Data Science style scaffold and grew MLOps-specific DVC/CI pieces around it.

**Possible improvements or risks:** Update/remove stale `make data`, make commands cross-platform, and align Makefile with DVC entrypoints.

### `README.md`

**Role:** Main project documentation.

**Why it matters:** Provides the strongest narrative explanation of project purpose, architecture, pipeline, serving, deployment, testing, limitations, and future work.

**Key dependencies/imports:** None.

**Exports/public surface:** Human-readable project guide.

**Used by:** Developers, reviewers, interviewers.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Notes |
|---|---|---|---|
| 1-70 | Overview/highlights/problem | Defines project as first-innings T20 score MLOps project | Establishes value proposition |
| 72-105 | Architecture diagram | Shows S3 -> DVC -> MLflow -> Flask -> metrics and CI/CD | Matches code structure |
| 141-176 | Data pipeline | Documents stages and raw YAML ingestion | Explains DVC stages |
| 177-226 | Features/modeling | Defines engineered features and XGBoost pipeline | Aligns with `feature_engineering.py` and `model_building.py` |
| 228-271 | MLflow/registry/promotion | Documents `CAPSTONE_TEST`, model name, tag contract | Aligns with register/promote scripts |
| 273-388 | Serving/deployment/observability | Documents Flask routes, Docker, Kubernetes, Prometheus | Aligns with app and manifests |
| 390-560 | Stack/run/test | Lists dependencies, local commands, env vars, tests | Useful quick start |
| 562-585 | Caveats/future/conclusion | Explicitly notes external dependencies and limitations | Good honesty in docs |

**Potential interview talking points:** Use README sections as the basis for a project walkthrough.

**Possible improvements or risks:** Ensure README notes the missing Gunicorn dependency and DVC lock/param mismatch.

### `deployment.yaml`

**Role:** Kubernetes Deployment and Service.

**Why it matters:** Encodes production-ish runtime assumptions for DigitalOcean Kubernetes.

**Key dependencies/imports:** Kubernetes apps/v1 Deployment, v1 Service, DigitalOcean registry image, Kubernetes Secret.

**Exports/public surface:** Deployment `flask-app`, Service `flask-app-service`.

**Used by:** GitHub Actions `kubectl apply -f deployment.yaml`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-20 | Deployment metadata/spec | Runs 2 replicas of image `registry.digitalocean.com/flask-app-container-registry/flask-app:latest` | Registry image | Pods | Pulls image | Uses mutable `latest` tag |
| 21-28 | Container port/env | Exposes 5000 and injects `CAPSTONE_TEST` from secret | Secret `capstone-secret` | Env var | None | Pod fails if secret missing |
| 31-45 | LoadBalancer Service | Exposes port 5000 to targetPort 5000 | Cluster/network | External service | Cloud LB provisioning | No annotations, probes, or resource limits |

**Potential interview talking points:** Shows how model-serving secrets are injected separately from image build.

**Possible improvements or risks:** Add readiness/liveness probes, requests/limits, immutable image tags, rollout strategy, HPA, and namespace config.

### `docs/Makefile`

**Role:** Sphinx documentation build wrapper for Unix-like systems.

**Why it matters:** Enables generated docs from `docs/*.rst`.

**Key dependencies/imports:** `sphinx-build`, make.

**Exports/public surface:** Targets `html`, `dirhtml`, `singlehtml`, `pickle`, `json`, `htmlhelp`, `qthelp`, `devhelp`, `epub`, `latex`, `latexpdf`, `text`, `man`, `texinfo`, `info`, `gettext`, `changes`, `linkcheck`, `doctest`.

**Used by:** Developers building docs.

**Detailed code/chunk walkthrough:** Standard Sphinx quickstart Makefile; configures `BUILDDIR=_build`, `ALLSPHINXOPTS`, and build targets.

**Potential interview talking points:** Docs scaffold exists but is not deeply populated.

**Possible improvements or risks:** Add real API docs and pipeline docs, or remove scaffold if unused.

### `docs/commands.rst`

**Role:** Documents Makefile commands.

**Why it matters:** Explains data sync commands.

**Key dependencies/imports:** Sphinx/reStructuredText.

**Exports/public surface:** `commands` docs page.

**Used by:** `docs/index.rst` toctree.

**Detailed code/chunk walkthrough:** Lines 1-10 define a `Commands` page and describe `make sync_data_to_s3` and `make sync_data_from_s3` using the placeholder bucket.

**Potential interview talking points:** Shows original scaffold documentation for operational commands.

**Possible improvements or risks:** Bucket docs are placeholder; update for actual DVC/S3 workflow.

### `docs/conf.py`

**Role:** Sphinx config.

**Why it matters:** Controls docs metadata and build outputs.

**Key dependencies/imports:** `os`, `sys`, Sphinx.

**Exports/public surface:** Sphinx config variables such as `project`, `version`, `release`, `html_theme`, `latex_documents`.

**Used by:** Sphinx build commands.

**Detailed code/chunk walkthrough:**

| Lines/Section | What It Does | Notes |
|---|---|---|
| 14-29 | Imports and extensions | Imports `os`, `sys`; no extensions enabled | Autodoc not enabled |
| 31-44 | Paths/source/master/project | Templates path, `.rst` suffix, master doc, project title | Project title is long descriptive name |
| 51-67 | Version/exclude patterns | Version/release `0.1`, excludes `_build` | Basic scaffold |
| 83-167 | HTML settings | Pygments style, default theme, static path, htmlhelp basename | Minimal default theme |
| 172-244 | LaTeX/man/texinfo | Output document metadata | Standard Sphinx quickstart |

**Potential interview talking points:** Documentation scaffold can be extended with autodoc for `src`.

**Possible improvements or risks:** Add `sphinx.ext.autodoc`, type hints docs, and pipeline diagrams.

### `docs/getting-started.rst`

**Role:** Placeholder setup documentation.

**Why it matters:** Indicates intended docs structure but not full project-specific content.

**Key dependencies/imports:** Sphinx/reStructuredText.

**Exports/public surface:** `getting-started` docs page.

**Used by:** `docs/index.rst`.

**Detailed code/chunk walkthrough:** Lines 1-6 state that setup and data commands should be described, but the content is generic placeholder text.

**Potential interview talking points:** Acknowledges docs gap and opportunity.

**Possible improvements or risks:** Replace with evidence-based quick start from README and this ALLINFO.

### `docs/index.rst`

**Role:** Sphinx documentation root.

**Why it matters:** Defines docs navigation.

**Key dependencies/imports:** Sphinx toctree.

**Exports/public surface:** Docs index page.

**Used by:** Sphinx.

**Detailed code/chunk walkthrough:** Lines 6-15 create title and include `getting-started` and `commands`; lines 19-24 add standard index/module/search links.

**Potential interview talking points:** Minimal docs skeleton.

**Possible improvements or risks:** Add module API pages and architecture sections.

### `docs/make.bat`

**Role:** Windows Sphinx build wrapper.

**Why it matters:** Provides parity with `docs/Makefile` for Windows users.

**Key dependencies/imports:** `sphinx-build`, Windows batch.

**Exports/public surface:** Sphinx build targets.

**Used by:** Windows developers.

**Detailed code/chunk walkthrough:** Standard Sphinx quickstart batch file; sets `SPHINXBUILD`, `BUILDDIR`, parses target argument, and dispatches to HTML, LaTeX, text, man, linkcheck, doctest, and other builders.

**Potential interview talking points:** Cross-platform docs build support exists.

**Possible improvements or risks:** Docs content is sparse.

### `dvc.lock`

**Role:** Records a locked DVC pipeline state.

**Why it matters:** Captures command, dependency hashes, output hashes/sizes, and param values from a previous successful run.

**Key dependencies/imports:** DVC.

**Exports/public surface:** Locked stage metadata.

**Used by:** DVC status/repro.

**Detailed code/chunk walkthrough:**

| Lines/Section | What It Does | Notes |
|---|---|---|
| 1-15 | `data_ingestion` lock | Tracks `src/data/data_ingestion.py` and `data/raw` output hash | Raw output size 1,676,639 bytes |
| 16-33 | `data_preprocessing` lock | Tracks raw dependency, script, and interim output | Interim output size 850,338 bytes |
| 34-54 | `feature_engineering` lock | Tracks interim, script, params, processed output | Records `test_size: 0.2`, not current `params.yaml` value |
| 55-76 | `model_building` lock | Tracks processed data, script, params, model pickle | Records `learning_rate: 0.2`, `max_depth: 12`, not current values |
| 77-96 | `model_evaluation` lock | Tracks model, script, report outputs | Metrics/report info generated but ignored in Git |
| 97-107 | `model_registration` lock | Tracks experiment info and register script | Registry side effects are external |

**Potential interview talking points:** DVC lockfiles provide reproducibility evidence but must stay synchronized with params.

**Possible improvements or risks:** Run `dvc repro` after parameter changes and commit updated lock if expected.

### `dvc.yaml`

**Role:** Defines the ML pipeline DAG.

**Why it matters:** This is the executable recipe for data-to-registry reproducibility.

**Key dependencies/imports:** DVC, Python scripts.

**Exports/public surface:** Stages `data_ingestion`, `data_preprocessing`, `feature_engineering`, `model_building`, `model_evaluation`, `model_registration`.

**Used by:** `dvc repro`, CI.

**Detailed code/chunk walkthrough:**

| Lines/Section | What It Does | Inputs | Outputs |
|---|---|---|---|
| 4-9 | Ingestion | `src/data/data_ingestion.py` | `data/raw` |
| 11-17 | Preprocessing | `data/raw`, script | `data/interim` |
| 19-27 | Feature engineering | `data/interim`, script, `feature_engineering.test_size` | `data/processed` |
| 29-39 | Model building | `data/processed`, script, model params | `models/model.pkl` |
| 41-49 | Evaluation | `models/model.pkl`, script | metric `reports/metrics.json`, output `reports/experiment_info.json` |
| 51-55 | Registration | report info, script | External MLflow registry side effect |

**Potential interview talking points:** Clean, sequential DVC pipeline maps directly to MLOps lifecycle stages.

**Possible improvements or risks:** Declare `eligible_cities.txt` and `flask_app/eligible_cities.txt` as outputs or generated artifacts to avoid hidden side effects.

### `eligible_cities.txt`

**Role:** Root-level list of cities that passed feature-engineering volume threshold.

**Why it matters:** Bridges training-derived city support into serving/UI.

**Key dependencies/imports:** Written by `src/features/feature_engineering.py`; read by `flask_app/app.py` when app is launched from repository root. Docker uses the copy under `flask_app/`.

**Exports/public surface:** Python-list-like text containing city names.

**Used by:** Flask app startup depending on working directory.

**Detailed code/chunk walkthrough:** One-line data artifact. It currently includes cities such as Johannesburg, London, Barbados, St Lucia, Cape Town, Nottingham, blank string, Durban, and Auckland.

**Potential interview talking points:** Shows model serving only offers cities with enough historical data.

**Possible improvements or risks:** Blank string is included as a city; store as JSON/YAML and validate schema.

### `flask_app/app.py`

**Role:** Flask inference app.

**Why it matters:** This is the runtime product surface: it loads the production model, renders UI, handles predictions, and exposes metrics.

**Key dependencies/imports:** `os`, `ast`, `time`, `logging`, pandas, Flask, MLflow, DagsHub, prometheus_client, `MlflowClient`.

**Exports/public surface:** Flask `app`, routes `/`, `/predict`, `/metrics`, helper `get_model_version_by_stage`.

**Used by:** `python app.py`, Gunicorn `app:app`, `tests/test_flask_app.py`, Docker/Kubernetes.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 3-27 | Imports and MLflow auth | Reads `CAPSTONE_TEST`, sets MLflow username/password, sets DagsHub tracking URI | Env token | MLflow auth state | Raises if token absent | Happens at import time |
| 38-45 | `get_model_version_by_stage` | Searches model versions and returns latest matching tag | Model name, stage tag | Version string | MLflow API call | Raises if no matching version |
| 47-53 | Global model loading | Resolves `my_model` production version and loads pyfunc model | MLflow registry | Global `model` | Network/model download | Startup is coupled to registry availability |
| 58-65 | Metrics/app setup | Creates custom Prometheus registry, counter, histogram, Flask app | None | `app`, metrics | None | Custom registry exposes only app metrics |
| 67-100 | `TEAMS` and `CITIES` | Hardcoded teams and file-loaded cities | `eligible_cities.txt` | Lists for UI/defaults | File read at import | Working directory matters |
| 102-121 | `home` | Renders form with sorted teams/cities and empty state | GET `/` | HTML response | Increments metrics | No auth |
| 123-159 | JSON branch of `predict` | Reads JSON, requires numeric model fields, predicts, returns JSON | JSON request | `predicted_score` or error | Model inference, metrics count | Does not range-check numeric values |
| 161-228 | Form branch of `predict` | Validates form, derives balls/wickets/crr, predicts, renders template | Form request | HTML result/error | Model inference, latency metric | Uses `int(overs_done * 6)`, so decimal overs are treated as decimal overs rather than cricket ball notation |
| 230-232 | `metrics` | Returns Prometheus text | GET `/metrics` | Metrics response | None | Does not increment request count for `/metrics` |
| 234-235 | Local run | Runs debug server on 0.0.0.0:5000 | Direct script execution | Dev server | Debug mode | Gunicorn path ignores this block |

**Potential interview talking points:** Registry-driven serving decouples deployment from hardcoded model files, while Prometheus gives basic operational visibility.

**Possible improvements or risks:** Move model loading to app factory for testability, add health/readiness endpoint, validate JSON ranges, avoid import-time external calls in tests, parse cricket overs more carefully, and remove unused `dagshub` import.

### `flask_app/eligible_cities.txt`

**Role:** Eligible city list copied into the Docker image.

**Why it matters:** `Dockerfile` copies `flask_app/` to `/app`; `app.py` reads `eligible_cities.txt` relative to `/app`, so this file enables container startup.

**Key dependencies/imports:** Written by feature engineering; read by Flask.

**Exports/public surface:** Python-list-like text city list.

**Used by:** Containerized Flask app.

**Detailed code/chunk walkthrough:** Same content/risks as root `eligible_cities.txt`.

**Potential interview talking points:** Serving artifacts must be packaged with the app when the model does not include UI-domain metadata.

**Possible improvements or risks:** Use a structured JSON artifact and include it in DVC outputs.

### `flask_app/requirements.txt`

**Role:** Slim serving dependency list.

**Why it matters:** Docker installs this file rather than the full training requirements.

**Key dependencies/imports:** DagsHub, Flask, MLflow, cloudpickle, numpy, pandas, psutil, scikit-learn, xgboost, prometheus_client.

**Exports/public surface:** pip install contract for serving image.

**Used by:** Dockerfile.

**Detailed code/chunk walkthrough:** Lines 1-11 pin/declare core runtime deps. Flask is `3.1.0`, MLflow is `2.15.0`, scikit-learn is `1.5.1`, XGBoost is `3.0.2`.

**Potential interview talking points:** Separating serving requirements reduces image size and attack surface compared with full training environment.

**Possible improvements or risks:** Add `gunicorn`; align versions with the saved model metadata if registry models require newer scikit-learn/MLflow.

### `flask_app/templates/index.html`

**Role:** Jinja2/Bootstrap prediction page.

**Why it matters:** This is the human-facing UI.

**Key dependencies/imports:** Bootstrap CSS/JS from CDN, Flask/Jinja context variables.

**Exports/public surface:** HTML form fields and result block.

**Used by:** `home()` and form branch of `predict()`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-14 | HTML head/CDN CSS | Sets metadata/title and Bootstrap CSS | CDN | Styled page | Browser request | Integrity placeholder may block load |
| 15-54 | Inline CSS | Gradient background, card styling, button colors, result card | None | Visual styles | None | Simple, not design-system based |
| 57-67 | Header/error | Displays title and optional error alert | `error_message` | HTML | None | Good validation feedback |
| 73-102 | Team/city selects | Renders batting, bowling, and city dropdowns | `teams`, `cities`, selected values | Form controls | None | Uses sorted values from app |
| 103-160 | Numeric inputs | Current score, overs, wickets, last five | Current state values | Validated form controls | Browser validation | Overs min/max 5-19; last-five max dynamic only after current_score present |
| 162-174 | Submit/result | Submit button and predicted score display | `result` | Result card | None | Result only appears when not `None` |
| 176-190 | Footer/CDN JS | Shows deployment/source note and Bootstrap JS | CDN | Footer and JS | Browser request | Integrity placeholder again |

**Potential interview talking points:** Simple server-rendered UI keeps the app lightweight and demo-friendly.

**Possible improvements or risks:** Fix CDN integrity attributes, add accessibility refinements, add client-side dependent validation for last-five max, and avoid relying on CDN for air-gapped demos.

### `model_dir/MLmodel`

**Role:** MLflow saved model descriptor.

**Why it matters:** Documents the tracked model artifact format and environment.

**Key dependencies/imports:** MLflow pyfunc and sklearn flavor.

**Exports/public surface:** `python_function` and `sklearn` flavors; model path `model.pkl`.

**Used by:** MLflow model loading when using this local artifact.

**Detailed code/chunk walkthrough:**

| Lines/Section | What It Does | Notes |
|---|---|---|
| 1-9 | pyfunc flavor | Loader `mlflow.sklearn`, model path `model.pkl`, Python 3.13.5 | Generated metadata |
| 10-14 | sklearn flavor | Pickled model `model.pkl`, cloudpickle serialization, sklearn 1.7.0 | Version differs from app requirements sklearn 1.5.1 |
| 15-20 | MLflow metadata | MLflow 3.1.1, model size 7,636,288 bytes, UUID, creation UTC | Useful artifact audit data |

**Potential interview talking points:** MLflow saved models carry environment metadata that should be considered during deployment.

**Possible improvements or risks:** Align serving dependency versions with the model artifact or rely on registry model environment management.

### `model_dir/conda.yaml`

**Role:** MLflow conda environment metadata.

**Why it matters:** Captures dependencies for recreating the saved model environment.

**Key dependencies/imports:** conda, pip packages.

**Exports/public surface:** Environment named `mlflow-env`.

**Used by:** MLflow environment recreation.

**Detailed code/chunk walkthrough:** Defines conda-forge, Python 3.13.5, pip <= 25.1, and pip deps including MLflow 3.1.1, cloudpickle 3.1.1, numpy 2.3.1, pandas 2.3.1, scikit-learn 1.7.0, scipy 1.16.0, xgboost 3.0.2.

**Potential interview talking points:** Artifact environment metadata helps explain reproducibility and version skew.

**Possible improvements or risks:** Version skew with app requirements can cause load/runtime issues.

### `model_dir/model.pkl`

**Role:** Binary serialized model artifact.

**Why it matters:** It is the local saved ML model payload referenced by `model_dir/MLmodel`.

**Key dependencies/imports:** cloudpickle/sklearn/XGBoost implied by metadata.

**Exports/public surface:** Pickled model bytes.

**Used by:** MLflow local artifact loading if `model_dir` is used.

**Detailed code/chunk walkthrough:** Binary file, 7,636,288 bytes. It should not be decoded line by line. The MLflow metadata identifies it as an sklearn-flavor pickled model with pyfunc `predict`.

**Potential interview talking points:** Binary artifacts are better managed by MLflow/DVC than direct code review.

**Possible improvements or risks:** Avoid committing large binary model artifacts unless intentionally needed; use DVC or model registry as source of truth.

### `model_dir/python_env.yaml`

**Role:** MLflow virtualenv environment descriptor.

**Why it matters:** Alternative environment recreation metadata.

**Key dependencies/imports:** Python 3.13.5, pip, setuptools, wheel, `requirements.txt`.

**Exports/public surface:** Python env specification.

**Used by:** MLflow environment tooling.

**Detailed code/chunk walkthrough:** Lines 1-7 specify Python version, build dependencies, and dependency reference to `requirements.txt`.

**Potential interview talking points:** MLflow stores both conda and virtualenv specs.

**Possible improvements or risks:** Same version skew concern as `conda.yaml`.

### `model_dir/requirements.txt`

**Role:** MLflow model dependency list.

**Why it matters:** Pin set for the saved model artifact.

**Key dependencies/imports:** MLflow, cloudpickle, numpy, pandas, psutil, scikit-learn, scipy, xgboost.

**Exports/public surface:** pip requirements for model environment.

**Used by:** `model_dir/python_env.yaml` and MLflow.

**Detailed code/chunk walkthrough:** Lines 1-8 pin model-specific dependencies, including MLflow 3.1.1 and scikit-learn 1.7.0.

**Potential interview talking points:** Model artifact dependencies can differ from application dependencies and should be checked.

**Possible improvements or risks:** Align with `flask_app/requirements.txt` or document why serving uses older versions.

### `notebooks/.gitkeep`

**Role:** Keeps `notebooks/` tracked even if notebooks are removed.

**Why it matters:** Preserves project structure.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Scaffolding artifact.

**Possible improvements or risks:** None.

### `notebooks/life-cycle.ipynb`

**Role:** Exploratory notebook for data extraction, cleaning, feature engineering, modeling, and MLflow experimentation.

**Why it matters:** It appears to be the research/prototyping source from which modular scripts were later extracted.

**Key dependencies/imports:** numpy, pandas, yaml, tqdm, ast, sklearn, xgboost, MLflow, DagsHub and other notebook-scoped imports.

**Exports/public surface:** Notebook cells, not package exports.

**Used by:** Human exploration; not imported by application code.

**Detailed code/chunk walkthrough:**

| Section/Cells | What It Does | Inputs | Outputs | Notes/Edge Cases |
|---|---|---|---|---|
| Cells 1-6 | Data extraction | Local YAML file list | Normalized `final_df` | Starts from local `notebooks/t20s` style data |
| Cells 7-38 | Cleaning and delivery extraction | Match-level dataframe | `dataset_level1.csv`, `dataset_level2.csv`, delivery rows | Prototype of `data_ingestion.py` and `data_preprocessing.py` |
| Cells 39-80 | Feature engineering | Delivery rows | `dataset_level3.csv`, model-ready features | Prototype of `feature_engineering.py` |
| Cells 81-97 | MLflow setup, split, model training, validation, experiment tracking | Final features | Model metrics/artifacts | Later modularized into `src/model` scripts |

Notebook metadata: nbformat 4.5, kernel display name `atlas310`, 97 cells.

**Potential interview talking points:** Shows iterative experimentation before modularizing into reproducible scripts.

**Possible improvements or risks:** Notebook outputs/large code cells can drift from production scripts; consider keeping a cleaned, narrative version.

### `notebooks/t20s/*.yaml` (137 tracked files)

**Role:** Raw T20 cricket scorecard data assets.

**Why it matters:** They provide inspectable sample source data compatible with the local helper `load_yaml_directory`, even though the current DVC ingestion path reads from S3.

**Key dependencies/imports:** Parsed by PyYAML and normalized by pandas in `data_ingestion.py`/notebook prototypes.

**Exports/public surface:** YAML records with top-level keys `meta`, `info`, and `innings`.

**Used by:** `notebooks/life-cycle.ipynb`; commented local fallback in `src/data/data_ingestion.py` (`load_yaml_directory(data_url='notebooks/t20s/')`).

**Detailed code/chunk walkthrough:** Data files, not executable code. A sampled file (`211028.yaml`) has `info` keys including city, dates, gender, match type, overs, teams, toss, umpires, venue, players, registry, and outcome. Detailed line-by-line code analysis is not applicable. Each file represents a match scorecard whose first innings can be transformed into delivery-level rows.

**Potential interview talking points:** The repository includes raw examples for explaining schema and feature derivation; production ingestion can swap local files for S3.

**Possible improvements or risks:** Data files can drift from S3 source; consider DVC-tracking the full dataset rather than committing samples directly.

The 137 tracked YAML files are individually accounted for in section 18.

### `params.yaml`

**Role:** Central parameter file for DVC-tracked pipeline stages.

**Why it matters:** Keeps train/test split and model hyperparameters outside code.

**Key dependencies/imports:** PyYAML loaders in feature/model scripts; DVC param tracking.

**Exports/public surface:** `feature_engineering.test_size`, `model_building.n_estimators`, `model_building.learning_rate`, `model_building.max_depth`.

**Used by:** `src/features/feature_engineering.py`, `src/model/model_building.py`, `dvc.yaml`.

**Detailed code/chunk walkthrough:** See section 7 parameter table.

**Potential interview talking points:** Hyperparameters are versioned with code and tracked by DVC.

**Possible improvements or risks:** Update `dvc.lock` to match current values; add validation for parameter ranges.

### `projectflow.txt`

**Role:** Manual project history and operations notes.

**Why it matters:** Captures how the project was scaffolded, connected to DagsHub, initialized with DVC, packaged, containerized, deployed to DigitalOcean, and monitored with Prometheus/Grafana.

**Key dependencies/imports:** conda, cookiecutter, DagsHub, DVC, DigitalOcean Spaces/DOKS, Docker, doctl, kubectl, Prometheus, Grafana.

**Exports/public surface:** Human runbook notes.

**Used by:** Developers/operators as reference.

**Detailed code/chunk walkthrough:**

| Section | What It Does | Notes |
|---|---|---|
| 1-18 | Project setup | Repo clone, conda env, cookiecutter scaffold, package rename |
| 19-28 | DagsHub/MLflow setup | Connect repo and install tracking libs |
| 29-55 | DVC setup | Initialize DVC, modular scripts, `dvc repro` |
| 56-70 | Object storage | DigitalOcean Space as S3-compatible remote |
| 71-90 | API and CI | Flask app, CI workflow, secrets |
| 92-120 | Docker/DigitalOcean credentials | Build/run image and deploy via DOKS |
| 123-180 | DOKS setup | doctl, kubectl, cluster creation, app deployment |
| 182-240 | Prometheus/Grafana | Manual monitoring server setup notes |
| 243-270 | Cleanup and operational commands | Delete K8s resources, cluster, droplets; scale app |

**Potential interview talking points:** Useful source for explaining infrastructure choices and project evolution.

**Possible improvements or risks:** Some paths mention `k8s/deployment.yaml`/`k8s/service.yaml`, but repo has root `deployment.yaml`; keep runbook synchronized.

### `pyproject.toml`

**Role:** Python build-system declaration.

**Why it matters:** Enables standards-based setuptools builds.

**Key dependencies/imports:** setuptools, wheel.

**Exports/public surface:** Build backend `setuptools.build_meta`.

**Used by:** pip/build tools.

**Detailed code/chunk walkthrough:** Lines 1-3 require `setuptools>=64.0.0` and `wheel`, and set `build-backend`.

**Potential interview talking points:** Minimal modern packaging support.

**Possible improvements or risks:** Add project metadata here if migrating from `setup.py`.

### `references/.gitkeep`

**Role:** Keeps references folder tracked.

**Why it matters:** Placeholder for external reference materials.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Cookiecutter scaffold artifact.

**Possible improvements or risks:** None.

### `reports/.gitignore`

**Role:** Ignores generated report artifacts.

**Why it matters:** `model_evaluation.py` writes `reports/metrics.json` and `reports/experiment_info.json`, but those are generated and ignored.

**Key dependencies/imports:** Git.

**Exports/public surface:** Ignore patterns `/experiment_info.json`, `/metrics.json`.

**Used by:** Git status/add.

**Detailed code/chunk walkthrough:** Lines 1-2 ignore the exact generated report files.

**Potential interview talking points:** Generated metrics are DVC/MLflow artifacts, not normal source.

**Possible improvements or risks:** DVC tracks metrics, but reviewers without DVC artifacts cannot inspect generated metric values locally.

### `reports/.gitkeep`

**Role:** Keeps reports folder tracked.

**Why it matters:** Ensures DVC/report output path exists in repo scaffold.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Generated reports are ignored but folder is preserved.

**Possible improvements or risks:** None.

### `reports/figures/.gitkeep`

**Role:** Keeps report figures folder tracked.

**Why it matters:** Placeholder for plots/figures.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Future visualization location.

**Possible improvements or risks:** None.

### `requirements.txt`

**Role:** Full training/pipeline dependency file.

**Why it matters:** CI and local setup install this before running DVC and tests.

**Key dependencies/imports:** XGBoost, LightGBM, CatBoost, boto3, DVC, dvc-s3, Flask, MLflow, DagsHub, pandas, numpy, scikit-learn, matplotlib, prometheus_client, and many transitive dependencies.

**Exports/public surface:** pip install requirements; editable local install via `-e .`.

**Used by:** README quick start, CI, Makefile.

**Detailed code/chunk walkthrough:** Lines 1-6 contain commented scaffold deps; lines 7-185 list pinned/unpinned Python dependencies; line 187 installs the local package editable.

**Potential interview talking points:** The full environment supports training, DVC, tracking, serving, and experimentation.

**Possible improvements or risks:** Mixed pinned and unpinned packages (`lightgbm`, `catboost`, `aiobotocore`, `awscli`, `botocore`, `prometheus_client`) can reduce reproducibility; full requirements are heavier than needed for serving.

### `scripts/promote_model.py`

**Role:** Promotes MLflow model versions by retagging.

**Why it matters:** It completes the registry-driven lifecycle by making a staging model available to production serving.

**Key dependencies/imports:** `os`, `mlflow`, `MlflowClient`.

**Exports/public surface:** `main()`.

**Used by:** CI and manual operations.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 10-17 | Auth | Reads `CAPSTONE_TEST`, maps it to MLflow username/password | Env token | MLflow auth env | Raises if missing | Token used as both username and password |
| 19-27 | Tracking/model constants | Sets DagsHub tracking URI, model name, tag values | Repo constants | MLflow client context | None | Hardcoded owner/repo/model |
| 29-39 | Find staging | Searches versions and chooses newest `stage=staging` | MLflow registry | Version number | API calls | Raises if no staging candidates |
| 41-56 | Retag production | Removes old production tags and sets new one | Model versions | Registry tag update | Mutates registry | Does not archive old production beyond tag removal |
| 58-62 | Print/run | Prints promoted version; script guard | CLI execution | Console output | None | Uses non-ASCII check in source output; doc does not copy it |

**Potential interview talking points:** Custom tag-based promotion is simple and explicit.

**Possible improvements or risks:** Add audit logging, support rollback, parameterize model name/stages, and consider MLflow aliases if available.

### `setup.py`

**Role:** Package metadata for editable install.

**Why it matters:** Enables `-e .` in `requirements.txt` so `src` imports work.

**Key dependencies/imports:** setuptools `find_packages`, `setup`.

**Exports/public surface:** Package name `src`, version `0.1.0`.

**Used by:** pip install.

**Detailed code/chunk walkthrough:** Lines 3-10 call `setup()` with package discovery, description, author, and MIT license.

**Potential interview talking points:** The project code is installable as a package, which makes imports stable in scripts.

**Possible improvements or risks:** Package name `src` is generic; use project-specific package name if publishing.

### `src/__init__.py`

**Role:** Marks `src` as a Python package.

**Why it matters:** Allows imports such as `from src.logger import logging`.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Python import system.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Package layout supports modular pipeline scripts.

**Possible improvements or risks:** None.

### `src/connections/__init__.py`

**Role:** Marks `src.connections` as a package.

**Why it matters:** Enables `from src.connections import s3_connection`.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** `src/data/data_ingestion.py`.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Data access is isolated from transformation logic.

**Possible improvements or risks:** Could expose stable connection classes if package API grows.

### `src/connections/config.json`

**Role:** SQL Server connection configuration.

**Why it matters:** Supports optional/legacy SQL Server ingestion helpers.

**Key dependencies/imports:** JSON parser in SQL Server connection modules.

**Exports/public surface:** Keys under `sql_server`: `server`, `database`, `table`, `username`, `pass`. Values are redacted.

**Used by:** `src/connections/ssms_connection.py`, `src/connections/ssms_connection_old.py`.

**Detailed code/chunk walkthrough:** JSON config file with sensitive connection details. Detailed values are not documented to protect secrets.

**Potential interview talking points:** Shows the project explored multiple data-source backends.

**Possible improvements or risks:** Do not commit real credentials; move values to environment variables or local ignored config.

### `src/connections/s3_connection.py`

**Role:** Current S3 data access helper.

**Why it matters:** This is the active dependency for DVC data ingestion.

**Key dependencies/imports:** `os` (unused), boto3, pandas, PyYAML, logging, custom `src.logger`, `StringIO`.

**Exports/public surface:** Class `s3_operations` with methods `fetch_file_from_s3` and `fetch_yaml_folder_from_s3`.

**Used by:** `src/data/data_ingestion.py`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 9-21 | `__init__` | Creates boto3 S3 client from bucket and credentials | Bucket, access key, secret key, region | Client wrapper | Network client setup | Credentials can be `None`; failure may happen on call |
| 23-35 | `fetch_file_from_s3` | Reads one CSV object into pandas | S3 key | DataFrame or `None` | S3 get_object | Catches all exceptions and returns `None` |
| 37-86 | `fetch_yaml_folder_from_s3` | Lists YAML keys under prefix, reads each, parses YAML, normalizes to DataFrame, assigns match ids | Prefix | Combined DataFrame or `None` | S3 list/get calls | Per-file parse errors are logged and skipped; if no files, empty DataFrame |

**Potential interview talking points:** Abstracts object storage reads away from ingestion transformation.

**Possible improvements or risks:** Validate credentials and non-empty results, preserve source file IDs, avoid class name lowercase style, and add tests/mocks.

### `src/connections/s3_connection_old.py`

**Role:** Older S3 CSV-only helper.

**Why it matters:** Historical/legacy version of S3 access logic.

**Key dependencies/imports:** boto3, pandas, logging/custom logger, `StringIO`.

**Exports/public surface:** Class `s3_operations`, method `fetch_file_from_s3`.

**Used by:** No active import found; likely retained for reference.

**Detailed code/chunk walkthrough:** Similar to current S3 helper but only fetches CSV, uses f-string logging, and contains commented example usage with placeholder credentials.

**Potential interview talking points:** Shows evolution from CSV S3 access to folder-level YAML ingestion.

**Possible improvements or risks:** Remove or archive to reduce confusion, or mark clearly as deprecated.

### `src/connections/ssms_connection.py`

**Role:** Current SQL Server helper for table and YAML blob loading.

**Why it matters:** Provides an alternative data access path not currently wired into DVC.

**Key dependencies/imports:** os, json, pyodbc, pandas, yaml, logging/custom logger.

**Exports/public surface:** Class `SSMSOperations` with `fetch_table_as_df` and `fetch_yaml_folder_from_ssms`.

**Used by:** Manual usage block only; no active pipeline import.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 9-43 | `__init__` | Loads JSON config, sets table/yaml config, connects with trusted SQL Server auth | `config.json`, local SQL Server access | `self.conn` | DB connection | Config contains sensitive fields but trusted connection ignores username/pass |
| 45-57 | `fetch_table_as_df` | Runs `SELECT * FROM <table>` into pandas | SQL table | DataFrame or `None` | DB query | Table name is config-driven string |
| 59-99 | `fetch_yaml_folder_from_ssms` | Reads YAML blobs from configured table, parses and concatenates | SQL rows | DataFrame or empty DataFrame/None | DB query | Per-row YAML parse errors logged |
| 104-117 | Usage example | Instantiates loader and prints YAML head | Manual run | Console output | DB calls | Runs only as script |

**Potential interview talking points:** Demonstrates extensible ingestion sources beyond S3.

**Possible improvements or risks:** Move secrets out of tracked config, parameterize connection auth, and protect against SQL injection from config values.

### `src/connections/ssms_connection_old.py`

**Role:** Older SQL Server table loader.

**Why it matters:** Historical simpler version of SQL Server ingestion.

**Key dependencies/imports:** pyodbc, pandas, json, os.

**Exports/public surface:** `main(config_path='config.json')`.

**Used by:** No active imports found.

**Detailed code/chunk walkthrough:** Loads config, prints paths/server/table, builds trusted connection string, runs `SELECT * FROM table`, returns DataFrame or `None`.

**Potential interview talking points:** Shows transition from a simple table fetch to a class-based loader with YAML support.

**Possible improvements or risks:** Prints connection string and config values; do not use with real secrets in logs.

### `src/data/.gitkeep`

**Role:** Keeps `src/data` directory tracked.

**Why it matters:** Placeholder alongside data scripts.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Scaffold artifact.

**Possible improvements or risks:** None.

### `src/data/__init__.py`

**Role:** Marks `src.data` as a package.

**Why it matters:** Supports package imports.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Python import system.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Modular data stage package.

**Possible improvements or risks:** None.

### `src/data/data_ingestion.py`

**Role:** Ingests match YAML data and extracts first-innings delivery rows.

**Why it matters:** It is the first executable DVC stage and defines the raw training data contract.

**Key dependencies/imports:** numpy (unused), pandas, os, tqdm, train_test_split (unused), yaml, logging/custom logger, ast (unused), `src.connections.s3_connection`.

**Exports/public surface:** `load_params`, `load_yaml_directory`, `load_data`, `extract_delivery_df`, `save_data`, `main`.

**Used by:** `dvc.yaml` stage `data_ingestion`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-14 | Imports/options | Sets pandas future option and imports helpers | Runtime env | Module dependencies | Logger config import creates log file | Several imports unused/duplicated |
| 17-32 | `load_params` | Reads YAML params with error handling | Param file path | dict | File read | Not used in `main` |
| 34-91 | `load_yaml_directory` | Loads local YAML files into normalized DataFrame with match ids | Directory path | DataFrame | File reads | Local path is commented out in `main`; parse errors skipped |
| 94-105 | `load_data` | Reads CSV into DataFrame | CSV path | DataFrame | File read | Not used in `main` |
| 111-195 | `extract_delivery_df` | Selects required match columns, filters male and 20-over matches, iterates first-innings deliveries into per-ball records | Match-level normalized DataFrame | Delivery DataFrame | Logs | Assumes `innings[0]['1st innings']['deliveries']`; increments internal match counter independent of source `match_id` |
| 198-208 | First `save_data` | Intended train/test CSV saver | Train/test DataFrames | CSVs | Writes files | Overwritten by second `save_data` definition |
| 210-237 | Second `save_data` | Writes one DataFrame to `<data_path>/raw/data.csv` by default | DataFrame, path, filename | CSV | Creates directory/file | Shadows earlier function |
| 239-256 | `main` | Reads AWS env vars, creates S3 helper, fetches `t20s` YAML folder, extracts deliveries, saves raw data | AWS env, S3 data | `data/raw/data.csv` | Network/file writes | Catches final exception and prints; may not fail process explicitly |

**Potential interview talking points:** The ingestion stage transforms nested scorecard YAML into supervised learning rows at delivery granularity.

**Possible improvements or risks:** Remove duplicate `save_data`, validate schema, preserve source file IDs, fail fast on empty S3 results, and add local fixture tests.

### `src/data/data_preprocessing.py`

**Role:** Cleans delivery-level data and derives bowling team.

**Why it matters:** It converts raw delivery rows into the normalized columns used by feature engineering.

**Key dependencies/imports:** logging/custom logger, pandas, os, ast.

**Exports/public surface:** `preprocess_dataframe`, `main`.

**Used by:** `dvc.yaml` stage `data_preprocessing`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 3-10 | Imports | Imports pandas/logging/os/ast | Runtime | Dependencies | Logger import creates log file | Duplicate imports |
| 11-31 | Function docstring | Defines expected columns and returned columns | Delivery DataFrame | Preprocessed DataFrame | None | Useful contract |
| 32-48 | Copy and `_find_bowler` | Parses `teams` string/list with `ast.literal_eval`, selects team not batting, drops `teams` | `teams`, `batting_team` | `bowling_team` | None | Fails if `teams` is not literal-list-like |
| 50-94 | Team filtering | Hardcoded valid teams; filters batting and bowling sides | Team names | Reduced DataFrame | None | Team list duplicated in Flask |
| 96-110 | Final selection/error | Selects modeling-relevant columns and logs | Filtered DataFrame | Output DataFrame | None | Raises on errors |
| 114-142 | `main` | Reads `data/raw/data.csv`, preprocesses, writes `data/interim/interim_data.csv` | Raw CSV | Interim CSV | Creates directory/file | Catches and prints final exceptions |

**Potential interview talking points:** This stage ensures the model only trains on teams with enough intended coverage and derives opponent context from raw scorecards.

**Possible improvements or risks:** Move valid teams to shared config, validate input schema, and avoid broad final catch swallowing failures.

### `src/features/.gitkeep`

**Role:** Keeps features folder tracked.

**Why it matters:** Placeholder alongside feature script.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Scaffold artifact.

**Possible improvements or risks:** None.

### `src/features/__init__.py`

**Role:** Marks features package.

**Why it matters:** Supports imports if feature functions are reused.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Python import system.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Modular stage package.

**Possible improvements or risks:** None.

### `src/features/feature_engineering.py`

**Role:** Converts cleaned delivery data into model-ready features and train/test splits.

**Why it matters:** This file defines the actual predictive feature set.

**Key dependencies/imports:** numpy, pandas, train_test_split, os, yaml, custom logger; CountVectorizer and pickle are imported but unused.

**Exports/public surface:** `load_params`, `load_data`, `engineer_and_split`, `save_data`, `main`.

**Used by:** `dvc.yaml` stage `feature_engineering`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 13-28 | `load_params` | Reads YAML params | `params.yaml` | dict | File read | Reused pattern |
| 30-42 | `load_data` | Reads CSV and fills NaNs with empty string | Interim CSV | DataFrame | File read | Empty strings affect later null checks |
| 44-80 | Start engineering | Copies data, imputes missing city from venue, drops venue | Interim DataFrame | DataFrame with city | None | After fillna(''), `isnull()` may miss originally missing values loaded through `load_data` |
| 82-98 | Eligible city filtering | Keeps cities with >600 deliveries and writes city lists | City counts | Filtered DataFrame; text files | Writes root and Flask city files | Outputs are not declared in `dvc.yaml`; blank city can survive |
| 100-126 | Score/balls/wickets/crr | Computes cumulative score, over/ball_no, balls left, cumulative dismissals, wickets left, current run rate | Delivery rows | Numeric features | None | Division by zero possible if ball 0 appears |
| 128-138 | Momentum and target | Rolling 30-ball `last_five`; match total `total_runs` | Runs grouped by match | `last_five`, `total_runs` | None | First 29 deliveries per match become NaN and are dropped |
| 140-164 | Final select/shuffle/split | Selects final feature columns, drops NA, shuffles, train/test splits | Engineered data, `test_size` | Train/test DataFrames | None | Returns full DataFrames including target |
| 171-179 | `save_data` | Writes DataFrame to CSV | DataFrame/path | CSV | Creates directories/files | No schema validation |
| 181-200 | `main` | Loads params and interim data, engineers split, saves `train_final.csv` and `test_final.csv` | `params.yaml`, interim CSV | Processed CSVs | File writes | Catches and prints final exceptions |

**Potential interview talking points:** Explain how live innings state maps to model features: current score, balls left, wickets left, current run rate, and last-five-over momentum.

**Possible improvements or risks:** Use cricket-valid over parsing, handle missing city consistently, declare city-list outputs, add feature unit tests, and remove unused imports.

### `src/logger/__init__.py`

**Role:** Configures root logging.

**Why it matters:** Pipeline scripts import `from src.logger import logging`, which initializes file and console logging.

**Key dependencies/imports:** logging, os, RotatingFileHandler, datetime, sys.

**Exports/public surface:** The standard `logging` module after root configuration.

**Used by:** Most `src` pipeline and connection scripts.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 7-17 | Constants/path | Defines log dir, timestamped file, max size, backups, creates `logs` directory | Current time/path | Log file path | Creates `logs/` | `logs` is ignored by `.gitignore` via `*.log` but folder may be local |
| 19-42 | `configure_logger` | Sets root logger DEBUG, adds rotating file handler and stdout handler at INFO | None | Configured root logger | Adds handlers | Re-imports can add duplicate handlers |
| 44-45 | Import-time config | Calls configure at import | Module import | Active logging | File handler opened | Import side effect |

**Potential interview talking points:** Centralized logging gives consistent pipeline diagnostics.

**Possible improvements or risks:** Guard against duplicate handlers and avoid import-time side effects in tests.

### `src/model/.gitkeep`

**Role:** Keeps model package folder tracked.

**Why it matters:** Placeholder alongside model scripts.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Scaffold artifact.

**Possible improvements or risks:** None.

### `src/model/__init__.py`

**Role:** Marks model package.

**Why it matters:** Supports modular imports.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Python import system.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Model lifecycle code is isolated under `src/model`.

**Possible improvements or risks:** None.

### `src/model/model_building.py`

**Role:** Builds and trains the regression pipeline.

**Why it matters:** It defines the estimator architecture and serialized model artifact.

**Key dependencies/imports:** pandas, pickle, yaml, pathlib.Path, custom logger, time, ColumnTransformer, OneHotEncoder, StandardScaler, Pipeline, XGBRegressor.

**Exports/public surface:** `load_data`, `load_params`, `build_and_train_model`, `save_model`, `main`.

**Used by:** `dvc.yaml` stage `model_building`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 13-24 | `load_data` | Reads training CSV | CSV path | DataFrame | File read | Parser errors logged |
| 26-41 | `load_params` | Reads YAML params | `params.yaml` | dict | File read | Reused pattern |
| 43-70 | Transformer setup | One-hot encodes categorical columns; passes remainder through | X_train columns | ColumnTransformer | None | `handle_unknown='ignore'` supports unseen categories |
| 72-89 | Pipeline setup | Reads hyperparams and builds `Pipeline(preprocess, scale, regressor)` | Params | sklearn Pipeline | None | Scaling one-hot columns is unusual but valid |
| 92-102 | Fit | Trains pipeline and logs elapsed time | X_train, y_train | Fitted pipeline | CPU work | No validation split here |
| 105-115 | `save_model` | Writes pickle under `models/model.pkl` | Fitted model | Pickle file | Creates `models/` | `file_path` is prefixed with `models/` regardless of caller |
| 117-137 | `main` | Loads params/data, splits X/y, trains, saves | `train_final.csv` | `models/model.pkl` | File writes | Loads params in `main` but only uses them indirectly again in builder |

**Potential interview talking points:** The model uses a practical tabular pipeline with categorical one-hot encoding and XGBoost regression.

**Possible improvements or risks:** Add model signature, validation metrics during training, deterministic XGBoost settings where possible, and unit tests for feature column expectations.

### `src/model/model_evaluation.py`

**Role:** Evaluates the trained model and logs artifacts to MLflow/DagsHub.

**Why it matters:** This is the bridge from local DVC-trained artifact to remote experiment tracking and registry metadata.

**Key dependencies/imports:** numpy, pandas, pickle, json, sklearn metrics, logging/custom logger, mlflow, dagshub, os, shutil.

**Exports/public surface:** `load_model`, `load_data`, `evaluate_model`, `save_metrics`, `save_model_info`, `main`.

**Used by:** `dvc.yaml` stage `model_evaluation`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 14-30 | MLflow auth/tracking | Reads `CAPSTONE_TEST`, sets MLflow auth env, sets tracking URI | Env token | MLflow context | Raises at import if missing | Makes module hard to import in tests without secret |
| 39-51 | `load_model` | Unpickles trained model | `models/model.pkl` | sklearn Pipeline | File read | Pickle loading executes trusted artifact code |
| 53-64 | `load_data` | Reads test CSV | `test_final.csv` | DataFrame | File read | Parser errors logged |
| 66-84 | `evaluate_model` | Predicts and computes R2, MAE, RMSE | Model, X_test, y_test | metrics dict | Model inference | No thresholding here |
| 86-105 | Save helpers | Writes metrics JSON and model info JSON | dict/run info | Files | File writes | Report files ignored by Git |
| 107-150 | `main` | Starts MLflow run, loads model/test data, logs metrics/params/artifacts, saves MLflow model in `model_dir`, writes experiment info | Token, model, data | Report files, MLflow artifacts | Deletes existing `model_dir`; writes new `model_dir` | `model_dir` is tracked, so running can modify tracked artifact files |

**Potential interview talking points:** Evaluation records metrics and model artifacts in MLflow, then writes run info used for registration.

**Possible improvements or risks:** Avoid deleting tracked `model_dir` during evaluation or make it generated/ignored; move auth into `main`; re-raise failures from `main`; add model signature/input example.

### `src/model/register_model.py`

**Role:** Registers an MLflow run artifact as a model version and tags it staging.

**Why it matters:** Turns experiment output into a registry-managed model candidate.

**Key dependencies/imports:** json, os, warnings, logging, mlflow, MlflowClient.

**Exports/public surface:** `load_model_info`, `register_model`, `main`.

**Used by:** `dvc.yaml` stage `model_registration`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 14-16 | Warning filters | Suppresses user warnings | None | Warning behavior | Global warning filters | Can hide useful warnings |
| 19-31 | `load_model_info` | Reads run id and artifact path JSON | `reports/experiment_info.json` | dict | File read | Required after evaluation |
| 34-51 | `register_model` | Builds `runs:/<run_id>/<model_path>`, registers, tags `stage=staging` | Model info | MLflow version | Registry mutation | No duplicate/version policy beyond MLflow default |
| 54-82 | `main` | Authenticates to DagsHub MLflow, loads info, registers `my_model` | `CAPSTONE_TEST` | Staging model version | Registry mutation | Hardcoded URI/model name |

**Potential interview talking points:** Separates evaluation from registry registration, making lifecycle stages explicit.

**Possible improvements or risks:** Parameterize model name, add rollback/audit, and avoid broad warning suppression.

### `src/visualization/.gitkeep`

**Role:** Keeps visualization folder tracked.

**Why it matters:** Placeholder for future visualizations.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Git.

**Detailed code/chunk walkthrough:** Empty placeholder.

**Potential interview talking points:** Scaffold artifact.

**Possible improvements or risks:** None.

### `src/visualization/__init__.py`

**Role:** Marks visualization package.

**Why it matters:** Supports future imports.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** Python import system.

**Detailed code/chunk walkthrough:** Empty package marker.

**Potential interview talking points:** Placeholder for plots/exploratory visual tools.

**Possible improvements or risks:** None.

### `src/visualization/visualize.py`

**Role:** Empty visualization module.

**Why it matters:** Placeholder from project scaffold.

**Key dependencies/imports:** None.

**Exports/public surface:** None.

**Used by:** No repository evidence of use.

**Detailed code/chunk walkthrough:** Empty file.

**Potential interview talking points:** Honest gap: visualizations are not implemented in code.

**Possible improvements or risks:** Remove if unused or add meaningful plotting/reporting utilities.

### `test_environment.py`

**Role:** Checks Python major version.

**Why it matters:** Used by Makefile to validate a Python 3 environment.

**Key dependencies/imports:** `sys`.

**Exports/public surface:** `main()`.

**Used by:** `make test_environment`, `make requirements`.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 3 | `REQUIRED_PYTHON = "python3"` | Declares required major version | None | Constant | None | Only major version checked |
| 6-21 | `main` | Compares running Python major to required major | `sys.version_info` | Print or exception | Console output | Does not check minor version 3.10 |
| 24-25 | Script guard | Runs check when executed | CLI | Exit status | None | Useful for Makefile |

**Potential interview talking points:** Lightweight environment smoke test.

**Possible improvements or risks:** Check Python 3.10 specifically if CI/Docker require it.

### `tests/test_flask_app.py`

**Role:** Integration tests for the Flask app.

**Why it matters:** Confirms the home page and prediction endpoint work at a high level.

**Key dependencies/imports:** unittest, json, `flask_app.app`.

**Exports/public surface:** `FlaskAppTests`.

**Used by:** CI and manual unittest.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 1-7 | Imports | Imports app from `flask_app.app` | `CAPSTONE_TEST`, registry | App/test client dependency | Loads model at import | External dependency before tests run |
| 11-14 | `setUpClass` | Creates Flask test client | app | `cls.client` | None | Shared client |
| 16-21 | `test_home_page` | GET `/`, expects 200 and title | App route | Assertion | Metrics increment | Depends on template title |
| 23-55 | `test_predict_endpoint` | Posts JSON payload, expects numeric plausible `predicted_score` | Loaded model | Assertion | Model inference | Does not mock model; may fail if registry/model unavailable |
| 57-58 | Script guard | Runs unittest | CLI | Test results | None | Standard |

**Potential interview talking points:** These are smoke/integration tests of deployed behavior rather than isolated unit tests.

**Possible improvements or risks:** Mock MLflow/model loading for unit tests; add form validation tests and `/metrics` test.

### `tests/test_model.py`

**Role:** Integration/performance tests for the registered model.

**Why it matters:** Checks that a staging model can load and meet basic regression thresholds on processed test data.

**Key dependencies/imports:** os, unittest, MLflow, pandas, sklearn metrics.

**Exports/public surface:** `TestCricketScorePredictor`.

**Used by:** CI and manual unittest.

**Detailed code/chunk walkthrough:**

| Lines/Section | Code Chunk | What It Does | Inputs | Outputs | Side Effects | Notes/Edge Cases |
|---|---|---|---|---|---|---|
| 12-41 | `setUpClass` | Authenticates to MLflow, finds staging model version, loads model, reads holdout data | `CAPSTONE_TEST`, MLflow registry, `data/processed/test_final.csv` | Class model/X/y | Network and file reads | Fails if generated data absent |
| 43-61 | `get_latest_model_version` | Searches versions and selects newest tag `stage=staging` | Model name, stage | Version string | MLflow API call | Raises if no candidate |
| 63-65 | `test_model_loaded` | Asserts model object exists | Loaded model | Assertion | None | Basic smoke test |
| 67-74 | `test_signature_matches` | If model has input signature, compare to DataFrame columns | Model metadata, X columns | Assertion or skip | None | Skips if no signature logged |
| 76-86 | `test_regression_performance` | Predicts on holdout and checks RMSE <= 8, MAE <= 8, R2 >= 0.90 | Model/test data | Assertions | Model inference | Thresholds may need recalibration as data changes |
| 89-90 | Script guard | Runs unittest | CLI | Test result | None | Standard |

**Potential interview talking points:** The test enforces both loadability and minimum model quality, not just code execution.

**Possible improvements or risks:** Add deterministic local fixture model/data, avoid hitting remote registry in every unit test, and store baseline metrics.

### `tox.ini`

**Role:** flake8 configuration.

**Why it matters:** Defines style thresholds for `make lint`.

**Key dependencies/imports:** flake8.

**Exports/public surface:** `max-line-length = 79`, `max-complexity = 10`.

**Used by:** `flake8 src`.

**Detailed code/chunk walkthrough:** Three-line config under `[flake8]`.

**Potential interview talking points:** Linting is configured but not run in CI workflow.

**Possible improvements or risks:** Add lint step to CI and consider formatter config.

## 11. Cross-Cutting Concerns

### Security And Secrets Handling

- Good: `.gitignore` excludes `.env`, `.envrc`, `.pypirc`, and `creds.txt`.
- Good: CI uses GitHub Secrets for AWS, DagsHub, and DigitalOcean values.
- Good: Kubernetes uses a Secret reference for `CAPSTONE_TEST`.
- Risk: `src/connections/config.json` is tracked and contains SQL Server connection fields including `username` and `pass`; values are not copied here. Move sensitive values to environment variables or ignored local config.
- Risk: App sets the same token as both MLflow username and password. This may be acceptable for DagsHub token auth but should be documented.
- Risk: CI creates a Kubernetes secret using a command-line literal; GitHub masks secrets, but shell history/log exposure should still be considered.

### Error Handling

- Helper functions frequently log and re-raise errors, which is good.
- Several `main()` functions catch broad exceptions, log/print, and do not re-raise. That can cause command success despite pipeline failure.
- S3 and SQL connection helpers often return `None` on failures; downstream functions may then fail with less helpful errors.

### Logging And Observability

- `src/logger/__init__.py` centralizes rotating file and console logging.
- Flask app configures basic logging and exposes `app_request_count` and `app_request_latency_seconds`.
- `/metrics` exists, but there is no Prometheus scrape config in repo-managed Kubernetes manifests.
- No structured logs, trace IDs, model input/output logging, drift metrics, or alert rules are implemented.

### Testing Strategy

- Tests are `unittest` based and integration-heavy.
- They validate registry model loading, optional model signature, regression thresholds, home route, and JSON prediction route.
- Missing: isolated tests for ingestion parsing, preprocessing, feature engineering, error paths, promotion logic, Docker startup, and Kubernetes manifest validation.

### Performance

- `extract_delivery_df` iterates rows and nested deliveries in Python; fine for small datasets but may be slow for large historical corpora.
- `feature_engineering.py` computes rolling last-five values in Python loops over groups; acceptable at current scale but may be optimized.
- Flask loads the model once globally, which is good for per-request latency.
- JSON/form inference builds a one-row DataFrame per request, normal for sklearn pipelines.

### Scalability

- Kubernetes has 2 replicas, but no HPA or resource requests/limits.
- App is stateless except external registry dependency and in-memory model.
- Startup depends on registry availability, which can slow or fail scale-up.
- DVC training pipeline is single-process scripts, not distributed.

### Accessibility

- Template uses labels tied to inputs and Bootstrap classes.
- Improvements: fix CDN integrity, add clearer error focus, ensure contrast after custom colors, and add aria-live for prediction/errors.

### Data Privacy

- Cricket scorecards are public-style sports data from repository context; no personal user data is stored by the app.
- Prediction requests are not persisted in repo code.
- If logs are extended, avoid logging secrets and sensitive request data.

### Dependency Management

- Full requirements are heavy and partly unpinned.
- Serving requirements are slimmer but missing Gunicorn.
- Saved model metadata versions differ from serving requirements.
- Consider lockfiles or constraints for reproducibility.

### Maintainability

- DVC stage separation is clean.
- Some constants are duplicated: valid teams in preprocessing and Flask app; city lists are written as text artifacts.
- Several legacy `_old.py` modules and placeholders remain.
- Import-time side effects reduce testability.

### Deployment Readiness

- Positive: Dockerfile, CI/CD, K8s manifest, secret injection, Prometheus endpoint.
- Gaps: missing Gunicorn dependency, no probes/resources, mutable image tag, no health endpoint, external registry dependency at startup.

### Failure Modes

- External service outages: S3, DagsHub/MLflow, DigitalOcean registry/cluster.
- Missing generated files: `data/processed/test_final.csv`, `models/model.pkl`, report info.
- Registry tag mistakes: no staging/production tag.
- Version mismatch: local model artifact deps vs app deps.
- Hidden pipeline side effects: eligible city files updated but not declared as DVC outs.

### Technical Debt

- Duplicate `save_data` in ingestion.
- Stale Makefile `data` target.
- NUL bytes in `.gitignore`.
- Empty/placeholder packages and docs.
- Legacy connection modules not marked deprecated.

## 12. Testing And Validation

### Test Frameworks

- Python `unittest`.
- `tox.ini` contains flake8 settings, but no tox test environment is defined.

### Existing Tests

| Test file | Behavior covered | External dependencies |
|---|---|---|
| `tests/test_model.py` | Loads newest `stage=staging` model, optional signature check, RMSE/MAE/R2 thresholds | `CAPSTONE_TEST`, MLflow/DagsHub, `data/processed/test_final.csv` |
| `tests/test_flask_app.py` | GET `/` status/title, POST `/predict` JSON prediction shape/plausibility | `CAPSTONE_TEST`, MLflow/DagsHub production model |
| `test_environment.py` | Python major version is 3 | Python runtime |

### Coverage Gaps

- YAML parsing and schema variations.
- S3 failure behavior and empty bucket behavior.
- `preprocess_dataframe` team derivation edge cases.
- `engineer_and_split` feature correctness, city filtering, rolling window behavior.
- Model training pipeline shape/columns without full training.
- MLflow registration/promotion with mocked client.
- Flask form validation failures.
- `/metrics` route.
- Docker image startup.
- Kubernetes manifest validation.

### Suggested High-Value Tests

1. Unit-test `extract_delivery_df` with a tiny YAML fixture containing a wicket and a no-wicket ball.
2. Unit-test `preprocess_dataframe` with two teams and verify derived `bowling_team`.
3. Unit-test `engineer_and_split` on a synthetic match with known runs/wickets to verify `current_score`, `balls_left`, `wickets_left`, `crr`, `last_five`, and `total_runs`.
4. Mock MLflow client in `get_model_version_by_stage`, `register_model`, and `promote_model.py`.
5. Mock global model in Flask tests so `/predict` can be tested without DagsHub.
6. Test form validation for overs, wickets, negative score, and last-five greater than current score.
7. Add CI linting using existing `tox.ini`.
8. Add a container smoke test that verifies Gunicorn is installed and `/metrics` responds.

## 13. Build, Deployment, And Operations

### Build Process

- Python package: `pip install -r requirements.txt` and `-e .`.
- ML pipeline: `dvc repro`.
- Serving image: `docker build -t flask-app:latest .`.
- Docs: `make html` inside `docs/` or `docs/make.bat html`.

### Runtime Process

- Container starts `gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app`.
- `app.py` authenticates to MLflow, resolves production model, loads model, reads eligible city list, and serves Flask routes.

### Deployment Clues

- DigitalOcean Container Registry path in `deployment.yaml`.
- GitHub Actions uses `digitalocean/action-doctl@v2`.
- DOKS kubeconfig fetched by `doctl kubernetes cluster kubeconfig save`.
- Kubernetes service type is `LoadBalancer`.

### Docker/Kubernetes/Cloud Config

- Docker image copies only `flask_app/`.
- Kubernetes Deployment uses 2 replicas and `imagePullPolicy: Always`.
- Secret `capstone-secret` supplies `CAPSTONE_TEST`.
- Service exposes TCP port 5000.

### CI/CD Config

The CI pipeline is push-based and linear:

```text
push -> install -> dvc repro -> model tests -> promote -> Flask tests -> build/push image -> deploy
```

This means model promotion and deployment are coupled to every successful push.

### Monitoring/Logging Config

- App exposes `/metrics`.
- Metrics: request count by method/endpoint and request latency by endpoint.
- Projectflow includes manual Prometheus and Grafana setup notes, but no repo-managed Prometheus/Grafana manifests.

### Operational Risks

- Missing Gunicorn in serving requirements.
- No readiness/liveness probes means Kubernetes may send traffic before model load is ready.
- Model registry outage can prevent pod startup and rollout.
- Mutable `latest` image tag makes rollbacks/audits harder.
- No resource limits can lead to noisy-neighbor or scheduling issues.

### Production Incident Debugging From This Codebase

1. Check pod status and logs for `CAPSTONE_TEST` missing, model load failure, or Gunicorn missing.
2. Check Kubernetes secret `capstone-secret`.
3. Check DagsHub/MLflow model `my_model` for a production-tagged version.
4. Hit `/metrics` and inspect request count/latency.
5. Reproduce locally with `cd flask_app` and `python app.py` using the same env vars.
6. Validate image contains `eligible_cities.txt` and installed dependencies.
7. Review GitHub Actions logs for DVC, tests, image push, and rollout steps.

## 14. How To Modify Or Extend This Project

### Add A New Feature

1. Identify whether it belongs in ingestion, preprocessing, feature engineering, model training, serving, or deployment.
2. If it changes model features, update `src/features/feature_engineering.py`, `flask_app/app.py` JSON/form input construction, tests, and README/docs.
3. Add parameter knobs to `params.yaml` when they should be versioned.
4. Update `dvc.yaml` deps/params/outs if files or parameters change.
5. Run `dvc repro` and update `dvc.lock` if changes are intentional.

### Add A New Route/Page/Endpoint

- Add route function to `flask_app/app.py`.
- Add or update template under `flask_app/templates/`.
- If route uses model input, keep feature names aligned with training columns.
- Add Flask tests in `tests/test_flask_app.py`.
- Consider metrics labels for the new endpoint.

### Add A New Data Model Or Config

- Prefer YAML/JSON config over hardcoded duplicated constants.
- Add config file to DVC params if it affects pipeline outputs.
- Keep secrets out of tracked config.
- Validate loaded config early and fail clearly.

### Add Tests

- Use `unittest` to match existing style, or introduce pytest deliberately.
- Mock external services for unit tests.
- Keep integration tests for MLflow/DagsHub as a separate CI job or mark them clearly.

### Debug Common Issues

- Pipeline stage failure: run the stage command directly from `dvc.yaml`.
- Feature mismatch: compare `train_final.csv` columns, Flask JSON DataFrame columns, and MLflow signature if available.
- Registry issue: inspect `my_model` tags `stage=staging` and `stage=production`.
- Docker issue: run container locally with `CAPSTONE_TEST` and verify Gunicorn dependency.

### Avoid Breaking Existing Patterns

- Preserve DVC stage boundaries.
- Keep model-ready feature names exactly aligned unless retraining and serving are updated together.
- Keep generated artifacts out of normal Git unless intentionally tracked.
- Do not copy secret values into docs or code.

## 15. Interview Preparation Pack

### 15.1 Elevator Pitches

**30-second pitch:** I built an end-to-end MLOps project that predicts the final first-innings score of a men's T20 cricket match from live innings state. It ingests YAML scorecards, creates delivery-level features, trains an XGBoost regression pipeline, tracks and registers models with MLflow on DagsHub, promotes versions with staging/production tags, and serves predictions through a Flask app with Prometheus metrics.

**60-second pitch:** This project is less about inventing a new cricket algorithm and more about showing a complete ML lifecycle. Raw T20 scorecards are ingested from S3-compatible storage, flattened into ball-by-ball rows, cleaned, transformed into features like current score, balls left, wickets left, current run rate, and last-five-over runs, then trained with a scikit-learn/XGBoost pipeline. DVC orchestrates reproducibility, MLflow/DagsHub handles experiment tracking and registry, a promotion script moves models from staging to production by tag, and Flask loads the production-tagged model at startup for browser and JSON predictions.

**2-minute technical pitch:** The architecture starts with Cricsheet-style YAML scorecards and a DVC pipeline. `data_ingestion.py` fetches YAML files from an S3-compatible bucket, normalizes them with pandas, filters men's 20-over matches, and extracts first-innings deliveries. `data_preprocessing.py` derives the bowling team from the teams list and filters supported international sides. `feature_engineering.py` creates a supervised row for each innings state, including cumulative score, balls remaining, wickets remaining, current run rate, last 30-ball scoring, and final total as the label. `model_building.py` trains a scikit-learn Pipeline with OneHotEncoder for teams/city, StandardScaler, and XGBRegressor. `model_evaluation.py` computes R2, MAE, and RMSE, saves metrics, and logs the model to MLflow on DagsHub. `register_model.py` registers it as `my_model` with a `stage=staging` tag, and `promote_model.py` retags the newest staging version as production. The Flask app resolves the newest `stage=production` model at startup, serves `/predict`, and exposes Prometheus metrics at `/metrics`. CI runs the pipeline, tests, promotion, image build, registry push, and Kubernetes deploy.

**Recruiter-friendly pitch:** I built a cricket score prediction app that demonstrates the full path from raw sports data to a deployed ML service. It includes data pipelines, model training, experiment tracking, automated testing, containerization, cloud deployment, and monitoring.

**Senior-engineer technical pitch:** The project demonstrates a pragmatic ML platform slice: DVC for deterministic stage orchestration, external object storage for raw data/artifacts, explicit model registry contracts using MLflow model-version tags, a lightweight Flask serving boundary, and CI/CD that ties training, validation, promotion, image build, and DOKS rollout together. The model is intentionally conventional; the interesting work is lifecycle control, artifact handoff, and identifying operational gaps such as import-time registry coupling, missing container dependencies, and incomplete health/probe infrastructure.

### 15.2 Architecture Questions And Answers

**Q: Why use DVC here?**  
A: The project has a clear sequence of data and model artifacts: raw delivery CSV, interim data, processed train/test data, model pickle, metrics, and experiment info. DVC makes those stages explicit and tracks dependencies/params so changes can be reproduced.

**Q: What are the main components?**  
A: S3 ingestion in `src/connections` and `src/data`, feature engineering in `src/features`, model lifecycle in `src/model`, promotion in `scripts`, serving in `flask_app`, CI/CD in `.github/workflows`, and deployment in `Dockerfile`/`deployment.yaml`.

**Q: How does data flow end to end?**  
A: YAML scorecards -> normalized match DataFrame -> delivery-level rows -> cleaned team/city data -> engineered model features -> train/test CSVs -> trained XGBoost pipeline -> MLflow artifact -> registered model -> production-tagged version -> Flask pyfunc inference.

**Q: Why does the Flask app load from MLflow instead of a local pickle?**  
A: The app is registry-driven. Loading the newest production-tagged version decouples deployment from manually bundling a local model and lets promotion control what production serves.

**Q: What would fail first under production traffic?**  
A: Based on repo evidence, likely startup or rollout rather than per-request inference: missing Gunicorn dependency, missing `CAPSTONE_TEST`, no production-tagged model, registry outage, or no readiness probe.

**Q: How would you scale this app?**  
A: The app is stateless after model load, so horizontal scaling works through more Kubernetes replicas. I would add readiness probes, resource requests/limits, HPA, immutable image tags, and possibly cache/preload models to avoid registry bottlenecks during scale-up.

**Q: Where are the bottlenecks in training?**  
A: Python loops in YAML/delivery extraction and rolling feature computation could be slow at larger scale. XGBoost training also grows with feature cardinality and row count.

**Q: How would you monitor it?**  
A: The app already exposes Prometheus request count and latency. I would add scrape manifests, dashboards, alerts, model error-rate metrics, input distribution/drift metrics, and health endpoints.

**Q: How does model promotion work?**  
A: `register_model.py` registers a model version and tags it `stage=staging`. `promote_model.py` finds the newest staging version, removes `stage=production` from existing production versions, and tags the chosen version as production.

**Q: How does CI/CD work?**  
A: On push, GitHub Actions installs dependencies, runs `dvc repro`, runs model tests, promotes the model, runs Flask tests, builds/tags/pushes Docker image, creates/updates the Kubernetes secret, applies `deployment.yaml`, and restarts the deployment.

### 15.3 Code-Level Questions And Answers

**Q: What does `extract_delivery_df` do?**  
A: It selects required match-level normalized columns, filters to male 20-over matches, iterates through first-innings deliveries, and emits one row per ball with match id, teams, batting team, ball, batsman, bowler, runs, dismissal, city, and venue.

**Q: Why is there a duplicate `save_data` in `data_ingestion.py`?**  
A: The second definition shadows the first in Python. The active `save_data` writes one DataFrame to `data/raw/data.csv`; the earlier train/test version is dead code and should be removed.

**Q: How is `bowling_team` derived?**  
A: `data_preprocessing.py` parses the `teams` field with `ast.literal_eval` and returns the team that is not equal to `batting_team`.

**Q: What does `last_five` mean?**  
A: It is a rolling 30-delivery sum of runs within each match, corresponding to the last five overs in a 6-ball-over match.

**Q: Why are early rows dropped in feature engineering?**  
A: Rolling 30-ball sums are NaN for the first 29 deliveries of a match, and `final.dropna()` removes those rows.

**Q: What columns does the model expect?**  
A: `batting_team`, `bowling_team`, `city`, `current_score`, `balls_left`, `wickets_left`, `crr`, and `last_five`.

**Q: What is the target column?**  
A: `total_runs`, computed as the total first-innings runs per match.

**Q: How are categorical variables handled?**  
A: `model_building.py` uses a `ColumnTransformer` with `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` for `batting_team`, `bowling_team`, and `city`.

**Q: What does `handle_unknown='ignore'` protect against?**  
A: It prevents inference failures when a categorical value was not seen during training, although predictions for unseen categories may be less reliable.

**Q: Why does `tests/test_flask_app.py` require MLflow access?**  
A: It imports `flask_app.app`, and that module loads the production model from MLflow at import time.

### 15.4 Debugging Questions And Answers

**Scenario: Flask app crashes with missing `CAPSTONE_TEST`.**  
Symptom: Import/startup raises environment variable error. Likely cause: token not set locally or Kubernetes secret missing. Inspect `flask_app/app.py`, `deployment.yaml`, GitHub Actions secret creation. Reproduce with `cd flask_app` and `python app.py`. Fix by setting env var or creating `capstone-secret`. Prevent by adding startup checks and deployment validation.

**Scenario: Docker container exits because Gunicorn is not found.**  
Symptom: Container command fails at startup. Likely cause: Dockerfile uses `gunicorn`, but `flask_app/requirements.txt` does not include it. Inspect `Dockerfile` and `flask_app/requirements.txt`. Fix by adding Gunicorn to serving requirements or changing command. Prevent with container smoke test.

**Scenario: `dvc repro` does not reflect new params.**  
Symptom: Lockfile values differ from `params.yaml` or DVC reports changed params. Likely cause: params changed without rerunning DVC. Inspect `params.yaml` and `dvc.lock`. Fix by running `dvc repro` and committing updated lock if expected.

**Scenario: Model promotion fails with no staging version.**  
Symptom: `ValueError` from `scripts/promote_model.py`. Likely cause: registration did not happen or tag key/value mismatch. Inspect MLflow registry, `src/model/register_model.py`, `reports/experiment_info.json`. Fix by rerunning evaluation/registration or correcting tags.

**Scenario: `/predict` JSON returns 400.**  
Symptom: Missing field error. Likely cause: JSON API requires `current_score`, `balls_left`, `wickets_left`, `crr`, and `last_five`. Inspect `flask_app/app.py:predict`. Fix request payload or add server-side derivation for JSON.

**Scenario: Form prediction seems off for overs like `10.5`.**  
Symptom: Balls left calculation may not match cricket notation. Likely cause: code treats overs as decimal and computes `int(overs_done * 6)`. Inspect form branch in `predict`. Fix by parsing overs and balls explicitly.

### 15.5 Design Tradeoff Questions And Answers

**Simplicity vs scalability:** The pipeline uses straightforward pandas scripts and DVC stages, which are easy to explain and reproduce at small scale. At larger scale, YAML parsing and feature loops may need distributed or vectorized processing.

**Local vs cloud assumptions:** Local code can run, but true reproduction depends on S3 and DagsHub credentials. This demonstrates real MLOps integration but makes offline onboarding harder.

**Sync vs async behavior:** Training and serving are synchronous. There are no background workers. Simpler to operate, but long model loading at startup can delay readiness.

**Type safety:** The project uses pandas DataFrames and runtime validation rather than static schemas. Simple but vulnerable to schema drift.

**State management:** Runtime state is global and loaded once. Efficient for inference, but import-time side effects complicate testing.

**Error handling:** Broad try/except logging gives visibility, but some `main()` functions may swallow failures. CI should rely on non-zero exits.

**Testing choices:** Integration tests prove real registry behavior, but they are brittle without secrets/network. Add unit tests with mocks for speed and reliability.

**Framework/library choices:** Flask is lightweight and adequate for a portfolio inference app. XGBoost is strong for tabular regression. DVC and MLflow are appropriate for MLOps demonstration.

**Performance choices:** One-row DataFrame inference is fine for demo traffic. For higher throughput, batch inference or a dedicated serving stack could help.

### 15.6 Behavioral / STAR Stories

**Building the project:**  
Situation: Needed to demonstrate more than a notebook model. Task: Build a complete ML lifecycle for T20 score prediction. Action: Modularized ingestion, preprocessing, feature engineering, training, evaluation, registration, serving, CI, and deployment. Result: A repo that can be explained as an end-to-end MLOps system. This is supported by `dvc.yaml`, `src/`, `flask_app/`, CI, Docker, and K8s files.

**Debugging a hard issue (suggested framing):**  
Situation: A deployed Flask app fails to start. Task: Identify whether it is app code, dependency, secret, or registry. Action: Check container logs, confirm `CAPSTONE_TEST`, verify `my_model` has a production tag, and inspect Docker requirements for Gunicorn. Result: The repo evidence points to a likely missing Gunicorn dependency and registry-startup coupling. Mark this as suggested framing unless you actually experienced it.

**Architectural decision:**  
Situation: Need to serve the right model without baking a pickle into the image. Task: Decouple model promotion from app image. Action: Use MLflow model-version tags and have Flask resolve `stage=production` at startup. Result: Promotion can change served model without changing application code.

**Improving reliability (suggested framing):**  
Situation: Tests depend on external services. Task: Make CI more reliable. Action: Add mocks for MLflow/model loading and keep integration tests separate. Result: Faster unit tests plus targeted registry tests. Suggested future story based on visible gaps.

**Learning a new tool/framework:**  
Situation: Needed reproducible ML stages. Task: Introduce DVC. Action: Created `dvc.yaml`, `params.yaml`, `.dvc/config`, and stage outputs. Result: Pipeline stages are explicit and CI can run `dvc repro`.

**Handling ambiguity:**  
Situation: Raw cricket YAML is nested and inconsistent. Task: Define a usable supervised target. Action: Extract first innings deliveries and compute final total as label. Result: Each row represents current innings state -> eventual score.

**Testing/validation:**  
Situation: Need confidence that registered models work. Task: Validate loadability and quality. Action: `tests/test_model.py` loads staging model and checks RMSE, MAE, and R2 thresholds. Result: CI can gate promotion/deployment on basic model performance.

**Deployment/production readiness:**  
Situation: Need a deployable demo. Task: Package and run app on Kubernetes. Action: Added Dockerfile, GitHub Actions, and `deployment.yaml`. Result: The project has a cloud deployment path, with known next steps around probes/resources/Gunicorn.

### 15.7 Explain This Project To...

**A recruiter:** It is a machine learning project that predicts cricket scores and, more importantly, shows the full engineering workflow to train, track, test, deploy, and monitor an ML model.

**A non-technical user:** You enter the current match situation, like teams, city, score, overs, wickets, and recent runs, and the app estimates what the first-innings final score will be.

**A junior developer:** The repo is split into steps. One script loads data, one cleans it, one creates features, one trains the model, one evaluates it, and a Flask app serves predictions.

**A senior engineer:** It is a compact ML platform slice with DVC orchestration, object storage, MLflow registry promotion, stateless Flask serving, CI/CD, and Kubernetes deployment, with clear improvement areas around test isolation, startup dependencies, and production hardening.

**A product manager:** The product value is fast score projection during a T20 innings. The technical value is that the model can be retrained, evaluated, promoted, and deployed in a repeatable workflow.

**A hiring manager:** The project demonstrates practical MLOps judgment: not just modeling, but data pipelines, registry workflows, automated tests, deployment, monitoring, and honest documentation of limitations.

**An ML/AI engineer:** The model is a tabular regression pipeline using engineered innings-state features and XGBoost. The interesting ML engineering is around feature construction, registry/version management, and serving consistency.

## 16. Glossary

| Term | Definition |
|---|---|
| T20 | Cricket format where each team normally faces 20 overs |
| Innings | One team's batting turn |
| First-innings total | Final score made by the team batting first |
| Delivery/ball | One bowled ball; the project creates one row per delivery |
| Over | Set of six legal deliveries in cricket |
| Current score | Cumulative runs so far in the innings |
| Balls left | Remaining deliveries out of 120 in a T20 innings |
| Wickets left | Remaining dismissals before all out |
| CRR | Current run rate |
| Last five | Runs in the last 30 deliveries, representing five overs |
| `total_runs` | Model target: final first-innings score |
| DVC | Data Version Control, used to define/reproduce ML pipeline stages |
| MLflow | Experiment tracking/model registry tool |
| DagsHub | Hosted platform used here for MLflow tracking/registry |
| `my_model` | Registered MLflow model name used by this project |
| `stage` tag | Custom MLflow model-version tag for `staging`/`production` |
| Pyfunc | MLflow generic Python model interface used for serving |
| DOCR | DigitalOcean Container Registry |
| DOKS | DigitalOcean Kubernetes |
| `CAPSTONE_TEST` | Environment variable containing DagsHub/MLflow token |
| `eligible_cities.txt` | Training-derived city list used by serving UI |

## 17. Risks, Gaps, And Improvement Roadmap

### Highest-Risk Code Areas

1. `flask_app/app.py` import-time registry/model loading.
2. `Dockerfile`/`flask_app/requirements.txt` mismatch around Gunicorn.
3. `src/model/model_evaluation.py` deleting/writing tracked `model_dir`.
4. `src/data/data_ingestion.py` assumptions about YAML shape and duplicate `save_data`.
5. External-service-dependent tests and CI promotion on every push.

### Missing Tests

- Pure data parsing, preprocessing, feature engineering, form validation, metrics endpoint, promotion logic, Docker startup, and K8s manifest checks.

### Security Concerns

- Tracked SQL config with credential fields.
- Command-line secret creation in CI.
- Need to ensure no real tokens are ever committed.

### Performance Concerns

- Python loops in ingestion and rolling feature computation.
- Full requirements install is heavy.
- Startup model download from MLflow can slow scaling.

### Maintainability Concerns

- Duplicated constants and city/team lists.
- Legacy `_old.py` files.
- Stale Makefile target.
- Sparse docs scaffold.
- `.gitignore` NUL bytes.

### Documentation Gaps

- Sphinx docs are mostly scaffold placeholders.
- No explicit local mock/offline development guide.
- No detailed production runbook in repo-managed docs beyond `projectflow.txt`.

### Improvements Ordered By Impact

1. Add Gunicorn to serving requirements and container smoke test.
2. Move Flask app to app factory with injectable/mockable model loader.
3. Add readiness/liveness probes and health endpoint.
4. Add unit tests for data/feature/model-serving logic with fixtures and mocks.
5. Move SQL config secrets out of tracked JSON.
6. Declare eligible city files as DVC outputs or generated artifacts.
7. Align `params.yaml`, `dvc.lock`, and model artifact dependency versions.
8. Add monitoring-as-code for Prometheus/Grafana or ServiceMonitor.
9. Parameterize model name/stage/registry URI.
10. Clean stale scaffold/legacy files.

### Improvements Ordered By Effort

1. Clean `.gitignore` NUL bytes.
2. Add `gunicorn` to `flask_app/requirements.txt`.
3. Add `/healthz` route.
4. Add `/metrics` test.
5. Remove duplicate `save_data`.
6. Move team list to shared config.
7. Add mocked model loader in tests.
8. Add DVC outs for city files.
9. Add Kubernetes probes/resources.
10. Split CI into train/test/deploy jobs with environment gates.

## 18. Coverage Checklist

- **Total tracked files analyzed:** 204.
- **Total tracked folders analyzed:** 20.
- **Tracked file source:** `git ls-files` using the Git executable at `C:\Program Files\Git\cmd\git.exe` because `git` was not on PATH.
- **Notable untracked files before creating this document:** None; `git status --short` returned no output.
- **Files covered in deep dive:** All tracked source/config/docs/runtime/test files have explicit entries. The 137 tracked match YAML files are covered as a repeated data-asset group and individually listed below.
- **Files covered only at high level:** `model_dir/model.pkl` because it is a binary serialized model; `notebooks/t20s/*.yaml` because they are repeated raw data assets, not source code; empty `.gitkeep`, `__init__.py`, and placeholder files because there is no code to walk through.
- **Files skipped:** None.
- **Secrets limitation:** `src/connections/config.json` values were not copied; only keys were documented.
- **Validation limitation:** Application tests were not run for this documentation-only change because they require external MLflow/DagsHub credentials and generated DVC outputs; the validation focus was repository inventory and file coverage.

### 18.1 Full Tracked File Checklist

All tracked files from `git ls-files` are accounted for:

- [x] `.dvc/.gitignore`
- [x] `.dvc/config`
- [x] `.dvcignore`
- [x] `.github/workflows/ci.yaml`
- [x] `.gitignore`
- [x] `Dockerfile`
- [x] `LICENSE`
- [x] `Makefile`
- [x] `README.md`
- [x] `deployment.yaml`
- [x] `docs/Makefile`
- [x] `docs/commands.rst`
- [x] `docs/conf.py`
- [x] `docs/getting-started.rst`
- [x] `docs/index.rst`
- [x] `docs/make.bat`
- [x] `dvc.lock`
- [x] `dvc.yaml`
- [x] `eligible_cities.txt`
- [x] `flask_app/app.py`
- [x] `flask_app/eligible_cities.txt`
- [x] `flask_app/requirements.txt`
- [x] `flask_app/templates/index.html`
- [x] `model_dir/MLmodel`
- [x] `model_dir/conda.yaml`
- [x] `model_dir/model.pkl`
- [x] `model_dir/python_env.yaml`
- [x] `model_dir/requirements.txt`
- [x] `notebooks/.gitkeep`
- [x] `notebooks/life-cycle.ipynb`
- [x] `notebooks/t20s/211028.yaml`
- [x] `notebooks/t20s/211048.yaml`
- [x] `notebooks/t20s/222678.yaml`
- [x] `notebooks/t20s/225263.yaml`
- [x] `notebooks/t20s/225271.yaml`
- [x] `notebooks/t20s/226374.yaml`
- [x] `notebooks/t20s/237242.yaml`
- [x] `notebooks/t20s/238195.yaml`
- [x] `notebooks/t20s/249227.yaml`
- [x] `notebooks/t20s/251487.yaml`
- [x] `notebooks/t20s/251488.yaml`
- [x] `notebooks/t20s/255954.yaml`
- [x] `notebooks/t20s/258463.yaml`
- [x] `notebooks/t20s/258464.yaml`
- [x] `notebooks/t20s/287853.yaml`
- [x] `notebooks/t20s/287854.yaml`
- [x] `notebooks/t20s/287855.yaml`
- [x] `notebooks/t20s/287856.yaml`
- [x] `notebooks/t20s/287857.yaml`
- [x] `notebooks/t20s/287858.yaml`
- [x] `notebooks/t20s/287860.yaml`
- [x] `notebooks/t20s/287861.yaml`
- [x] `notebooks/t20s/287862.yaml`
- [x] `notebooks/t20s/287863.yaml`
- [x] `notebooks/t20s/287864.yaml`
- [x] `notebooks/t20s/287865.yaml`
- [x] `notebooks/t20s/287866.yaml`
- [x] `notebooks/t20s/287867.yaml`
- [x] `notebooks/t20s/287868.yaml`
- [x] `notebooks/t20s/287869.yaml`
- [x] `notebooks/t20s/287870.yaml`
- [x] `notebooks/t20s/287871.yaml`
- [x] `notebooks/t20s/287872.yaml`
- [x] `notebooks/t20s/287873.yaml`
- [x] `notebooks/t20s/287874.yaml`
- [x] `notebooks/t20s/287875.yaml`
- [x] `notebooks/t20s/287876.yaml`
- [x] `notebooks/t20s/287877.yaml`
- [x] `notebooks/t20s/287878.yaml`
- [x] `notebooks/t20s/287879.yaml`
- [x] `notebooks/t20s/291343.yaml`
- [x] `notebooks/t20s/291356.yaml`
- [x] `notebooks/t20s/296903.yaml`
- [x] `notebooks/t20s/297800.yaml`
- [x] `notebooks/t20s/298795.yaml`
- [x] `notebooks/t20s/298804.yaml`
- [x] `notebooks/t20s/300435.yaml`
- [x] `notebooks/t20s/300436.yaml`
- [x] `notebooks/t20s/306987.yaml`
- [x] `notebooks/t20s/306989.yaml`
- [x] `notebooks/t20s/306991.yaml`
- [x] `notebooks/t20s/319112.yaml`
- [x] `notebooks/t20s/319142.yaml`
- [x] `notebooks/t20s/343764.yaml`
- [x] `notebooks/t20s/350050.yaml`
- [x] `notebooks/t20s/350347.yaml`
- [x] `notebooks/t20s/350475.yaml`
- [x] `notebooks/t20s/350476.yaml`
- [x] `notebooks/t20s/351694.yaml`
- [x] `notebooks/t20s/351695.yaml`
- [x] `notebooks/t20s/351696.yaml`
- [x] `notebooks/t20s/352674.yaml`
- [x] `notebooks/t20s/354456.yaml`
- [x] `notebooks/t20s/355988.yaml`
- [x] `notebooks/t20s/355989.yaml`
- [x] `notebooks/t20s/355991.yaml`
- [x] `notebooks/t20s/355992.yaml`
- [x] `notebooks/t20s/355993.yaml`
- [x] `notebooks/t20s/355994.yaml`
- [x] `notebooks/t20s/355995.yaml`
- [x] `notebooks/t20s/355996.yaml`
- [x] `notebooks/t20s/355997.yaml`
- [x] `notebooks/t20s/355998.yaml`
- [x] `notebooks/t20s/355999.yaml`
- [x] `notebooks/t20s/356000.yaml`
- [x] `notebooks/t20s/356001.yaml`
- [x] `notebooks/t20s/356002.yaml`
- [x] `notebooks/t20s/356003.yaml`
- [x] `notebooks/t20s/356004.yaml`
- [x] `notebooks/t20s/356005.yaml`
- [x] `notebooks/t20s/356006.yaml`
- [x] `notebooks/t20s/356007.yaml`
- [x] `notebooks/t20s/356008.yaml`
- [x] `notebooks/t20s/356009.yaml`
- [x] `notebooks/t20s/356010.yaml`
- [x] `notebooks/t20s/356011.yaml`
- [x] `notebooks/t20s/356012.yaml`
- [x] `notebooks/t20s/356013.yaml`
- [x] `notebooks/t20s/356014.yaml`
- [x] `notebooks/t20s/356015.yaml`
- [x] `notebooks/t20s/356016.yaml`
- [x] `notebooks/t20s/356017.yaml`
- [x] `notebooks/t20s/361530.yaml`
- [x] `notebooks/t20s/361531.yaml`
- [x] `notebooks/t20s/361656.yaml`
- [x] `notebooks/t20s/361660.yaml`
- [x] `notebooks/t20s/366622.yaml`
- [x] `notebooks/t20s/366707.yaml`
- [x] `notebooks/t20s/366708.yaml`
- [x] `notebooks/t20s/386494.yaml`
- [x] `notebooks/t20s/386535.yaml`
- [x] `notebooks/t20s/387563.yaml`
- [x] `notebooks/t20s/387564.yaml`
- [x] `notebooks/t20s/391794.yaml`
- [x] `notebooks/t20s/392615.yaml`
- [x] `notebooks/t20s/401076.yaml`
- [x] `notebooks/t20s/403375.yaml`
- [x] `notebooks/t20s/403385.yaml`
- [x] `notebooks/t20s/403386.yaml`
- [x] `notebooks/t20s/406197.yaml`
- [x] `notebooks/t20s/406198.yaml`
- [x] `notebooks/t20s/406207.yaml`
- [x] `notebooks/t20s/412677.yaml`
- [x] `notebooks/t20s/412678.yaml`
- [x] `notebooks/t20s/412680.yaml`
- [x] `notebooks/t20s/412681.yaml`
- [x] `notebooks/t20s/412682.yaml`
- [x] `notebooks/t20s/412683.yaml`
- [x] `notebooks/t20s/412684.yaml`
- [x] `notebooks/t20s/412685.yaml`
- [x] `notebooks/t20s/412686.yaml`
- [x] `notebooks/t20s/412688.yaml`
- [x] `notebooks/t20s/412689.yaml`
- [x] `notebooks/t20s/412690.yaml`
- [x] `notebooks/t20s/412691.yaml`
- [x] `notebooks/t20s/412692.yaml`
- [x] `notebooks/t20s/412693.yaml`
- [x] `notebooks/t20s/412694.yaml`
- [x] `notebooks/t20s/412695.yaml`
- [x] `notebooks/t20s/412696.yaml`
- [x] `notebooks/t20s/412697.yaml`
- [x] `notebooks/t20s/412698.yaml`
- [x] `notebooks/t20s/412699.yaml`
- [x] `notebooks/t20s/412700.yaml`
- [x] `notebooks/t20s/412701.yaml`
- [x] `notebooks/t20s/412702.yaml`
- [x] `notebooks/t20s/412703.yaml`
- [x] `params.yaml`
- [x] `projectflow.txt`
- [x] `pyproject.toml`
- [x] `references/.gitkeep`
- [x] `reports/.gitignore`
- [x] `reports/.gitkeep`
- [x] `reports/figures/.gitkeep`
- [x] `requirements.txt`
- [x] `scripts/promote_model.py`
- [x] `setup.py`
- [x] `src/__init__.py`
- [x] `src/connections/__init__.py`
- [x] `src/connections/config.json`
- [x] `src/connections/s3_connection.py`
- [x] `src/connections/s3_connection_old.py`
- [x] `src/connections/ssms_connection.py`
- [x] `src/connections/ssms_connection_old.py`
- [x] `src/data/.gitkeep`
- [x] `src/data/__init__.py`
- [x] `src/data/data_ingestion.py`
- [x] `src/data/data_preprocessing.py`
- [x] `src/features/.gitkeep`
- [x] `src/features/__init__.py`
- [x] `src/features/feature_engineering.py`
- [x] `src/logger/__init__.py`
- [x] `src/model/.gitkeep`
- [x] `src/model/__init__.py`
- [x] `src/model/model_building.py`
- [x] `src/model/model_evaluation.py`
- [x] `src/model/register_model.py`
- [x] `src/visualization/.gitkeep`
- [x] `src/visualization/__init__.py`
- [x] `src/visualization/visualize.py`
- [x] `test_environment.py`
- [x] `tests/test_flask_app.py`
- [x] `tests/test_model.py`
- [x] `tox.ini`

