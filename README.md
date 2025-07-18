# Cricket T20 Score Predictor — End‑to‑End MLOps Case Study

> **Live demo:** [http://144.126.254.108:5000/](http://144.126.254.108:5000/)
> **Prometheus:** [http://64.225.85.3:9090/](http://64.225.85.3:9090/)  |  **Grafana:** [http://152.42.157.72/](http://152.42.157.72/)

---

## 1 • Why this project?

Cricket T20 scores swing wildly from ball to ball. Accurate, real‑time projections can help commentators, fantasy‑sports analysts, and data‑driven fans. This repository shows **how to turn a pure machine‑learning idea into an enterprise‑grade, cloud‑native product**—covering the full spectrum from raw data to monitored production API.

*Everything is built and deployed on **DigitalOcean** to keep costs predictable, yet the workflow remains cloud‑agnostic.*

---

## 2 • Key Skills & Technologies

| Domain               | Stack                                                                           |
| -------------------- | ------------------------------------------------------------------------------- |
| **Data Science**     | Pandas · NumPy · Seaborn · Matplotlib · XGBoost · Scikit‑Learn                  |
| **MLOps**            | Cookiecutter Data‑Science · DVC · MLflow (⌂ DagsHub)                            |
| **Dev & Ops**        | Docker · GitHub Actions CI/CD · DigitalOcean Kubernetes (DOKS) · DO Spaces (S3) |
| **Observability**    | Prometheus (/metrics endpoint) · Grafana dashboards                             |
| **Coding Standards** | Modular OOP Python (src/), logger + unit tests, pre‑commit, flake8              |

---

## 3 • Data & Feature Engineering

| Step                   | Details                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Source**             | Ball‑by‑ball YAML files for every men’s T20 International up to *July 2025* from **[Cricsheet](https://cricsheet.org/)**                                  |
| **Raw → Bronze**       | Parse YAML ➜ tidy per‑ball DataFrame (overs, ball#, batter, bowler, runs, wickets)                                                                        |
| **Silver**             | Added context features (current run‑rate, required run‑rate, wickets in hand, phase, venue‑encoded, opposition‑strength bins, Power‑play indicator, etc.) |
| **Gold / Model input** | One‑hot / target‑oriented encoding, scaling via ColumnTransformer, saved as DVC artifact                                                                  |

Each transformation stage is an explicit **DVC pipeline stage** (see `dvc.yaml`) so anyone can reproduce the exact training set with one command:

```bash
dvc repro   # DAG guarantees determinism
```

---

## 4 • Model Development

* **Algorithm:** XGBoost Regressor — chosen for non‑linear interactions & speed on tabular data.
* **Validation:** Stratified 5‑fold split on match‑id to avoid data leakage across overs.
* **Best run (logged in MLflow):**

  * *RMSE:* **4.052 runs**
  * *MAE:* 1.906 runs
  * *R²:* 0.985
* Hyper‑parameters, metrics, artifacts, and the production model are versioned in **MLflow on DagsHub** (`runs/`, `models/`).

---

## 5 • Software Architecture

```text
src/
 ├── data_ingestion.py      # Cricsheet downloader & parser
 ├── data_preprocessing.py  # cleaning / formatting
 ├── feature_engineering.py # context feature builders
 ├── model_building.py      # training pipeline class
 ├── model_evaluation.py    # metric utils + plots
 ├── register_model.py      # push to MLflow
 ├── logger/                # structured logging
 └── config/params.yaml     # all hyper‑params & paths
```

Each module is **stateless & unit‑tested**, making local debugging and CI validation straightforward.

---

## 6 • Serving Layer (Flask API)

* `POST /predict` — JSON body with current match scenario ➜ returns projected innings total.
* `GET /metrics` — Prometheus‑compatible exposition of latency, request‑count, and model‑score drift (inference vs. ground truth where available).

The API is wrapped in a **Slim Alpine Docker image (\~120 MB)** built via multi‑stage build.

---

## 7 • Cloud‑Native Deployment (DigitalOcean)

### 7.1 Container Registry & CI

1. GitHub Action (`ci.yaml`) lints, tests, builds the image, and pushes to **Docker Hub** (or **DOCR**).
2. On successful push, the workflow triggers `kubectl apply` to update the live Deployment.

### 7.2 Kubernetes Objects

* **Deployment** (rolling updates, 1‑3 replicas)
* **Service LoadBalancer** — automatic external IP (`144.126.254.108`)
* **Secret** — DagsHub token & DVC remote creds injected as env vars
* **PVC** (optional) — mounts model cache if warm‑start required

All manifests live under `k8s/` and can be recreated with:

```bash
doctl kubernetes cluster create ...   # see projectflow.txt
kubectl apply -f k8s/
```

---

## 8 • Monitoring & Alerting

| Component      | What it does                                                                | Where                    |
| -------------- | --------------------------------------------------------------------------- | ------------------------ |
| **Prometheus** | Scrapes `/metrics` every 15 s; stores TS data                               | DO Droplet 1 (port 9090) |
| **Grafana**    | Dashboards: traffic, p‑95 latency, error‑rate,<br> model drift, Pod CPU/RAM | DO Droplet 2 (port 3000) |
| **Alerts**     | High latency, 5xx spike, RMSE drift > 20 %: Slack webhook                   | Prometheus rules         |

Both VMs are provisioned via simple user‑data scripts (see `infra/prometheus/` & `infra/grafana/`). No AWS CloudWatch or IAM overhead.

---

## 9 • Local Setup (Mac/Win/Linux)

```bash
git clone https://github.com/srikarashankara/cricket-t20-score-predictor.git
cd cricket-t20-score-predictor
conda env create -f environment.yml   # or use requirements.txt
conda activate atlas

# Re‑create data & train
 dvc pull               # if you just want the processed dataset
 dvc repro              # OR regenerate from raw YAMLs

# Run notebook experiments (optional)
 jupyter lab

# Serve locally
 cd flask_app
 python app.py          # http://127.0.0.1:5000/docs
```

Dockerised one‑liner:

```bash
docker run -p 8888:5000 srikarashankara/cricket-app:latest
```

---

## 10 • Re‑deploy in Your Own Cloud (Quick Guide)

All commands (cluster creation, registry login, secrets) are captured step‑by‑step in **[`projectflow.txt`](projectflow.txt)**. Replace my domain names and token values with your own and run through Sections 24‑36.

---

## 11 • Future Work

* **Ensemble models** (LightGBM, CatBoost) with stacked meta‑learner.
* **Real‑time feature store** (Redis Streams) to feed live broadcasts.
* **Canary deployments** with Argo Rollouts to minimise risk.
* **Terraform** modules to codify all DO resources.

---

## 12 • Contact & Links

* **LinkedIn:** [https://www.linkedin.com/in/srikarashankara/](https://www.linkedin.com/in/srikarashankara/)
* **Email:** [mailto\:srikarashankara@outlook.com](mailto:srikarashankara@outlook.com)

---
