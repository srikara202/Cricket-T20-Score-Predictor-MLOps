# Cricket T20 Score Predictor - MLOps Implementation

This project demonstrates a complete MLOps pipeline using cricket T20 score prediction as a practical use case. The primary focus is on showcasing modern MLOps practices, tools, and deployment strategies rather than the cricket prediction model itself.

## Why This Project?

As a learner of data science, I wanted to build something that goes beyond just training a model in a Jupyter notebook. This project demonstrates the entire machine learning lifecycle - from data ingestion to production monitoring. I chose cricket T20 score prediction because it provides an interesting real-world dataset, but the MLOps architecture could easily be adapted for any Machine Learning problem.

Project is hosted at [http://144.126.254.108:5000/](http://144.126.254.108:5000/)

## MLOps Architecture Overview

This project implements a production-ready MLOps pipeline with the following components:

- **Experiment Management**: MLflow for tracking experiments, model versions, and artifacts
- **Data Pipeline**: DVC for data versioning and pipeline automation
- **Code Management**: Git/GitHub with proper branching and collaboration workflows
- **CI/CD**: GitHub Actions for automated testing, building, and deployment
- **Containerization**: Docker for consistent deployment environments
- **Orchestration**: Kubernetes for scalable, cloud-native deployment
- **Monitoring**: Prometheus + Grafana for observability and alerting
- **Storage**: AWS S3-compatible storage for data artifacts

## Project Structure

The project follows the Cookiecutter Data Science template with some modifications:

```
├── .github/workflows/          # GitHub Actions CI/CD pipelines
├── src/                        # Source code modules
│   ├── logger/                 # Structured logging utilities
│   ├── data_ingestion.py       # Data collection and initial processing
│   ├── data_preprocessing.py   # Data cleaning and validation
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── model_building.py       # Model training pipeline
│   ├── model_evaluation.py     # Model validation and metrics
│   └── register_model.py       # Model registration to MLflow
├── flask_app/                  # Production API service
├── deployment.yaml             # Kubernetes deployment manifests
├── tests/                      # Unit and integration tests
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Configuration parameters
└── requirements.txt            # Python dependencies
```

## MLOps Tools and Technologies Used

### 1. Project Structure & Code Organization
- **Cookiecutter Data Science**: Standardized project template for reproducibility
- **Modular Python Architecture**: Object-oriented design with separation of concerns
- **Git/GitHub**: Version control with feature branching and pull request workflows

### 2. Data Management & Pipeline Automation
- **DVC (Data Version Control)**: Tracks data versions, creates reproducible pipelines
- **AWS S3**: Centralized data storage with versioning capabilities
- **Pipeline Automation**: Automated data processing stages with dependency management

### 3. Experiment Tracking & Model Management
- **MLflow**: Comprehensive experiment tracking, model registry, and artifact storage
- **DagsHub Integration**: Cloud-hosted MLflow with Git integration
- **Model Versioning**: Automatic model versioning with metadata and lineage tracking

### 4. Continuous Integration & Deployment
- **GitHub Actions**: Automated CI/CD pipelines triggered by code changes
- **Docker**: Containerized applications for consistent deployments
- **Multi-stage Builds**: Optimized Docker images for production

### 5. Cloud-Native Deployment
- **Kubernetes**: Container orchestration for scalability and reliability
- **DigitalOcean Kubernetes (DOKS)**: Managed Kubernetes service
- **LoadBalancer Services**: Automatic external IP assignment and traffic distribution

### 6. Monitoring & Observability
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Visualization dashboards and alerting
- **Custom Metrics**: Application-specific metrics for model performance monitoring

## Key MLOps Practices Demonstrated

### Reproducible Pipelines
The entire data processing and model training pipeline is defined in `dvc.yaml`. Anyone can reproduce the exact same results by running:

```bash
dvc repro
```

This ensures that data transformations, feature engineering, and model training are completely reproducible across different environments.

### Experiment Tracking
Every model training run is automatically logged to MLflow with:
- Hyperparameters and configuration
- Training and validation metrics
- Model artifacts and dependencies
- Data versioning information

### Automated CI/CD
The GitHub Actions workflow automatically:
1. Runs unit tests on every pull request
2. Builds and tests Docker images
3. Deploys to staging/production environments
4. Updates Kubernetes deployments with zero downtime

### Infrastructure as Code
All deployment configurations are version-controlled:
- Kubernetes manifests define the production environment
- Docker configurations ensure consistent runtime environments
- DVC pipelines define data processing steps

### Model Monitoring
The production API exposes Prometheus metrics for:
- Request latency and throughput
- Model prediction confidence
- Data drift detection
- System resource utilization

## Getting Started

### Prerequisites

- Python 3.10+
- Docker Desktop
- AWS CLI (for S3 access)
- kubectl (for Kubernetes deployment)

### Local Deployment

```bash
# Clone and setup environment
git clone https://github.com/srikara202/Cricket-T20-Score-Predictor-MLOps.git
cd Cricket-T20-Score-Predictor-MLOps
conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt

# Create a dagshub account and add its API key to the CAPSTONE_TEST environment variable

# linux
export CAPSTONE_TEST=[keyhere]

# windows
set CAPSTONE_TEST=[keyhere]
```

Download the t20 international YAML data (bunch of YAML files) from [cricsheet](https://cricsheet.org/downloads/), Set up AWS IAM user and S3 bucket and upload your data there. Add the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY. Then:

```bash
# Initialize DVC and run pipeline
dvc init
dvc repro

# Start local API server
cd flask_app
python app.py
```

### Kubernetes CLuster Deployment and Monitoring

---

#### 1. Provision your Kubernetes infrastructure

1. **Create a DigitalOcean Kubernetes Cluster**

   * In the DO Control Panel → Kubernetes → Create Cluster
   * Region: e.g. **BLR1**
   * Kubernetes version: **1.33.1-do.1**
   * Node pool: 2 nodes × 2 vCPU, 4 GB RAM (you can size this however you like)

2. **Create a Container Registry (DOCR)**

   * In the DO Control Panel → Container Registry → Create Registry
   * Name it e.g. `flask-app-container-registry`
   * Region: **BLR1**
   * Note your registry endpoint:

     ```
     registry.digitalocean.com/flask-app-container-registry
     ```

3. **Grant your DO Kubernetes cluster pull-access to DOCR**

   * In the registry’s **Settings** tab → Integrations → Kubernetes → “Connect” → select your cluster → “Allow pull from this registry”

---

#### 2. CI/CD: build & deploy via GitHub Actions

Create **`.github/workflows/ci.yaml`** in your repo:

```yaml
name: CI → DOCR → DOKS

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install deps & run tests
        run: |
          pip install -r requirements.txt
          pytest

      # ── Build & Push Docker image ────────────────────────────────
      - name: Log in to DOCR
        uses: docker/login-action@v2
        with:
          registry: registry.digitalocean.com
          username: ${{ secrets.DOCR_USERNAME }}
          password: ${{ secrets.DOCR_TOKEN }}

      - name: Build & push image
        run: |
          IMAGE=registry.digitalocean.com/flask-app-container-registry/flask-app:latest
          docker build -t $IMAGE .
          docker push    $IMAGE

      # ── Deploy to DOKS ────────────────────────────────────────────
      - name: Set up doctl
        uses: digitalocean/action-doctl@v2
        with:
          token: ${{ secrets.DIGITALOCEAN_ACCESS_TOKEN }}

      - name: Fetch kubeconfig
        run: |
          doctl kubernetes cluster kubeconfig save flask-app-cluster

      - name: kubectl apply
        run: |
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml
```

> **GitHub Secrets you’ll need:**
>
> * `DOCR_USERNAME` & `DOCR_TOKEN` → your DOCR creds
> * `DIGITALOCEAN_ACCESS_TOKEN` → with read/write permissions on your DO resources

---

#### 3. Kubernetes manifests

Put these under `k8s/` in your repo:

##### `deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
  labels:
    app: flask-app
spec:
  replicas: 2
  selector:
    matchLabels: { app: flask-app }
  template:
    metadata:
      labels: { app: flask-app }
    spec:
      containers:
      - name: flask-app
        image: registry.digitalocean.com/flask-app-container-registry/flask-app:latest
        ports: [{ containerPort: 5000 }]
        resources:
          requests: { cpu: "250m", memory: "256Mi" }
          limits:   { cpu: "1",    memory: "512Mi" }
        readinessProbe:
          httpGet: { path: "/", port: 5000 }
          initialDelaySeconds: 5
          periodSeconds: 10
        env:
        - name: CAPSTONE_TEST
          valueFrom:
            secretKeyRef:
              name: capstone-secret
              key: CAPSTONE_TEST
```

##### `service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: flask-app-service
  labels: { app: flask-app }
spec:
  type: LoadBalancer
  selector: { app: flask-app }
  ports:
    - name: http
      port: 5000
      targetPort: 5000
```

1. **`kubectl apply -f k8s/`** will stand up your app behind a DO LoadBalancer.
2. **`kubectl get svc`** will show you an **EXTERNAL-IP** you can browse at `http://<EXTERNAL-IP>:5000/`.

---

#### 4. Install Prometheus & Grafana via Helm

We used the upstream **kube-prometheus-stack** chart (Prometheus Operator + Grafana):

```bash
# 1. Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 2. Create monitoring namespace
kubectl create namespace monitoring

# 3. Install the chart
helm install prometheus \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.service.type=LoadBalancer \
  --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.release=prometheus
```

* This deploys:

  * **Prometheus Operator**
  * A **Prometheus** instance
  * **Alertmanager**
  * **Grafana** (with its own LoadBalancer service)
  * Node Exporter, Kube-State-Metrics, and default and CRD-based ServiceMonitors

---

#### 5. Access Grafana & build dashboards

1. **Find Grafana’s endpoint**:

   ```bash
   kubectl -n monitoring get svc prometheus-grafana
   ```

   It will have a **LoadBalancer Ingress** IP.

2. **Browse** `http://<GRAFANA-LB-IP>:80/`

   * Default login: `admin` / `prom-operator` (or whatever the chart notes)
   * Add your Prometheus data source if it isn’t auto-configured:

     * URL: `http://prometheus-operated.monitoring.svc.cluster.local:9090`
   * Import community dashboards or create your own queries:

     ```promql
     rate(http_requests_total{job="flask-app-service"}[1m])
     ```
---

## Monitoring and Observability

The project includes comprehensive monitoring:

- **Prometheus Server**: Collects metrics from the API endpoints
- **Grafana Dashboards**: Visualizes system and model performance
- **Alerting Rules**: Notifications for anomalies and performance issues

Access monitoring:
- Prometheus: `http://<prometheus-server>:9090`
- Grafana: `http://<grafana-server>:3000`

## What I Learned

Building this project taught me several important lessons about MLOps:

1. **Automation is Critical**: Manual deployment processes are error-prone and don't scale
2. **Monitoring is Essential**: You can't manage what you don't measure
3. **Infrastructure as Code**: Version-controlled infrastructure prevents configuration drift
4. **Data Versioning**: Just as important as code versioning for ML projects
5. **Security**: Proper secret management and access controls are non-negotiable

## Technical Specifications

- **Model**: XGBoost Regressor with hyperparameter tuning
- **API Framework**: Flask with Prometheus metrics integration
- **Container Runtime**: Docker with Alpine Linux base
- **Orchestration**: Kubernetes with rolling updates
- **Storage**: S3-compatible object storage for artifacts
- **Monitoring**: Prometheus + Grafana stack

This project demonstrates that MLOps isn't just about deploying models - it's about building sustainable, scalable systems that can evolve with business needs while maintaining reliability and observability.

---
1. LinkedIn: https://www.linkedin.com/in/srikarashankara/
2. Email: srikarashankara@outlook.com
---
*Built to showcase MLOps best practices with real-world applicability*