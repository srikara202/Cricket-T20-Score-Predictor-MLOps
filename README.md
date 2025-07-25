# Cricket T20 Score Predictor - MLOps Implementation

This project demonstrates a complete MLOps pipeline using cricket T20 score prediction as a practical use case. The primary focus is on showcasing modern MLOps practices, tools, and deployment strategies rather than the cricket prediction model itself.

## Why This Project?

As a learner of data science, I wanted to build something that goes beyond just training a model in a Jupyter notebook. This project demonstrates the entire machine learning lifecycle - from data ingestion to production monitoring. I chose cricket T20 score prediction because it provides an interesting real-world dataset, but the MLOps architecture could easily be adapted for any Machine Learning problem.

Project is hosted at [http://144.126.254.108:5000/](http://144.126.254.108:5000/) (disabled now due to heavy billing from DigitalOcean)

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
├── deployment.yaml             # Kubernetes deployment and service manifests
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

## CI/CD Pipeline Details

The project implements a comprehensive CI/CD pipeline using GitHub Actions that automates testing, building, and deployment:

### Pipeline Stages

**1. Automated Testing:**
- Runs DVC pipeline to validate data processing
- Executes unit tests for model validation (`test_model.py`)
- Tests Flask application endpoints (`test_flask_app.py`)
- Only proceeds to deployment if all tests pass

**2. Model Promotion:**
- Automatically promotes successful models to production in MLflow
- Ensures only validated models reach production environment

**3. Container Building:**
- Builds optimized Docker images using multi-stage builds
- Tags and pushes to DigitalOcean Container Registry (DOCR)
- Implements caching strategies for faster builds

**4. Kubernetes Deployment:**
- Performs rolling updates with zero downtime
- Creates/updates Kubernetes secrets for environment variables
- Automatically restarts deployments to pull latest images

### Security & Configuration

**Environment Variables Management:**
- Sensitive data stored as GitHub Secrets
- Kubernetes secrets created dynamically during deployment
- No hardcoded credentials in codebase

**Required GitHub Secrets:**
```
DO_TOKEN                 # DigitalOcean API access token
DO_REGISTRY             # Container registry URL
DO_CLUSTER_NAME         # Kubernetes cluster name
AWS_ACCESS_KEY_ID       # S3 storage access
AWS_SECRET_ACCESS_KEY   # S3 storage secret
CAPSTONE_TEST           # MLflow/DagsHub authentication
```

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

## DigitalOcean Infrastructure Setup & Deployment

### Prerequisites

Before deploying to DigitalOcean, ensure you have:
- DigitalOcean account with billing enabled
- `doctl` CLI tool installed
- `kubectl` installed locally
- Docker Desktop running

### Step 1: Install and Configure DigitalOcean CLI

**Install doctl:**
```bash
# macOS (Homebrew)
brew install doctl

# Windows (Chocolatey)
choco install doctl

# Linux (Snap)
sudo snap install doctl

# Or download binary from: https://github.com/digitalocean/doctl/releases
```

**Authenticate with DigitalOcean:**
```bash
# Get your API token from: https://cloud.digitalocean.com/account/api/tokens
doctl auth init --access-token YOUR_DO_TOKEN_HERE

# Verify authentication
doctl account get
```

### Step 2: Create DigitalOcean Resources

**Create Kubernetes Cluster (DOKS):**
```bash
# Create cluster with 2 nodes
doctl kubernetes cluster create flask-app-cluster \
    --region blr1 \
    --version 1.33.1-do.1 \
    --size s-2vcpu-4gb \
    --node-pool "name=flask-app-nodes;count=2;size=s-2vcpu-4gb" \
    --auto-upgrade=true \
    --maintenance-window start=04:00,day=sunday

# This takes about 5-10 minutes to provision
```

**Create Container Registry (DOCR):**
```bash
# Create container registry
doctl registry create flask-app-container-registry --region blr1

# Enable registry integration with your cluster
doctl kubernetes cluster registry add flask-app-cluster flask-app-container-registry
```

**Configure kubectl:**
```bash
# Download cluster credentials
doctl kubernetes cluster kubeconfig save flask-app-cluster

# Verify cluster connection
kubectl get nodes
kubectl cluster-info
```

### Step 3: Configure GitHub Repository

**Required GitHub Secrets:**

Navigate to your GitHub repo → Settings → Secrets and variables → Actions, and add:

```bash
# DigitalOcean secrets
DO_TOKEN=your_digitalocean_api_token
DO_REGISTRY=registry.digitalocean.com/flask-app-container-registry
DO_CLUSTER_NAME=flask-app-cluster

# AWS S3 secrets (for data storage)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# MLflow/DagsHub token
CAPSTONE_TEST=your_dagshub_token
```

### Step 4: Deploy Application

**Manual Deployment (First Time):**
```bash
# Clone your repo
git clone https://github.com/your-username/Cricket-T20-Score-Predictor-MLOps.git
cd Cricket-T20-Score-Predictor-MLOps

# Apply Kubernetes manifests
kubectl apply -f deployment.yaml

# Check deployment status
kubectl get pods -l app=flask-app
kubectl get services

# Get external IP (may take a few minutes)
kubectl get svc flask-app-service -w
```

**Automatic Deployment via CI/CD:**
- Simply push code to main branch
- GitHub Actions will automatically test, build, and deploy
- Monitor progress in Actions tab of your GitHub repo

### Step 5: Set Up Monitoring Stack

**Install Helm (if not already installed):**
```bash
# macOS
brew install helm

# Windows
choco install kubernetes-helm

# Linux
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

**Deploy Prometheus & Grafana:**
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --set grafana.service.type=LoadBalancer \
    --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.release=prometheus \
    --set grafana.adminPassword=admin123 \
    --wait

# Get Grafana external IP
kubectl -n monitoring get svc prometheus-grafana -w
```

**Access Monitoring:**
```bash
# Get Grafana URL
echo "Grafana: http://$(kubectl -n monitoring get svc prometheus-grafana -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):80"

# Default login: admin / admin123
# Import dashboard ID: 315 (Kubernetes Cluster Monitoring)
```

### Step 6: Useful Commands

**Check Application Status:**
```bash
# View pods
kubectl get pods -l app=flask-app

# View logs
kubectl logs -l app=flask-app -f

# Describe service
kubectl describe svc flask-app-service

# Scale deployment
kubectl scale deployment flask-app --replicas=3
```

**Monitoring Commands:**
```bash
# Check Prometheus targets
kubectl -n monitoring port-forward svc/prometheus-operated 9090:9090
# Visit: http://localhost:9090/targets

# Access Grafana locally
kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80
# Visit: http://localhost:3000
```

### Step 7: Cost Management & Cleanup

**Monitor Costs:**
- Check DigitalOcean billing dashboard regularly
- DOKS cluster costs ~$24/month for 2 x 2vCPU nodes
- LoadBalancer costs ~$12/month each
- Container registry is free up to 5GB

**Cleanup Resources (Important!):**
```bash
# Delete monitoring stack
helm uninstall prometheus -n monitoring
kubectl delete namespace monitoring

# Delete application
kubectl delete -f deployment.yaml

# Delete cluster (THIS DELETES EVERYTHING!)
doctl kubernetes cluster delete flask-app-cluster

# Delete container registry
doctl registry delete flask-app-container-registry

# Verify cleanup
doctl kubernetes cluster list
doctl registry list
```

### Troubleshooting

**Common Issues:**

1. **Pods stuck in Pending:** Check node resources with `kubectl describe nodes`
2. **ImagePullBackOff:** Verify DOCR integration with `kubectl get secrets`
3. **Service no external IP:** Check LoadBalancer with `kubectl describe svc flask-app-service`
4. **CI/CD fails:** Verify GitHub secrets are correctly set

**Useful Debug Commands:**
```bash
# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Debug pod issues
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous

# Test connectivity
kubectl run debug --image=busybox -it --rm -- /bin/sh
```

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