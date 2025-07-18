# Cricket T20 Score Predictor (MLOps Project)

## Project Overview

This project aims to predict the innings score of a men's T20 international cricket match using ball-by-ball historical data taken from [the cricsheet website](https://cricsheet.org/). The core objective is to demonstrate advanced Data Science and MLOps techniques, including data ingestion, preprocessing, feature engineering, model training, evaluation, deployment, monitoring, and alerting.

The final application is containerized using docker and deployed on a Kubernetes cluster hosted on DigitalOcean and features integrated monitoring and alerting with Prometheus and Grafana.

**Live Application URL:** [http://144.126.254.108:5000/](http://144.126.254.108:5000/)


---

## Technical Skills Demonstrated

* **Data Science**: Data Cleaning, Exploratory Data Analysis, Feature Engineering, Model Building (XGBoost Regression), Model Evaluation (RMSE, MAE, R² Score)
* **Programming Languages**: Python
* **Frameworks and Tools**: Pandas, NumPy, Scikit-Learn, XGBoost, Flask, Cookiecutter
* **Version Control and Collaboration**: Git, GitHub, DagsHub
* **Experiment Tracking**: MLflow integrated with DagsHub
* **Containerization & Orchestration**: Docker, Kubernetes (DigitalOcean)
* **Cloud Services**: DigitalOcean Kubernetes (DOKS), DigitalOcean Spaces (S3 equivalent)
* **Monitoring & Alerting**: Prometheus, Grafana
* **CI/CD Pipelines**: GitHub Actions

---

## Project Workflow

### Step-by-Step Setup and Execution:

### 1. Project Structure and Initialization

* Created and cloned GitHub repository.
* Set up Conda environment (`atlas`) and installed project dependencies using Cookiecutter Data Science Template.

### 2. Experiment Tracking & Data Versioning

* Integrated MLflow via DagsHub for experiment tracking.
* Initialized Data Version Control (DVC) to track data pipelines and model artifacts.
* Configured DigitalOcean Spaces as remote storage for DVC.

### 3. Data Processing & Model Development

* Implemented modular Object-Oriented Programming (OOP) code structure:

  * **Data Ingestion**: Loading ball-by-ball data
  * **Data Preprocessing**: Handling missing values, feature selection
  * **Feature Engineering**: Creating meaningful predictors for innings score prediction
  * **Model Building**: Developed regression models using XGBoost
  * **Model Evaluation**: Evaluated performance using RMSE, MAE, and R² metrics
  * **Model Registration**: Managed models via MLflow

### 4. Application Development & Containerization

* Built RESTful Flask API to serve predictions.
* Generated requirements with `pipreqs` and containerized the application using Docker.

### 5. Deployment on DigitalOcean Kubernetes

* Created Kubernetes cluster on DigitalOcean.
* Configured Kubernetes resources including Deployment, Service (LoadBalancer), and Secrets.
* Successfully deployed and exposed the Flask application publicly.

### 6. Monitoring with Prometheus & Grafana

* Configured Prometheus to scrape metrics from the Flask application.
* Deployed Prometheus on DigitalOcean VM instance.
* Set up Grafana dashboard to visualize real-time metrics and alerts from Prometheus data sources.

**Monitoring URLs:**

* Prometheus: [http://64.225.85.3:9090/](http://64.225.85.3:9090/)
* Grafana: [http://152.42.157.72/](http://152.42.157.72/)

### 7. CI/CD Pipeline

* Implemented automated CI/CD pipelines using GitHub Actions.
* Automates the build, test, and deployment processes to DigitalOcean Kubernetes clusters.

---

## AWS vs DigitalOcean Deployment Comparison

This project initially considered AWS for deployment but transitioned to DigitalOcean for ease of setup and cost efficiency. Key replacements:

| AWS Service       | DigitalOcean Equivalent         |
| ----------------- | ------------------------------- |
| Amazon EKS        | DigitalOcean Kubernetes (DOKS)  |
| Amazon S3         | DigitalOcean Spaces             |
| AWS IAM & Secrets | Kubernetes Secrets              |
| AWS CloudWatch    | Prometheus & Grafana            |
| Amazon ECR        | Docker Hub / Container Registry |

---

## Setup Instructions

### Clone the repository:

```bash
git clone https://github.com/yourusername/cricket-t20-score-predictor.git
```

### Setup Environment:

```bash
conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt
```

### DVC Configuration:

* Set up DigitalOcean Spaces and configure DVC:

```bash
dvc remote add -d myremote s3://<digitalocean-space>
dvc push
```

### Run Application Locally:

```bash
cd flask_app
python app.py
```

### Docker Setup:

```bash
docker build -t cricket-app:latest .
docker run -p 8888:5000 -e DVC_REMOTE=myremote cricket-app:latest
```

---

## Kubernetes & Monitoring

Detailed Kubernetes configurations and YAML files are provided in the repository. Instructions for setting up Prometheus and Grafana are included in the monitoring folder.

---

## Cleanup

* Delete Kubernetes deployments and services
* Remove DigitalOcean Kubernetes clusters and Spaces (if necessary)

---

## Contact

For questions or suggestions, feel free to contact me:

* LinkedIn: [Srikara Shankara](https://www.linkedin.com/in/srikarashankara/)
* Email: [srikarashankara@outlook.com](srikarashankara@outlook.com)

---

This project effectively showcases end-to-end skills in Data Science and MLOps, emphasizing real-world applicability, maintainability, scalability, and robust deployment practices.
