# app.py
import os
import time
import logging
import pandas as pd
from flask import Flask, render_template, request
import mlflow
import dagshub
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

# ─────────────────────────────────────────────────────────────────────────────
# Local DagsHub + MLflow setup
# ─────────────────────────────────────────────────────────────────────────────
# mlflow.set_tracking_uri("https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow")
# dagshub.init(repo_owner="srikara202",repo_name="Cricket-T20-Score-Predictor-MLOps",mlflow=True)
# -------------------------------------------------------------------------------------
# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "srikara202"
repo_name = "Cricket-T20-Score-Predictor-MLOps"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


def get_latest_model_version(model_name: str) -> str:
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name)
    return versions[0].version if versions else None

MODEL_NAME    = "my_model"
MODEL_VERSION = get_latest_model_version(MODEL_NAME)
MODEL_URI     = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Loading model from %s", MODEL_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────
registry = CollectorRegistry()
REQUEST_COUNT   = Counter("app_request_count",   "Total HTTP requests", ["method","endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)

# ─────────────────────────────────────────────────────────────────────────────
# Flask setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

TEAMS = [
    "Australia","India","Bangladesh","New Zealand","South Africa",
    "England","West Indies","Afghanistan","Pakistan","Sri Lanka"
]
CITIES = [
    "Colombo","Mirpur","Johannesburg","Dubai","Auckland","Cape Town",
    "London","Pallekele","Barbados","Sydney","Melbourne","Durban",
    "St Lucia","Wellington","Lauderhill","Hamilton","Centurion",
    "Manchester","Abu Dhabi","Mumbai","Nottingham","Southampton",
    "Mount Maunganui","Chittagong","Kolkata","Lahore","Delhi",
    "Nagpur","Chandigarh","Adelaide","Bangalore","St Kitts",
    "Cardiff","Christchurch","Trinidad"
]

@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()

    # On GET, no prior inputs or result
    context = dict(
        teams=sorted(TEAMS),
        cities=sorted(CITIES),
        result=None,
        batting_team=None,
        bowling_team=None,
        city=None,
        current_score=None,
        overs=None,
        wickets=None,
        last_five=None,
    )
    response = render_template("index.html", **context)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    # Pull everything out of the form    
    batting_team  = request.form.get("batting_team")
    bowling_team  = request.form.get("bowling_team")
    city          = request.form.get("city")
    current_score = request.form.get("current_score", type=int)
    overs_done    = request.form.get("overs", type=float)
    wickets_out   = request.form.get("wickets", type=int)
    last_five     = request.form.get("last_five", type=int)

    try:
        # Derived features
        balls_bowled = int(overs_done * 6)
        balls_left   = max(120 - balls_bowled, 0)
        wickets_left = max(10 - wickets_out, 0)
        crr = round(current_score / overs_done, 2) if overs_done > 0 else 0.0

        input_df = pd.DataFrame([{
            "batting_team":  batting_team,
            "bowling_team":  bowling_team,
            "city":          city,
            "current_score": current_score,
            "balls_left":    balls_left,
            "wickets_left":  wickets_left,
            "crr":           crr,
            "last_five":     last_five
        }])

        pred = model.predict(input_df)[0]
        prediction = int(round(pred))
        result = prediction

    except Exception as e:
        logging.exception("Prediction error: %s", e)
        result = f"Error: {e}"

    # Render back **with** all inputs and result so the form stays filled
    context = dict(
        teams=sorted(TEAMS),
        cities=sorted(CITIES),
        result=result,
        batting_team=batting_team,
        bowling_team=bowling_team,
        city=city,
        current_score=current_score,
        overs=overs_done,
        wickets=wickets_out,
        last_five=last_five,
    )
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
    return render_template("index.html", **context)

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
