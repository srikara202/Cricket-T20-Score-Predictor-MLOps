# flask_app/app.py
#!/usr/bin/env python3
import os
import ast
import time
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify
import mlflow
import dagshub
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from mlflow import MlflowClient

# ─────────────────────────────────────────────────────────────────────────────
# MLflow + DagsHub setup
# ─────────────────────────────────────────────────────────────────────────────
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "srikara202"
repo_name = "Cricket-T20-Score-Predictor-MLOps"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# -------------------------------------------------------------------------------------
# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow')
# dagshub.init(repo_owner='srikara202', repo_name='Cricket-T20-Score-Predictor-MLOps', mlflow=True)
# -------------------------------------------------------------------------------------


def get_model_version_by_stage(model_name: str, stage: str) -> str:
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name = '{model_name}'")
    stage_versions = [mv for mv in all_versions if mv.tags.get("stage") == stage]
    if not stage_versions:
        raise ValueError(f"No model version found tagged stage='{stage}'")
    chosen = max(stage_versions, key=lambda mv: mv.last_updated_timestamp)
    return chosen.version

MODEL_NAME    = "my_model"
MODEL_VERSION = get_model_version_by_stage(MODEL_NAME, stage="production")
MODEL_URI     = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Loading production model version %s from %s", MODEL_VERSION, MODEL_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────────────────────────────────────
registry = CollectorRegistry()
REQUEST_COUNT   = Counter("app_request_count",   "Total HTTP requests", ["method","endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

TEAMS = [
    "Australia","India","Bangladesh","New Zealand","South Africa",
    "England","West Indies","Pakistan","Sri Lanka"
]

with open('eligible_cities.txt', 'r') as file:
    my_list = [line.strip() for line in file]
CITIES = ast.literal_eval(my_list[0])

@app.route("/", methods=["GET"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()
    context = dict(
        teams=sorted(TEAMS),
        cities=sorted(CITIES),
        result=None,
        error_message=None,
        batting_team=None,
        bowling_team=None,
        city=None,
        current_score=None,
        overs=None,
        wickets=None,
        last_five=None,
    )
    resp = render_template("index.html", **context)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return resp

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    # JSON-based API (unchanged)...
    if request.is_json:
        # ... existing JSON flow ...
        payload = request.get_json()
        batting  = payload.get("batting_team", TEAMS[0])
        bowling  = payload.get("bowling_team", TEAMS[1] if TEAMS[1] != batting else TEAMS[0])
        city     = payload.get("city", CITIES[0])
        try:
            current_score = payload["current_score"]
            balls_left    = payload["balls_left"]
            wickets_left  = payload["wickets_left"]
            crr           = payload["crr"]
            last_five     = payload["last_five"]
        except KeyError as e:
            return jsonify({"error": f"Missing field {e}"}), 400
        try:
            input_df = pd.DataFrame([{
                "batting_team":  batting,
                "bowling_team":  bowling,
                "city":          city,
                "current_score": current_score,
                "balls_left":    balls_left,
                "wickets_left":  wickets_left,
                "crr":           crr,
                "last_five":     last_five
            }])
            pred = model.predict(input_df)[0]
            score = int(round(pred))
            return jsonify({"predicted_score": score})
        except Exception as e:
            logging.exception("Prediction error (JSON): %s", e)
            return jsonify({"error": str(e)}), 500

    # ─────────────────────────────────────────────────────────────────────────
    # form-based flow with validation
    batting_team  = request.form.get("batting_team")
    bowling_team  = request.form.get("bowling_team")
    city          = request.form.get("city")
    current_score = request.form.get("current_score", type=int)
    overs_done    = request.form.get("overs", type=float)
    wickets_out   = request.form.get("wickets", type=int)
    last_five     = request.form.get("last_five", type=int)

    error_message = None
    # validate each field
    if current_score is None:
        error_message = "Current score is required and must be a number."
    elif current_score < 0:
        error_message = "Current score must be 0 or a positive integer."
    elif overs_done is None:
        error_message = "Overs done is required and must be a number."
    elif overs_done < 5.0 or overs_done > 19.0:
        error_message = "Overs done must be between 5 and 19 (inclusive)."
    elif wickets_out is None:
        error_message = "Wickets out is required and must be a whole number."
    elif wickets_out < 0 or wickets_out > 9:
        error_message = "Wickets out must be a whole number between 0 and 9."
    elif last_five is None:
        error_message = "Runs in last 5 overs is required and must be a number."
    elif last_five < 0 or last_five > current_score:
        error_message = "Runs in last 5 overs must be non-negative and no more than the current score."

    result = None
    if not error_message:
        try:
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
            result = int(round(pred))
        except Exception as e:
            logging.exception("Prediction error: %s", e)
            error_message = f"Prediction error: {e}"

    context = dict(
        teams=sorted(TEAMS),
        cities=sorted(CITIES),
        result=result,
        error_message=error_message,
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
