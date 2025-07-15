import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
import logging
import mlflow
import dagshub
import os
import shutil
from src.logger import logging


# # Below code block is for production use
# # -------------------------------------------------------------------------------------
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

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/srikara202/Cricket-T20-Score-Predictor-MLOps.mlflow')
# dagshub.init(repo_owner='srikara202', repo_name='Cricket-T20-Score-Predictor-MLOps', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        r2 = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = root_mean_squared_error(y_test,y_pred)

        metrics_dict = {
            'r2_score':r2,
            'mean_absolute_error':mae,
            'root_mean_squared_error':rmse
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            pipe = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_final.csv')
            
            X_test = test_data.drop(columns=['total_runs'])
            y_test = test_data['total_runs']

            metrics = evaluate_model(pipe, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(pipe, 'get_params'):
                params = pipe.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            model_dir = "model_dir"
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            # Now save + upload
            mlflow.sklearn.save_model(sk_model=pipe, path=model_dir)
            mlflow.log_artifacts(model_dir, artifact_path="model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
