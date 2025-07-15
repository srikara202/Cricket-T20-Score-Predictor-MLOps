import pandas as pd
import pickle
import yaml
from src.logger import logging
from time import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


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

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def build_and_train_model(X_train, y_train):
    """
    Build a preprocessing + XGBoost regression pipeline and train it.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with columns ['batting_team','bowling_team','city', ...].
    y_train : pd.Series or array-like
        Training target (total runs).

    Returns
    -------
    Pipeline
        The fitted sklearn Pipeline containing preprocessing and the trained XGBRegressor.
    """
    try:
        logging.info("Starting model build & training")

        # 1. Define the column transformer
        transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(sparse_output=False, drop='first'),
                 ['batting_team', 'bowling_team', 'city'])
            ],
            remainder='passthrough'
        )
        logging.info("Initialized ColumnTransformer for one-hot encoding")

        # 1.5. Load Params
        params = load_params('params.yaml')
        hps = params['model_building']
        n_estimators = hps['n_estimators']
        learning_rate = hps['learning_rate']
        max_depth = hps['max_depth']

        # 2. Define the pipeline steps
        pipeline = Pipeline(steps=[
            ('preprocess', transformer),
            ('scale', StandardScaler()),
            ('regressor', XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=1
            ))
        ])
        logging.info("Pipeline created with StandardScaler and XGBRegressor")

        # 3. Train the model
        start = time()
        pipeline.fit(X_train, y_train)
        elapsed = time() - start
        logging.info("Model training complete (%.2f seconds)", elapsed)

        return pipeline

    except Exception as e:
        logging.error("Error during model build or training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        params = load_params('params.yaml')
        hps = params['model_building']
        n_estimators = hps['n_estimators']
        learning_rate = hps['learning_rate']
        max_depth = hps['max_depth']
        train_data = load_data('./data/processed/train_final.csv')
        X_train = train_data.drop(columns=['total_runs'])
        y_train = train_data['total_runs']

        clf = build_and_train_model(X_train, y_train)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()