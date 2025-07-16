# feature engineering
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle


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

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def engineer_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 1
):
    """
    Perform feature engineering on the delivery-level DataFrame and split into train/test.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least ['match_id','batting_team','bowling_team',
        'ball','runs','player_dismissed','city','venue'].
    test_size : float, default=0.2
        Fraction of data to reserve for testing.
    random_state : int, default=1
        Random seed for reproducibility (shuffling & split).

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """
    try:
        data = df.copy()
        logging.info("Starting feature engineering: %d rows", len(data))

        # 1. Impute city from venue when missing
        data['city'] = np.where(
            data['city'].isnull(),
            data['venue'].str.split().apply(lambda parts: parts[0]),
            data['city']
        )
        logging.info("Imputed 'city' for %d null entries", data['city'].isnull().sum())

        # 2. Drop the now­-redundant venue column
        data.drop(columns=['venue'], inplace=True)
        logging.info("Dropped 'venue' column")

        # 3. Filter to only cities with >600 deliveries and save them in a txt file
        counts = data['city'].value_counts()


        eligible = counts[counts > 600].index.tolist()
        with open("flask_app/eligible_cities.txt", "w") as output:
            output.write(str(eligible))
        with open("eligible_cities.txt", "w") as output:
            output.write(str(eligible))
        logging.info("saved eligible cities")
        
        before = len(data)
        data = data[data['city'].isin(eligible)]
        logging.info(
            "Filtered cities to %d eligible (>%d rows): %d -> %d rows",
            len(eligible), 600, before, len(data)
        )

        # 4. Cumulative score so far in the innings
        data['current_score'] = data.groupby('match_id')['runs'].cumsum()
        logging.info("Computed 'current_score'")

        # 5. Split ball into over and ball_no
        splits = data['ball'].astype(str).str.split('.', expand=True)
        data['over'] = splits[0].astype(int)
        data['ball_no'] = splits[1].astype(int)
        logging.info("Extracted 'over' and 'ball_no'")

        # 6. Compute balls bowled & balls left (max 120)
        data['balls_bowled'] = data['over'] * 6 + data['ball_no']
        data['balls_left'] = (120 - data['balls_bowled']).clip(lower=0)
        logging.info("Calculated 'balls_bowled' and 'balls_left'")

        # 7. Encode dismissals & compute wickets remaining
        data['player_dismissed'] = (
            data['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
            .astype(int)
        )
        data['player_dismissed'] = data.groupby('match_id')['player_dismissed'].cumsum()
        data['wickets_left'] = 10 - data['player_dismissed']
        logging.info("Encoded 'player_dismissed' and computed 'wickets_left'")

        # 8. Current run rate
        data['crr'] = (data['current_score'] * 6 / data['balls_bowled']).round(2)
        logging.info("Calculated 'crr'")

        # 9. Rolling sum of last 30 deliveries
        last_five = []
        for _, grp in data.groupby('match_id'):
            last_five.extend(grp['runs'].rolling(window=30).sum().tolist())
        data['last_five'] = last_five
        logging.info("Computed 'last_five' with window=30")

        # 10. Total runs per match
        total = data.groupby('match_id')['runs'].sum().reset_index(name='total_runs')
        data = data.merge(total, on='match_id')
        logging.info("Merged 'total_runs' per match")

        # 11. Select final features & drop any NA
        final = data[[
            'batting_team','bowling_team','city','current_score',
            'balls_left','wickets_left','crr','last_five','total_runs'
        ]].copy()
        before = len(final)
        final.dropna(inplace=True)
        logging.info("Dropped NA: %d -> %d rows", before, len(final))

        # 12. Shuffle
        final = final.sample(frac=1, random_state=random_state).reset_index(drop=True)
        logging.info("Shuffled final data")

        # 13. Split into X/y and train/test
        final_train, final_test = train_test_split(
            final,
            test_size=test_size,
            random_state=random_state
        )
        logging.info(
            "Performed train/test split with test_size=%.2f; Train=%d rows, Test=%d rows",
            test_size, len(final_train), len(final_test)
        )

        return final_train, final_test

    except Exception as e:
        logging.error("Error in feature engineering pipeline: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params['feature_engineering']['test_size']
        # test_size = 0.2

        # train_data = load_data('./data/interim/train_processed.csv')
        # test_data = load_data('./data/interim/test_processed.csv')
        interim_data = load_data('./data/interim/interim_data.csv')

        train_df, test_df = engineer_and_split(interim_data,test_size=test_size)

        save_data(train_df, os.path.join("./data", "processed", "train_final.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_final.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()