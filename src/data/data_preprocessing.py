# data preprocessing

import logging
import pandas as pd
import os
from src.logger import logging
import ast
import logging
import pandas as pd

def preprocess_dataframe(delivery_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the delivery-level DataFrame:
      1. Derive 'bowling_team' by finding the non-batting side.
      2. Drop the original 'teams' column.
      3. Filter to only the 10 international sides.
      4. Select & return the final columns.

    Parameters
    ----------
    delivery_df : pd.DataFrame
        Must contain columns ['teams', 'batting_team', 'ball', 'runs',
        'player_dismissed', 'city', 'venue'].

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with columns
        ['match_id','batting_team','bowling_team','ball','runs',
         'player_dismissed','city','venue'].
    """
    try:
        df = delivery_df.copy()
        logging.info("Starting preprocessing: %d rows", len(df))

        # 1. Derive bowling_team
        def _find_bowler(row):
            for team in ast.literal_eval(row['teams']):
                if team != row['batting_team']:
                    return team
            return None

        df['bowling_team'] = df.apply(_find_bowler, axis=1)
        logging.info("Derived 'bowling_team' column")

        # 2. Drop the original teams list
        df.drop(columns=['teams'], inplace=True)
        logging.info("Dropped 'teams' column")

        # 3. Filter to the 10 ICC full member teams
        valid_teams = {
            'Australia','India','Bangladesh','New Zealand','South Africa',
            'England','West Indies','Pakistan','Sri Lanka'
        }

        before = len(df)
        df = df[df['batting_team'].isin(valid_teams)]
        logging.info(
            "Filtered batting_team to valid list: %d → %d rows",
            before, len(df)
        )

        before = len(df)
        df = df[df['bowling_team'].isin(valid_teams)]
        logging.info(
            "Filtered bowling_team to valid list: %d → %d rows",
            before, len(df)
        )

        # 4. Select final columns
        output = df[[
            'match_id','batting_team','bowling_team',
            'ball','runs','player_dismissed','city','venue'
        ]].copy()
        logging.info(
            "Final selection done: %d rows × %d columns",
            output.shape[0], output.shape[1]
        )

        return output

    except Exception as e:
        logging.error("Error during preprocessing delivery_df: %s", e)
        raise



def main():
    try:
        # # Fetch the data from data/raw
        # train_data = pd.read_csv('./data/raw/train.csv')
        # test_data = pd.read_csv('./data/raw/test.csv')
        data_df = pd.read_csv('./data/raw/data.csv')
        logging.info('data loaded properly')

        # Transform the data
        # train_processed_data = preprocess_dataframe(train_data)
        # test_processed_data = preprocess_dataframe(test_data)
        processed_data = preprocess_dataframe(data_df)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        # train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        # test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        processed_data.to_csv(os.path.join(data_path, "interim_data.csv"), index=False)

        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()