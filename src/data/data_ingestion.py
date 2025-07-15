# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
import ast
from src.connections import s3_connection


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

def load_yaml_directory(data_url: str) -> pd.DataFrame:
    """
    Load all YAML files from a directory into a single pandas DataFrame.
    
    Each file is parsed, normalized, given a 'match_id' based on load order,
    and concatenated into the returned DataFrame.
    
    Parameters
    ----------
    data_url : str
        Path to the directory containing YAML files.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concatenated contents of all YAML files.
    
    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    Exception
        For any other unexpected errors.
    """
    try:
        # Gather YAML filenames
        filenames = [
            os.path.join(data_url, fname)
            for fname in os.listdir(data_url)
            if fname.lower().endswith(('.yaml', '.yml'))
        ]
        logging.info("Found %d YAML files in %s", len(filenames), data_url)
        
        final_df = pd.DataFrame()
        # Load each file
        for idx, filepath in enumerate(tqdm(filenames, desc="Loading YAML files"), start=1):
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                df = pd.json_normalize(data)
                df['match_id'] = idx
                final_df = pd.concat([final_df, df], ignore_index=True)
                logging.info("Loaded file %s (match_id=%d)", filepath, idx)
            except yaml.YAMLError as ye:
                logging.error("YAML parse error in %s: %s", filepath, ye)
                # skip this file but continue processing others
            except Exception as ie:
                logging.error("Unexpected error processing %s: %s", filepath, ie)
        
        logging.info("Successfully loaded %d total records", len(final_df))
        return final_df

    except FileNotFoundError as fe:
        logging.error("Directory not found: %s", fe)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading YAML directory: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

import logging
from tqdm import tqdm
import pandas as pd

def extract_delivery_df(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a DataFrame of match‐level YAML data into a delivery‐level DataFrame.

    Steps:
      1. Select & copy only the needed columns.
      2. Filter to men's T20 matches.
      3. Loop through each match and each ball:
         • build a dict record per delivery
         • handle missing 'wicket' info gracefully
      4. Concatenate into a single delivery_df.

    Parameters
    ----------
    final_df : pd.DataFrame
        DataFrame containing match‐level data with at least these columns:
        ['innings','info.dates','info.gender','info.match_type',
         'info.outcome.winner','info.overs','info.player_of_match',
         'info.teams','info.toss.decision','info.toss.winner',
         'info.umpires','info.venue','match_id','info.city']

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row is one ball delivery, with columns:
        ['match_id','teams','batting_team','ball','batsman','bowler',
         'runs','player_dismissed','city','venue'].
    """
    try:
        # 1. Select & copy
        cols = [
            'innings','info.dates','info.gender','info.match_type',
            'info.outcome.winner','info.overs','info.player_of_match',
            'info.teams','info.toss.decision','info.toss.winner',
            'info.umpires','info.venue','match_id','info.city'
        ]
        df = final_df[cols].copy()
        logging.info("Selected %d columns", len(cols))

        # 2. Filter to male, T20
        df = df[df['info.gender'] == 'male']
        df.drop(columns=['info.gender'], inplace=True)
        logging.info("Filtered to male matches: %d rows remain", len(df))

        df = df[df['info.overs'] == 20]
        df.drop(columns=['info.overs','info.match_type'], inplace=True)
        logging.info("Filtered to T20 matches: %d rows remain", len(df))

    except KeyError as ke:
        logging.error("Missing expected column: %s", ke)
        raise
    except Exception as e:
        logging.error("Error during initial filtering: %s", e)
        raise

    # 3. Extract per‐ball records
    records = []
    match_counter = 1
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting deliveries"):
        try:
            deliveries = row['innings'][0]['1st innings']['deliveries']
            for ball in deliveries:
                for ball_num, details in ball.items():
                    records.append({
                        'match_id':     match_counter,
                        'teams':        row['info.teams'],
                        'batting_team': row['innings'][0]['1st innings']['team'],
                        'ball':         ball_num,
                        'batsman':      details['batsman'],
                        'bowler':       details['bowler'],
                        'runs':         details['runs']['total'],
                        'player_dismissed': details.get('wicket', {}).get('player_out', '0'),
                        'city':         row['info.city'],
                        'venue':        row['info.venue']
                    })
            logging.info("Processed match_id=%d: %d deliveries", match_counter, len(deliveries))
        except Exception as e:
            logging.error("Error processing match_id=%d: %s", match_counter, e)
        finally:
            match_counter += 1

    # 4. Build final DataFrame
    delivery_df = pd.DataFrame(records)
    logging.info("Extraction complete: %d total deliveries", len(delivery_df))
    return delivery_df


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def save_data(
    data: pd.DataFrame,
    data_path: str,
    filename: str = "data.csv"
) -> None:
    """
    Save a single DataFrame to CSV under <data_path>/raw.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to save.
    data_path : str
        Base directory in which to create a 'raw' folder.
    filename : str, optional
        Name of the CSV file (default is 'data.csv').
    """
    try:
        raw_data_dir = os.path.join(data_path, "raw")
        os.makedirs(raw_data_dir, exist_ok=True)
        full_path = os.path.join(raw_data_dir, filename)
        
        data.to_csv(full_path, index=False)
        logging.debug("Saved DataFrame with %d rows × %d cols to %s",
                      data.shape[0], data.shape[1], full_path)
    except Exception as e:
        logging.error("❌ Failed to save data to %s: %s", raw_data_dir, e)
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2
        
        df = load_yaml_directory(data_url='notebooks/t20s/')
        # s3 = s3_connection.s3_operations("t20s", "aws_access_key", "aws_secret_access_key")
        # df = s3.fetch_yaml_folder_from_s3("t20s")



        final_df = extract_delivery_df(df)
        # train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(final_df, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()