import os
import boto3
import pandas as pd
import yaml
import logging
from src.logger import logging  # your custom logger
from io import StringIO

class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        """
        Initialize the s3_operations class with AWS credentials and S3 bucket details.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("✅ Initialized S3 operations for bucket '%s'", bucket_name)

    def fetch_file_from_s3(self, file_key: str) -> pd.DataFrame:
        """
        Fetches a single CSV file from S3 and returns it as a DataFrame.
        """
        try:
            logging.info("🔍 Fetching CSV '%s' from S3 bucket '%s'...", file_key, self.bucket_name)
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info("✅ Loaded CSV '%s' (%d rows)", file_key, len(df))
            return df
        except Exception as e:
            logging.exception("❌ Failed to fetch CSV '%s': %s", file_key, e)
            return None

    def fetch_yaml_folder_from_s3(self, folder_prefix: str) -> pd.DataFrame:
        """
        Fetches all YAML files under the given S3 “folder” prefix, parses them,
        and concatenates into one DataFrame with a 'match_id' per file.

        :param folder_prefix: e.g. 't20s/' or 't20s' (we'll normalize it)
        :return: DataFrame or None on complete failure
        """
        # ensure trailing slash
        prefix = folder_prefix.rstrip('/') + '/'
        try:
            logging.info("🔍 Listing objects under prefix '%s'...", prefix)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            yaml_keys = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.lower().endswith(('.yaml', '.yml')):
                        yaml_keys.append(key)

            logging.info("✅ Found %d YAML files under '%s'", len(yaml_keys), prefix)

            final_df = pd.DataFrame()
            for idx, key in enumerate(yaml_keys, start=1):
                try:
                    logging.info("🔄 Fetching YAML '%s' (match_id=%d)...", key, idx)
                    obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                    text = obj['Body'].read().decode('utf-8')
                    data = yaml.safe_load(text)

                    df = pd.json_normalize(data)
                    df['match_id'] = idx
                    final_df = pd.concat([final_df, df], ignore_index=True)

                    logging.info("✅ Parsed '%s' (%d records)", key, len(df))

                except yaml.YAMLError as ye:
                    logging.error("⚠️ YAML parse error in '%s': %s", key, ye)
                except Exception as e:
                    logging.error("⚠️ Error processing '%s': %s", key, e)

            logging.info("🎉 Completed loading %d total records from %d files",
                         len(final_df), len(yaml_keys))
            return final_df

        except Exception as e:
            logging.exception("❌ Failed to fetch YAML folder '%s': %s", prefix, e)
            return None
