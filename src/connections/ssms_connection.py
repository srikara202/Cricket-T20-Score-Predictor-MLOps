import os
import json
import pyodbc
import pandas as pd
import yaml
import logging
from src.logger import logging  # your existing logger

class SSMSOperations:
    def __init__(self, config_path="config.json"):
        """
        Initialize SSMSOperations by loading connection & table config.
        """
        # locate config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(script_dir, config_path)

        with open(cfg_file, "r") as f:
            config = json.load(f)

        sql_cfg = config["sql_server"]
        self.server   = sql_cfg["server"]
        self.database = sql_cfg["database"]
        self.table    = sql_cfg["table"]

        # optional: yaml‐files table config
        yaml_cfg = config.get("yaml_folder", {})
        self.yaml_table   = yaml_cfg.get("table", "t20s_yaml")
        self.id_column    = yaml_cfg.get("id_column", "file_id")
        self.yaml_column  = yaml_cfg.get("yaml_column", "yaml_content")

        conn_str = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"Trusted_Connection=yes;"
        )
        try:
            self.conn = pyodbc.connect(conn_str)
            logging.info("✅ Connected to SQL Server %s/%s", self.server, self.database)
        except Exception as e:
            logging.exception("❌ Could not connect to SQL Server: %s", e)
            raise

    def fetch_table_as_df(self) -> pd.DataFrame:
        """
        Fetches the configured table and returns it as a DataFrame.
        """
        try:
            query = f"SELECT * FROM {self.table}"
            logging.info("🔍 Running query: %s", query)
            df = pd.read_sql(query, self.conn)
            logging.info("✅ Loaded %d rows from %s", len(df), self.table)
            return df
        except Exception as e:
            logging.exception("❌ Failed to fetch table %s: %s", self.table, e)
            return None

    def fetch_yaml_folder_from_ssms(self) -> pd.DataFrame:
        """
        Fetches all YAML blobs from the configured yaml_table, parses, and concatenates them.
        Expects columns (id_column, yaml_column) in yaml_table.
        """
        records = []
        try:
            sql = (
                f"SELECT {self.id_column}, {self.yaml_column} "
                f"FROM {self.yaml_table}"
            )
            logging.info("🔍 Querying YAML table: %s", sql)
            cursor = self.conn.cursor()
            cursor.execute(sql)

            rows = cursor.fetchall()
            logging.info("✅ Retrieved %d YAML rows from %s", len(rows), self.yaml_table)

            for file_id, yaml_text in rows:
                try:
                    data = yaml.safe_load(yaml_text)
                    df = pd.json_normalize(data)
                    df["match_id"] = file_id
                    records.append(df)
                    logging.info("✅ Parsed YAML id=%s (%d records)", file_id, len(df))
                except yaml.YAMLError as ye:
                    logging.error("⚠️ YAML parse error for id=%s: %s", file_id, ye)
                except Exception as ie:
                    logging.error("⚠️ Unexpected error parsing id=%s: %s", file_id, ie)

        except Exception as e:
            logging.exception("❌ Failed to fetch YAML folder from SSMS: %s", e)
            return None

        if records:
            final_df = pd.concat(records, ignore_index=True)
            logging.info("🎉 Combined into final DataFrame with %d total rows", len(final_df))
            return final_df
        else:
            logging.warning("⚠️ No valid YAML records parsed; returning empty DataFrame")
            return pd.DataFrame()

# ---------------------------
# Usage example
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    loader = SSMSOperations("config.json")

    # 1) Fetch your normal table
    df = loader.fetch_table_as_df()

    # 2) Fetch & parse all YAML files stored in your yaml_table
    yaml_df = loader.fetch_yaml_folder_from_ssms()
    print(yaml_df.head())