#!/usr/bin/env python3
"""
This script orchestrates the entire pipeline for fetching, processing,
and inserting on-chain governance data into the database. It also
verifies the integrity of the data insertion by comparing source
indexes with database indexes.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
import psycopg2
import json
import time

# Load environment variables from .env file
load_dotenv()

# Setup project root for imports
# Assumes the script is in src/texttosql/
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from src.data.onchain_data_hook import OnchainDataHook
    from src.texttosql.flatten_all_data import DataFlattener
    from src.texttosql.create_one_table import CSVCombiner
    from src.texttosql.insert_into_postgres import PostgresInserter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that the script is run from a location where 'src' module is accessible,")
    print(f"or that the project root '{project_root}' is correctly added to PYTHONPATH.")
    sys.exit(1)


# --- Configuration ---
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory structure
ONCHAIN_DATA_DIR = os.getenv("ONCHAIN_DATA_DIR")
if not ONCHAIN_DATA_DIR:
    logger.error("❌ Environment variable ONCHAIN_DATA_DIR is not set. Please set it to the desired data directory.")
    sys.exit(1)

DATA_ROOT = Path(ONCHAIN_DATA_DIR).resolve()
JSON_DIR = DATA_ROOT / "current"
CSV_DIR = DATA_ROOT / "csv_files"
ONE_TABLE_DIR = DATA_ROOT / "one_table"
COMBINED_CSV_FILENAME = "combined_governance_data.csv"
COMBINED_CSV_PATH = ONE_TABLE_DIR / COMBINED_CSV_FILENAME

# Ensure directories exist
DATA_ROOT.mkdir(exist_ok=True, parents=True)
JSON_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
ONE_TABLE_DIR.mkdir(exist_ok=True)


def fetch_onchain_data():
    """Fetches the latest on-chain data."""
    logger.info("--- Step 1: Fetching on-chain data ---")
    try:
        hook = OnchainDataHook(data_root_dir=str(DATA_ROOT))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        success = hook.fetch_new_data(timestamp)
        if success:
            logger.info("✅ On-chain data fetched successfully.")
        else:
            logger.error("❌ Failed to fetch on-chain data.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during data fetching: {e}", exc_info=True)
        sys.exit(1)


def flatten_json_to_csv():
    """Flattens the fetched JSON files into CSVs."""
    logger.info("\n--- Step 2: Flattening JSON data to CSV ---")
    try:
        flattener = DataFlattener(data_dir=str(JSON_DIR), output_data_dir=str(DATA_ROOT))
        results = flattener.process_all_files()
        if results['successful'] > 0:
            logger.info(f"✅ Successfully flattened {results['successful']} JSON files.")
        else:
            logger.error("❌ No JSON files were successfully flattened.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during JSON flattening: {e}", exc_info=True)
        sys.exit(1)


def combine_csvs_to_one_table():
    """Combines all individual CSVs into a single master CSV."""
    logger.info("\n--- Step 3: Combining CSV files into a single table ---")
    try:
        combiner = CSVCombiner(csv_directory=str(CSV_DIR), output_directory=str(ONE_TABLE_DIR))
        results = combiner.run_combination()
        if results.get('combined_csv') and results['combined_csv'].exists():
            logger.info(f"✅ Successfully combined CSVs into '{results['combined_csv']}'.")
        else:
            logger.error("❌ Failed to combine CSV files.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during CSV combination: {e}", exc_info=True)
        sys.exit(1)


def get_combined_csv_data(csv_path: Path) -> pd.DataFrame:
    """Reads the combined CSV and returns it as a DataFrame."""
    logger.info("\n--- Step 4: Reading combined CSV data ---")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"✅ Loaded {len(df)} rows from combined CSV.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while reading the combined CSV: {e}", exc_info=True)
        sys.exit(1)


def insert_data_to_database(csv_path: Path):
    """Inserts the data from the combined CSV into the PostgreSQL database."""
    logger.info("\n--- Step 5: Inserting data into PostgreSQL database ---")
    try:
        inserter = PostgresInserter()
        # We drop the existing table to ensure a clean slate for comparison.
        success = inserter.run_full_import(csv_path, drop_existing=True)
        if success:
            logger.info("✅ Data inserted into database successfully.")
        else:
            logger.error("❌ Failed to insert data into the database.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during database insertion: {e}", exc_info=True)
        sys.exit(1)


def get_database_row_indexes() -> set:
    """Queries the database to get all unique row_indexes that were inserted."""
    logger.info("\n--- Step 6: Querying database for existing row_indexes ---")
    db_row_indexes = set()
    try:
        inserter = PostgresInserter()
        with inserter.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists first
                cur.execute("SELECT to_regclass(%s)", (inserter.table_name,))
                if cur.fetchone()[0] is None:
                    logger.warning(f"Table '{inserter.table_name}' does not exist. Assuming all rows are new.")
                    return db_row_indexes

                # Check if column exists
                cur.execute("""
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name=%s AND column_name='row_index'
                """, (inserter.table_name,))
                if cur.fetchone() is None:
                    logger.warning(f"'row_index' column not found in table '{inserter.table_name}'. Assuming all rows are new.")
                    return db_row_indexes

                logger.info(f"Querying table '{inserter.table_name}' for 'row_index' column...")
                cur.execute(f'SELECT "row_index" FROM {inserter.table_name};')
                # Fetch all rows and handle potential None values
                rows = cur.fetchall()
                for row in rows:
                    if row[0] is not None:
                        # Convert to string to match the type from pandas
                        db_row_indexes.add(str(row[0]))

        logger.info(f"✅ Found {len(db_row_indexes)} unique row_indexes in the database.")
        return db_row_indexes
    except psycopg2.Error as e:
        logger.warning(f"Could not connect to or query database for row_indexes: {e}")
        logger.warning("Proceeding as if database is empty. All rows will be considered new.")
        return db_row_indexes
    except Exception as e:
        logger.error(f"An error occurred while querying database row_indexes: {e}", exc_info=True)
        sys.exit(1)


def save_new_rows_to_json(source_df: pd.DataFrame, db_row_indexes: set):
    """Filters the source dataframe for new rows and saves their row_index and createdat to a JSON file."""
    logger.info("\n--- Step 7: Filtering for new rows to insert and saving to JSON ---")

    if 'row_index' not in source_df.columns:
        logger.error("❌ 'row_index' column not found in the source DataFrame.")
        sys.exit(1)

    if 'createdat' not in source_df.columns:
        logger.error("❌ 'createdat' column not found in the source DataFrame. Cannot create JSON file.")
        sys.exit(1)

    new_rows_df = source_df[~source_df['row_index'].isin(db_row_indexes)].copy()

    if new_rows_df.empty:
        logger.info("✅ No new rows to insert. The database is up-to-date.")
    else:
        output_filename = ONE_TABLE_DIR / "new_rows_to_insert.json"
        logger.warning(f"Found {len(new_rows_df)} new rows to be inserted.")

        # Select the required columns and create a list of dictionaries
        output_data = new_rows_df[['row_index', 'createdat']].to_dict(orient='records')

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"✅ Successfully saved {len(new_rows_df)} new row indexes and createdat to '{output_filename}'.")
        except Exception as e:
            logger.error(f"Error saving new rows to JSON: {e}", exc_info=True)


def main():
    """Main function to run the entire data pipeline."""
    logger.info("🚀 Starting the on-chain data processing pipeline...")
    
    # Step 1: Fetch data from on-chain sources
    fetch_onchain_data()
    
    # Step 2: Flatten the raw JSON data into multiple CSV files
    flatten_json_to_csv()
    
    # Step 3: Combine all CSV files into one large table
    combine_csvs_to_one_table()
    
    # Step 4: Get the list of original indexes from the combined CSV
    source_dataframe = get_combined_csv_data(COMBINED_CSV_PATH)
    
    # Step 5: Insert the combined data into the database
    # insert_data_to_database(COMBINED_CSV_PATH)
    
    # Step 6: Get the list of indexes that were successfully inserted
    database_row_indexes = get_database_row_indexes()
    
    # Step 7: Compare the lists and report any discrepancies
    save_new_rows_to_json(source_dataframe, database_row_indexes)
    
    logger.info("\n🏁 Pipeline finished.")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start} seconds")
