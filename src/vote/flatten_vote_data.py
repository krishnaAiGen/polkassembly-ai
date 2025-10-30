"""
In this code, we iterate through all the keys of a json, find unique keys and make a .csv file flatetning all the keys and values.
"""
import json
import pandas as pd
import os
import logging
from typing import Dict, Any, List

def get_value_from_path(d: Dict[str, Any], path: str) -> Any:
    """
    Gets a value from a nested dictionary using a dot-separated path.
    """
    keys = path.split('.')
    val = d
    for key in keys:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return None  # Path not found
    return val

def process_vote_data(json_file_path: str, output_dir: str):
    """
    Processes a JSON file of vote data, extracts specific fields, and saves it as a CSV.
    """
    logger.info(f"Reading vote data from {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vote_list: List[Dict[str, Any]]
    if isinstance(data, dict) and 'items' in data and isinstance(data['items'], list):
        vote_list = data['items']
    elif isinstance(data, list):
        vote_list = data
    else:
        logger.error("JSON file is not in the expected format (must be a list of objects or an object with an 'items' list).")
        return

    if not vote_list:
        logger.warning("No vote records found in the JSON file.")
        return

    logger.info(f"Found {len(vote_list)} vote records to process.")

    # User-defined list of columns to extract
    fields = [
        "balance.value", "createdAt", "decision", "isDelegated", "postDetails.content",
        "postDetails.dataSource", "postDetails.id", "postDetails.index", "postDetails.metrics",
        "postDetails.network", "postDetails.proposalType", "postDetails.title", "postDetails.userId",
        "postDetails.publicUser.addresses", "postDetails.publicUser.id", "postDetails.publicUser.username",
        "postDetails.publicUser.rank", "postDetails.publicUser.profileScore", "voter", "voterAddress",
        "lockPeriod"  # Added to be available for calculation and in the output
    ]

    extracted_data = []
    for record in vote_list:
        new_record = {}
        
        # Get balance and lockPeriod for calculation
        balance_value = get_value_from_path(record, "balance.value")
        lock_period = get_value_from_path(record, "lockPeriod")

        # Calculate the adjusted balance based on lockPeriod
        adjusted_balance = None
        try:
            # Coerce to numeric, turning non-numbers into NaN
            balance_numeric = pd.to_numeric(balance_value, errors='coerce')
            lock_numeric = pd.to_numeric(lock_period, errors='coerce')

            if pd.notna(balance_numeric) and pd.notna(lock_numeric):
                multiplier = 0.1 if lock_numeric == 0 else lock_numeric
                adjusted_balance = balance_numeric * multiplier
        except (TypeError, ValueError):
            pass  # Keep adjusted_balance as None if there's an issue

        for field_path in fields:
            new_column_name = field_path.replace('.', '_')

            # Use the calculated value for balance.value, otherwise extract normally
            if field_path == "balance.value":
                value = adjusted_balance
            else:
                value = get_value_from_path(record, field_path)

            # If the value is a dict or list, convert it to a JSON string
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            new_record[new_column_name] = value
        extracted_data.append(new_record)

    df = pd.DataFrame(extracted_data)
    
    # Report any columns that are entirely missing
    desired_columns_underscored = [f.replace('.', '_') for f in fields]
    found_cols = set(df.columns)
    
    missing_cols_report = []
    for i, col_name in enumerate(desired_columns_underscored):
        # Check if the column was created and if it contains any non-null values
        if col_name not in found_cols or df[col_name].isnull().all():
            missing_cols_report.append(fields[i])
            
    if missing_cols_report:
        logger.warning(f"The following requested columns were not found or were empty in all records: {missing_cols_report}")

    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns.")

    # Create 'index_voterAddress' column from 'postDetails_index' and 'voterAddress'
    index_col = 'postDetails_index'
    voter_col = 'voterAddress'
    
    if index_col in df.columns and voter_col in df.columns:
        logger.info(f"Creating 'index_voterAddress' column from '{index_col}' and '{voter_col}'.")
        df['index_voterAddress'] = df[index_col].astype(str).fillna('') + '_' + df[voter_col].astype(str).fillna('')
    else:
        missing_for_concat = []
        if index_col not in df.columns: missing_for_concat.append("'postDetails.index'")
        if voter_col not in df.columns: missing_for_concat.append("'voterAddress'")
        if missing_for_concat:
            logger.warning(f"Could not create 'index_voterAddress' column because {', '.join(missing_for_concat)} were not found.")

    # Save to CSV in the specified output directory
    base_filename = os.path.basename(json_file_path)
    csv_filename = os.path.splitext(base_filename)[0] + '.csv'
    output_path = os.path.join(output_dir, csv_filename)

    logger.info(f"Saving filtered data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Successfully saved filtered CSV file to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # File paths from environment (do not hardcode paths)
    from dotenv import load_dotenv
    load_dotenv()
    input_file = os.getenv("VOTING_DATA_INPUT_FILE", "")
    output_directory = os.getenv("VOTING_DATA_DIR", "")
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
    else:
        process_vote_data(input_file, output_directory)

