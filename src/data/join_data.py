#!/usr/bin/env python3
"""
Utility module to concatenate data from static and dynamic sources.
Handles both JSON and TXT files from specified directories.
"""

import os
import json
from typing import Dict, List, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data source paths - these will be set by the user
static_data_source: str = ""
dynamic_data_source: str = ""

def read_json_file(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """Read a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {}

def read_text_file(file_path: str) -> str:
    """Read a text file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def concatenate_directory_data(directory: str, combine_by_subfolder: bool = False) -> Dict[str, Any]:
    """
    Concatenate all JSON and text files in a directory and its subdirectories.
    If combine_by_subfolder is True, combines files within each subfolder separately.
    Returns a dictionary with 'json_data' and 'text_data' keys.
    """
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return {"json_data": [], "text_data": ""}

    json_data = []
    text_data = []

    if combine_by_subfolder:
        # Process each subdirectory separately
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        logger.info(f"Found {len(subdirs)} subdirectories: {subdirs}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(directory, subdir)
            logger.info(f"Processing subdirectory: {subdir}")
            
            subdir_json_data = []
            subdir_text_data = []
            
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_lower = file.lower()

                    try:
                        if file_lower.endswith('.json'):
                            data = read_json_file(file_path)
                            if isinstance(data, list):
                                subdir_json_data.extend(data)
                            else:
                                subdir_json_data.append(data)
                            logger.info(f"Successfully processed JSON file: {file_path}")

                        elif file_lower.endswith('.txt'):
                            content = read_text_file(file_path)
                            if content:
                                # Add file separator for better organization
                                file_header = f"\n\n--- FILE: {os.path.basename(file_path)} ---\n"
                                subdir_text_data.append(file_header + content)
                            logger.info(f"Successfully processed text file: {file_path}")

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
            
            # Combine subdirectory data
            if subdir_text_data:
                combined_text = f"\n\n=== {subdir.upper()} DOCUMENTATION ===\n" + "\n".join(subdir_text_data)
                text_data.append(combined_text)
                logger.info(f"Combined {len(subdir_text_data)} text files from {subdir}")
            
            if subdir_json_data:
                json_data.extend(subdir_json_data)
                logger.info(f"Added {len(subdir_json_data)} JSON entries from {subdir}")
    else:
        # Original behavior - process all files together
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                try:
                    if file_lower.endswith('.json'):
                        data = read_json_file(file_path)
                        if isinstance(data, list):
                            json_data.extend(data)
                        else:
                            json_data.append(data)
                        logger.info(f"Successfully processed JSON file: {file_path}")

                    elif file_lower.endswith('.txt'):
                        content = read_text_file(file_path)
                        if content:
                            text_data.append(content)
                        logger.info(f"Successfully processed text file: {file_path}")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

    return {
        "json_data": json_data,
        "text_data": "\n\n".join(text_data) if text_data else ""
    }

def get_static_data() -> Dict[str, Any]:
    """Get concatenated data from static data source with subfolder combination."""
    if not static_data_source:
        logger.error("Static data source path not set")
        return {"json_data": [], "text_data": ""}
    
    logger.info(f"Processing static data from: {static_data_source}")
    return concatenate_directory_data(static_data_source, combine_by_subfolder=True)

def get_dynamic_data() -> Dict[str, Any]:
    """Get concatenated data from dynamic data source."""
    if not dynamic_data_source:
        logger.error("Dynamic data source path not set")
        return {"json_data": [], "text_data": ""}
    
    logger.info(f"Processing dynamic data from: {dynamic_data_source}")
    return concatenate_directory_data(dynamic_data_source)

def save_concatenated_data(output_dir: str) -> None:
    """
    Save concatenated data from both sources to separate files.
    Creates 'static' and 'dynamic' subdirectories in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process static data
    static_data = get_static_data()
    static_dir = os.path.join(output_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    if static_data["json_data"]:
        with open(os.path.join(static_dir, 'combined.json'), 'w', encoding='utf-8') as f:
            json.dump(static_data["json_data"], f, indent=2)
    
    if static_data["text_data"]:
        with open(os.path.join(static_dir, 'combined.txt'), 'w', encoding='utf-8') as f:
            f.write(static_data["text_data"])
    
    # Process dynamic data
    dynamic_data = get_dynamic_data()
    dynamic_dir = os.path.join(output_dir, 'dynamic')
    os.makedirs(dynamic_dir, exist_ok=True)
    
    if dynamic_data["json_data"]:
        with open(os.path.join(dynamic_dir, 'combined.json'), 'w', encoding='utf-8') as f:
            json.dump(dynamic_data["json_data"], f, indent=2)
    
    if dynamic_data["text_data"]:
        with open(os.path.join(dynamic_dir, 'combined.txt'), 'w', encoding='utf-8') as f:
            f.write(dynamic_data["text_data"])

    logger.info(f"Concatenated data saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # 1. Set the data source paths from environment (do not hardcode paths)
    from dotenv import load_dotenv
    load_dotenv()
    static_data_source = os.getenv("STATIC_DATA_SOURCE", "")
    dynamic_data_source = os.getenv("DYNAMIC_DATA_SOURCE", "")
    output_dir = os.getenv("JOINED_OUTPUT_DIR", "")

    if not static_data_source or not dynamic_data_source or not output_dir:
        logger.error("Please set STATIC_DATA_SOURCE, DYNAMIC_DATA_SOURCE, and JOINED_OUTPUT_DIR in your environment.")
    else:
        # 2. Save concatenated data
        save_concatenated_data(output_dir)