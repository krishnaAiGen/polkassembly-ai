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

def concatenate_directory_data(directory: str) -> Dict[str, Any]:
    """
    Concatenate all JSON and text files in a directory and its subdirectories.
    Returns a dictionary with 'json_data' and 'text_data' keys.
    """
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return {"json_data": [], "text_data": ""}

    json_data = []
    text_data = []

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
    """Get concatenated data from static data source."""
    if not static_data_source:
        logger.error("Static data source path not set")
        return {"json_data": [], "text_data": ""}
    
    logger.info(f"Processing static data from: {static_data_source}")
    return concatenate_directory_data(static_data_source)

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
    # 1. Set the data source paths
    static_data_source = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/static_sources"
    dynamic_data_source = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/dynamic_kusama_polka"
    
    # 2. Save concatenated data
    save_concatenated_data("/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/joined_data") 