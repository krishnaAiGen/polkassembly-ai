import os
import json
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and process data from static (.txt) and dynamic (.json) sources"""
    
    def __init__(self, static_data_path: str, dynamic_data_path: str):
        self.static_data_path = static_data_path
        self.dynamic_data_path = dynamic_data_path
    
    def find_all_files(self, directory: str, extension: str) -> List[str]:
        """Recursively find all files with given extension in directory and subdirectories"""
        file_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        
        return file_paths
    
    def load_static_data(self) -> List[Dict[str, Any]]:
        """Load all .txt files from static data source and subdirectories"""
        logger.info(f"Loading static data from: {self.static_data_path}")
        
        if not os.path.exists(self.static_data_path):
            logger.warning(f"Static data path does not exist: {self.static_data_path}")
            return []
        
        # Find all .txt files recursively
        txt_files = self.find_all_files(self.static_data_path, '.txt')
        logger.info(f"Found {len(txt_files)} .txt files")
        
        static_data = []
        
        for file_path in txt_files:
            try:
                data = self.load_text_file(file_path)
                if data:
                    static_data.append(data)
            except Exception as e:
                logger.error(f"Error loading static file {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(static_data)} static data items")
        return static_data
    
    def load_dynamic_data(self) -> List[Dict[str, Any]]:
        """Load all .json files from dynamic data source and subdirectories"""
        logger.info(f"Loading dynamic data from: {self.dynamic_data_path}")
        
        if not os.path.exists(self.dynamic_data_path):
            logger.warning(f"Dynamic data path does not exist: {self.dynamic_data_path}")
            return []
        
        # Find all .json files recursively
        json_files = self.find_all_files(self.dynamic_data_path, '.json')
        logger.info(f"Found {len(json_files)} .json files")
        
        dynamic_data = []
        
        for file_path in json_files:
            try:
                data = self.load_json_file(file_path)
                if data:
                    dynamic_data.append(data)
            except Exception as e:
                logger.error(f"Error loading dynamic file {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(dynamic_data)} dynamic data items")
        return dynamic_data
    
    def load_text_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single text file and extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            metadata = {}
            text_content = ""
            
            # Parse metadata from the beginning of text files
            if len(lines) > 0 and lines[0].startswith('Title: '):
                metadata['title'] = lines[0].replace('Title: ', '').strip()
            
            if len(lines) > 1 and lines[1].startswith('URL: '):
                metadata['url'] = lines[1].replace('URL: ', '').strip()
            
            if len(lines) > 2 and (lines[2].startswith('Description: ') or lines[2].startswith('Type: ')):
                if lines[2].startswith('Description: '):
                    metadata['description'] = lines[2].replace('Description: ', '').strip()
                elif lines[2].startswith('Type: '):
                    metadata['type'] = lines[2].replace('Type: ', '').strip()
                    if len(lines) > 3 and lines[3].startswith('Description: '):
                        metadata['description'] = lines[3].replace('Description: ', '').strip()
            
            # Find the separator line and extract main content
            separator_found = False
            for i, line in enumerate(lines):
                if '========' in line or '=======' in line:
                    text_content = '\n'.join(lines[i+1:]).strip()
                    separator_found = True
                    break
            
            if not separator_found:
                # If no separator found, use all content after metadata
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        start_idx = i + 1
                        break
                text_content = '\n'.join(lines[start_idx:]).strip()
            
            # Add source information
            metadata['source'] = 'static'
            metadata['file_path'] = file_path
            metadata['relative_path'] = os.path.relpath(file_path, self.static_data_path)
            
            return {
                'content': text_content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return None
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single JSON file and extract relevant content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content from JSON structure
            content_parts = []
            
            if 'title' in data:
                content_parts.append(f"Title: {data['title']}")
            
            if 'description' in data:
                content_parts.append(f"Description: {data['description']}")
            
            if 'content' in data and data['content']:
                content_parts.append(data['content'])
            
            # Add topics if available (for forum data)
            if 'topics' in data and data['topics']:
                topics_text = "Related Topics:\n"
                for topic in data['topics'][:10]:  # Limit to first 10 topics
                    topics_text += f"- {topic.get('title', '')}\n"
                content_parts.append(topics_text)
            
            content = '\n\n'.join(content_parts)
            
            metadata = {
                'title': data.get('title', ''),
                'url': data.get('url', ''),
                'description': data.get('description', ''),
                'source': 'dynamic',
                'file_path': file_path,
                'relative_path': os.path.relpath(file_path, self.dynamic_data_path)
            }
            
            # Add any additional metadata from JSON
            for key, value in data.items():
                if key not in ['title', 'url', 'description', 'content', 'topics'] and isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get status information about available data"""
        static_files = self.find_all_files(self.static_data_path, '.txt') if os.path.exists(self.static_data_path) else []
        dynamic_files = self.find_all_files(self.dynamic_data_path, '.json') if os.path.exists(self.dynamic_data_path) else []
        
        return {
            'static_data': {
                'path': self.static_data_path,
                'exists': os.path.exists(self.static_data_path),
                'file_count': len(static_files),
                'files': static_files[:10]  # Show first 10 files
            },
            'dynamic_data': {
                'path': self.dynamic_data_path,
                'exists': os.path.exists(self.dynamic_data_path),
                'file_count': len(dynamic_files),
                'files': dynamic_files[:10]  # Show first 10 files
            }
        } 
    

    