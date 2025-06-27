import os
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and process data from Polkadot network and wiki sources"""
    
    def __init__(self, polkadot_network_path: str, polkadot_wiki_path: str):
        self.polkadot_network_path = polkadot_network_path
        self.polkadot_wiki_path = polkadot_wiki_path
    
    def load_text_file(self, file_path: str) -> Dict[str, str]:
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
            
            metadata['source'] = 'polkadot_network' if 'polkadot_network' in file_path else 'polkadot_wiki'
            metadata['file_path'] = file_path
            
            return {
                'content': text_content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def load_json_file(self, file_path: str) -> Dict[str, str]:
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
                'source': 'polkadot_network' if 'polkadot_network' in file_path else 'polkadot_wiki',
                'file_path': file_path,
                'content_type': data.get('content_type', data.get('site_type', 'unknown'))
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict[str, str]]:
        """Load all documents from both data sources"""
        documents = []
        
        # Load from polkadot_network
        if os.path.exists(self.polkadot_network_path):
            logger.info(f"Loading documents from {self.polkadot_network_path}")
            for file_name in os.listdir(self.polkadot_network_path):
                file_path = os.path.join(self.polkadot_network_path, file_name)
                if os.path.isfile(file_path):
                    if file_name.endswith('.txt'):
                        doc = self.load_text_file(file_path)
                        if doc and doc['content'].strip():
                            documents.append(doc)
                    elif file_name.endswith('.json'):
                        # Skip JSON files that have corresponding TXT files
                        txt_file = file_name.replace('.json', '.txt')
                        txt_path = os.path.join(self.polkadot_network_path, txt_file)
                        if not os.path.exists(txt_path):
                            doc = self.load_json_file(file_path)
                            if doc and doc['content'].strip():
                                documents.append(doc)
        
        # Load from polkadot_wiki
        if os.path.exists(self.polkadot_wiki_path):
            logger.info(f"Loading documents from {self.polkadot_wiki_path}")
            for file_name in os.listdir(self.polkadot_wiki_path):
                file_path = os.path.join(self.polkadot_wiki_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.txt'):
                    doc = self.load_text_file(file_path)
                    if doc and doc['content'].strip():
                        documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents in total")
        return documents
    
    def get_document_stats(self, documents: List[Dict[str, str]]) -> Dict[str, int]:
        """Get statistics about loaded documents"""
        stats = {
            'total_documents': len(documents),
            'polkadot_network_docs': 0,
            'polkadot_wiki_docs': 0,
            'total_characters': 0,
            'average_doc_length': 0
        }
        
        for doc in documents:
            source = doc['metadata']['source']
            if source == 'polkadot_network':
                stats['polkadot_network_docs'] += 1
            elif source == 'polkadot_wiki':
                stats['polkadot_wiki_docs'] += 1
            
            stats['total_characters'] += len(doc['content'])
        
        if len(documents) > 0:
            stats['average_doc_length'] = stats['total_characters'] // len(documents)
        
        return stats 