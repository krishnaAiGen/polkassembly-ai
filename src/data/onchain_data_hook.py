#!/usr/bin/env python3
"""
Onchain Data Hook - Automated data fetching and embedding update system
Runs daily at 12:00:00 UTC to check for new onchain data and update embeddings if needed
"""

import os
import sys
import json
import hashlib
import schedule
import time
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import from our modules
from .onchain_data import fetch_onchain_data, PolkassemblyDataFetcher, SupportedNetworks
from ..utils.embeddings import EmbeddingManager
from ..utils.text_chunker import TextChunker
from ..rag.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('src/data/onchain_data_hook.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OnchainDataHook:
    """Automated system for fetching onchain data and updating embeddings"""
    
    def __init__(self, data_root_dir: str = "src/data", max_items_per_type: int = 1000):
        self.data_root_dir = Path(data_root_dir)
        self.max_items_per_type = max_items_per_type
        self.current_data_dir = self.data_root_dir / "current"
        self.archive_dir = self.data_root_dir / "archive"
        self.hash_file = self.data_root_dir / "data_hashes.json"
        
        # Ensure directories exist
        self._setup_directories()
        
        logger.info(f"OnchainDataHook initialized with data root: {self.data_root_dir}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        self.data_root_dir.mkdir(exist_ok=True)
        self.current_data_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Create log file if it doesn't exist
        log_file = self.data_root_dir / "onchain_data_hook.log"
        if not log_file.exists():
            log_file.touch()
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        
        try:
            with open(filepath, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return ""
    
    def calculate_directory_hash(self, directory: Path) -> Dict[str, str]:
        """Calculate hashes for all JSON files in a directory"""
        hashes = {}
        
        if not directory.exists():
            return hashes
        
        for json_file in directory.glob("*.json"):
            file_hash = self.calculate_file_hash(json_file)
            if file_hash:
                hashes[json_file.name] = file_hash
        
        return hashes
    
    def load_previous_hashes(self) -> Dict[str, str]:
        """Load previous file hashes from storage"""
        if not self.hash_file.exists():
            return {}
        
        try:
            with open(self.hash_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading previous hashes: {e}")
            return {}
    
    def save_current_hashes(self, hashes: Dict[str, str]):
        """Save current file hashes to storage"""
        try:
            with open(self.hash_file, 'w') as f:
                json.dump(hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving hashes: {e}")
    
    def compare_data_changes(self, new_hashes: Dict[str, str], old_hashes: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Compare new and old hashes to detect changes"""
        changes_detected = False
        changed_files = []
        
        # Check for new files or changed files
        for filename, new_hash in new_hashes.items():
            old_hash = old_hashes.get(filename, "")
            if new_hash != old_hash:
                changes_detected = True
                changed_files.append(filename)
                if old_hash:
                    logger.info(f"Changed file detected: {filename}")
                else:
                    logger.info(f"New file detected: {filename}")
        
        # Check for deleted files
        for filename in old_hashes:
            if filename not in new_hashes:
                changes_detected = True
                changed_files.append(f"DELETED: {filename}")
                logger.info(f"Deleted file detected: {filename}")
        
        return changes_detected, changed_files
    
    def archive_current_data(self, timestamp: str):
        """Archive current data to timestamped directory"""
        if not self.current_data_dir.exists() or not list(self.current_data_dir.glob("*.json")):
            logger.info("No current data to archive")
            return
        
        archive_path = self.archive_dir / f"data_{timestamp}"
        archive_path.mkdir(exist_ok=True)
        
        # Copy all JSON files to archive
        for json_file in self.current_data_dir.glob("*.json"):
            shutil.copy2(json_file, archive_path / json_file.name)
        
        logger.info(f"Archived current data to {archive_path}")
    
    def fetch_new_data(self, timestamp: str) -> bool:
        """Fetch new onchain data"""
        logger.info("Starting onchain data fetch...")
        
        # Create temporary directory for new data
        temp_dir = self.data_root_dir / f"temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Fetch data for each network
            for network in SupportedNetworks:
                try:
                    logger.info(f"Fetching data for {network.value}...")
                    
                    # Initialize fetcher with temp directory
                    fetcher = PolkassemblyDataFetcher(network=network.value)
                    fetcher.data_dir = str(temp_dir)  # Override data directory
                    
                    # Fetch and save all data
                    fetcher.fetch_and_save_all_data(max_items_per_type=self.max_items_per_type)
                    
                    logger.info(f"Completed data fetch for {network.value}")
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {network.value}: {e}")
                    continue
            
            # Move temp data to current directory
            if list(temp_dir.glob("*.json")):
                # Clear current directory
                for json_file in self.current_data_dir.glob("*.json"):
                    json_file.unlink()
                
                # Move new files to current
                for json_file in temp_dir.glob("*.json"):
                    shutil.move(str(json_file), self.current_data_dir / json_file.name)
                
                temp_dir.rmdir()  # Remove empty temp directory
                return True
            else:
                logger.warning("No new data fetched")
                temp_dir.rmdir()
                return False
                
        except Exception as e:
            logger.error(f"Error during data fetch: {e}")
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False
    
    def create_dynamic_embeddings_from_data(self) -> bool:
        """Create embeddings from current data directory"""
        try:
            logger.info("Starting dynamic embeddings creation...")
            
            # Load documents from current data
            documents = self.load_dynamic_data(str(self.current_data_dir))
            if not documents:
                logger.warning("No documents found for embedding creation")
                return False
            
            # Initialize components
            chunker = TextChunker(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            
            embedding_manager = EmbeddingManager(
                openai_api_key=Config.OPENAI_API_KEY,
                collection_name=Config.CHROMA_DYNAMIC_COLLECTION_NAME,
                chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY
            )
            
            # Clear existing collection
            logger.info("Clearing existing dynamic embeddings collection...")
            embedding_manager.clear_collection()
            
            # Process documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            
            # Create embeddings
            success = embedding_manager.add_chunks_to_collection(all_chunks)
            
            if success:
                logger.info("Successfully created dynamic embeddings")
                return True
            else:
                logger.error("Failed to create dynamic embeddings")
                return False
                
        except Exception as e:
            logger.error(f"Error creating dynamic embeddings: {e}")
            return False
    
    def load_dynamic_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load dynamic data from JSON files (similar to create_dynamic_embeddings.py)"""
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_path}")
            return documents
        
        # Process all JSON files in the directory
        for file_path in data_path.glob("*.json"):
            try:
                # Extract network from filename
                filename = file_path.name.lower()
                if filename.startswith("polkadot_"):
                    network = "polkadot"
                elif filename.startswith("kusama_"):
                    network = "kusama"
                else:
                    logger.warning(f"Skipping file with unknown network prefix: {file_path}")
                    continue
                
                if file_path.name == "fetch_summary.json":
                    continue
                
                # Load JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process each item in the file
                for item in data.get('items', []):
                    # Extract content parts
                    content_parts = []
                    
                    if item.get('title'):
                        content_parts.append(f"Title: {item['title']}")
                    
                    if item.get('content'):
                        content_parts.append(item['content'])
                    
                    if item.get('onChainInfo'):
                        info = item['onChainInfo']
                        content_parts.append(f"\nOn-chain Information:")
                        if info.get('proposer'):
                            content_parts.append(f"Proposer: {info['proposer']}")
                        if info.get('status'):
                            content_parts.append(f"Status: {info['status']}")
                        if info.get('hash'):
                            content_parts.append(f"Hash: {info['hash']}")
                    
                    content = '\n\n'.join(content_parts)
                    
                    # Create metadata
                    metadata = {
                        'title': item.get('title', ''),
                        'network': network,
                        'proposalType': item.get('proposalType', ''),
                        'index': item.get('index', ''),
                        'createdAt': item.get('createdAt', ''),
                        'source': 'polkassembly',
                        'data_type': 'dynamic',
                        'file_path': str(file_path)
                    }
                    
                    if content.strip():
                        documents.append({
                            'content': content,
                            'metadata': metadata
                        })
                        
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
        
        return documents
    
    def run_daily_update(self):
        """Main function to run daily data update and embedding refresh"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        logger.info("="*60)
        logger.info(f"STARTING DAILY ONCHAIN DATA UPDATE - {timestamp}")
        logger.info("="*60)
        
        try:
            # Step 1: Archive current data
            self.archive_current_data(timestamp)
            
            # Step 2: Fetch new data
            fetch_success = self.fetch_new_data(timestamp)
            if not fetch_success:
                logger.error("Failed to fetch new data. Skipping embedding update.")
                return
            
            # Step 3: Calculate hashes and compare
            new_hashes = self.calculate_directory_hash(self.current_data_dir)
            old_hashes = self.load_previous_hashes()
            
            changes_detected, changed_files = self.compare_data_changes(new_hashes, old_hashes)
            
            # Step 4: Update embeddings if changes detected
            if changes_detected:
                logger.info(f"Data changes detected in {len(changed_files)} files:")
                for file in changed_files:
                    logger.info(f"  - {file}")
                
                logger.info("Updating dynamic embeddings...")
                embedding_success = self.create_dynamic_embeddings_from_data()
                
                if embedding_success:
                    # Save new hashes only if embedding update was successful
                    self.save_current_hashes(new_hashes)
                    logger.info("✅ Dynamic embeddings updated successfully!")
                else:
                    logger.error("❌ Failed to update dynamic embeddings")
            else:
                logger.info("No data changes detected. Skipping embedding update.")
                # Still save hashes for consistency
                self.save_current_hashes(new_hashes)
            
            logger.info("="*60)
            logger.info("DAILY ONCHAIN DATA UPDATE COMPLETED")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error during daily update: {e}")
    
    def start_scheduler(self):
        """Start the scheduler to run daily at 12:00 UTC"""
        logger.info("Starting onchain data hook scheduler...")
        logger.info("Scheduled to run daily at 12:00:00 UTC")
        
        # Schedule daily run at 12:00 UTC
        schedule.every().day.at("12:00:00").do(self.run_daily_update)
        
        # Also run immediately for testing (optional)
        logger.info("Running initial data fetch...")
        self.run_daily_update()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Onchain Data Hook - Automated data fetching and embedding updates")
    parser.add_argument("--data-root", type=str, default="src/data", help="Root directory for data storage")
    parser.add_argument("--max-items", type=int, default=1000, help="Maximum items per proposal type")
    parser.add_argument("--run-once", action="store_true", help="Run once instead of starting scheduler")
    
    args = parser.parse_args()
    
    # Initialize hook
    hook = OnchainDataHook(
        data_root_dir=args.data_root,
        max_items_per_type=args.max_items
    )
    
    if args.run_once:
        # Run once and exit
        hook.run_daily_update()
    else:
        # Start continuous scheduler
        hook.start_scheduler()

if __name__ == "__main__":
    main()
