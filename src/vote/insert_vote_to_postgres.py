"""
Inserting the vote data to postgres
"""
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# It's better to set up the path to import from the project root.
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.texttosql.insert_into_postgres import PostgresInserter

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function to insert vote data."""
    
    # --- Configuration ---
    csv_file = Path("/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/voting_data/polkadot_votes_20250922_153403.csv")
    schema_file = Path("/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/onchain_data/vote/vote_schema_info.json")
    table_name = "voting_data"
    
    if not csv_file.exists():
        logger.error(f"Vote data CSV file not found at: {csv_file}")
        logger.error("Please run flatten_vote_data.py first to generate the CSV.")
        return
        
    if not schema_file.exists():
        logger.error(f"Vote schema info file not found at: {schema_file}")
        return

    try:
        # Create inserter with vote-specific configuration
        inserter = PostgresInserter(
            table_name=table_name,
            schema_path=str(schema_file)
        )
        
        # Ask user about dropping the existing table, or use environment variable
        drop_existing_env = os.getenv('POSTGRES_DROP_EXISTING', 'false').lower()
        if drop_existing_env == 'true':
            drop_existing = True
        else:
            response = input(f"Drop existing table '{table_name}' if it exists? (y/N): ").lower()
            drop_existing = response in ['y', 'yes']
        
        # Run the full import process
        success = inserter.run_full_import(csv_file, drop_existing=drop_existing)
        
        if success:
            logger.info(f"🎉 Vote data import to '{table_name}' completed successfully!")
        else:
            logger.error(f"❌ Vote data import to '{table_name}' failed!")
            
    except KeyboardInterrupt:
        logger.info("Import cancelled by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()