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
    
    # --- Configuration --- (do not hardcode paths)
    from dotenv import load_dotenv
    load_dotenv()
    csv_file = Path(os.getenv("VOTING_CSV_FILE", ""))
    schema_file = Path(os.getenv("VOTE_SCHEMA_PATH", ""))
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
        
        # Drop behavior controlled by environment variable only (non-interactive)
        drop_existing = os.getenv('POSTGRES_DROP_EXISTING', 'false').lower() == 'true'
        
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