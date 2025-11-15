#!/usr/bin/env python3
"""
Dynamic Embeddings Indexer - Reads directly from PostgreSQL and indexes into Chroma

This module ingests data from governance_data and voting_data tables
directly into the dynamic Chroma collection without intermediate JSON files.
"""

import os
import sys
import psycopg2
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import logging
from dotenv import load_dotenv
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.embeddings import EmbeddingManager
from src.rag.config import Config

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicEmbeddingsIndexer:
    """Indexes governance and voting data into the dynamic Chroma collection"""
    
    def __init__(self):
        """Initialize the indexer with database configs and Chroma client"""
        
        # Governance DB configuration
        self.gov_db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DATABASE'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Voting DB configuration
        self.vote_db_config = {
            'host': os.getenv('POSTGRES_HOST_PA'),
            'port': int(os.getenv('POSTGRES_PORT_PA', '5432')),
            'database': os.getenv('POSTGRES_DATABASE_PA'),
            'user': os.getenv('POSTGRES_USER_PA'),
            'password': os.getenv('POSTGRES_PASSWORD_PA')
        }
        
        # Validate configurations
        self._validate_config()
        
        # Initialize Chroma dynamic collection
        logger.info("Initializing dynamic Chroma collection...")
        self.embedding_manager = None
        self._init_dynamic_collection()
        
        logger.info("Dynamic Embeddings Indexer initialized successfully")
    
    def _validate_config(self):
        """Validate that required environment variables are set"""
        gov_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        vote_vars = ['POSTGRES_HOST_PA', 'POSTGRES_DATABASE_PA', 'POSTGRES_USER_PA', 'POSTGRES_PASSWORD_PA']
        
        missing_gov = [v for v in gov_vars if not os.getenv(v)]
        missing_vote = [v for v in vote_vars if not os.getenv(v)]
        
        if missing_gov:
            logger.warning(f"Missing governance DB vars: {missing_gov}")
        if missing_vote:
            logger.warning(f"Missing voting DB vars: {missing_vote}")
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY is required for embeddings generation")
    
    def _init_dynamic_collection(self):
        """Initialize or get the dynamic Chroma collection"""
        try:
            self.embedding_manager = EmbeddingManager(
                openai_api_key=Config.OPENAI_API_KEY,
                embedding_model=Config.OPENAI_EMBEDDING_MODEL,
                chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                collection_name=Config.CHROMA_DYNAMIC_COLLECTION_NAME
            )
            logger.info(f"Loaded dynamic collection: {Config.CHROMA_DYNAMIC_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize dynamic collection: {e}")
            raise
    
    def get_dynamic_collection(self):
        """Get the dynamic Chroma collection"""
        if not self.embedding_manager or not self.embedding_manager.collection:
            raise RuntimeError("Dynamic collection not initialized")
        return self.embedding_manager.collection
    
    @contextmanager
    def get_gov_connection(self):
        """Context manager for governance database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.gov_db_config)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Governance DB connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_vote_connection(self):
        """Context manager for voting database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.vote_db_config)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Voting DB connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def index_governance_dynamic_embeddings(self, limit: Optional[int] = None, batch_size: int = 500) -> int:
        """
        Index governance data from the governance_data table into dynamic Chroma collection
        
        Args:
            limit: Optional limit on number of rows to index
            batch_size: Number of rows to process in each batch
            
        Returns:
            Total number of governance rows indexed
        """
        logger.info("Starting governance data indexing...")
        total_indexed = 0
        
        try:
            with self.get_gov_connection() as conn:
                cursor = conn.cursor()
                
                # Query to select governance data
                # Use DISTINCT ON to get one row per (network, proposal_index) combination
                query = """
                    SELECT DISTINCT ON ("source_network", "index")
                        "index" as proposal_index,
                        "source_network" as network,
                        "title",
                        "content",
                        "onchaininfo_status" as status,
                        "createdat" as created_at,
                        "source_proposal_type" as proposal_type
                    FROM governance_data
                    WHERE "index" IS NOT NULL
                        AND "source_network" IS NOT NULL
                        AND "createdat" IS NOT NULL
                    ORDER BY "source_network", "index", "createdat" DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                logger.info(f"Executing governance query (limit: {limit or 'none'})...")
                cursor.execute(query)
                
                # Process in batches
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    
                    batch_chunks = []
                    for row in rows:
                        proposal_index, network, title, content, status, created_at, proposal_type = row
                        
                        # Build stable document ID
                        doc_id = f"gov:{network}:{proposal_index}"
                        
                        # Build document text
                        document = f"""Network: {network}
Proposal #{proposal_index}
Title: {title or 'N/A'}
Type: {proposal_type or 'N/A'}
Status: {status or 'N/A'}
Created: {created_at}

Description: {content[:2000] if content else 'N/A'}"""
                        
                        # Build metadata
                        metadata = {
                            'source_db': 'governance_db',
                            'table': 'governance_data',
                            'network': str(network),
                            'proposal_index': str(proposal_index),
                            'proposal_type': str(proposal_type or 'N/A'),
                            'status': str(status or 'N/A'),
                            'created_at': str(created_at),
                            'doc_type': 'governance'
                        }
                        
                        batch_chunks.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata
                        })
                    
                    # Add batch to Chroma
                    if batch_chunks:
                        self._add_batch_to_chroma(batch_chunks)
                        total_indexed += len(batch_chunks)
                        logger.info(f"Indexed {total_indexed} governance rows...")
                
                cursor.close()
            
            logger.info(f"✅ Successfully indexed {total_indexed} governance rows")
            return total_indexed
            
        except Exception as e:
            logger.error(f"Error indexing governance data: {e}")
            raise
    
    def index_voting_dynamic_embeddings(self, limit: Optional[int] = None, batch_size: int = 500) -> int:
        """
        Index voting data from the flattened_conviction_votes table into dynamic Chroma collection
        
        Args:
            limit: Optional limit on number of rows to index
            batch_size: Number of rows to process in each batch
            
        Returns:
            Total number of voting rows indexed
        """
        logger.info("Starting voting data indexing...")
        total_indexed = 0
        
        try:
            with self.get_vote_connection() as conn:
                cursor = conn.cursor()
                
                # Query to select voting data
                # Note: flattened_conviction_votes doesn't have a network column
                query = """
                    SELECT 
                        main.id,
                        main.voter,
                        main.proposal_index,
                        main.decision,
                        main.created_at,
                        cv.self_voting_power,
                        main.type
                    FROM flattened_conviction_votes AS main
                    LEFT JOIN conviction_vote AS cv ON main.parent_vote_id = cv.id
                    WHERE main.voter IS NOT NULL
                        AND main.proposal_index IS NOT NULL
                        AND main.created_at IS NOT NULL
                    ORDER BY main.created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                logger.info(f"Executing voting query (limit: {limit or 'none'})...")
                cursor.execute(query)
                
                # Process in batches
                while True:
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    
                    batch_chunks = []
                    for row in rows:
                        vote_id, voter, proposal_index, decision, created_at, voting_power, vote_type = row
                        
                        # Derive network from vote type if available (e.g., "ReferendumV2" might indicate network)
                        # For now, we'll use "unknown" or the vote_type as a fallback
                        network = "polkadot"  # Default assumption since this is polkadot data
                        
                        # Build stable document ID
                        doc_id = f"vote:{network}:{proposal_index}:{voter}:{vote_id}"
                        
                        # Build document text
                        document = f"""Network: {network}
Voter: {voter}
Proposal #{proposal_index}
Decision: {decision or 'N/A'}
Voting Power: {voting_power or 'N/A'}
Type: {vote_type or 'N/A'}
Created: {created_at}"""
                        
                        # Build metadata
                        metadata = {
                            'source_db': 'voting_db',
                            'table': 'voting_data',
                            'network': str(network),
                            'proposal_index': str(proposal_index),
                            'voter': str(voter),
                            'decision': str(decision or 'N/A'),
                            'vote_type': str(vote_type or 'N/A'),
                            'created_at': str(created_at),
                            'doc_type': 'vote'
                        }
                        
                        batch_chunks.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata
                        })
                    
                    # Add batch to Chroma
                    if batch_chunks:
                        self._add_batch_to_chroma(batch_chunks)
                        total_indexed += len(batch_chunks)
                        logger.info(f"Indexed {total_indexed} voting rows...")
                
                cursor.close()
            
            logger.info(f"✅ Successfully indexed {total_indexed} voting rows")
            return total_indexed
            
        except Exception as e:
            logger.error(f"Error indexing voting data: {e}")
            raise
    
    def _add_batch_to_chroma(self, chunks: List[Dict[str, Any]]):
        """
        Add a batch of chunks to the dynamic Chroma collection
        
        Args:
            chunks: List of dicts with 'id', 'content', and 'metadata'
        """
        try:
            # Extract data
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Generate embeddings and add to collection
            # Using the existing EmbeddingManager's add_chunks_to_collection would regenerate embeddings
            # So we'll use direct collection.add with our own embedding generation
            embeddings = self.embedding_manager.generate_embeddings(documents)
            
            # Add to collection (this will upsert if IDs already exist)
            collection = self.get_dynamic_collection()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error(f"Error adding batch to Chroma: {e}")
            raise
    
    def index_all_dynamic_embeddings(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Index both governance and voting data into the dynamic Chroma collection
        
        Args:
            limit: Optional limit on number of rows to index per table
            
        Returns:
            Dict with counts: {"governance_indexed": int, "voting_indexed": int}
        """
        logger.info("=" * 60)
        logger.info("Starting full dynamic embeddings indexing")
        logger.info("=" * 60)
        
        results = {
            'governance_indexed': 0,
            'voting_indexed': 0
        }
        
        # Index governance data
        try:
            results['governance_indexed'] = self.index_governance_dynamic_embeddings(limit=limit)
        except Exception as e:
            logger.error(f"Failed to index governance data: {e}")
        
        # Index voting data
        try:
            results['voting_indexed'] = self.index_voting_dynamic_embeddings(limit=limit)
        except Exception as e:
            logger.error(f"Failed to index voting data: {e}")
        
        # Get final stats
        try:
            collection = self.get_dynamic_collection()
            total_docs = collection.count()
            logger.info(f"\n📊 Final Collection Stats:")
            logger.info(f"  Total documents in collection: {total_docs}")
            logger.info(f"  Governance indexed this run: {results['governance_indexed']}")
            logger.info(f"  Voting indexed this run: {results['voting_indexed']}")
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
        
        logger.info("=" * 60)
        logger.info("Indexing complete")
        logger.info("=" * 60)
        
        return results


def main():
    """Main entry point for the indexer"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Index governance and voting data into dynamic Chroma collection"
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of rows to index per table (for testing)',
        default=None
    )
    parser.add_argument(
        '--governance-only',
        action='store_true',
        help='Index only governance data'
    )
    parser.add_argument(
        '--voting-only',
        action='store_true',
        help='Index only voting data'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the collection after indexing'
    )
    
    args = parser.parse_args()
    
    try:
        indexer = DynamicEmbeddingsIndexer()
        
        if args.governance_only:
            count = indexer.index_governance_dynamic_embeddings(limit=args.limit)
            print(f"\n✅ Indexed {count} governance rows")
        elif args.voting_only:
            count = indexer.index_voting_dynamic_embeddings(limit=args.limit)
            print(f"\n✅ Indexed {count} voting rows")
        else:
            results = indexer.index_all_dynamic_embeddings(limit=args.limit)
            print(f"\n✅ Indexing complete:")
            print(f"   Governance: {results['governance_indexed']} rows")
            print(f"   Voting: {results['voting_indexed']} rows")
        
        if args.verify:
            print("\n🔍 Verifying collection...")
            collection = indexer.get_dynamic_collection()
            count = collection.count()
            print(f"   Total documents in collection: {count}")
            
            # Test query
            if count > 0:
                test_results = collection.query(
                    query_texts=["recent proposals"],
                    n_results=min(3, count)
                )
                print(f"   Sample query returned {len(test_results['documents'][0])} results")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

