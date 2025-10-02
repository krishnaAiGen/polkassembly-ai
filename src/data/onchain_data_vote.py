import requests
import json
import os
import time
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProposalType(Enum):
    DEMOCRACY_PROPOSAL = "DemocracyProposal"
    TECH_COMMITTEE_PROPOSAL = "TechCommitteeProposal"
    TREASURY_PROPOSAL = "TreasuryProposal"
    REFERENDUM = "Referendum"
    COUNCIL_MOTION = "CouncilMotion"
    BOUNTY = "Bounty"
    TIP = "Tip"
    CHILD_BOUNTY = "ChildBounty"
    REFERENDUM_V2 = "ReferendumV2"
    FELLOWSHIP_REFERENDUM = "FellowshipReferendum"

class OriginType(Enum):
    AUCTION_ADMIN = "AuctionAdmin"
    BIG_SPENDER = "BigSpender"
    BIG_TIPPER = "BigTipper"
    CANDIDATES = "Candidates"
    EXPERTS = "Experts"
    FELLOWS = "Fellows"
    FELLOWSHIP_ADMIN = "FellowshipAdmin"
    GENERAL_ADMIN = "GeneralAdmin"
    GRAND_MASTERS = "GrandMasters"
    LEASE_ADMIN = "LeaseAdmin"
    MASTERS = "Masters"
    MEDIUM_SPENDER = "MediumSpender"
    MEMBERS = "Members"
    PROFICIENTS = "Proficients"
    REFERENDUM_CANCELLER = "ReferendumCanceller"
    REFERENDUM_KILLER = "ReferendumKiller"
    ROOT = "Root"
    SENIOR_EXPERTS = "SeniorExperts"
    SENIOR_FELLOWS = "SeniorFellows"
    SENIOR_MASTERS = "SeniorMasters"
    SMALL_SPENDER = "SmallSpender"
    SMALL_TIPPER = "SmallTipper"
    STAKING_ADMIN = "StakingAdmin"
    TREASURER = "Treasurer"
    WHITELISTED_CALLER = "WhitelistedCaller"
    WISH_FOR_CHANGE = "WishForChange"
    FAST_GENERAL_ADMIN = "FastGeneralAdmin"

class SupportedNetworks(Enum):
    POLKADOT = "polkadot"
    KUSAMA = "kusama"

class PolkassemblyDataFetcher:
    def __init__(self, network: str = "polkadot", data_dir: str = None):
        self.network = network
        self.base_url = f"https://{network}.polkassembly.io/api/v2"
        self.headers = {
            'Content-Type': 'application/json',
        }
        # Use specified data directory or default to the requested path
        if data_dir:
            self.data_dir = data_dir
        else:
            # Default to the specified onchain_data directory
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(script_dir, "data", "onchain_data")
        
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_posts(self, proposal_type: ProposalType, origin_type: Optional[OriginType] = None, 
                   limit: int = 50, offset: int = 1) -> Dict[str, Any]:
        """Fetch posts from Polkassembly API"""
        url = f"{self.base_url}/{proposal_type.value}"

        params = {
            'limit': limit,
            'page': offset
        }
        
        if origin_type:
            params['origin'] = origin_type.value

        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching posts for {proposal_type.value}: {e}")
            return {}

    def fetch_all_posts_for_type(self, proposal_type: ProposalType, 
                                origin_type: Optional[OriginType] = None, 
                                max_items: int = 1000) -> List[Dict]:
        """Fetch all posts for a specific proposal type with pagination"""
        all_posts = []
        offset = 1
        limit = 50

        logger.info(f"Fetching {proposal_type.value} posts for {self.network}...")
        
        while len(all_posts) < max_items:
            response_data = self.fetch_posts(proposal_type, origin_type, limit, offset)
            
            if not response_data or 'items' not in response_data:
                break
                
            posts = response_data['items']
            if not posts:
                break

            total_count = response_data['totalCount']
            if total_count:
                max_items = total_count
                
            all_posts.extend(posts)
            offset += 1
            time.sleep(0.1)  # Rate limiting
            
            logger.info(f"Fetched {len(all_posts)} {proposal_type.value} posts so far...")

        return all_posts[:max_items]

    def save_to_file(self, data: List[Dict], filename: str):
        """Save data to JSON file in the data directory"""
        filepath = os.path.join(self.data_dir, filename)
        
        # Add metadata
        file_data = {
            'network': self.network,
            'timestamp': datetime.now().isoformat(),
            'total_items': len(data),
            'items': data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} items to {filepath}")

    def fetch_votes(self, page: int = 1, limit: int = 50) -> Dict[str, Any]:
        """Fetch votes from Polkassembly API"""
        url = f"{self.base_url}/votes/all"
        
        params = {
            'page': page,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching votes: {e}")
            return {}

    def fetch_all_votes_batched(self, max_items: int = None, save_batch_size: int = 1000) -> int:
        """Fetch ALL available votes in batches and save immediately to avoid memory issues"""
        page = 1
        limit = 100  # API limit per request
        total_saved = 0
        current_batch = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Fetching ALL available votes for {self.network}, saving every {save_batch_size} votes...")
        
        while True:
            response_data = self.fetch_votes(page=page, limit=limit)
            
            if not response_data or 'items' not in response_data:
                logger.warning("No more data available from API")
                break
                
            votes = response_data['items']
            if not votes:
                logger.info("No votes returned in this batch - reached end of data")
                break

            total_count = response_data.get('totalCount', 0)
            if total_count and page == 1:  # Only log this once
                logger.info(f"API reports {total_count} total votes available")
                if max_items and max_items < total_count:
                    logger.info(f"Will fetch up to {max_items} votes as requested")
            
            # Add votes to current batch
            current_batch.extend(votes)
            
            # Save batch when it reaches save_batch_size
            if len(current_batch) >= save_batch_size:
                # Save the full batch
                batch_number = (total_saved // save_batch_size) + 1
                filename = f"{self.network}_votes_batch_{batch_number}_{timestamp}.json"
                batch_to_save = current_batch[:save_batch_size]
                self.save_to_file(batch_to_save, filename)
                
                total_saved += len(batch_to_save)
                logger.info(f"Saved batch {batch_number}: {len(batch_to_save)} votes (Total saved: {total_saved})")
                
                # Keep remaining votes for next batch
                current_batch = current_batch[save_batch_size:]
                
                # Check if we've reached max_items
                if max_items and total_saved >= max_items:
                    logger.info(f"Reached maximum requested items: {max_items}")
                    break
            
            # Check if we got fewer votes than requested (indicates end of data)
            if len(votes) < limit:
                logger.info("Received fewer votes than requested - reached end of available data")
                break
            
            page += 1
            time.sleep(0.2)  # Rate limiting
            
            if page % 50 == 0:  # Log progress every 50 pages
                logger.info(f"Processed {page * limit} API records, saved {total_saved} votes so far...")

        # Save any remaining votes in the last batch
        if current_batch:
            batch_number = (total_saved // save_batch_size) + 1
            filename = f"{self.network}_votes_batch_{batch_number}_{timestamp}.json"
            self.save_to_file(current_batch, filename)
            total_saved += len(current_batch)
            logger.info(f"Saved final batch {batch_number}: {len(current_batch)} votes (Total saved: {total_saved})")

        logger.info(f"Finished fetching ALL available votes. Total votes saved: {total_saved}")
        return total_saved

    def fetch_all_votes(self, max_items: int = 1000) -> List[Dict]:
        """Fetch all votes with pagination (legacy method for compatibility)"""
        all_votes = []
        page = 1
        limit = 100

        logger.info(f"Fetching votes for {self.network}...")
        
        while len(all_votes) < max_items:
            response_data = self.fetch_votes(page=page, limit=limit)
            
            if not response_data or 'items' not in response_data:
                break
                
            votes = response_data['items']
            if not votes:
                break

            total_count = response_data.get('totalCount', 0)
            if total_count:
                max_items = total_count
                
            all_votes.extend(votes)
            page += 1
            time.sleep(0.2)  # Rate limiting
            
            logger.info(f"Fetched {len(all_votes)} votes so far...")

        return all_votes[:max_items]

    def fetch_and_save_all_data(self, max_items_per_type: int = 1000):
        """Fetch all data types and save to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for proposal_type in ProposalType:
            logger.info(f"proposal_type is {proposal_type}")
            try:
                if proposal_type == ProposalType.REFERENDUM_V2:
                    # ReferendumV2 requires origin_type parameter
                    for origin_type in OriginType:
                        try:
                            data = self.fetch_all_posts_for_type(
                                proposal_type=proposal_type,
                                origin_type=origin_type,
                                max_items=max_items_per_type
                            )
                            
                            if data:
                                filename = f"{self.network}_{proposal_type.value}_{origin_type.value}_{timestamp}.json"
                                self.save_to_file(data, filename)
                        except Exception as e:
                            logger.error(f"Error processing {proposal_type.value} with {origin_type.value}: {e}")
                            continue
                else:
                    # Other proposal types don't require origin_type
                    data = self.fetch_all_posts_for_type(
                        proposal_type=proposal_type,
                        max_items=max_items_per_type
                    )
                    
                    if data:
                        filename = f"{self.network}_{proposal_type.value}_{timestamp}.json"
                        self.save_to_file(data, filename)
                        
            except Exception as e:
                logger.error(f"Error processing {proposal_type.value}: {e}")
                continue

def fetch_votes_data(network: str = "polkadot", data_dir: str = None, max_items: int = None, save_batch_size: int = 1000):
    """Main function to fetch ALL votes data for a specific network in batches"""
    # Use the specified directory path
    if not data_dir:
        data_dir = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/voting_data"
    
    logger.info(f"Storing votes data in: {data_dir}")
    
    try:
        if max_items:
            logger.info(f"Starting votes fetch for {network} (max {max_items} votes)...")
        else:
            logger.info(f"Starting votes fetch for {network} (ALL available votes)...")
        
        # Initialize fetcher with specified data directory
        fetcher = PolkassemblyDataFetcher(network=network, data_dir=data_dir)
        
        # Fetch votes in batches and save immediately
        total_saved = fetcher.fetch_all_votes_batched(max_items=max_items, save_batch_size=save_batch_size)
        
        if total_saved > 0:
            logger.info(f"Completed votes fetch for {network}. Total votes saved: {total_saved}")
        else:
            logger.warning(f"No votes data fetched for {network}")
            
    except Exception as e:
        logger.error(f"Error processing votes for network {network}: {e}")

def fetch_onchain_data(max_items_per_type: int = 1000, data_dir: str = None):
    """Main function to fetch onchain data for all supported networks"""
    # Use the specified directory path
    if not data_dir:
        data_dir = "/Users/krishnayadav/Documents/test_projects/polkassembly-ai-v2/polkassembly-ai/data/onchain_data"
    
    logger.info(f"Storing onchain data in: {data_dir}")
    
    for network in SupportedNetworks:
        logger.info(f"network is {network}")
        try:
            logger.info(f"Starting data fetch for {network.value}...")
            
            # Initialize fetcher with specified data directory
            fetcher = PolkassemblyDataFetcher(network=network.value, data_dir=data_dir)
            
            # Fetch and save all data
            fetcher.fetch_and_save_all_data(max_items_per_type=max_items_per_type)
            
            logger.info(f"Completed data fetch for {network.value}")
            
        except Exception as e:
            logger.error(f"Error processing network {network.value}: {e}")
            continue

if __name__ == "__main__":
    # Fetch ALL available votes for polkadot in memory-efficient batches
    # max_items=None: Fetch ALL available votes (no limit)
    # save_batch_size=1000: Save every 1000 votes to manage memory
    fetch_votes_data(network="polkadot", max_items=None, save_batch_size=1000) 
    
    # Fetch data for all networks and proposal types
    # fetch_onchain_data(max_items_per_type=10)  # Adjust as needed