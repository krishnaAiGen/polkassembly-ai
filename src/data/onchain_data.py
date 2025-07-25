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
    def __init__(self, network: str = "polkadot"):
        self.network = network
        self.base_url = f"https://{network}.polkassembly.io/api/v2"
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.data_dir = "data/dynamic_kusama_polka"
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_posts(self, proposal_type: ProposalType, origin_type: Optional[OriginType] = None, 
                   limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Fetch posts from Polkassembly API"""
        url = f"{self.base_url}/{proposal_type.value}"

        params = {
            'limit': limit,
            'offset': offset
        }
        
        if origin_type:
            params['origin_type'] = origin_type.value

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
        offset = 0
        limit = 10

        logger.info(f"Fetching {proposal_type.value} posts for {self.network}...")
        
        while len(all_posts) < max_items:
            response_data = self.fetch_posts(proposal_type, origin_type, limit, offset)
            
            if not response_data or 'items' not in response_data:
                break
                
            posts = response_data['items']
            if not posts:
                break
                
            all_posts.extend(posts)
            offset += limit
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

    def fetch_and_save_all_data(self, max_items_per_type: int = 1000):
        """Fetch all data types and save to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for proposal_type in ProposalType:
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

def fetch_onchain_data(max_items_per_type: int = 1000):
    """Main function to fetch onchain data for all supported networks"""
    for network in SupportedNetworks:
        try:
            logger.info(f"Starting data fetch for {network.value}...")
            
            # Initialize fetcher
            fetcher = PolkassemblyDataFetcher(network=network.value)
            
            # Fetch and save all data
            fetcher.fetch_and_save_all_data(max_items_per_type=max_items_per_type)
            
            logger.info(f"Completed data fetch for {network.value}")
            
        except Exception as e:
            logger.error(f"Error processing network {network.value}: {e}")
            continue

if __name__ == "__main__":
    # Fetch data for all networks and proposal types
    fetch_onchain_data(max_items_per_type=10)  # Adjust as needed