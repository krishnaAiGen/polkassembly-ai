import requests
import json
import os
from typing import List, Dict, Any, Tuple
import time
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportedNetworks(Enum):
    POLKADOT = "polkadot"
    KUSAMA = "kusama"

class PolkassemblyDataProcessor:
    """Process data from Polkassembly API"""
    
    def __init__(self, network: str = "polkadot"):
        self.network = network
        self.base_url = "https://api.polkassembly.io/api/v1"
        
        # Proposal types to fetch
        self.proposal_types = [
            "Democracy",
            "TechCommitteeProposal",
            "TreasuryProposal",
            "Referendum",
            "CouncilMotion",
            "Tip",
            "Bounty",
            "ChildBounty",
            "DemocracyProposal",
            "ReferendumV2",
            "FellowshipReferendum"
        ]
        
        # ReferendumV2 origins (only for Polkadot network)
        self.referendum_v2_origins = [
            "Root",
            "WhitelistedCaller",
            "StakingAdmin",
            "Treasurer",
            "LeaseAdmin",
            "FellowshipAdmin",
            "GeneralAdmin",
            "AuctionAdmin",
            "ReferendumCanceller",
            "ReferendumKiller",
            "SmallTipper",
            "BigTipper",
            "SmallSpender",
            "MediumSpender",
            "BigSpender",
            "WishForChange",
            "FastGeneralAdmin",
            "Candidates",
            "Members",
            "Proficients",
            "Fellows",
            "SeniorFellows",
            "Experts",
            "SeniorExperts",
            "Masters",
            "SeniorMasters",
            "GrandMasters"
        ]
    
    def fetch_proposal_data(self, proposal_type: str, origin: str = None) -> Dict[str, Any]:
        """Fetch proposal data from Polkassembly API"""
        import requests
        import time
        
        # Construct URL
        if proposal_type == "ReferendumV2" and origin:
            url = f"{self.base_url}/listing/{proposal_type.lower()}/{origin}?network={self.network}&listingLimit=10"
        else:
            url = f"{self.base_url}/listing/{proposal_type.lower()}?network={self.network}&listingLimit=10"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Add delay to respect rate limits
            time.sleep(0.5)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {proposal_type} data: {e}")
            return {"items": [], "totalCount": 0}
    
    def fetch_all_proposal_data(self, max_items: int = 1000) -> Dict[str, Any]:
        """Fetch all proposal data from Polkassembly API"""
        all_data = {"items": [], "totalCount": 0}
        total_fetched = 0
        
        logger.info(f"Fetching data for {self.network} network...")
        
        for proposal_type in self.proposal_types:
            if total_fetched >= max_items:
                break
                
            logger.info(f"Fetching {proposal_type} data...")
            
            if proposal_type == "ReferendumV2" and self.network == "polkadot":
                # Handle ReferendumV2 with different origins
                for origin in self.referendum_v2_origins:
                    if total_fetched >= max_items:
                        break
                        
                    data = self.fetch_proposal_data(proposal_type, origin)
                    if data["items"]:
                        all_data["items"].extend(data["items"])
                        all_data["totalCount"] += data["totalCount"]
                        total_fetched += len(data["items"])
                        logger.info(f"  {origin}: {len(data['items'])} items")
            else:
                data = self.fetch_proposal_data(proposal_type)
                if data["items"]:
                    all_data["items"].extend(data["items"])
                    all_data["totalCount"] += data["totalCount"]
                    total_fetched += len(data["items"])
                    logger.info(f"  {proposal_type}: {len(data['items'])} items")
        
        logger.info(f"Total fetched: {total_fetched} items")
        return all_data

def fetch_onchain_data(network: str = "polkadot", max_items: int = 1000) -> Dict[str, Any]:
    """Fetch onchain data from Polkassembly API"""
    processor = PolkassemblyDataProcessor(network)
    return processor.fetch_all_proposal_data(max_items)

# Legacy function for backward compatibility
def get_onchain_data_status(use_multi_collection: bool = False) -> Dict[str, Any]:
    """Legacy function - returns empty status"""
    return {
        "ready": False,
        "data": {"exists": False, "info": {"total_files": 0}},
        "embeddings": {"exists": False, "info": {"total_chunks": 0}},
        "multi_collection": use_multi_collection
    }