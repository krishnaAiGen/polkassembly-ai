import requests
import json
from typing import List, Dict, Any
import time
from utils.embeddings import EmbeddingManager
from utils.data_loader import DataLoader
from enum import Enum
from rag.config import Config

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


class PolkassemblyDataProcessor:
    def __init__(self, network: str = "polkadot"):
        self.base_url = f"https://{network}.polkassembly.io/api/v2"
        self.headers = {
            'Content-Type': 'application/json',
        }
        
    def fetch_posts(self, proposal_type: ProposalType = ProposalType.REFERENDUM_V2, origin_type: OriginType = OriginType.ROOT, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Fetch posts from Polkassembly API"""
        url = f"{self.base_url}/{proposal_type.value}"
        
        params = {
            # 'limit': limit,
            'offset': offset,
            'origin_type': origin_type.value
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            print(response.json())
            print(response.json().get('items', []))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts: {e}")
            return []
    
    # def fetch_proposals(self, network: str = "polkadot", limit: int = 100, offset: int = 0) -> List[Dict]:
    #     """Fetch proposals from Polkassembly API"""
    #     url = f"{self.base_url}/proposals"
        
    #     params = {
    #         'network': network,
    #         'limit': limit,
    #         'offset': offset
    #     }
        
    #     try:
    #         response = requests.get(url, params=params, headers=self.headers)
    #         response.raise_for_status()
    #         return response.json().get('proposals', [])
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error fetching proposals: {e}")
    #         return []
    
    # def fetch_all_data(self, network: str = "polkadot", max_items: int = 1000) -> Dict[str, List]:
        """Fetch all posts and proposals with pagination"""
        all_posts = []
        all_proposals = []
        
        # Fetch posts
        offset = 0
        limit = 100
        
        print("Fetching posts...")
        while len(all_posts) < max_items:
            posts = self.fetch_posts(network, limit, offset)
            if not posts:
                break
            all_posts.extend(posts)
            offset += limit
            time.sleep(0.1)  # Rate limiting
            print(f"Fetched {len(all_posts)} posts so far...")
        
        # Fetch proposals
        offset = 0
        print("Fetching proposals...")
        while len(all_proposals) < max_items:
            proposals = self.fetch_proposals(network, limit, offset)
            if not proposals:
                break
            all_proposals.extend(proposals)
            offset += limit
            time.sleep(0.1)  # Rate limiting
            print(f"Fetched {len(all_proposals)} proposals so far...")
        
        return {
            'posts': all_posts[:max_items],
            'proposals': all_proposals[:max_items]
        }
    
def fetch_onchain_data(max_items: int = 5000):

    for network in SupportedNetworks:
        # Initialize processor
        processor = PolkassemblyDataProcessor(network=network.value)
        embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            embedding_model=Config.OPENAI_EMBEDDING_MODEL,
            chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            collection_name=Config.CHROMA_COLLECTION_NAME
        )

        # Fetch data
        print(f"Fetching data from Polkassembly for {network.value}...")
    
        for proposal_type in ProposalType:
            if proposal_type == ProposalType.REFERENDUM_V2:
                for origin_type in OriginType:
                    data = processor.fetch_posts(proposal_type=proposal_type, origin_type=origin_type, limit=max_items)
                    print(f"Fetched {len(data)} items for {network.value} {proposal_type.value} {origin_type.value}")
                    print(data)
                    success = embedding_manager.add_onchain_data_to_collection(data)
                    print(f"Successfully added {success} items to the collection")
            else:
                data = processor.fetch_posts(proposal_type=proposal_type, limit=max_items)
                success = embedding_manager.add_onchain_data_to_collection(data)

        print(f"Fetched {len(data)} items for {network.value}")
        print(f"Successfully added {success} items to the collection")