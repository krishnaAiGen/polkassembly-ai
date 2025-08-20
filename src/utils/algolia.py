import os
import asyncio
from algoliasearch.search.client import SearchClient
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PolkassemblySearch:
    def __init__(self):
        """
        Initialize Algolia search client with environment variables
        Make sure to set ALGOLIA_APP_ID and ALGOLIA_SEARCH_API_KEY
        """
        self.app_id = os.getenv('ALGOLIA_APP_ID')
        self.search_api_key = os.getenv('ALGOLIA_SEARCH_API_KEY')
        self.index_name = 'polkassembly_v2_posts'
        
        self.searchable_attributes = [
            'index',
            'title',
            'proposer',
            'proposalType',
            'parsedContent'
        ]
        
        if not self.app_id or not self.search_api_key:
            raise ValueError("Missing Algolia credentials. Please set ALGOLIA_APP_ID and ALGOLIA_SEARCH_API_KEY environment variables.")
        
        self.client = SearchClient(self.app_id, self.search_api_key)

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def configure_index_settings(self):
        """
        Configure the Algolia index settings with the desired searchable attributes and ranking
        
        This method should be called once to set up the index properly
        """
        try:
            index = self.client.init_index(self.index_name)
            settings = {
                "searchableAttributes": self.searchable_attributes,
                "ranking": ["typo", "geo", "words", "filters", "proximity", "attribute", "exact", "custom"]
            }
            await index.set_settings(settings)
            print(f"Successfully configured index {self.index_name} with searchable attributes: {self.searchable_attributes}")
            return True
        except Exception as e:
            print(f"Error configuring index settings: {str(e)}")
            return False

    async def close(self):
        """Close the client session properly"""
        await self.client.close()

    async def search_recent_posts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for posts and return top 5 most recent results
        
        Args:
            query (str): Search keyword/query
            limit (int): Number of results to return (default: 5)
            
        Returns:
            List[Dict[str, Any]]: List of search results with post data
        """
        try:
            search_params = {
                "requests": [
                    {
                        "indexName": self.index_name,
                        "query": query,
                        "hitsPerPage": limit,
                        "attributesToRetrieve": [
                            'objectID',
                            'title',
                            'proposer',
                            'index',
                            'proposalType',
                            'createdAtTimestamp',
                            'updatedAtTimestamp',
                            'parsedContent',
                            'network',
                            'origin',
                            'tags'
                        ]
                    }
                ]
            }
            
            print(f"Searching for '{query}' in index '{self.index_name}' with app_id: {self.app_id}")
            response = await self.client.search(search_method_params=search_params)
            
            
            try:
                results_list = response.results
                
                if results_list and len(results_list) > 0:
                    first_result = results_list[0]
                    
                    if hasattr(first_result, 'model_dump'):
                        result_dict = first_result.model_dump()
                        hits = result_dict.get('hits', [])
                    elif hasattr(first_result, 'actual_instance'):
                        actual = first_result.actual_instance
                        if hasattr(actual, 'model_dump'):
                            actual_dict = actual.model_dump()
                            hits = actual_dict.get('hits', [])
                        else:
                            hits = []
                    else:
                        hits = []
                else:
                    hits = []
            except Exception as e:
                print(f"Error accessing hits: {str(e)}")
                hits = []
                
            formatted_results = []
            for hit in hits:
                formatted_post = {
                    'objectID': hit.get('objectID'),
                    'title': hit.get('title'),
                    'proposer': hit.get('proposer'),
                    'index': hit.get('index'),
                    'proposalType': hit.get('proposalType'),
                    'createdAtTimestamp': hit.get('createdAtTimestamp'),
                    'updatedAtTimestamp': hit.get('updatedAtTimestamp'),
                    'parsedContent': hit.get('parsedContent'),
                    'network': hit.get('network'),
                    'origin': hit.get('origin'),
                    'tags': hit.get('tags', []),
                    'content_preview': self._get_content_preview(hit.get('parsedContent', '')),
                    'highlighted_title': hit.get('_highlightResult', {}).get('title', {}).get('value', hit.get('title')),
                }
                formatted_results.append(formatted_post)
            formatted_results = sorted(formatted_results, 
                                     key=lambda x: x.get('createdAtTimestamp', 0), 
                                     reverse=True)
            return formatted_results
            
        except Exception as e:
            print(f"Error searching Algolia: {str(e)}")
            return []

    async def search_with_filters(self, query: str, proposal_type: Optional[str] = None, 
                                network: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search with additional filters
        
        Args:
            query (str): Search keyword/query
            proposal_type (str, optional): Filter by proposal type (e.g., "ReferendumV2")
            network (str, optional): Filter by network
            limit (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Filtered search results
        """
        try:
            filters = []
            if proposal_type:
                filters.append(f'proposalType:"{proposal_type}"')
            if network:
                filters.append(f'network:"{network}"')
            
            search_request = {
                "indexName": self.index_name,
                "query": query,
                "hitsPerPage": limit,
                "attributesToRetrieve": [
                    'objectID',
                    'title', 
                    'proposer',
                    'index',
                    'proposalType',
                    'createdAtTimestamp',
                    'updatedAtTimestamp',
                    'parsedContent',
                    'network',
                    'origin',
                    'tags'
                ]
            }
            
            if filters:
                search_request['filters'] = ' AND '.join(filters)
            
            search_params = {"requests": [search_request]}
            
            response = await self.client.search(search_method_params=search_params)
            
            try:
                results_list = response.results
                if results_list and len(results_list) > 0:
                    first_result = results_list[0]
                    
                    if hasattr(first_result, 'model_dump'):
                        result_dict = first_result.model_dump()
                        hits = result_dict.get('hits', [])
                    elif hasattr(first_result, 'actual_instance'):
                        actual = first_result.actual_instance
                        if hasattr(actual, 'model_dump'):
                            actual_dict = actual.model_dump()
                            hits = actual_dict.get('hits', [])
                        else:
                            hits = []
                    else:
                        hits = []
                else:
                    hits = []
            except Exception as e:
                print(f"Error accessing hits in search_with_filters: {str(e)}")
                hits = []
            
            return self._format_results(hits)
            
        except Exception as e:
            print(f"Error searching with filters: {str(e)}")
            return []

    async def get_post_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific post by its objectID
        
        Args:
            object_id (str): The objectID of the post
            
        Returns:
            Optional[Dict[str, Any]]: Post data or None if not found
        """
        try:
            response = await self.client.get_object(
                index_name=self.index_name,
                object_id=object_id
            )
            return self._format_single_result(response)
        except Exception as e:
            print(f"Error getting post by ID {object_id}: {str(e)}")
            return None

    async def search_by_tags(self, query: str, tags: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search posts that contain specific tags
        
        Args:
            query (str): Search keyword/query
            tags (List[str]): List of tags to filter by
            limit (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with specified tags
        """
        try:
            tag_filters = [f'tags:"{tag}"' for tag in tags]
            filters_string = ' OR '.join(tag_filters) if len(tag_filters) > 1 else tag_filters[0] if tag_filters else ""
            
            search_request = {
                "indexName": self.index_name,
                "query": query,
                "hitsPerPage": limit,
                "attributesToRetrieve": [
                    'objectID', 'title', 'proposer', 'index', 'proposalType',
                    'createdAtTimestamp', 'updatedAtTimestamp', 'parsedContent', 'network', 'origin', 'tags'
                ]
            }
            
            if filters_string:
                search_request['filters'] = filters_string
            
            search_params = {"requests": [search_request]}
            response = await self.client.search(search_method_params=search_params)
            
            try:
                results_list = response.results
                if results_list and len(results_list) > 0:
                    first_result = results_list[0]
                    
                    if hasattr(first_result, 'model_dump'):
                        result_dict = first_result.model_dump()
                        hits = result_dict.get('hits', [])
                    elif hasattr(first_result, 'actual_instance'):
                        actual = first_result.actual_instance
                        if hasattr(actual, 'model_dump'):
                            actual_dict = actual.model_dump()
                            hits = actual_dict.get('hits', [])
                        else:
                            hits = []
                    else:
                        hits = []
                else:
                    hits = []
            except Exception as e:
                print(f"Error accessing hits in search_by_tags: {str(e)}")
                hits = []
            
            return self._format_results(hits)
            
        except Exception as e:
            print(f"Error searching by tags: {str(e)}")
            return []

    def _format_results(self, hits: List[Dict]) -> List[Dict[str, Any]]:
        """Format search results"""
        formatted_results = []
        for hit in hits:
            formatted_results.append(self._format_single_result(hit))
        return formatted_results

    def _format_single_result(self, hit) -> Dict[str, Any]:
        """Format a single search result"""
       
        hit_dict = {}
        if hasattr(hit, 'model_dump'):
            hit_dict = hit.model_dump()
        elif hasattr(hit, 'get'):
            hit_dict = hit
        else:
           
            try:
                hit_dict = dict(hit)
            except (TypeError, ValueError):
                print(f"Warning: Could not convert hit of type {type(hit).__name__} to dictionary")
                return {}
        
       
        result = {}
        
        if 'object_id' in hit_dict and not hit_dict.get('objectID'):
            result['objectID'] = hit_dict['object_id']
        
        print(f"Debug: hit_dict has updatedAtTimestamp: {'updatedAtTimestamp' in hit_dict}")
        if 'updatedAtTimestamp' in hit_dict:
            print(f"Debug: updatedAtTimestamp value: {hit_dict['updatedAtTimestamp']}")
        
        for key, value in hit_dict.items():
            if key != 'object_id' or 'objectID' not in result:  
                result[key] = value
                
        print(f"Debug: result has updatedAtTimestamp after copy: {'updatedAtTimestamp' in result}")
                
        result['formatted_date'] = self._format_timestamp(hit_dict.get('createdAtTimestamp'))
        result['updated_date'] = self._format_timestamp(hit_dict.get('updatedAtTimestamp'))
        
        result['content_preview'] = self._get_content_preview(hit_dict.get('parsedContent', ''), preserve_markdown=True)
        
        highlight_result = hit_dict.get('highlight_result', {}) or hit_dict.get('_highlightResult', {}) or {}
        if highlight_result and 'title' in highlight_result and 'value' in highlight_result['title']:
            result['highlighted_title'] = highlight_result['title']['value']
        else:
            result['highlighted_title'] = hit_dict.get('title', '')
            
        return result

    def _get_content_preview(self, content: str, max_length: int = 200, preserve_markdown: bool = True) -> str:
        """
        Get a preview of the content
        
        Args:
            content (str): The content to preview
            max_length (int): Maximum length of the preview
            preserve_markdown (bool): Whether to preserve markdown formatting
            
        Returns:
            str: The content preview
        """
        if not content:
            return ""
        
        preview = content
        
        if not preserve_markdown:
            preview = preview.replace('#', '').replace('**', '').replace('*', '')
        
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview.strip()

    def _format_timestamp(self, timestamp: Optional[int]) -> str:
        """Format timestamp to readable date"""
        if not timestamp:
            return "Unknown date"
        
        try:
           
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return "Invalid date"



async def search_posts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Quick search function with proper session management
    
    Args:
        query (str): Search keyword
        limit (int): Number of results (default: 5)
        
    Returns:
        List[Dict[str, Any]]: Search results
    """
    async with PolkassemblySearch() as searcher:
        return await searcher.search_recent_posts(query, limit)


async def search_by_type(query: str, proposal_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search with proposal type filter
    
    Args:
        query (str): Search keyword
        proposal_type (str): Type of proposal to filter by
        limit (int): Number of results
        
    Returns:
        List[Dict[str, Any]]: Filtered search results
    """
    async with PolkassemblySearch() as searcher:
        return await searcher.search_with_filters(query, proposal_type, None, limit)


async def search_by_network(query: str, network: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search with network filter
    
    Args:
        query (str): Search keyword
        network (str): Network to filter by
        limit (int): Number of results
        
    Returns:
        List[Dict[str, Any]]: Network filtered search results
    """
    async with PolkassemblySearch() as searcher:
        return await searcher.search_with_filters(query, None, network, limit)



def search_posts_sync(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Synchronous version of search_posts"""
    return asyncio.run(search_posts(query, limit))


def search_by_type_sync(query: str, proposal_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Synchronous version of search_by_type"""
    return asyncio.run(search_by_type(query, proposal_type, limit))


def search_by_network_sync(query: str, network: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Synchronous version of search_by_network"""
    return asyncio.run(search_by_network(query, network, limit))


def configure_index_settings_sync() -> bool:
    """Synchronous version of configure_index_settings"""
    async def _configure():
        async with PolkassemblySearch() as searcher:
            return await searcher.configure_index_settings()
    return asyncio.run(_configure())



if __name__ == "__main__":
    async def main():
        """Example usage of the search functions"""
        try:            
            results = await search_posts("clarys", 5)
                                 
            print(f"\nFound {len(results)} results:")
            for i, post in enumerate(results, 1):
                print(f"\n{i}. {post.get('title', 'Untitled')}")
                print(f"   Type: {post.get('proposalType', 'Unknown')}")
                print(f"   Network: {post.get('network', 'Unknown')}")
                print(f"   Index: {post.get('index', 'Unknown')}")
                
                if 'formatted_date' in post:
                    print(f"   Created: {post['formatted_date']}")
                elif 'createdAtTimestamp' in post and post['createdAtTimestamp']:
                    print(f"   Created: {datetime.fromtimestamp(post['createdAtTimestamp']).strftime('%Y-%m-%d %H:%M:%S') if post['createdAtTimestamp'] else 'Unknown'}")
                else:
                    print(f"   Created: Unknown")
                
                if 'updated_date' in post and post['updated_date'] != "Unknown date":
                    print(f"   Updated: {post['updated_date']}")
                elif 'updatedAtTimestamp' in post and post.get('updatedAtTimestamp'):
                    try:
                        updated_time = datetime.fromtimestamp(post['updatedAtTimestamp'])
                        print(f"   Updated: {updated_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        print(f"   Error formatting updatedAtTimestamp: {e}")
                
                print(f"   Proposer: {post.get('proposer', 'Unknown')}")
                print(f"   Origin: {post.get('origin', 'Unknown')}")
                print(f"   Tags: {', '.join(post.get('tags', [])) if post.get('tags') else 'None'}")
                if post.get('content_preview'):
                    print(f"   Preview: {post['content_preview']}")
                    
        except Exception as e:
            print(f"Error in example: {str(e)}")
    
    
    asyncio.run(main())