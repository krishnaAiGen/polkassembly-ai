import os
from typing import Dict, List, Union, Tuple
import aiohttp
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_high_scoring_results(response: Dict, min_score: float = 0.7) -> Tuple[str, List[Dict]]:
    """
    Parse Tavily API response and return answer and 2-4 high-scoring sources in structured format.
    
    Args:
        response: Raw response from Tavily API
        min_score: Minimum score threshold (default: 0.7)
    
    Returns:
        Tuple containing (answer, list of source dictionaries with title, url, source_type, similarity_score)
    """
    if not response or "results" not in response:
        return "", []
    
    # Get the AI-generated answer
    answer = response.get("answer", "")
    
    # Filter results by score and structure as dictionaries
    sources = []
    for result in response.get("results", []):
        score = result.get("score", 0)
        if score >= min_score:
            url = result.get("url", "")
            title = result.get("title", "")
            
            if url:  # Only add if URL exists
                source = {
                    'title': title or url,  # Use URL as fallback title if title is empty
                    'url': url,
                    'source_type': 'web_search',
                    'similarity_score': score
                }
                sources.append(source)
    
    # Limit to 2-4 sources
    if len(sources) < 2:
        return answer, sources  # Return all if less than 2
    elif len(sources) > 4:
        return answer, sources[:4]  # Return top 4 if more than 4
    else:
        return answer, sources  # Return all if between 2-4

async def search_tavily(query: str, min_score: float = 0.7) -> Tuple[str, List[Dict]]:
    """
    Perform an asynchronous search using Tavily API.
    
    Args:
        query: Search query string
        min_score: Minimum score threshold for filtering results
        
    Returns:
        Tuple containing (answer, list of source dictionaries with title, url, source_type, similarity_score)
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY not found in environment variables")
        return "", []
    
    # Get timeout from environment variable
    timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
    
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }
    
    try:
        # Create timeout object for aiohttp
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                raw_results = await response.json()
                
                # Parse and return answer and structured sources
                return parse_high_scoring_results(raw_results, min_score)
                
    except aiohttp.ClientError as e:
        print(f"Error making request: {e}")
        return "", []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "", []

# Example usage with asyncio
# async def main():
#     answer, sources = await search_tavily("What is Polkassembly?")
#     print("Answer:", answer)
#     print("\nSource URLs:", sources)

# if __name__ == "__main__":
#     asyncio.run(main())