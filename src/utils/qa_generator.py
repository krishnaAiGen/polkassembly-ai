import openai
import logging
import json
from typing import List, Dict, Any, Optional
import re
from .web_search import search_tavily
from dotenv import load_dotenv
import os
import numpy as np
from .gemini import GeminiClient
import time
import sys
from datetime import datetime

from .mem0_memory import get_memory_manager, add_user_query, add_assistant_response
# Content guardrails now handled by Bedrock guardrails in the API endpoint
from .slack_bot import SlackBot

# Load environment variables
load_dotenv()

# Helper function to print model usage in green
def print_model_usage(model_name: str, purpose: str):
    """Print model usage information in green color"""
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'
    print(f"{GREEN}{BOLD}🤖 Using {model_name} for {purpose}{END}")

# Get Gemini timeout from environment
GEMINI_TIMEOUT = float(os.getenv('GEMINI_TIMEOUT', '30'))

# Import ask_question from the texttosql module
from ..texttosql.query_api import ask_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class QAGenerator:
    """Generate answers using OpenAI based on retrieved document chunks"""
    
    def __init__(self, 
                 openai_api_key: str,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000,
                 enable_web_search: bool = True,
                 web_search_context_size: str = "high",
                 enable_memory: bool = True):
        
        self.openai_api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_web_search = enable_web_search
        self.web_search_context_size = web_search_context_size
        self.enable_memory = enable_memory
        
        # Timeout configuration
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
        
        # Initialize OpenAI client with timeout
        self.client = openai.OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
        
        # Initialize Gemini client (optional) with timeout
        try:
            self.gemini_client = GeminiClient(timeout=GEMINI_TIMEOUT)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.warning(f"Gemini client initialization failed: {e}")
            logger.info("Continuing without Gemini client (OpenAI only mode)")
            self.gemini_client = None
        
        # Content guardrails now handled by Bedrock guardrails in the API endpoint
        logger.info("Content moderation will be handled by Bedrock guardrails")
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager() if enable_memory else None
        if self.memory_manager and self.memory_manager.enabled:
            logger.info("Memory functionality enabled")
        else:
            logger.info("Memory functionality disabled")
        
        # Initialize Slack bot for error notifications
        try:
            self.slack_bot = SlackBot()
            logger.info("Slack bot initialized for error notifications")
        except Exception as e:
            logger.warning(f"Slack bot initialization failed: {e}")
            logger.info("Continuing without Slack notifications")
            self.slack_bot = None
    
    def send_error_to_slack(self, query: str, error: str, error_source: str = "Klara") -> None:
        """Send error notification to Slack channel"""
        if not self.slack_bot:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            error_context = {
                "query": query,
                "timestamp": timestamp,
                "error_source": error_source,
                "error_details": str(error)
            }
            
            self.slack_bot.post_error_to_slack(
                "Query processing failed", 
                context=error_context
            )
            logger.info("Error notification sent to Slack")
        except Exception as slack_error:
            logger.error(f"Failed to send error notification to Slack: {slack_error}")
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
        """
        Create context string from retrieved chunks
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Format chunk with source information
            source_info = ""
            metadata = chunk.get('metadata', {})
            
            if metadata.get('title'):
                source_info += f"Title: {metadata['title']}\n"
            if metadata.get('url'):
                source_info += f"URL: {metadata['url']}\n"
            if metadata.get('source'):
                source_info += f"Source: {metadata['source']}\n"
            
            chunk_text = f"--- Document {i+1} ---\n{source_info}\nContent:\n{chunk['content']}\n\n"
            
            # Check if adding this chunk would exceed the max length
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return ''.join(context_parts)
    
    async def generate_answer_with_web_search(self, query: str, user_id: str = "default_user", conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate answer using OpenAI with web search-like knowledge
        
        Args:
            query: User's question
            user_id: User identifier for memory operations
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Using web search fallback for query: '{query[:50]}...'")
            
            # Get memory context and add query to memory if memory is enabled
            memory_context = ""
            if self.memory_manager and self.memory_manager.enabled:
                memory_context = self.memory_manager.get_memory_context(query, user_id)
                self.memory_manager.add_user_query(query, user_id)
            
            # Create a comprehensive system prompt for web search-like responses
            system_prompt = self._get_default_system_prompt()
            
            # Create user prompt with memory context and conversation history
            user_prompt_parts = []
            
            # Add conversation history if available
            if conversation_history:
                user_prompt_parts.append(f"Conversation History: {conversation_history}")
            
            if memory_context:
                user_prompt_parts.append(f"Previous conversation context:\n{memory_context}")
            
            user_prompt_parts.append(f"Current Question about Polkadot: {query}")
            user_prompt = "\n\n".join(user_prompt_parts)
            
            if os.getenv("WEB_SEARCH"):
                try:
                    #do the web search first
                    answer, sources = await search_tavily(query)
                except:
                    #fall back to openAI
                    logger.info("Web search failed, falling back to OpenAI")
                    response = self.client.chat.completions.create(
                        model="gpt-4o" if "gpt-4" in self.model or self.model == "gpt-3.5-turbo" else self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,  # Lower temperature for more factual responses
                        max_tokens=self.max_tokens
                    )
                    answer = response.choices[0].message.content
                    sources = []  # Initialize empty sources for fallback
            else:
                #no web search
                response = self.client.chat.completions.create(
                    model="gpt-4o" if "gpt-4" in self.model or self.model == "gpt-3.5-turbo" else self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more factual responses
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
                sources = []
            
            # Clean any example.com URLs from the response
            answer = self.clean_example_urls(answer)
            
            # Apply content guardrails but preserve markdown formatting
            # answer = self.guardrails.sanitize_response(answer)
            
            # Add assistant response to memory if memory is enabled
            if self.memory_manager and self.memory_manager.enabled:
                self.memory_manager.add_assistant_response(answer, user_id)
            
            # Create web search-style sources
            if len(sources) > 0:
                sources = sources[:np.random.randint(1, 4)]
            else:
                sources = [
                    {
                        'title': 'Polkadot Knowledge Base',
                        'url': 'https://polkadot.network',
                        'source_type': 'web_search',
                        'similarity_score': 0.9
                    },
                    {
                        'title': 'Polkadot Wiki',
                        'url': 'https://wiki.polkadot.network',
                        'source_type': 'web_search',
                        'similarity_score': 0.9
                    },
                    {
                        'title': 'Polkadot Documentation',
                        'url': 'https://docs.polkadot.network',
                        'source_type': 'web_search',
                        'similarity_score': 0.9
                    }
                ]
            
            # Generate follow-up questions for web search results
            follow_up_questions = self._get_fallback_follow_ups(query)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 0.8,  # High confidence for knowledge-based responses
                'follow_up_questions': follow_up_questions,
                'context_used': True,
                'model_used': 'gpt-4o',
                'chunks_used': 0,
                'search_method': 'web_search'
            }
            
        except Exception as e:
            logger.error(f"Error with web search fallback: {e}")
            return {
                'answer': "I encountered an error while searching for information. Please try again or rephrase your question.",
                'sources': [
                    {
                        'title': 'Polkadot Network',
                        'url': 'https://polkadot.network',
                        'source_type': 'web_search',
                        'similarity_score': 1.0
                    }
                ],
                'confidence': 0.0,
                'follow_up_questions': self._get_fallback_follow_ups(query),
                'context_used': False,
                'error': str(e),
                'search_method': 'web_search_failed'
            }
    
    def _is_greeting_message(self, query: str) -> bool:
        """
        Check if the query is a greeting or introduction message
        
        Args:
            query: User's question
            
        Returns:
            True if it's a greeting message
        """
        greeting_keywords = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'what\'s up', 'whats up', 'wassup', 'howdy',
            'intro', 'introduction', 'who are you', 'what are you',
            'what do you do', 'what is this', 'help', 'start'
        ]
        
        query_lower = query.lower().strip()
        
        # Check for exact matches only (no startswith to avoid false positives like "highest" matching "help")
        for keyword in greeting_keywords:
            if query_lower == keyword:
                return True
        
        # Check for short queries that might be greetings (but exclude data queries)
        if len(query_lower.split()) <= 3:
            for keyword in greeting_keywords:
                if keyword in query_lower:
                    # Check if it's actually a data query by looking for data-related indicators
                    data_indicators = ['number', 'count', 'total', 'sum', 'average', 'highest', 'lowest', 
                                       'vote', 'votes', 'voter', 'proposal', 'referendum', 'treasury',
                                       'month', 'year', 'date', 'time', 'day', 'week']
                    has_data_indicator = any(indicator in query_lower for indicator in data_indicators)
                    
                    # Only treat as greeting if it doesn't contain data indicators
                    if not has_data_indicator:
                        return True
        
        return False
    
    def _get_polkassembly_introduction(self) -> Dict[str, Any]:
        """
        Generate introduction response about Polkassembly
        
        Returns:
            Dictionary with introduction answer and metadata
        """
        introduction = """Hello! I'm **Klara** 👋 – your AI-powered governance assistant for **Polkadot** and **Kusama**!

I'm here to help you explore the governance ecosystem through **Polkassembly**, making it easy to query on-chain data, analyze proposals, and understand the voting process—all in natural language.

**What I can help you with:**

🗳 **Governance Data** - Query proposals, referenda, bounties, and treasury activities

📊 **Voting Analysis** - Track voter behavior, delegation, and voting power

💰 **Treasury Insights** - Explore funding proposals and beneficiary data

🧭 **Platform Guidance** - Learn how to use Polkassembly features and OpenGov

**Who I'm built for:**

- Community members exploring proposals
- Delegates analyzing voting patterns  
- Builders tracking treasury activities
- Researchers studying governance trends

**Try asking me things like:**

- "Show all active referenda on Polkadot"
- "Who voted on referendum 472?"
- "List treasury proposals above 100k DOT"
- "How does conviction voting work?"

**Useful Links:**

- **Klara Guide**: [klara.polkassembly.io/guide](https://klara.polkassembly.io/guide) - If you want detailed guidance on how to use Klara, follow this doc
- **Polkadot Governance**: [polkadot.polkassembly.io](https://polkadot.polkassembly.io)
- **Kusama Governance**: [kusama.polkassembly.io](https://kusama.polkassembly.io)
- **Documentation**: [docs.polkassembly.io](https://docs.polkassembly.io)"""

        sources = [
            {
                'title': 'Polkassembly Main Platform',
                'url': 'https://polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Polkadot Governance on Polkassembly',
                'url': 'https://polkadot.polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Kusama Governance on Polkassembly',
                'url': 'https://kusama.polkassembly.io',
                'source_type': 'platform',
                'similarity_score': 1.0
            },
            {
                'title': 'Polkassembly Documentation',
                'url': 'https://docs.polkassembly.io',
                'source_type': 'documentation',
                'similarity_score': 1.0
            }
        ]

        # Generate greeting-specific follow-up questions
        greeting_follow_ups = [
            "How does Polkadot governance work?",
            "What are parachains and how do they work?",
            "How can I start staking DOT tokens?"
        ]
        
        return {
            'answer': introduction,
            'sources': sources,
            'confidence': 1.0,
            'follow_up_questions': greeting_follow_ups,
            'context_used': True,
            'model_used': 'polkassembly_intro',
            'chunks_used': 0,
            'search_method': 'greeting_response'
        }

    def remove_double_asterisks(self, text):
        return text.replace("**", "").replace("-", "")
    
    def clean_example_urls(self, text):
        """
        Remove any lines containing links that don't contain the polkassembly S3 bucket URL.
        Only allows links from https://polkassembly-ai.s3.us-east-1.amazonaws.com
        """
        import re
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Pattern to match image markdown: ![alt text](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        for line in lines:
            should_include_line = True
            
            # Check if line contains image markdown
            image_matches = re.findall(image_pattern, line)
            
            if image_matches:
                for alt_text, url in image_matches:
                    # Only keep links that contain our S3 bucket URL
                    if 'https://polkassembly-ai.s3.us-east-1.amazonaws.com' not in url:
                        should_include_line = False
                        logger.info(f"Removed image line with invalid URL: {line.strip()}")
                        break
            
            if should_include_line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    

    
    def _fallback_to_static_response(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Fallback response when dynamic query processing fails
        """
        return {
            'answer': "I encountered an issue fetching the specific proposal data you requested. Please try rephrasing your question or check if the proposal ID is correct. You can also try asking about general Polkadot governance topics.",
            'sources': [
                {
                    'title': 'Polkadot Governance',
                    'url': 'https://polkadot.polkassembly.io',
                    'source_type': 'platform',
                    'similarity_score': 0.8
                }
            ],
            'confidence': 0.3,
            'context_used': False,
            'model_used': self.model,
            'chunks_used': 0,
            'search_method': 'dynamic_fallback',
            'error': 'Failed to fetch proposal data'
        }

    def analyze_query_with_memory(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Analyze query with conversation history to add memory/context awareness"""
        try:
            # Early return if no context needed
            if not conversation_history or not self.gemini_client:
                return query
            
            # Optimize: Only use recent conversation history (last 5-10 messages)
            # This saves tokens and focuses on relevant context
            max_history_length = 5
            recent_history = conversation_history[-max_history_length:] if len(conversation_history) > max_history_length else conversation_history
            
            # Convert conversation history to serializable format
            serializable_history = []
            for msg in recent_history:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    serializable_history.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict):
                    serializable_history.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", str(msg))
                    })
                else:
                    serializable_history.append({
                        "role": "user",
                        "content": str(msg)
                    })
            
            # Improved prompt with better structure and examples
            analysis_prompt = f"""You are a query context analyzer. Your job is to rewrite incomplete or contextual queries into complete, standalone queries.

CONVERSATION HISTORY:
{self._format_conversation_history(serializable_history)}

CURRENT USER QUERY: "{query}"

INSTRUCTIONS:
1. If the current query is complete and standalone → return it unchanged
2. If the query references previous context (e.g., "what about June?", "show recent ones", "their titles too") → rewrite to be complete
3. Preserve the user's intent and style
4. Keep technical terms and column names consistent with previous queries

EXAMPLES:

Example 1:
Previous: "Give me total number of referendums in July 2025"
Current: "what about June?"
Output: "Give me total number of referendums in June 2025"

Example 2:
Previous: "Show top 10 proposals by vote count"
Current: "include their titles too"
Output: "Show top 10 proposals with their titles by vote count"

Example 3:
Previous: "List all treasury proposals"
Current: "filter for amount > 1000"
Output: "List all treasury proposals with amount > 1000"

Example 4:
Current: "Show me all active referendums"
Output: "Show me all active referendums" (unchanged - already complete)

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{"analyzed_query": "your rewritten query here"}}

No explanations, no markdown, just the JSON."""

            # Get analysis from Gemini with retry logic
            gemini_response = self._get_gemini_response_with_retry(analysis_prompt)
            
            # Parse response with better error handling
            analyzed_query = self._parse_gemini_response(gemini_response, query)
            
            # Validation: ensure analyzed query is not empty or just whitespace
            if not analyzed_query or analyzed_query.strip() == "":
                logger.warning("Analyzed query is empty, returning original")
                return query
            
            # Log only if query was actually modified
            if analyzed_query.lower().strip() != query.lower().strip():
                logger.info(f"Query modified: '{query}' → '{analyzed_query}'")
            
            return analyzed_query
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}", exc_info=True)
            return query

    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history in a readable way"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for i, msg in enumerate(history, 1):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"{i}. {role}: {content}")
        
        return "\n".join(formatted)

    def _get_gemini_response_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """Get response from Gemini with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.get_response(prompt)
                if response and response.strip():
                    return response
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Brief pause before retry
        
        raise Exception("All Gemini API attempts failed")

    def _parse_gemini_response(self, response: str, fallback_query: str) -> str:
        """Parse Gemini JSON response with multiple fallback strategies"""
        try:
            # Strategy 1: Direct JSON parse
            response_json = json.loads(response.strip())
            return response_json.get("analyzed_query", fallback_query)
        
        except json.JSONDecodeError:
            # Strategy 2: Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    response_json = json.loads(json_match.group(1))
                    return response_json.get("analyzed_query", fallback_query)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Find JSON object in text
            json_match = re.search(r'\{[^}]*"analyzed_query"[^}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    response_json = json.loads(json_match.group(0))
                    return response_json.get("analyzed_query", fallback_query)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Extract quoted text after "analyzed_query"
            quote_match = re.search(r'analyzed_query["\s:]+(["\'])(.*?)\1', response, re.DOTALL)
            if quote_match:
                return quote_match.group(2).strip()
            
            # Final fallback: log and return original
            logger.warning(f"Could not parse Gemini response: {response[:200]}")
            return fallback_query

    def route_query(self, query):
        """
        Route query to appropriate data source using Gemini AI analysis.
        Returns a dictionary with data_source and table (if applicable).
        """
        import json
        
        # Prompt for Gemini to analyze the query
        analysis_prompt = f"""
            Analyze this user query and determine which data source and table it should use:

            Query: "{query}"

            Data Sources:
            
            1. STATIC - For general educational/informational topics:
            - Governance/OpenGov concepts and explanations
            - Ambassador Programme information  
            - Parachains & AnV explanations
            - Hyperbridge, JAM definitions
            - Dashboard information (Frequency, People, Stellaswap)
            - Wiki pages, how-to guides, tutorials
            - General "what is" or "how does" questions without specific data requests

            2. ONCHAIN - For blockchain data queries. Choose the appropriate table:
            
            a) Table: "governance_data" - Use when query involves:
                - Proposal information (title, content, description, hash)
                - Proposal metadata (status, type, dates, network)
                - Proposer addresses and beneficiaries
                - Financial amounts (rewards, transfers, beneficiary amounts)
                - Proposal metrics (likes, dislikes, comments)
                - Voting decisions on proposals(Aye/Nay/Abstain)
                - Proposal filtering by:
                    * Index/ID numbers (e.g., "proposal 1679", "referendum 234")
                    * Dates (created, updated, before/after/between dates)
                    * Network (Kusama/Polkadot)
                    * Type (ReferendumV2, Bounty, Treasury, etc.)
                    * Status (Executed, Rejected, Deciding, etc.)
                    * Data source (subsquare/polkassembly)
                - Proposal counts, aggregations, or summaries
                - Keywords: "proposal", "referendum", "bounty", "treasury", "motion"
                
                Examples:
                - "Show me recent proposals"
                - "Find proposals with amount > 10000 DOT"
                - "What's the status of proposal 1679?"
                - "Count rejected proposals in 2024"
                - "Show treasury proposals by address X"
                - "List proposals about topic Y"

            b) Table: "voting_data" - Use when query involves:
                - Voter information and voter accounts
                - Vote delegation (delegated votes, delegation chains)
                - Voting power or locked amounts
                - Lock periods or conviction voting
                - Vote timestamps (created_at, removed_at)
                - Voter counts or voting statistics
                - Voting patterns or behavior analysis
                - Keywords: "voter", "vote", "voting", "delegat", "conviction", "lock period"
                
                Examples:
                - "How many voters participated in proposal 123?"
                - "Show me votes with >1000 DOT voting power"
                - "Count total voters in July 2024"
                - "Who voted Aye on referendum 456?"
                - "List delegated votes for proposal X"
                - "What was voter Y's decision on proposal Z?"
                - "Show me votes with 6x conviction"

            Decision Logic:
            - If query asks about VOTER behavior, decisions, or participation → "voting_data"
            - If query asks about PROPOSAL details, status, or metadata → "governance_data"
            - If query is conceptual/educational without specific data requests → "STATIC"
            - If query mentions both proposals AND voters, prioritize based on main intent:
            * Main focus on "how people voted" → "voting_data"
            * Main focus on "proposal details" → "governance_data"

            Respond ONLY with valid JSON in this exact format:
            {{
                "data_source": "STATIC" or "ONCHAIN",
                "table": "governance_data" or "voting_data" (only if data_source is ONCHAIN, otherwise null)
            }}
        """

        try:
            # Use Gemini to analyze the query
            if self.gemini_client:
                # Get the actual model name from the client
                model_name = getattr(self.gemini_client, 'model_name', 'Gemini')
                print_model_usage(f"{model_name}", "query routing")
                response = self.gemini_client.get_response(analysis_prompt)
                
                # Try to parse JSON response
                try:
                    result = json.loads(response.strip())
                    data_source = result.get('data_source', 'STATIC')
                    table = result.get('table', None)
                    
                    return {
                        'data_source': data_source,
                        'table': table if data_source == 'ONCHAIN' else None
                    }
                    
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract from response text
                    if "onchain" in response.lower():
                        # Try to determine table from response content
                        if "voting" in response.lower() or "voter" in response.lower():
                            table = "voting_data"
                        else:
                            table = "governance_data"
                        return {
                            'data_source': 'ONCHAIN',
                            'table': table
                        }
                    else:
                        return {
                            'data_source': 'STATIC',
                            'table': None
                        }
            else:
                # Fallback if no Gemini client
                return {
                    'data_source': 'STATIC',
                    'table': None
                }
                
        except Exception as e:
            logger.error(f"Error in route_query: {e}")
            # Default fallback
            return {
                'data_source': 'STATIC',
                'table': None
            }
        

    async def generate_answer(self, 
                       query: str, 
                       chunks: List[Dict[str, Any]], 
                       custom_prompt: Optional[str] = None,
                       user_id: str = "default_user",
                       conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved chunks, with web search fallback

        CRITICAL FORMATTING RULES:
        ✅ PROFESSIONAL FORMATTING REQUIREMENTS:
        - Use plain text without any markdown symbols (**, *, ##, -, etc.)
        - ALWAYS add line breaks between numbered steps
        - ALWAYS add line breaks between bullet points
        - Use numbered lists (1. 2. 3.) for step-by-step instructions with line breaks
        - Use simple bullet points without dashes or symbols
        - Write in clean, professional sentences
        - Use quotation marks for emphasis instead of bold/italic
        
        Args:
            query: User's question
            chunks: Retrieved document chunks
            custom_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Print user query in green color
            print(f"\033[92m📝 User Query: {query}\033[0m")
            
            # Check if this is a greeting message
            if self._is_greeting_message(query):
                logger.info("Detected greeting message, providing Polkassembly introduction")
                
                # Handle memory for greeting
                if self.memory_manager and self.memory_manager.enabled:
                    self.memory_manager.add_user_query(query, user_id)
                
                greeting_response = self._get_polkassembly_introduction()
                
                # Add greeting response to memory
                if self.memory_manager and self.memory_manager.enabled:
                    self.memory_manager.add_assistant_response(greeting_response['answer'], user_id)
                
                return greeting_response
            
            # Analyze query with conversation history first to get context-aware query
            analyzed_query = self.analyze_query_with_memory(query, conversation_history)
            logger.info(f"Original Query: {query}, Analyzed query: {analyzed_query}")
            
            # Route the analyzed query to determine data source
            logger.info(f"Routing analyzed query: '{analyzed_query[:50]}...'")
            try:
                route_result = self.route_query(analyzed_query)
                route_result_data_source = route_result.get('data_source')
                route_result_table = route_result.get('table')
                logger.info(f"Route result: {route_result_data_source}")
            except Exception as route_error:
                logger.error(f"Error in query routing: {route_error}")
                # Default to static processing if routing fails
                route_result_data_source = 'STATIC'
                route_result_table = None
                logger.info("Falling back to static data processing due to routing error")
            
            # Handle dynamic data source (ONCHAIN queries)
            if route_result_data_source == 'ONCHAIN':
                try:
                    # Use the ask_question function from query_api with conversation history
                    sql_result = ask_question(analyzed_query, conversation_history, route_result_table)
                    
                    # Format response to match expected structure
                    return {
                        'answer': sql_result.get('natural_response', 'No response available'),
                        'sources': [],  # SQL queries don't have traditional sources
                        'chunks_used': 0,
                        'search_method': 'sql_query',
                        'sql_query': sql_result.get('sql_queries', []),
                        'result_count': sql_result.get('result_count', 0),
                        'success': sql_result.get('success', False)
                    }
                except Exception as e:
                    logger.error(f"Error in SQL query processing: {e}")
                    # Return user-friendly error response instead of continuing to static processing
                    return {
                        'answer': "I'm sorry, I encountered an error processing your database query. Please try rephrasing your question or try again later.",
                        'sources': [],
                        'chunks_used': 0,
                        'search_method': 'sql_error_fallback',
                        'sql_query': [],
                        'result_count': 0,
                        'success': False,
                        'follow_up_questions': [
                            "How does Polkadot's governance system work?",
                            "What are the benefits of staking DOT tokens?",
                            "How do parachains connect to Polkadot?"
                        ]
                    }
            
            # Continue with static data processing (existing logic)
            # Note: analyzed_query is already available from the routing step above
            
            # Create context from chunks (increased length limit for large chunks)
            try:
                context = self.create_context_from_chunks(chunks, max_context_length=8000)
                print("context from chunks", context)
                print("context after strip", context.strip())
            except Exception as context_error:
                logger.error(f"Error creating context from chunks: {context_error}")
                # Return error response if context creation fails
                return {
                    'answer': "I'm sorry, I encountered an error processing your request. Please try again.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'search_method': 'context_error'
                }
            
            # Check if we have sufficient context
            has_sufficient_context = (
                context.strip() and 
                len(chunks) > 0 and 
                any(chunk.get('similarity_score', 0) > float(os.getenv("SIMILARITY_THRESHOLD")) for chunk in chunks)
            )

            # If no sufficient context, diagnose the specific issue
            if not has_sufficient_context:
                max_score = max([chunk.get('similarity_score', 0) for chunk in chunks]) if chunks else 0
                has_good_similarity = any(chunk.get('similarity_score', 0) > 0.6 for chunk in chunks)
                has_chunks = len(chunks) > 0
                has_content = bool(context.strip())
                
                # Diagnose the specific problem
                if not has_chunks:
                    reason = "No relevant chunks found"
                elif not has_good_similarity:
                    reason = f"Low similarity scores (max: {max_score:.3f} < 0.6)"
                elif not has_content:
                    reason = f"Retrieved chunks contain no usable content (similarity: {max_score:.3f})"
                else:
                    reason = "Unknown context issue"
                
                logger.info(f"Insufficient context: {reason}. Query is outside knowledge boundary.")
                
                # Generate helpful follow-up questions for better guidance
                follow_up_questions = [
                    "How does Polkadot's governance system work?",
                    "What are parachains and how do they connect to Polkadot?",
                    "How can I stake DOT tokens on Polkadot?",
                    "What is the difference between Polkadot and Kusama?"
                ]
                
                return {
                    'answer': "I couldn't find sufficient information to answer your question accurately. This query appears to be outside my knowledge boundary for Polkadot-related topics. Please try rephrasing your question or ask about a more specific Polkadot topic.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': self.model,
                    'chunks_used': len(chunks),
                    'follow_up_questions': follow_up_questions,
                    'search_method': 'insufficient_context',
                    'max_similarity_score': max_score,
                    'similarity_threshold': 0.6,
                    'failure_reason': reason
                }
            
            # If no context at all, return appropriate message
            if not context.strip():
                if self.enable_web_search:
                    # This shouldn't happen since we would have used web search above
                    return {
                        'answer': "I couldn't find relevant information in the Polkadot knowledge base to answer your question. Please try rephrasing your query or ask about a different topic related to Polkadot.",
                        'sources': [],
                        'confidence': 0.0,
                        'context_used': False,
                        'search_method': 'local_only'
                    }
                else:
                    return {
                        'answer': "I could not find sufficient information about the query. Please try rephrasing your query or ask about a different topic related to Polkaseembly.",
                        'sources': [],
                        'confidence': 0.0,
                        'context_used': False,
                        'search_method': 'local_only'
                    }
            
            # Get memory context if memory is enabled
            memory_context = ""
            try:
                if self.memory_manager and self.memory_manager.enabled:
                    memory_context = self.memory_manager.get_memory_context(analyzed_query, user_id)
                    # Add analyzed query to memory
                    self.memory_manager.add_user_query(analyzed_query, user_id)
            except Exception as memory_error:
                logger.warning(f"Error in memory operations: {memory_error}")
                # Continue without memory context
                memory_context = ""
            
            # Create system prompt
            try:
                system_prompt = custom_prompt or self._get_default_system_prompt()
            except Exception as system_prompt_error:
                logger.warning(f"Error creating system prompt: {system_prompt_error}")
                system_prompt = "You are a helpful AI assistant for Polkadot-related questions."
            
            # Create user prompt with context, memory, and analyzed query
            print("context before going to openAI prompt", context)
            try:
                user_prompt = self._create_user_prompt(analyzed_query, context, memory_context, conversation_history)
            except Exception as user_prompt_error:
                logger.error(f"Error creating user prompt: {user_prompt_error}")
                # Return error response if prompt creation fails
                return {
                    'answer': "I'm sorry, I encountered an error preparing your request. Please try again.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'search_method': 'prompt_error'
                }
            
            # Generate response using selected AI service (exclusive)
            answer = None
            openai_enabled = os.getenv("ENABLE_OPENAI", "").lower() == "true"
            gemini_enabled = os.getenv("ENABLE_GEMINI", "").lower() == "true"
            system_prompt = self._get_default_system_prompt()
            
            try:
                if openai_enabled:
                    # Use OpenAI exclusively
                    print_model_usage("GPT-3.5-turbo", "response generation (static data)")
                    logger.info("Using OpenAI for response generation")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    answer = response.choices[0].message.content
                    logger.info("OpenAI response received successfully")
                
                elif gemini_enabled and self.gemini_client:
                    # Use Gemini exclusively  
                    model_name = getattr(self.gemini_client, 'model_name', 'Gemini')
                    print_model_usage(f"{model_name}", "response generation (static data)")
                    logger.info("Using Gemini for response generation")
                    answer = self.gemini_client.get_response(system_prompt + "\n\n" + user_prompt)
                    logger.info("Gemini response received successfully")
                    
                else:
                    # Fallback to OpenAI if no service is enabled
                    print_model_usage("GPT-3.5-turbo", "response generation fallback (static data)")
                    logger.warning("No AI service explicitly enabled, falling back to OpenAI")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    answer = response.choices[0].message.content
                    logger.info("OpenAI fallback response received successfully")
            except Exception as llm_error:
                logger.error(f"Error in LLM response generation: {llm_error}")
                # Return a user-friendly error response
                return {
                    'answer': "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment.",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'model_used': 'error',
                    'chunks_used': len(chunks),
                    'follow_up_questions': [
                        "How does Polkadot's governance system work?",
                        "What are the benefits of staking DOT tokens?",
                        "How do parachains connect to Polkadot?"
                    ],
                    'search_method': 'error_fallback'
                }
            print("--------answer without strip-------\n", answer)
            
            # Clean any example.com URLs from the response
            try:
                answer = self.clean_example_urls(answer)
                print("--------answer after cleaning example.com URLs-------\n", answer)
            except Exception as clean_error:
                logger.warning(f"Error cleaning URLs from response: {clean_error}")
                # Continue with original answer
            
            # Add assistant response to memory if memory is enabled
            try:
                if self.memory_manager and self.memory_manager.enabled:
                    self.memory_manager.add_assistant_response(answer, user_id)
            except Exception as memory_error:
                logger.warning(f"Error adding assistant response to memory: {memory_error}")
                # Continue without memory update
            
            # Extract sources from chunks
            try:
                sources = self._extract_sources(chunks)
            except Exception as source_error:
                logger.warning(f"Error extracting sources: {source_error}")
                sources = []
            
            # Estimate confidence based on number of chunks and similarity scores
            try:
                confidence = self._estimate_confidence(chunks)
            except Exception as confidence_error:
                logger.warning(f"Error estimating confidence: {confidence_error}")
                confidence = 0.5  # Default confidence
            
            # Generate follow-up questions
            try:
                follow_up_questions = self._generate_follow_up_questions(query, chunks, answer)
            except Exception as followup_error:
                logger.warning(f"Error generating follow-up questions: {followup_error}")
                follow_up_questions = [
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?",
                    "How do parachains connect to Polkadot?"
                ]
            
            result = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'follow_up_questions': follow_up_questions,
                'context_used': True,
                'model_used': self.model,
                'chunks_used': len(chunks),
                'search_method': 'local_knowledge'
            }
            
            logger.info(f"Generated answer for query: '{query[:50]}...' using {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Send error notification to Slack
            self.send_error_to_slack(query, str(e))
            
            # If local generation fails and web search is enabled, try web search as fallback
            if self.enable_web_search:
                logger.info("Local generation failed, trying web search fallback")
                return await self.generate_answer_with_web_search(query, user_id, conversation_history)
            
            # Return user-friendly error response
            return {
                'answer': "I'm having trouble processing your query. Please try again or rephrase your question in the next prompt.",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'model_used': self.model,
                'chunks_used': 0,
                'search_method': 'error_fallback',
                'error': True,
                'follow_up_questions': [
                    "How does Polkadot's governance system work?",
                    "What are the benefits of staking DOT tokens?", 
                    "How do parachains connect to Polkadot?"
                ]
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the QA system"""

        return """You are a helpful AI assistant specialized in answering questions about Polkadot, the blockchain platform. 

If conversation history is provided, consider it when answering. If the current question is a follow-up to previous queries, provide relevant context from previous responses. If the current question is standalone, answer independently.

You will be provided with context from Polkadot documentation and forum posts. Please follow these guidelines:

                ✅ PROFESSIONAL FORMATTING REQUIREMENTS:
                - ALWAYS add line breaks between numbered steps
                - ALWAYS add line breaks between bullet points
                - Use numbered lists (1. 2. 3.) for step-by-step instructions with line breaks
                - Use simple bullet points without dashes or symbols
                - Write in clean, professional sentences
                - Use quotation marks for emphasis instead of bold/italic
                - PRESERVE image markdown exactly as provided: ![Step Image](https://...) - keep this format unchanged
                - Include ALL images from the context in your response at the appropriate steps.
                - If there is subsqure in your output, the omit any link related to subsquare in your output and nudge polkassembly.

            ## STEP-BY-STEP FORMATTING (MANDATORY):

            When providing numbered instructions, ALWAYS format like this:

            To stake DOT tokens and earn rewards:

            1. Create and fund your wallet with DOT tokens
               ![Step Image](https://example.com/image1.jpg)

            2. Access a staking interface (Polkadot.js, Polkassembly, etc.)
               ![Step Image](https://example.com/image2.jpg)

            3. Select reliable validators based on commission and performance

            4. Nominate your chosen validators with your desired amount

            5. Monitor your staking rewards and validator performance

            CRITICAL: Always include any images that appear in the context - they are essential visual guides!

            ## BULLET POINT FORMATTING:

            When listing features or benefits:

            Key benefits include:

            - Passive income** through staking rewards (typically 10-15% APY)
            - Network security** participation and decentralization support  
            - Governance rights** to vote on network proposals

            ## WHAT TO AVOID:

            NEVER format like this (bad example):
            "### How to stake DOT: 1. Create wallet 2. Select validators 3. Nominate tokens"
            "Never use https://example.com/image1.jpg or https://example.com/image2.jpg, in your output, if there is such, then remove it in the output"
            
            **Remember**: Answer as if you have direct expertise about Polkadot. Start directly with content, use proper line breaks between steps, and provide helpful, accurate, and **professionally formatted** information."""
    
    def _create_user_prompt(self, query: str, context: str, memory_context: str = "", conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create the user prompt with query, context, memory, and conversation history"""
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append(f"Conversation History: {conversation_history}")
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append(f"Memory Context:\n{memory_context}")
        
        # Add document context
        if context:
            prompt_parts.append(f"Context Information:\n{context}")
        
        prompt_parts.append(f"Current Question: {query}")
        
        prompt_parts.append("Answer the question directly without mentioning the context, sources, documentation, or previous conversations. Do not start with phrases like \"Based on the provided context\", \"According to the documentation\", \"From the Polkadot Wiki\", \"From our previous conversation\", etc. Simply provide the answer as if you have direct knowledge of the topic.\n\nCRITICAL FORMATTING REQUIREMENTS:\n- NEVER start with headers (##, ###)\n- Start directly with answer content\n- ALWAYS add line breaks between numbered steps (1. step one [LINE BREAK] 2. step two [LINE BREAK])\n- ALWAYS add line breaks between bullet points\n- Use professional markdown formatting throughout\n- IMPORTANT: Include all images from the context using exact markdown format: ![Step Image](url)")
        
        return "\n\n".join(prompt_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract and filter source information from chunks"""
        sources = []
        seen_urls = set()
        
        # Sort chunks by similarity score
        sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        for chunk in sorted_chunks[:3]:
            metadata = chunk.get('metadata', {})
            
            source = {
                'title': metadata.get('title', 'Unknown Title'),
                'url': metadata.get('url', ''),
                'source_type': metadata.get('source', 'unknown'),
                'similarity_score': chunk.get('similarity_score', 0.0)
            }
            
            if source['url'] and source['url'] not in seen_urls:
                sources.append(source)
                seen_urls.add(source['url'])
            elif not source['url'] and len(sources) < 2:
                sources.append(source)
            
            if len(sources) >= 2 and any(s['url'] for s in sources):
                break
        
        # Return sources without additional filtering (Bedrock guardrails handle content moderation)
        return sources
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate confidence based on chunks and similarity scores"""
        if not chunks:
            return 0.0
        
        # Base confidence on average similarity score and number of chunks
        similarity_scores = [chunk.get('similarity_score', 0.0) for chunk in chunks]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Adjust confidence based on number of chunks (more chunks = higher confidence, up to a point)
        chunk_bonus = min(len(chunks) * 0.1, 0.3)
        
        confidence = min(avg_similarity + chunk_bonus, 1.0)
        return round(confidence, 2)
    
    def generate_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """Generate a summary of the retrieved chunks"""
        try:
            if not chunks:
                return "No relevant information found."
            
            # Create a condensed context
            context = self.create_context_from_chunks(chunks, max_context_length=2000)
            
            summary_prompt = """Please provide a brief summary of the following Polkadot-related information using proper markdown formatting.

IMPORTANT FORMATTING RULES:
- DO NOT start with headers (##, ###)
- Start directly with the summary content
- Use **bold** for key terms, *italics* for technical concepts
- Add line breaks between bullet points if used
- Keep it concise and professional

{context}

Summary:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": summary_prompt.format(context=context)}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Clean any example.com URLs from the summary
            summary = self.clean_example_urls(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary."

    def _generate_follow_up_questions(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> List[str]:
        """
        Generate 2-3 follow-up questions based on the query, context, and answer
        
        Args:
            query: Original user query
            chunks: Retrieved chunks that were used
            answer: Generated answer
            
        Returns:
            List of follow-up questions
        """
        try:
            # Extract key topics from chunks
            topics = set()
            for chunk in chunks[:3]:  # Only use top 3 chunks for topics
                metadata = chunk.get('metadata', {})
                title = metadata.get('title', '')
                content = chunk.get('content', '')
                
                # Extract potential topics from titles and content
                if title:
                    topics.add(title.split(' - ')[0])  # Get main topic before dash
                
                # Look for common Polkadot-related terms
                polkadot_terms = [
                    'parachain', 'relay chain', 'governance', 'staking', 'validator',
                    'nominator', 'treasury', 'referendum', 'proposal', 'council',
                    'DOT', 'KSM', 'kusama', 'democracy', 'xcm', 'bridge',
                    'consensus', 'runtime', 'substrate', 'crowdloan', 'auction'
                ]
                
                content_lower = content.lower()
                for term in polkadot_terms:
                    if term.lower() in content_lower and term.lower() not in query.lower():
                        topics.add(term)
            
            # Create context for follow-up generation
            topics_list = list(topics)[:5]  # Limit to 5 most relevant topics
            topics_str = ', '.join(topics_list) if topics_list else 'Polkadot ecosystem'
            
            # Generate follow-up questions using OpenAI
            follow_up_prompt = f"""Based on this query: "{query}"
And these related topics: {topics_str}

Generate exactly 3 short, relevant follow-up questions that a user might want to ask next. Each question should:
- Be directly related to the original query or mentioned topics
- Be concise (under 15 words)
- Explore different aspects or dive deeper
- Be practical and useful

Format: Return only the 3 questions, one per line, without numbers or bullets."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            follow_up_text = response.choices[0].message.content.strip()
            
            # Clean any example.com URLs from the follow-up questions
            follow_up_text = self.clean_example_urls(follow_up_text)
            
            follow_up_questions = [q.strip() for q in follow_up_text.split('\n') if q.strip()]
            
            # Ensure we have 2-3 questions, fallback if needed
            if len(follow_up_questions) < 2:
                follow_up_questions = self._get_fallback_follow_ups(query)
            elif len(follow_up_questions) > 3:
                follow_up_questions = follow_up_questions[:3]
            
            return follow_up_questions
            
        except Exception as e:
            logger.warning(f"Error generating follow-up questions: {e}")
            return self._get_fallback_follow_ups(query)
    
    def _get_helpful_follow_ups(self) -> List[str]:
        """Get helpful follow-up questions that guide users to appropriate topics"""
        helpful_questions = [
            "How does Polkadot's governance system work?",
            "What are the benefits of staking DOT tokens?",
            "How do parachains communicate with each other?",
            "What makes Polkadot different from other blockchains?",
            "How can I participate in Polkadot governance?",
            "What are the risks and rewards of DOT staking?",
        ]
        
        import random
        return random.sample(helpful_questions, 3)

    def _get_fallback_follow_ups(self, query: str) -> List[str]:
        """
        Get fallback follow-up questions based on common Polkadot topics
        
        Args:
            query: Original user query
            
        Returns:
            List of fallback follow-up questions
        """
        query_lower = query.lower()
        
        # Topic-based follow-ups
        if any(term in query_lower for term in ['governance', 'vote', 'proposal', 'referendum']):
            return [
                "How do I participate in Polkadot governance?",
                "What are the different governance tracks?",
                "How does voting power work in OpenGov?"
            ]
        elif any(term in query_lower for term in ['staking', 'validator', 'nominator']):
            return [
                "What are the risks of staking DOT?",
                "How do I choose good validators?",
                "What is the minimum amount needed to stake?"
            ]
        elif any(term in query_lower for term in ['parachain', 'slot', 'auction']):
            return [
                "How do parachain auctions work?",
                "What is the difference between parachains and parathreads?",
                "How do parachains communicate with each other?"
            ]
        elif any(term in query_lower for term in ['dot', 'token', 'price', 'economics']):
            return [
                "What are the main uses of DOT token?",
                "How does DOT inflation work?",
                "Where can I buy and store DOT?"
            ]
        else:
            # General follow-ups
            return [
                "How does Polkadot differ from other blockchains?",
                "What are the main benefits of using Polkadot?",
                "How can I get started with Polkadot?"
            ]