"""
Natural Language Query to SQL Converter and Executor

This module converts natural language queries to SQL, executes them against PostgreSQL,
and returns natural language responses using OpenAI API.
"""

import os
import json
import logging
import psycopg2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from contextlib import contextmanager
import pandas as pd

# Add tiktoken for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not available, using approximate token counting")

# Add Gemini client
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from gemini import GeminiClient
except ImportError:
    GeminiClient = None
    print("Warning: Gemini client not available")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Query2SQL:
    def __init__(self):
        """Initialize the Query2SQL converter with database and OpenAI connections"""
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DATABASE'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Validate database configuration
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Timeout configuration
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
        
        self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
        
        # Initialize Gemini client (optional fallback)
        self.gemini_client = None
        if GeminiClient is not None:
            try:
                self.gemini_client = GeminiClient(timeout=self.api_timeout)
                logger.info("Gemini client initialized successfully as fallback")
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")
                logger.info("Continuing without Gemini fallback (OpenAI only mode)")
        
        self.table_name = os.getenv('POSTGRES_TABLE_NAME', 'governance_data')
        
        # Load schema information
        self.schema_info = self._load_schema_info()
        self.table_schema = self._get_table_schema()
        
        logger.info(f"Initialized Query2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def _load_schema_info(self) -> Dict[str, Dict[str, str]]:
        """Load schema information from schema_info.json"""
        schema_path_str = os.getenv('POSTGRES_SCHEMA_PATH')
        if not schema_path_str:
            raise ValueError("POSTGRES_SCHEMA_PATH environment variable is required")
        
        schema_path = Path(schema_path_str)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema info file not found at {schema_path}")
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Check if the schema has a 'columns' key (new format) or is direct (old format)
            if 'columns' in schema_data:
                columns_data = schema_data['columns']
                logger.info(f"Loaded schema information (new format) from {schema_path}")
                return columns_data
            else:
                logger.info(f"Loaded schema information (old format) from {schema_path}")
                return schema_data
            
        except Exception as e:
            logger.error(f"Error loading schema info: {e}")
            raise
    
    def format_amount_by_asset_id(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format amounts based on assetId rules:
        - If assetId is NaN/None: keep amount as is, it is DOT
        - If assetId is 1984: remove 6 zeros (divide by 1,000,000) USDT
        - If assetId is 1337: remove 6 zeros (divide by 1,000,000)  USDC
        - If assetId is 30: remove 3 zeros (divide by 1,000) DED
        """
        if not results:
            return results
        
        formatted_results = []
        
        for result in results:
            formatted_result = result.copy()
            
            # Check if this result has amount and assetId fields
            amount_field = None
            asset_id_field = None
            
            # Find amount and assetId fields (they might have different names)
            for key in result.keys():
                if 'amount' in key.lower() and 'beneficiaries' in key.lower():
                    amount_field = key
                elif 'assetid' in key.lower() and 'beneficiaries' in key.lower():
                    asset_id_field = key
            
            # If we found both fields, apply formatting
            if amount_field and asset_id_field:
                amount_value = result.get(amount_field)
                asset_id_value = result.get(asset_id_field)
                
                # Only format if amount exists and is not None
                if amount_value is not None and str(amount_value) not in ['', 'None', 'NaN']:
                    try:
                        # Convert amount to float for processing
                        amount_float = float(amount_value)
                        
                        # Apply formatting based on assetId
                        if asset_id_value is not None and str(asset_id_value) not in ['', 'None', 'NaN']:
                            asset_id_int = int(float(asset_id_value))
                            
                            if asset_id_int == 1984:
                                # Remove 6 zeros (divide by 1,000,000) - USDT
                                formatted_amount = amount_float / 1_000_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDT"
                                
                            elif asset_id_int == 1337:
                                # Remove 6 zeros (divide by 1,000,000) - USDC
                                formatted_amount = amount_float / 1_000_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} USDC"
                                
                            elif asset_id_int == 30:
                                # Remove 3 zeros (divide by 1,000) - DED
                                formatted_amount = amount_float / 1_000
                                formatted_result[f"{amount_field}_formatted"] = f"{formatted_amount:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{formatted_amount:,.2f} DED"
                                
                            else:
                                # Unknown assetId, keep original
                                formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                                formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} (Asset ID: {asset_id_int})"
                        else:
                            # AssetId is NaN/None, keep original amount - DOT
                            formatted_result[f"{amount_field}_formatted"] = f"{amount_float:,.2f}"
                            formatted_result[f"{amount_field}_display"] = f"{amount_float:,.2f} DOT"
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not format amount {amount_value} with assetId {asset_id_value}: {e}")
                        # Keep original values if formatting fails
                        pass
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def add_proposal_links(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add proposal links to results based on proposal type, network, and index/id.
        Link generation rules:
        - ReferendumV2 + polkadot: https://polkadot.polkassembly.io/referenda/{proposal_id}
        - Discussion + polkadot: https://polkadot.polkassembly.io/post/{proposal_id}
        - ReferendumV2 + kusama: https://kusama.polkassembly.io/referenda/{proposal_id}
        - Discussion + kusama: https://kusama.polkassembly.io/post/{proposal_id}
        """
        if not results:
            return results
        
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Find relevant fields for link generation
            proposal_id = None
            network = None
            proposal_type = None
            title = None
            
            # Extract proposal ID (try multiple possible field names)
            for key in result.keys():
                if key.lower() in ['index', 'proposal_index', 'id', 'proposal_id']:
                    proposal_id = result.get(key)
                    break
            
            # Extract network (try multiple field names)
            for key in result.keys():
                if 'network' in key.lower():
                    network = result.get(key)
                    if network:
                        network = str(network).lower()
                    break
            
            # If network not found in results, default to polkadot (most common)
            if not network:
                network = 'polkadot'
                logger.debug(f"Network not found in result, defaulting to 'polkadot'")
            
            # Extract proposal type (try multiple field names)
            for key in result.keys():
                if 'proposal_type' in key.lower() or 'proposaltype' in key.lower() or key.lower() == 'type':
                    proposal_type = result.get(key)
                    break
            
            # If proposal type not found in results, default to ReferendumV2 (most common)
            if not proposal_type:
                proposal_type = 'ReferendumV2'
                logger.debug(f"Proposal type not found in result, defaulting to 'ReferendumV2'")
            
            # Extract title
            for key in result.keys():
                if key.lower() == 'title':
                    title = result.get(key)
                    break
            
            # Debug logging
            logger.debug(f"Link generation - ID: {proposal_id}, Network: {network}, Type: {proposal_type}, Title: {title}")
            
            # Generate link if we have the required information
            if proposal_id is not None and network and proposal_type:
                try:
                    # Clean and validate proposal_id
                    proposal_id_clean = str(proposal_id).strip()
                    if proposal_id_clean and proposal_id_clean != 'None' and proposal_id_clean != 'NaN':
                        
                        # Generate link based on type and network
                        link = None
                        
                        if network in ['polkadot'] and proposal_type in ['ReferendumV2']:
                            link = f"https://polkadot.polkassembly.io/referenda/{proposal_id_clean}"
                        elif network in ['polkadot'] and proposal_type in ['Discussion']:
                            link = f"https://polkadot.polkassembly.io/post/{proposal_id_clean}"
                        elif network in ['kusama'] and proposal_type in ['ReferendumV2']:
                            link = f"https://kusama.polkassembly.io/referenda/{proposal_id_clean}"
                        elif network in ['kusama'] and proposal_type in ['Discussion']:
                            link = f"https://kusama.polkassembly.io/post/{proposal_id_clean}"
                        
                        # Add link to result if generated
                        if link:
                            enhanced_result['proposal_link'] = link
                            
                            # Also create a display version with title if available
                            if title and str(title).strip() and str(title).strip() != 'None':
                                enhanced_result['proposal_link_display'] = f"[{title}]({link})"
                            else:
                                enhanced_result['proposal_link_display'] = f"[Proposal {proposal_id_clean}]({link})"
                            
                            logger.debug(f"Generated proposal link: {link}")
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not generate proposal link for ID {proposal_id}: {e}")
                    pass
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken or approximate counting
        """
        if tiktoken:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error with tiktoken: {e}, using approximate counting")
        
        # Approximate token counting (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def trim_prompt_to_fit_tokens(self, system_prompt: str, max_tokens: int = 8192, completion_tokens: int = 800, buffer_tokens: int = 200) -> str:
        """
        Trim the system prompt to fit within token limits
        
        Args:
            system_prompt: The full system prompt
            max_tokens: Maximum tokens allowed by the model (default: 8192 for GPT-4)
            completion_tokens: Tokens reserved for completion (default: 800)
            buffer_tokens: Safety buffer tokens (default: 200)
        
        Returns:
            Trimmed system prompt that fits within limits
        """
        # Calculate available tokens for the prompt
        available_tokens = max_tokens - completion_tokens - buffer_tokens
        
        # Count current tokens
        current_tokens = self.count_tokens(system_prompt)
        
        logger.info(f"Token analysis - Current: {current_tokens}, Available: {available_tokens}, Max: {max_tokens}")
        
        # If within limits, return as-is
        if current_tokens <= available_tokens:
            return system_prompt
        
        # Need to trim - calculate target length
        target_length = int(len(system_prompt) * (available_tokens / current_tokens))
        
        # Find good places to cut (preserve important sections)
        lines = system_prompt.split('\n')
        
        # Priority sections to keep (in order of importance)
        essential_sections = [
            'COLUMN SELECTION STRATEGY:',
            'EXAMPLE QUERIES:',
            'CRITICAL NULL VALUE HANDLING:',
            'NaN VALUE HANDLING:',
            'Very very Important Rule:'
        ]
        
        # Build trimmed prompt by keeping essential sections
        trimmed_lines = []
        current_length = 0
        
        # First pass: add essential sections
        for line in lines:
            if any(section in line for section in essential_sections):
                # Found essential section, add it and some context
                section_start = lines.index(line)
                # Add this section and next 10 lines (or until next section)
                for i in range(section_start, min(section_start + 10, len(lines))):
                    if lines[i] not in trimmed_lines:
                        test_length = current_length + len(lines[i]) + 1
                        if test_length < target_length:
                            trimmed_lines.append(lines[i])
                            current_length = test_length
                        else:
                            break
        
        # Second pass: add other important lines if space allows
        for line in lines:
            if line not in trimmed_lines and current_length + len(line) + 1 < target_length:
                # Skip very long lines (likely examples that can be shortened)
                if len(line) < 200:
                    trimmed_lines.append(line)
                    current_length += len(line) + 1
        
        trimmed_prompt = '\n'.join(trimmed_lines)
        
        # Final token check
        final_tokens = self.count_tokens(trimmed_prompt)
        logger.info(f"Prompt trimmed: {current_tokens} -> {final_tokens} tokens (target: {available_tokens})")
        
        return trimmed_prompt
    
    def _get_table_schema(self) -> str:
        """Load schema directly from JSON file"""
        try:
            schema_path_str = os.getenv('POSTGRES_SCHEMA_PATH')
            if not schema_path_str:
                raise ValueError("POSTGRES_SCHEMA_PATH environment variable not set")
            
            schema_path = Path(schema_path_str)
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
            # Load the entire JSON file
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            # Return schema_data directly as string
            logger.info(f"Loaded schema directly from JSON file: {schema_path}")
            return str(schema_data)
            
        except Exception as e:
            logger.error(f"Error loading schema directly from JSON: {e}")
            # Fallback to old method if loading fails
            logger.warning("Falling back to old schema generation method")
            return self._get_table_schema_fallback()
    
    def _get_table_schema_fallback(self) -> str:
        """Fallback method to generate schema from loaded schema_info"""
        schema_parts = []
        schema_parts.append(f"Table: {self.table_name}")
        schema_parts.append("\nColumns:")
        
        for column, info in self.schema_info.items():
            # Handle both old and new schema formats
            if isinstance(info, dict):
                # New format: {"type": "TEXT", "description": "..."}
                data_type = info.get('type', info.get('data_type', 'unknown'))
                description = info.get('description', info.get('Description', 'No description'))
            else:
                # Fallback for unexpected format
                data_type = 'unknown'
                description = str(info)
            
            schema_parts.append(f"  - {column} ({data_type}): {description}")
        
        return "\n".join(schema_parts)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with timeout"""
        conn = None
        try:
            # Add timeout to database connection
            db_config_with_timeout = self.db_config.copy()
            db_config_with_timeout['connect_timeout'] = int(self.api_timeout)
            
            conn = psycopg2.connect(**db_config_with_timeout)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def generate_sql_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convert natural language query to SQL using OpenAI - can return multiple queries"""

        # print("Schema of table is", self.table_schema)

        try:

            
            system_prompt = f"""You are a PostgreSQL expert. Convert natural language queries into optimized SQL queries.

CONVERSATION CONTEXT:
Conversation history: {conversation_history}
- If current query is a follow-up: Generate SQL that builds upon or references previous context
- If current query is standalone: Generate SQL independently
- Use your judgment to determine query relationships


DATABASE SCHEMA:
{self.table_schema}


            CORE SQL GUIDELINES:
            1. Use ONLY existing columns from the schema above
            2. Table name: {self.table_name}
            3. Use proper PostgreSQL syntax with double quotes for column names
            4. Apply appropriate LIMIT clauses (typically 10 for lists, no limit for counts)

            DATA FILTERING RULES:
            5. Network filtering: Use 'source_network' column (values: 'polkadot', 'kusama')
            6. Proposal types: Use 'source_proposal_type' column  
            7. Proposal IDs: Use 'index' column
            8. Date filtering: Use DATE_TRUNC() for month/year, direct comparison for specific dates
            9. Text search: Use ILIKE for case-insensitive matching with % wildcards
            10. When you filter data by taking keywords from query itself. Some you can take from title, however see the
                param supported in the DATABASE SCHEMA and use the nearest matching param. 
                For example: 
                -can you show me some treasury proposals currently in voting
                -Don't use SELECT "title", "index", "onchaininfo_status", "createdat" FROM governance_data WHERE "source_proposal_type" ILIKE \'%treasury%\' AND "onchaininfo_status" = \'Voting\' LIMIT 10;
                -Don't use "onchaininfo_status" = \'Voting\' since Voting is not in params, use nearest which can be "onchaininfo_status" = \'Deciding\'
                -You can find all possible supported params in description of DATABSE SCHEMA.


            CRITICAL NULL VALUE HANDLING:
            10. Many columns contain NULL values - use IS NOT NULL when querying for specific values
            11. For amount queries (highest, lowest, etc.): ALWAYS add IS NOT NULL condition
            12. For date-based queries: Consider adding IS NOT NULL for 'createdat' if needed
            13. Key columns with NULLs: amounts, addresses, vote metrics, dates

            NaN VALUE HANDLING:
            14. Some columns also contain 'NaN' string values - use != 'NaN' condition along with IS NOT NULL
            15. For amount/numeric queries: Add both IS NOT NULL AND != 'NaN' conditions
            16. When ordering by numeric columns: Use CAST(column AS FLOAT) for proper numeric sorting
            17. Example: WHERE "amount" IS NOT NULL AND "amount" != 'NaN' ORDER BY CAST("amount" AS FLOAT) DESC
            
            MULTIPLE QUERIES STRATEGY:
            - If query asks for COUNT and EXAMPLES (like "how many proposals and name a few"), return 2 queries:
              Query 1: COUNT query to get the total number
              Query 2: SELECT query to get examples with details
            - If query asks only for count, return 1 COUNT query
            - If query asks only for examples/list, return 1 SELECT query
            - Return queries as a JSON array: ["query1", "query2"]
            
            COLUMN SELECTION STRATEGY:
            - For general queries: SELECT key columns like "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type", "content"
            - For searches: Focus on "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type", "content"
            - For FINANCIAL/AMOUNT queries: ALWAYS include "onchaininfo_beneficiaries_0_assetid" along with "onchaininfo_beneficiaries_0_amount". Both fields are must required at any cost.
            - Avoid SELECT * unless specifically needed - it causes long responses. Only use when somebody asks fro more info on proposals, referenda ID.
            - But, if somebody ask, proposals in voting then also use other attributes such as DecisionDepositPlaced, Submitted, ConfirmStarted, ConfirmAborted along with Deciding.
            
            EXAMPLE QUERIES:
            Single Query Examples:
             - "Show me recent proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type" FROM {self.table_name} ORDER BY "createdat" DESC LIMIT 10;
            - "Find Kusama proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type" FROM {self.table_name} WHERE "source_network" = 'kusama' LIMIT 10;
            - "What treasury proposals exist?" -> SELECT "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type" FROM {self.table_name} WHERE "source_proposal_type" ILIKE '%treasury%' LIMIT 10;
            - "Tell me about clarys proposal" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "content" ILIKE '%clarys%' OR "title" ILIKE '%clarys%' LIMIT 10;
            - "Tell me about subsquare proposal" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "content" ILIKE '%subsquare%' OR "title" ILIKE '%subsquare%' LIMIT 10;
            - "Give me the details of the proposal with id 123456" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "index" = 123456 LIMIT 10;
            - "Give me some recent proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat", "source_network", "source_proposal_type", "content" FROM {self.table_name} ORDER BY "createdat" DESC LIMIT 10;
            - "Give me proposals after 2024-01-01" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "createdat" > '2024-01-01' LIMIT 10;
            - "Give me proposals before 2024-01-01" -> SELECT "title", "index", "onchaininfo_status", "createdat" , "content" FROM {self.table_name} WHERE "createdat" < '2024-01-01' LIMIT 10;
            - "Give me proposals between dates" -> SELECT "title", "index", "onchaininfo_status", "createdat" , "content" FROM {self.table_name} WHERE "createdat" BETWEEN '2024-01-01' AND '2024-01-02' LIMIT 10;
            - "Count total proposals" -> SELECT COUNT(*) as total_proposals FROM {self.table_name};
            - "Show me proposal amounts" -> SELECT "title", "onchaininfo_beneficiaries_0_assetid", "index", "onchaininfo_beneficiaries_0_amount", "createdat" FROM {self.table_name} WHERE "onchaininfo_beneficiaries_0_amount" IS NOT NULL LIMIT 10;
            
            Multiple Query Examples:
            - "How many proposals in August 2025 and name a few?" -> ["SELECT COUNT(*) as total_count FROM {self.table_name} WHERE DATE_TRUNC('month', \"createdat\") = '2025-08-01';", "SELECT \"title\", \"index\", \"onchaininfo_status\", \"createdat\" FROM {self.table_name} WHERE DATE_TRUNC('month', \"createdat\") = '2025-08-01' LIMIT 10;"]
            - "How many Kusama proposals exist and show some examples?" -> ["SELECT COUNT(*) as kusama_count FROM {self.table_name} WHERE \"source_network\" = 'kusama';", "SELECT \"title\", \"index\", \"onchaininfo_status\", \"createdat\" FROM {self.table_name} WHERE \"source_network\" = 'kusama' LIMIT 10;"]
            
            Null Results
            - Some columns has NULL and NaN values and for some queries like 
            - tell me the proposal who had asked for highest amount in the month of august 2025, use NOT NULL and != NaN to get correct result. Do your own thinking and generate the query where NOT NULL and !=NaN is needed. Example:
                    SELECT
                        "title",
                        "index",
                        "onchaininfo_beneficiaries_0_assetid",
                        "onchaininfo_beneficiaries_0_amount",
                        "createdat"
                    FROM
                        governance_data
                    WHERE
                        DATE_TRUNC('month', "createdat") = '2025-08-01'
                        AND "onchaininfo_beneficiaries_0_amount" IS NOT NULL
                        AND "onchaininfo_beneficiaries_0_amount" != 'NaN'
                    ORDER BY
                        CAST("onchaininfo_beneficiaries_0_amount" AS FLOAT) DESC
                    LIMIT 1;

            - Columns with NULL/NaN values: ['publicuser_profiledetails_publicsociallinks_0_platform', 'history_1_title', 'linkedpost_indexorhash', 'tags_1_network', 'index', 'onchaininfo_votemetrics', 'hash', 'content', 'onchaininfo_beneficiaries_0_assetid', 'publicuser_addresses_3', 'userid', 'history_0_title', 'onchaininfo_prepareperiodendsat', 'history_2_createdat_seconds', 'tags_0_network', 'publicuser_addresses_2', 'topic', 'onchaininfo_proposer', 'history_2_content', 'poll', 'publicuser_profiledetails_title', 'publicuser_profiledetails_publicsociallinks_0_url', 'onchaininfo_index', 'history_1_createdat_nanoseconds', 'onchaininfo_beneficiaries_0_amount', 'history_1_content', 'onchaininfo_votemetrics_bareayes_value', 'publicuser_profilescore', 'onchaininfo_decisionperiodendsat', 'history_0_content', 'tags_0_lastusedat', 'tags_2_lastusedat', 'tags_2_value', 'id', 'tags_1_value', 'history_0_createdat_seconds', 'updatedat', 'onchaininfo_origin', 'publicuser_profiledetails_coverimage', 'onchaininfo_votemetrics_nay_count', 'onchaininfo_votemetrics_aye_count', 'createdat', 'history_2_title', 'onchaininfo_votemetrics_aye_value', 'publicuser_profiledetails_bio', 'history_0_createdat_nanoseconds', 'publicuser_profiledetails_image', 'linkedpost_proposaltype', 'onchaininfo_beneficiaries_0_address', 'publicuser_addresses_0', 'publicuser_rank', 'tags_1_lastusedat', 'tags_0_value', 'publicuser_addresses_1', 'onchaininfo_votemetrics_nay_value', 'onchaininfo_votemetrics_support_value', 'onchaininfo_hash', 'onchaininfo_reward', 'publicuser_id', 'onchaininfo_description', 'publicuser_addresses_4', 'history_1_createdat_seconds', 'onchaininfo_curator', 'history_2_createdat_nanoseconds', 'publicuser_createdat', 'publicuser_username', 'tags_2_network']

            Very very Important Rule:
            - For every query you generate, you must add a filter of source_proposal_type = 'ReferendumV2' unless, otherwise, specified that somebody needs info on ChildBounty, FellowshipReferendum and Bounty.
            
            Natural Language Query: {natural_query}
            
            SQL Query:
            """
            
            # Trim prompt to fit token limits
            system_prompt = self.trim_prompt_to_fit_tokens(system_prompt, max_tokens=8192, completion_tokens=800, buffer_tokens=200)
            
            # print(f"sql prompt, {system_prompt}")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert. Generate SQL queries based on the provided schema. For complex queries requiring both count and examples, return a JSON array of queries. For simple queries, return a JSON array with one query. Always return valid JSON format."},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean up the response
            response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
            
            try:
                # Try to parse as JSON array
                import json
                sql_queries = json.loads(response_content)
                
                # Ensure it's a list
                if isinstance(sql_queries, str):
                    sql_queries = [sql_queries]
                elif not isinstance(sql_queries, list):
                    sql_queries = [str(sql_queries)]
                    
                logger.info(f"Generated {len(sql_queries)} SQL queries: {sql_queries}")
                return sql_queries
                
            except json.JSONDecodeError:
                # Fallback: treat as single query
                logger.warning("Could not parse JSON, treating as single query")
                single_query = response_content.strip()
                logger.info(f"Generated single SQL query: {single_query}")
                return [single_query]
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise
    
    def execute_sql_queries(self, sql_queries: List[str]) -> List[Tuple[List[Dict[str, Any]], List[str]]]:
        """Execute multiple SQL queries against PostgreSQL database"""
        all_results = []
        
        try:
            with self.get_connection() as conn:
                for i, sql_query in enumerate(sql_queries):
                    logger.info(f"Executing query {i+1}/{len(sql_queries)}: {sql_query}")
                    
                    # Use pandas for easier data handling
                    df = pd.read_sql_query(sql_query, conn)
                    
                    # Convert DataFrame to list of dictionaries
                    results = df.to_dict('records')
                    columns = df.columns.tolist()
                    
                    all_results.append((results, columns))
                    logger.info(f"Query {i+1} executed successfully. Retrieved {len(results)} rows")
                
                return all_results
                
        except Exception as e:
            logger.error(f"Error executing SQL queries: {e}")
            logger.error(f"Queries: {sql_queries}")
            raise
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Execute single SQL query - wrapper for backwards compatibility"""
        results = self.execute_sql_queries([sql_query])
        return results[0]
    
    def generate_natural_response_multiple(self, natural_query: str, sql_queries: List[str], 
                                         all_results: List[Tuple[List[Dict[str, Any]], List[str]]], 
                                         conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from multiple query results - tries Gemini first, then OpenAI"""
        try:
        
            # Combine results from all queries
            combined_summary = f"Executed {len(sql_queries)} queries for: {natural_query}\n\n"
            
            for i, (sql_query, (results, columns)) in enumerate(zip(sql_queries, all_results)):
                result_count = len(results)
                combined_summary += f"Query {i+1}: {sql_query}\n"
                combined_summary += f"Results: {result_count} rows\n"
                
                # Add sample data from this query
                if results:
                    # Pass full data to AI model - let the AI decide how to summarize if needed
                    sample_results = results[:5]  # First 5 results
                    trimmed_results = []
                    
                    for result in sample_results:
                        trimmed_result = {}
                        for key, value in result.items():
                            # Only trim extremely long fields (over 2000 characters) to prevent token issues
                            if isinstance(value, str) and len(value) > 2000:
                                trimmed_result[key] = value[:2000] + "... [truncated]"
                            else:
                                trimmed_result[key] = value
                        trimmed_results.append(trimmed_result)
                    
                    # Add structured info
                    for j, result in enumerate(trimmed_results):
                        item_info = []
                        # Include ALL fields from SQL results - no filtering whatsoever
                        for key, value in result.items():
                            if value is not None and str(value) != 'None' and str(value).strip():
                                # Just clean up the key name for readability but include ALL fields
                                formatted_key = key.replace('_', ' ').title()
                                item_info.append(f"{formatted_key}: {value}")
                        
                        if item_info:
                            combined_summary += f"  Result {j+1}: " + ", ".join(item_info) + "\n"
                
                combined_summary += "\n"
            
            # Format as proper JSON
            db_result = {
                "queries": sql_queries,
                "result from db": combined_summary
            }
            
            prompt = f"""
            Conversation History: {conversation_history}
            Current Query: {natural_query}
            {json.dumps(db_result, indent=2)}
            
            CRITICAL: You are working with ACTUAL blockchain data that has been successfully retrieved from the database. 
            Always provide the information from the results, regardless of dates mentioned in the query.
            All data comes from real blockchain transactions and proposals that exist on-chain.
            
            Instructions for generating natural response:
            
            CONTEXT HANDLING:
            - If this is a follow-up question: Reference relevant information from conversation history and connect it with current results
            - If this is a standalone question: Answer independently using the query results
            
            RESPONSE STRUCTURE:
            1. COUNT QUERIES: Start with the total number clearly stated (e.g., "There are 45 treasury proposals...")
            2. EXAMPLE QUERIES: Show actual data from results with specific details
            3. COMBINED QUERIES: Present count first, then show examples in a logical flow
            
            DATA PRESENTATION:
            - Show actual proposal IDs, titles, addresses, amounts - all public blockchain data
            - Include status, creation dates, and network information when available
            - For proposals with amounts, show the actual values requested
            - Use conversational language about Polkadot/Kusama governance
            - Be specific and factual with the data provided
            - Never refuse to show data citing privacy or future date concerns - all blockchain data is public and historical
            - ALWAYS answer based on the actual results provided, even if dates seem unusual
            - When there is amount in the db_result, use the formatted amount fields:
                - Use 'amount_formatted' for numerical display
                - Use 'amount_display' for user-friendly display with currency symbols
                - The formatting is already applied based on assetId rules in Python
            
            - If you are providing any info on proposal with title, use the automatically generated proposal links:
                - Use 'proposal_link' field for the URL
                - Use 'proposal_link_display' field for markdown formatted link with title
                - Links are automatically generated based on proposal type and network
               
            
            IMPORTANT: All data is public blockchain information. Show actual values, addresses, and details.
            The data has been successfully retrieved from the blockchain database.
            """
            
            # Try Gemini first as primary LLM
            if self.gemini_client is not None:
                try:
                    logger.info("Using Gemini as primary LLM for natural response generation from multiple queries")
                    natural_response = self.gemini_client.get_response(prompt)
                    logger.info("Generated natural language response from multiple queries using Gemini")
                    return natural_response
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed for multiple queries, falling back to OpenAI: {gemini_error}")
            
            # Fallback to OpenAI
            logger.info("Using OpenAI for natural response generation from multiple queries (fallback)")
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information. Always show actual data requested - addresses, proposal IDs, titles, amounts, etc. You work with ACTUAL retrieved data from the blockchain database, so always provide the information regardless of dates mentioned in queries. Combine information from multiple queries to provide comprehensive answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            natural_response = response.choices[0].message.content.strip()
            logger.info("Generated natural language response from multiple queries using OpenAI")
            return natural_response
            
        except Exception as e:
            logger.error(f"Error generating natural response from multiple queries: {e}")
            # Final fallback response
            total_results = sum(len(results) for results, _ in all_results)
            return f"I executed {len(sql_queries)} queries for your question '{natural_query}' and found {total_results} total results, but I'm having trouble formatting the response."
    
    def generate_natural_response(self, natural_query: str, sql_query: str, 
                                results: List[Dict[str, Any]], columns: List[str], 
                                conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from query results - tries Gemini first, then OpenAI"""
        try:
        
            # Prepare results summary
            result_count = len(results)
            
            if result_count == 0:
                return f"I couldn't find any data matching your query: '{natural_query}'. The database might not contain the specific information you're looking for."
            
            # Limit results for OpenAI context (show first 5 rows max)
            sample_results = results[:10]
            
            # Pass full data to AI model - let the AI decide how to summarize if needed
            trimmed_results = []
            for result in sample_results:
                trimmed_result = {}
                for key, value in result.items():
                    # Only trim extremely long fields (over 2000 characters) to prevent token issues
                    if isinstance(value, str) and len(value) > 2000:
                        trimmed_result[key] = value[:2000] + "... [truncated]"
                    else:
                        trimmed_result[key] = value
                trimmed_results.append(trimmed_result)
            
            # Focus on key columns for summary
            key_columns = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['title', 'index', 'status', 'network', 'type', 'createdat', 'amount'])]
            
            # Create a summary of the data with trimmed content
            results_summary = {
                "total_rows": result_count,
                "key_columns": key_columns[:10],  # Limit columns shown
                "sample_data": trimmed_results,
                "showing": f"Showing {min(5, result_count)} of {result_count} results" if result_count > 5 else f"Showing all {result_count} results"
            }
            
            # Create a more concise summary for the prompt
            summary_text = f"Found {result_count} results. "
            
            # Create a structured summary of key data points
            if trimmed_results:
                summary_items = []
                for i, result in enumerate(trimmed_results[:10]):  # Show up to 10 examples
                    item_info = []
                    
                    # Include ALL fields from SQL results - no filtering whatsoever
                    for key, value in result.items():
                        if value is not None and str(value) != 'None' and str(value).strip():
                            # Just clean up the key name for readability but include ALL fields
                            formatted_key = key.replace('_', ' ').title()
                            item_info.append(f"{formatted_key}: {value}")
                    
                    if item_info:
                        summary_items.append(f"#{i+1}: " + ", ".join(item_info))
                    elif len(result) == 1:
                        # Handle single-column results (like just proposer addresses)
                        key = list(result.keys())[0]
                        value = result[key]
                        if value and value != 'None':
                            summary_items.append(f"#{i+1}: {key} = {value}")
                
                if summary_items:
                    summary_text += "\nExamples:\n" + "\n".join(summary_items)
            
            # Format as proper JSON
            db_result = {
                "query": sql_query,
                "result from db": summary_text
            }

            # print(f"\n\n db_result: {db_result}")
            
            prompt = f"""
            Current Query: {natural_query}
            {json.dumps(db_result, indent=2)}
            
            CRITICAL: You are working with ACTUAL blockchain data that has been successfully retrieved from the database. 
            Always provide the information from the results, regardless of dates mentioned in the query.
            All data comes from real blockchain transactions and proposals that exist on-chain.
            
            Instructions for generating natural response:
            
            CONTEXT HANDLING:
            - If this is a follow-up question: Reference relevant information from conversation history and connect it with current results
            - If this is a standalone question: Answer independently using the query results
            
            RESPONSE GUIDELINES:
            1. COUNT QUERIES: State the exact number clearly (e.g., "There are 23 proposals...")
            2. LIST QUERIES: Show actual data with specific details from up to 10 results
            3. DETAIL QUERIES: Include all relevant information like titles, IDs, status, dates, amounts
            4. ADDRESS QUERIES: Show actual blockchain addresses - these are public on-chain data
            5. AMOUNT QUERIES: Display exact values requested in proposals
            6. 20-300 is the ideal word output count. If the output will be too long, provide in summarization form instead of listing all the data.
            7. If you are providing any info on proposal with title, use the automatically generated proposal links:
                - Use 'proposal_link' field for the URL
                - Use 'proposal_link_display' field for markdown formatted link with title
                - Links are automatically generated based on proposal type and network
               
            
            DATA PRESENTATION:
            - Use conversational language about Polkadot/Kusama governance
            - Be specific and factual with all data provided
            - Show actual proposal IDs, titles, addresses, amounts - all public blockchain information
            - Include network (Polkadot/Kusama), status, and creation dates when available
            - Never refuse to show data citing privacy or future date concerns - all blockchain data is public and historical
            - ALWAYS answer based on the actual results provided, even if dates seem unusual
            - When there is amount in the db_result, use the formatted amount fields:
                - Use 'amount_formatted' for numerical display
                - Use 'amount_display' for user-friendly display with currency symbols
                - The formatting is already applied based on assetId rules in Python

            Focus on providing accurate, specific information from the query results. The data has been successfully retrieved from the blockchain database.
            """
            
            # Debug: Log the results being passed to Gemini
            logger.info(f"Results being passed to Gemini for natural response: {summary_text[:500]}...")
            # print(f"\n\n prompt: {prompt}")
            
            # Try Gemini first as primary LLM
            if self.gemini_client is not None:
                try:
                    logger.info("Using Gemini as primary LLM for natural response generation")
                    natural_response = self.gemini_client.get_response(prompt)
                    # logger.info(f"repsonse from gemini is: {natural_response}")
                    logger.info("Generated natural language response using Gemini")
                    return natural_response
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed, falling back to OpenAI: {gemini_error}")
            
            # Fallback to OpenAI
            logger.info("Using OpenAI for natural response generation (fallback)")
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information including addresses, proposal IDs, and transaction details. Always show the actual data requested - never refuse due to privacy concerns as this is all public information. Provide clear, helpful explanations with actual values, addresses, and details from the results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            natural_response = response.choices[0].message.content.strip()
            logger.info("Generated natural language response using OpenAI")
            return natural_response
            
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            # Final fallback response
            return f"I found {len(results)} results for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the database."
    
    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None, table: Optional[str] = None) -> Dict[str, Any]:
        """Main method to process a natural language query end-to-end"""
        try:
            logger.info(f"Processing query: {natural_query}")
            
            # Step 1: Generate SQL queries (can be multiple)
            sql_queries = self.generate_sql_query(natural_query, conversation_history)
            
            # Step 2: Execute SQL queries
            if len(sql_queries) == 1:
                # Single query - use existing flow for backwards compatibility
                results, columns = self.execute_sql_query(sql_queries[0])
                
                # Format amounts in results
                results = self.format_amount_by_asset_id(results)
                
                # Add proposal links to results
                results = self.add_proposal_links(results)
                # print(f"results after formatting amounts: {results}")
                
                # Step 3: Generate natural language response
                natural_response = self.generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0],
                    "sql_queries": sql_queries,
                    "result_count": len(results),
                    "results": results,
                    "columns": columns,
                    "natural_response": natural_response,
                    "success": True
                }
            else:
                # Multiple queries
                all_results = self.execute_sql_queries(sql_queries)
                
                # Format amounts and add proposal links in all results
                formatted_all_results = []
                for results, columns in all_results:
                    # Format amounts
                    formatted_results = self.format_amount_by_asset_id(results)
                    # Add proposal links
                    enhanced_results = self.add_proposal_links(formatted_results)
                    formatted_all_results.append((enhanced_results, columns))
                all_results = formatted_all_results
                
                # Step 3: Generate natural language response from multiple results
                natural_response = self.generate_natural_response_multiple(
                    natural_query, sql_queries, all_results, conversation_history
                )
                
                # Combine all results for response
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if columns not in combined_columns:
                        combined_columns.extend(columns)
                    total_result_count += len(results)
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),  # Combined for backwards compatibility
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": list(set(combined_columns)),
                    "all_results": all_results,  # Detailed results per query
                    "natural_response": natural_response,
                    "success": True
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": f"I'm sorry, I encountered an error processing your query: '{natural_query}'. Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    count = cur.fetchone()[0]
                    logger.info(f"Connection test successful. Table {self.table_name} has {count:,} rows")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def debug_schema(self):
        """Debug method to inspect loaded schema"""
        print(f"Schema info type: {type(self.schema_info)}")
        print(f"Schema info keys: {list(self.schema_info.keys())[:10]}")  # First 10 keys
        
        if self.schema_info:
            first_key = list(self.schema_info.keys())[0]
            first_value = self.schema_info[first_key]
            print(f"First item: {first_key} -> {first_value} (type: {type(first_value)})")
        
        print(f"\nTable schema preview:\n{self.table_schema[:500]}...")

class VoteQuery2SQL:
    def __init__(self):
        """Initialize the VoteQuery2SQL converter specifically for voting data"""
        
        # Database configuration
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST_PA'),
            'port': int(os.getenv('POSTGRES_PORT_PA', '5432')),
            'database': os.getenv('POSTGRES_DATABASE_PA'),
            'user': os.getenv('POSTGRES_USER_PA'),
            'password': os.getenv('POSTGRES_PASSWORD_PA')
        }
        
        # Validate database configuration
        required_vars = ['POSTGRES_HOST_PA', 'POSTGRES_PORT_PA', 'POSTGRES_USER_PA', 'POSTGRES_PASSWORD_PA']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Timeout configuration
        self.api_timeout = float(os.getenv('API_TIMEOUT', '10'))  # Default 10 seconds
        
        self.openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.api_timeout)
        
        # Initialize Gemini client (optional fallback)
        self.gemini_client = None
        if GeminiClient is not None:
            try:
                self.gemini_client = GeminiClient(timeout=self.api_timeout)
                logger.info("Gemini client initialized successfully as fallback")
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")
                logger.info("Continuing without Gemini fallback (OpenAI only mode)")
        
        self.table_name = 'flattened_conviction_votes'
        
        # Load schema information for voting data
        self.schema_info = self._load_vote_schema_info()
        self.table_schema = self._get_table_schema()
        
        logger.info(f"Initialized VoteQuery2SQL for table: {self.table_name}")
        logger.info(f"Loaded schema for {len(self.schema_info)} columns")
    
    def _load_vote_schema_info(self) -> Dict[str, Dict[str, str]]:
        """Load schema information from vote schema file"""
        schema_path_str = os.getenv('POSTGRES_SCHEMA_VOTE_PATH')
        if not schema_path_str:
            raise ValueError("POSTGRES_SCHEMA_VOTE_PATH environment variable is required")
        
        schema_path = Path(schema_path_str)
        if not schema_path.exists():
            raise FileNotFoundError(f"Vote schema info file not found at {schema_path}")
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Check if the schema has a 'columns' key (new format) or is direct (old format)
            if 'columns' in schema_data:
                columns_data = schema_data['columns']
                logger.info(f"Loaded vote schema information (new format) from {schema_path}")
                return columns_data
            else:
                # Old format - data is directly the columns
                logger.info(f"Loaded vote schema information (old format) from {schema_path}")
                return schema_data
                
        except Exception as e:
            logger.error(f"Error loading vote schema info: {e}")
            raise
    
    def _get_table_schema(self) -> str:
        """Generate table schema description for prompt"""
        if not self.schema_info:
            return "No schema information available"
        
        schema_lines = [f"Table: {self.table_name}"]
        schema_lines.append("Columns:")
        
        for column_name, column_info in self.schema_info.items():
            data_type = column_info.get('data_type', 'unknown')
            description = column_info.get('description', 'No description')
            schema_lines.append(f"  - {column_name} ({data_type}): {description}")
        
        return "\n".join(schema_lines)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"PostgreSQL version: {version}")
                    
                    # Test if voting_data table exists
                    cur.execute(f"SELECT to_regclass('{self.table_name}');")
                    table_exists = cur.fetchone()[0] is not None
                    if table_exists:
                        logger.info(f"Table {self.table_name} exists")
                    else:
                        logger.warning(f"Table {self.table_name} does not exist")
                    
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text using tiktoken or approximate counting
        """
        if tiktoken:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error with tiktoken: {e}, using approximate counting")
        
        # Approximate token counting (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def trim_prompt_to_fit_tokens(self, system_prompt: str, max_tokens: int = 8192, completion_tokens: int = 800, buffer_tokens: int = 200) -> str:
        """
        Trim the system prompt to fit within token limits
        """
        # Calculate available tokens for the prompt
        available_tokens = max_tokens - completion_tokens - buffer_tokens
        
        # Count current tokens
        current_tokens = self.count_tokens(system_prompt)
        
        logger.info(f"VoteQuery2SQL Token analysis - Current: {current_tokens}, Available: {available_tokens}, Max: {max_tokens}")
        
        # If within limits, return as-is
        if current_tokens <= available_tokens:
            return system_prompt
        
        # Need to trim - calculate target length
        target_length = int(len(system_prompt) * (available_tokens / current_tokens))
        
        # Simple trimming for voting queries (keep essential parts)
        lines = system_prompt.split('\n')
        essential_keywords = ['DATABASE SCHEMA:', 'EXAMPLE VOTING QUERIES', 'Natural Language Query:']
        
        trimmed_lines = []
        current_length = 0
        
        for line in lines:
            if any(keyword in line for keyword in essential_keywords) or current_length + len(line) + 1 < target_length:
                trimmed_lines.append(line)
                current_length += len(line) + 1
            elif current_length >= target_length:
                break
        
        trimmed_prompt = '\n'.join(trimmed_lines)
        final_tokens = self.count_tokens(trimmed_prompt)
        logger.info(f"VoteQuery2SQL Prompt trimmed: {current_tokens} -> {final_tokens} tokens (target: {available_tokens})")
        
        return trimmed_prompt

    def generate_sql_query_for_voting(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Convert natural language query to SQL for voting data"""
        try:
            system_prompt = f"""
You are a PostgreSQL expert specializing in voting data analysis. Convert natural language queries into optimized SQL queries for voting data.

DATABASE SCHEMA:
Main Table: {self.table_name}
{self.table_schema}

Related Table: conviction_vote
- Contains "self_voting_power" (voting power/balance for each vote)
- Joined via "parent_vote_id" (foreign key in {self.table_name}) → "id" (primary key in conviction_vote)

CORE SQL GUIDELINES:
1. Use ONLY existing columns from the schema above.
2. Main table name: {self.table_name}
3. Use proper PostgreSQL syntax with double quotes for column names.
4. Apply appropriate LIMIT clauses (typically 10 for lists; no LIMIT for counts/aggregates).
5. Always order explicitly when returning recent items (e.g., ORDER BY main."created_at" DESC).

JOIN REQUIREMENTS:
6. When querying voting power/balance, JOIN with conviction_vote table:
   FROM {self.table_name} AS main
   LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
7. Use "cv.self_voting_power" for all voting power queries (replaces "balance").
8. Always use table aliases (main, cv) to avoid ambiguity.

VOTING DATA SPECIFIC RULES:
9. Voter information: Use "main.voter".
10. Proposal identification: Use "main.proposal_index" or "main.proposal_id" for proposal/referendum IDs.
11. Voting decisions: Use "main.decision" (values like 'aye', 'nay', 'abstain' — case-insensitive compare with ILIKE when needed).
12. Voting power: Use "cv.self_voting_power" (FLOAT). When querying voting power, always include the JOIN with conviction_vote.
13. Delegation: Use "main.is_delegated" (BOOLEAN) and "main.delegated_to" for target account.
14. Date filtering: Use "main.created_at" for when the vote was cast; use "main.removed_at" to exclude revoked/invalidated votes (e.g., WHERE main."removed_at" IS NULL for "active" votes).
15. Proposal types: Use "main.type" (e.g., 'ReferendumV2', 'Treasury', 'Fellowship').
16. Lock period / conviction: Use "main.lock_period" for conviction or lock-time–related queries.

CRITICAL NULL VALUE HANDLING:
17. Many columns may be NULL — add IS NOT NULL when the query semantically requires a value.
18. For voting power queries: ALWAYS add "cv.self_voting_power IS NOT NULL" and include JOIN with conviction_vote.
19. For date-based queries: Consider adding "main.created_at IS NOT NULL".
20. When filtering by proposal or voter, consider "main.proposal_index IS NOT NULL" and/or "main.voter IS NOT NULL" as appropriate.

MULTIPLE QUERIES STRATEGY:
- If the user asks for COUNT and EXAMPLES (e.g., "how many voters and show some"), return 2 queries:
  • Query 1: COUNT query to get the total number
  • Query 2: SELECT query to get examples with details
- If the user asks only for a count, return 1 COUNT query.
- If the user asks only for a list/examples, return 1 SELECT query.
- Return queries as a JSON array: ["query1", "query2"].

COLUMN SELECTION STRATEGY:
- General lists: select key columns like "main.voter", "main.decision", "cv.self_voting_power", "main.created_at", "main.proposal_index", "main.type", "main.is_delegated".
- Voter analysis: focus on "main.voter", "cv.self_voting_power", "main.decision", "main.is_delegated", "main.delegated_to", "main.created_at".
- Proposal analysis: include "main.proposal_index", "main.type", "main.created_at", "main.decision", "cv.self_voting_power".
- Avoid SELECT * unless absolutely necessary.

EXAMPLE VOTING QUERIES (WITH CORRECT JOIN):

Single Query Examples:
- "Show me recent votes"
  -> SELECT main."voter", main."decision", cv."self_voting_power", main."created_at", main."proposal_index", main."type"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE main."created_at" IS NOT NULL
     ORDER BY main."created_at" DESC
     LIMIT 10;

- "How many voters in July 2025?"
  -> SELECT COUNT(DISTINCT main."voter") AS unique_voters
     FROM {self.table_name} AS main
     WHERE main."created_at" IS NOT NULL
       AND DATE_TRUNC('month', main."created_at") = '2025-07-01';

- "Voters with more than 1000 DOT voting power"
  -> SELECT main."voter", cv."self_voting_power"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE cv."self_voting_power" IS NOT NULL
       AND cv."self_voting_power" > 1000
     ORDER BY cv."self_voting_power" DESC
     LIMIT 10;

- "Votes on proposal 123"
  -> SELECT main."voter", main."decision", cv."self_voting_power", main."created_at"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE main."proposal_index" = 123;

- "Show delegated votes"
  -> SELECT main."voter", main."delegated_to", main."decision", cv."self_voting_power", main."proposal_index"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE main."is_delegated" = TRUE
     LIMIT 10;

- "Active votes only (exclude removed)"
  -> SELECT main."voter", main."decision", cv."self_voting_power", main."created_at", main."proposal_index"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE main."removed_at" IS NULL
     ORDER BY main."created_at" DESC
     LIMIT 10;

- "Votes with conviction lock period >= 4"
  -> SELECT main."voter", main."decision", cv."self_voting_power", main."lock_period", main."proposal_index"
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE main."lock_period" IS NOT NULL AND main."lock_period" >= 4
     ORDER BY main."lock_period" DESC
     LIMIT 10;

- "Top voters by voting power"
  -> SELECT main."voter", SUM(cv."self_voting_power") AS total_voting_power, COUNT(*) AS vote_count
     FROM {self.table_name} AS main
     LEFT JOIN conviction_vote AS cv ON main."parent_vote_id" = cv."id"
     WHERE cv."self_voting_power" IS NOT NULL
     GROUP BY main."voter"
     ORDER BY total_voting_power DESC
     LIMIT 10;

Multiple Query Example:
- "How many voters in July and show some?"
  -> [
       "SELECT COUNT(DISTINCT main.\"voter\") AS total_voters FROM {self.table_name} AS main WHERE main.\"created_at\" IS NOT NULL AND DATE_TRUNC('month', main.\"created_at\") = '2025-07-01';",
       "SELECT main.\"voter\", main.\"decision\", cv.\"self_voting_power\", main.\"created_at\", main.\"proposal_index\" FROM {self.table_name} AS main LEFT JOIN conviction_vote AS cv ON main.\"parent_vote_id\" = cv.\"id\" WHERE main.\"created_at\" IS NOT NULL AND DATE_TRUNC('month', main.\"created_at\") = '2025-07-01' ORDER BY main.\"created_at\" DESC LIMIT 10;"
     ]

Natural Language Query: {natural_query}

SQL Query:
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert specializing in voting data. Generate SQL queries based on the provided schema. For complex queries requiring both count and examples, return a JSON array of queries. For simple queries, return a JSON array with one query. Always return valid JSON format."},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean up the response
            response_content = response_content.replace('```json', '').replace('```sql', '').replace('```', '').strip()
            
            try:
                # Try to parse as JSON array
                import json
                sql_queries = json.loads(response_content)
                
                # Ensure it's a list
                if isinstance(sql_queries, str):
                    sql_queries = [sql_queries]
                elif not isinstance(sql_queries, list):
                    sql_queries = [str(sql_queries)]
                    
                logger.info(f"Generated {len(sql_queries)} SQL queries for voting data: {sql_queries}")
                return sql_queries
                
            except json.JSONDecodeError:
                # Fallback: treat as single query
                logger.warning("Could not parse JSON, treating as single query")
                single_query = response_content.strip()
                logger.info(f"Generated single SQL query for voting data: {single_query}")
                return [single_query]
            
        except Exception as e:
            logger.error(f"Error generating SQL query for voting data: {e}")
            return ["SELECT COUNT(*) FROM voting_data;"]  # Fallback query

    def execute_sql_query(self, sql_query: str) -> Tuple[List[List[Any]], List[str]]:
        """Execute SQL query and return results with column names"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    logger.info(f"Executing SQL: {sql_query}")
                    cur.execute(sql_query)
                    
                    # Get column names
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    
                    # Fetch results
                    results = cur.fetchall()
                    
                    logger.info(f"Query executed successfully: {len(results)} rows returned")
                    return results, columns
                    
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return [], []

    def execute_sql_queries(self, sql_queries: List[str]) -> List[Tuple[List[List[Any]], List[str]]]:
        """Execute multiple SQL queries and return all results"""
        all_results = []
        for i, query in enumerate(sql_queries):
            logger.info(f"Executing query {i+1}/{len(sql_queries)}")
            results, columns = self.execute_sql_query(query)
            all_results.append((results, columns))
        return all_results

    def generate_natural_response(self, natural_query: str, sql_query: str, results: List[List[Any]], 
                                columns: List[str], conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate natural language response from SQL results for voting data"""
        try:
            # If no results, provide a helpful message
            if not results:
                return f"I didn't find any voting records matching your query '{natural_query}'. This could mean there are no votes matching your criteria, or the voting data might not contain the specific information you're looking for."
            
            # Convert results to a more readable format
            if len(results) <= 20:  # For small result sets, include details
                results_summary = f"Found {len(results)} voting records"
                sample_data = results[:10]  # Show first 10 results
            else:
                results_summary = f"Found {len(results)} voting records (showing first 10)"
                sample_data = results[:10]
            
            # Determine response style based on conversation history
            has_context = conversation_history and len(conversation_history) > 0
            context_info = ""
            if has_context:
                # Extract relevant context from conversation history
                recent_topics = []
                for msg in conversation_history[-3:]:  # Last 3 messages
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if content and len(content) > 10:
                            recent_topics.append(content[:100])
                
                if recent_topics:
                    context_info = f"Previous conversation topics: {'; '.join(recent_topics)}"
            
            # Create context for the AI to generate response
            context_prompt = f"""
            Convert these voting query results into a natural, conversational response.
            
            Original Question: {natural_query}
            SQL Query: {sql_query}
            
            Results Summary: {results_summary}
            Columns: {columns}
            Sample Data: {sample_data}
            
            {context_info}
            
            RESPONSE STYLE GUIDELINES:
            1. BE CONCISE: Give direct, to-the-point answers unless the question specifically asks for detailed analysis
            2. ANSWER FIRST: Start with the direct answer to the user's question
            3. MINIMAL CONTEXT: Only add insights/context if:
               - The user explicitly asks for analysis or insights
               - The conversation history shows they want detailed explanations
               - The question is complex and requires context to understand
            4. AVOID SPECULATION: Don't add "could suggest" or "might indicate" unless specifically asked for analysis
            5. NUMBERS: Present key numbers clearly but don't over-explain their significance unless asked
            6. If you receive proposal_index in result. Then you should must make a link like below:
                - https://polkadot.polkassembly.io/referenda/{{proposal_index}} 
            7. When you receive voting self_voting_power, then always remove 9 zero from it. For ex: 10000000000 becomes 1 DOT. DOT is the unit here.
               
            
            EXAMPLES:
            - Question: "What proposal got the most votes?" 
              Good: "Proposal 1424 received the highest voting power with 2,955,235,968,346,605,113."
              Bad: "The proposal that received the highest number of votes... This indicates... It's interesting to note..."
            
            - Question: "Analyze voting patterns for treasury proposals"
              Good: [Longer response with analysis since "analyze" was requested]
            
            Response:
            """
            
            # Trim prompt to fit token limits
            context_prompt = self.trim_prompt_to_fit_tokens(context_prompt, max_tokens=8192, completion_tokens=800, buffer_tokens=200)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise, direct answers about voting data. Be brief and to-the-point unless the user specifically asks for detailed analysis or insights. Start with the direct answer, then add context only if needed."},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            # Fallback response
            return f"I found {len(results)} voting records for your query '{natural_query}', but I'm having trouble formatting the response. Here's a summary: The query returned {len(results)} rows from the voting database."

    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Main method to process a natural language query for voting data"""
        try:
            logger.info(f"Processing voting query: {natural_query}")
            
            # Step 1: Generate SQL queries (can be multiple)
            sql_queries = self.generate_sql_query_for_voting(natural_query, conversation_history)
            
            # Step 2: Execute SQL queries
            if len(sql_queries) == 1:
                # Single query
                results, columns = self.execute_sql_query(sql_queries[0])
                
                # Step 3: Generate natural language response
                natural_response = self.generate_natural_response(
                    natural_query, sql_queries[0], results, columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": sql_queries[0],
                    "sql_queries": sql_queries,
                    "result_count": len(results),
                    "results": results,
                    "columns": columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
            else:
                # Multiple queries
                all_results = self.execute_sql_queries(sql_queries)
                
                # Combine all results for response
                combined_results = []
                combined_columns = []
                total_result_count = 0
                
                for results, columns in all_results:
                    combined_results.extend(results)
                    if not combined_columns:  # Use columns from first query
                        combined_columns = columns
                    total_result_count += len(results)
                
                # Generate natural language response from multiple results
                natural_response = self.generate_natural_response(
                    natural_query, "; ".join(sql_queries), combined_results, combined_columns, conversation_history
                )
                
                return {
                    "original_query": natural_query,
                    "sql_query": "; ".join(sql_queries),
                    "sql_queries": sql_queries,
                    "result_count": total_result_count,
                    "results": combined_results,
                    "columns": combined_columns,
                    "natural_response": natural_response,
                    "success": True,
                    "table": "voting_data"
                }
                
        except Exception as e:
            logger.error(f"Error processing voting query: {e}")
            return {
                "original_query": natural_query,
                "sql_query": None,
                "sql_queries": [],
                "result_count": 0,
                "results": [],
                "columns": [],
                "natural_response": f"I'm having trouble processing your voting query. Please try again or rephrase your question.",
                "success": False,
                "error": str(e),
                "table": "voting_data"
            }

def main():
    """Example usage and testing"""
    try:
        # Initialize the query processor
        query_processor = Query2SQL()
        
        # Test connection
        if not query_processor.test_connection():
            print("❌ Database connection failed!")
            return
        
        print("✅ Database connection successful!")
        print(f"📊 Table: {query_processor.table_name}")
        print(f"📋 Schema columns: {len(query_processor.schema_info)}")
        print()
        
        # Example queries to test
        example_queries = [
            "Show me the 10 most recent proposals",
            "How many Kusama proposals are there?",
            "What treasury proposals exist?",
            "Find proposals created in 2024",
            "Show me active referendums"
        ]
        
        print("🤖 Testing example queries:")
        print("=" * 60)
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 40)
            
            result = query_processor.process_query(query)
            
            if result["success"]:
                print(f"✅ SQL: {result['sql_query']}")
                print(f"📊 Results: {result['result_count']} rows")
                print(f"💬 Response: {result['natural_response'][:200]}...")
            else:
                print(f"❌ Error: {result['error']}")
        
        print("\n" + "=" * 60)
        print("🎉 Testing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()