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
            
            # Format the schema for OpenAI prompt
            schema_parts = []
            table_name = schema_data.get('table_name', self.table_name)
            schema_parts.append(f"Table: {table_name}")
            schema_parts.append("\nColumns:")
            
            columns_data = schema_data.get('columns', {})
            for column, info in columns_data.items():
                data_type = info.get('type', 'unknown')
                description = info.get('description', 'No description')
                possible_values = info.get('possible_values_description', [])
                
                # Add column with type and description
                schema_line = f"  - {column} ({data_type}): {description}"
                
                # Add possible values if they exist
                if possible_values:
                    values_text = "; ".join(possible_values)
                    schema_line += f" Possible values: {values_text}"
                
                schema_parts.append(schema_line)
            
            schema_text = "\n".join(schema_parts)
            logger.info(f"Loaded schema directly from JSON file: {schema_path}")
            return schema_text
            
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
            - For general queries: SELECT key columns like "title", "index", "onchaininfo_status", "createdat", "source_network"
            - For specific proposal details: Include "content" only when explicitly asked for details/content
            - For searches: Focus on "title", "index", "onchaininfo_status", "createdat" 
            - Avoid SELECT * unless specifically needed - it causes long responses
            - Use SELECT "title", "index", "onchaininfo_status", "createdat" for most queries
            - If you searching for index keep LIMIT upto 10 as there can be multi proposals with same index
            - But, if somebody ask, proposals in voting then also use other attributes such as DecisionDepositPlaced, Submitted, ConfirmStarted, ConfirmAborted along with Deciding.
            
            EXAMPLE QUERIES:
            Single Query Examples:
             - "Show me recent proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat", "source_network" FROM {self.table_name} ORDER BY "createdat" DESC LIMIT 10;
            - "Find Kusama proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} WHERE "source_network" = 'kusama' LIMIT 10;
            - "What treasury proposals exist?" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} WHERE "source_proposal_type" ILIKE '%treasury%' LIMIT 10;
            - "Tell me about clarys proposal" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "content" ILIKE '%clarys%' OR "title" ILIKE '%clarys%' LIMIT 10;
            - "Tell me about subsquare proposal" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "content" ILIKE '%subsquare%' OR "title" ILIKE '%subsquare%' LIMIT 10;
            - "Give me the details of the proposal with id 123456" -> SELECT "title", "index", "onchaininfo_status", "createdat", "content" FROM {self.table_name} WHERE "index" = 123456 LIMIT 10;
            - "Give me some recent proposals" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} ORDER BY "createdat" DESC LIMIT 10;
            - "Give me proposals after 2024-01-01" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} WHERE "createdat" > '2024-01-01' LIMIT 10;
            - "Give me proposals before 2024-01-01" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} WHERE "createdat" < '2024-01-01' LIMIT 10;
            - "Give me proposals between dates" -> SELECT "title", "index", "onchaininfo_status", "createdat" FROM {self.table_name} WHERE "createdat" BETWEEN '2024-01-01' AND '2024-01-02' LIMIT 10;
            - "Count total proposals" -> SELECT COUNT(*) as total_proposals FROM {self.table_name};
            - "Show me proposal amounts" -> SELECT "title", "index", "onchaininfo_beneficiaries_0_amount", "createdat" FROM {self.table_name} WHERE "onchaininfo_beneficiaries_0_amount" IS NOT NULL LIMIT 10;
            
            Multiple Query Examples:
            - "How many proposals in August 2025 and name a few?" -> ["SELECT COUNT(*) as total_count FROM {self.table_name} WHERE DATE_TRUNC('month', \"createdat\") = '2025-08-01';", "SELECT \"title\", \"index\", \"onchaininfo_status\", \"createdat\" FROM {self.table_name} WHERE DATE_TRUNC('month', \"createdat\") = '2025-08-01' LIMIT 10;"]
            - "How many Kusama proposals exist and show some examples?" -> ["SELECT COUNT(*) as kusama_count FROM {self.table_name} WHERE \"source_network\" = 'kusama';", "SELECT \"title\", \"index\", \"onchaininfo_status\", \"createdat\" FROM {self.table_name} WHERE \"source_network\" = 'kusama' LIMIT 10;"]
            
            Null Results
            - Some columns has NULL and NaN values and for some queries like 
            - tell me the proposal who had asked for highest amount in the month of august 2025, use NOT NULL and != NaN to get correct result. Do your own thinking and generate the query where NOT NULL and !=NaN is needed. Example:
                    SELECT
                        "title",
                        "index",
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
            
            prompt = f"""
            Conversation History: {conversation_history}
            Current Query: {natural_query}
            Multiple Query Results: {combined_summary}
            
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
            
            IMPORTANT: All data is public blockchain information. Show actual values, addresses, and details.
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
                    {"role": "system", "content": "You are a knowledgeable assistant specializing in blockchain governance data. All data you work with is public blockchain information. Always show actual data requested - addresses, proposal IDs, titles, amounts, etc. Combine information from multiple queries to provide comprehensive answers."},
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
            
            prompt = f"""
            Conversation History: {conversation_history}
            Current Query: {natural_query}
            Results: {summary_text}
            
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
            6. 20-300 is the idead word ouput count. If the ouput will be too long than that, provide in summarization form instead of listing all the data.
            
            DATA PRESENTATION:
            - Use conversational language about Polkadot/Kusama governance
            - Be specific and factual with all data provided
            - Show actual proposal IDs, titles, addresses, amounts - all public blockchain information
            - Include network (Polkadot/Kusama), status, and creation dates when available
            - Never refuse to show data citing privacy concerns - all blockchain data is public
            
            Focus on providing accurate, specific information from the query results.
            """
            # print(f"\n\n prompt: {prompt}")
            
            # Try Gemini first as primary LLM
            if self.gemini_client is not None:
                try:
                    logger.info("Using Gemini as primary LLM for natural response generation")
                    natural_response = self.gemini_client.get_response(prompt)
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
    
    def process_query(self, natural_query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Main method to process a natural language query end-to-end"""
        try:
            logger.info(f"Processing query: {natural_query}")
            
            # Step 1: Generate SQL queries (can be multiple)
            sql_queries = self.generate_sql_query(natural_query, conversation_history)
            
            # Step 2: Execute SQL queries
            if len(sql_queries) == 1:
                # Single query - use existing flow for backwards compatibility
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
                    "success": True
                }
            else:
                # Multiple queries
                all_results = self.execute_sql_queries(sql_queries)
                
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