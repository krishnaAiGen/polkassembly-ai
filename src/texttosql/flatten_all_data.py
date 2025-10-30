#!/usr/bin/env python3
"""
Comprehensive script to flatten all JSON files in the onchain_data directory,
generate CSV files, and create metadata.json files with schema information.
"""

import json
import pandas as pd
import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
from openai import OpenAI
from datetime import datetime
import logging
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFlattener:
    def __init__(self, data_dir: str, output_data_dir: str = None, openai_api_key: str = None):
        self.data_dir = Path(data_dir)
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI API key configured")
        else:
            self.openai_client = None
            logger.warning("No OpenAI API key provided. Metadata generation will be limited.")
        
        if output_data_dir:
            output_path = Path(output_data_dir)
        else:
            output_path = Path(__file__).parent.parent / "data"
        
        # Create output directories - CSV files go to all_csv subdirectory
        self.csv_dir = output_path / "all_csv"
        self.metadata_dir = output_path / "metadata"
        self.csv_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_dir.mkdir(exist_ok=True, parents=True)
    
    def sanitize_key(self, key: str) -> str:
        """Keep it SQL/CSV friendly: lower, underscores, alnum only"""
        key = str(key).replace(" ", "_").replace("-", "_")
        key = re.sub(r"[^0-9a-zA-Z_]", "_", key)
        key = re.sub(r"_+", "_", key)
        return key.strip("_").lower()
    
    def flatten_all(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """
        Fully flatten dicts and lists:
          dict key 'k' -> prefix_k
          list index i -> prefix_i
        Example: {'beneficiaries': [{'amount': '1'}]}
          -> {'beneficiaries_0_amount': '1'}
        """
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                k2 = self.sanitize_key(k)
                new_prefix = f"{prefix}_{k2}" if prefix else k2
                if isinstance(v, (dict, list)):
                    out.update(self.flatten_all(v, new_prefix))
                else:
                    out[new_prefix] = v
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}_{i}" if prefix else str(i)
                if isinstance(v, (dict, list)):
                    out.update(self.flatten_all(v, new_prefix))
                else:
                    out[new_prefix] = v
        else:
            if prefix:
                out[prefix] = obj
        return out
    
    def analyze_csv_for_metadata(self, df: pd.DataFrame, network: str, proposal_type: str) -> Dict[str, Any]:
        """Analyze CSV data to create comprehensive metadata for query routing"""
        analysis = {}
        
        # Basic statistics
        analysis['total_rows'] = len(df)
        analysis['total_columns'] = len(df.columns)
        
        # Column analysis
        column_details = {}
        key_columns = []
        date_columns = []
        numeric_columns = []
        status_columns = []
        content_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            non_null_values = df[col].dropna()
            
            # Determine column category
            if any(pattern in col_lower for pattern in ['id', 'hash', 'address', 'proposer', 'beneficiary']):
                key_columns.append(col)
            elif any(pattern in col_lower for pattern in ['created_at', 'updated_at', 'timestamp', 'date', 'time']):
                date_columns.append(col)
            elif any(pattern in col_lower for pattern in ['amount', 'reward', 'balance', 'value', 'count', 'index']):
                numeric_columns.append(col)
            elif any(pattern in col_lower for pattern in ['status', 'state', 'phase']):
                status_columns.append(col)
            elif any(pattern in col_lower for pattern in ['title', 'content', 'description', 'name']):
                content_columns.append(col)
            
            # Get sample values and statistics
            sample_values = non_null_values.unique()[:5].tolist()
            unique_count = non_null_values.nunique()
            
            column_details[col] = {
                'data_type': str(df[col].dtype),
                'non_null_count': len(non_null_values),
                'null_count': df[col].isnull().sum(),
                'unique_count': unique_count,
                'sample_values': sample_values,
                'is_key_field': col in key_columns,
                'is_date_field': col in date_columns,
                'is_numeric_field': col in numeric_columns,
                'is_status_field': col in status_columns,
                'is_content_field': col in content_columns
            }
        
        analysis['column_categories'] = {
            'key_columns': key_columns,
            'date_columns': date_columns,
            'numeric_columns': numeric_columns,
            'status_columns': status_columns,
            'content_columns': content_columns
        }
        
        # Status analysis for query routing
        if status_columns:
            status_analysis = {}
            for status_col in status_columns:
                status_counts = df[status_col].value_counts().to_dict()
                status_analysis[status_col] = status_counts
            analysis['status_distribution'] = status_analysis
        
        # Date range analysis
        if date_columns:
            date_analysis = {}
            for date_col in date_columns:
                try:
                    # Try to parse dates
                    pd_dates = pd.to_datetime(df[date_col], errors='coerce')
                    valid_dates = pd_dates.dropna()
                    if len(valid_dates) > 0:
                        date_analysis[date_col] = {
                            'earliest': valid_dates.min().isoformat(),
                            'latest': valid_dates.max().isoformat(),
                            'total_dates': len(valid_dates)
                        }
                except:
                    pass
            analysis['date_ranges'] = date_analysis
        
        # Network and proposal type specific analysis
        if 'network' in df.columns:
            network_counts = df['network'].value_counts().to_dict()
            analysis['network_distribution'] = network_counts
        
        if 'proposal_type' in df.columns:
            proposal_counts = df['proposal_type'].value_counts().to_dict()
            analysis['proposal_type_distribution'] = proposal_counts
        
        # Content analysis for search
        if content_columns:
            content_analysis = {}
            for content_col in content_columns:
                # Get sample content for understanding
                sample_content = df[content_col].dropna().head(3).tolist()
                content_analysis[content_col] = {
                    'sample_content': sample_content,
                    'avg_length': df[content_col].str.len().mean() if df[content_col].dtype == 'object' else None
                }
            analysis['content_samples'] = content_analysis
        
        analysis['column_details'] = column_details
        return analysis
    

    
    def extract_sample_data(self, df: pd.DataFrame, max_samples: int = 5) -> Dict[str, Any]:
        """Extract sample data for metadata generation"""
        sample_data = {}
        for col in df.columns:
            # Get non-null values
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Get unique values (up to max_samples)
                unique_values = non_null_values.unique()[:max_samples]
                sample_data[col] = {
                    'data_type': str(df[col].dtype),
                    'non_null_count': len(non_null_values),
                    'null_count': df[col].isnull().sum(),
                    'unique_count': non_null_values.nunique(),
                    'sample_values': unique_values.tolist()
                }
        return sample_data
    
    def generate_metadata_with_openai(self, filename: str, sample_data: Dict[str, Any], 
                                    network: str, proposal_type: str, total_items: int,
                                    csv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata using OpenAI API with CSV data analysis"""
        if not self.openai_client:
            return self.generate_basic_metadata(filename, sample_data, network, proposal_type, total_items, csv_analysis)
        
        try:
            # Create a comprehensive prompt for OpenAI with actual data samples
            table_name = filename.replace('.json', '').replace('.csv', '')
            categories = csv_analysis['column_categories']
            
            # Get first 10 rows of actual data for better context
            sample_rows = []
            for col in list(sample_data.keys())[:15]:  # Top 15 columns
                col_data = sample_data[col]
                sample_rows.append({
                    "column": col,
                    "type": col_data['data_type'],
                    "sample_values": col_data['sample_values'][:5],
                    "non_null_count": col_data['non_null_count']
                })
            
            prompt = f"""
            Generate COMPACT metadata for query routing decisions for this Polkadot/Kusama governance data.
            
            Table: {table_name}
            Network: {network}
            Proposal Type: {proposal_type}
            Total Records: {total_items}
            
            ACTUAL COLUMN DATA (first 15 columns with samples):
            {json.dumps(sample_rows, indent=2)}
            
            Column Categories:
            - Key columns: {categories['key_columns'][:5]}
            - Status columns: {categories['status_columns'][:3]}
            - Date columns: {categories['date_columns'][:3]}
            - Content columns: {categories['content_columns'][:3]}
            - Numeric columns: {categories['numeric_columns'][:3]}
            
            Generate a JSON response with this structure:
            {{
                "table_name": "{table_name}",
                "network": "{network}",
                "proposal_type": "{proposal_type}",
                "description": "Brief description of what this table contains",
                "record_count": {total_items},
                "key_fields": ["top 5 most important queryable fields based on actual columns"],
                "routing_keywords": ["8-10 keywords that indicate queries should use this table"],
                "query_capabilities": ["status_filtering", "date_filtering", "content_search", "id_lookup", "financial_analysis"],
                "example_queries": [
                    "10-15 NATURAL LANGUAGE queries that users would ask (NOT SQL)",
                    "Examples: 'Show me active proposals', 'What referendums passed last month?'",
                    "Make them conversational and human-like",
                    "Focus on governance questions users would actually ask"
                ]
            }}
            
            IMPORTANT: 
            1. Generate HUMAN NATURAL LANGUAGE queries, NOT SQL
            2. Make them conversational: "Show me...", "What are...", "How many...", "Find..."
            3. Include queries for: status, dates, content search, financial analysis, governance
            4. Use terms users would actually say, not technical column names
            5. Focus on Polkadot/Kusama governance questions people would ask
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data engineering expert specializing in blockchain and governance data. Provide clear, actionable metadata for database design, querying, and query routing decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            try:
                # Try to extract JSON from the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    metadata = json.loads(content[json_start:json_end])
                else:
                    # Fallback to basic metadata
                    metadata = self.generate_basic_metadata(filename, sample_data, network, proposal_type, total_items, csv_analysis)
                    metadata['openai_generated_description'] = content
            except json.JSONDecodeError:
                # Fallback to basic metadata
                metadata = self.generate_basic_metadata(filename, sample_data, network, proposal_type, total_items, csv_analysis)
                metadata['openai_generated_description'] = content
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata with OpenAI for {filename}: {e}")
            return self.generate_basic_metadata(filename, sample_data, network, proposal_type, total_items, csv_analysis)
    
    def generate_basic_metadata(self, filename: str, sample_data: Dict[str, Any], 
                               network: str, proposal_type: str, total_items: int,
                               csv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compact metadata for query routing"""
        
        # Generate table name from CSV filename
        table_name = filename.replace('.json', '').replace('.csv', '')
        
        # Create compact description for routing
        description = f"{proposal_type} data from {network} network"
        if 'ReferendumV2' in proposal_type:
            description += " - OpenGov referendums, voting, tracks"
        elif 'TreasuryProposal' in proposal_type:
            description += " - Treasury spending, beneficiaries, amounts"
        elif 'Bounty' in proposal_type:
            description += " - Community bounties, rewards, curators"
        
        # Identify key query fields
        categories = csv_analysis['column_categories']
        query_fields = []
        if categories['status_columns']:
            query_fields.extend(categories['status_columns'][:2])
        if categories['key_columns']:
            query_fields.extend([col for col in categories['key_columns'] if 'id' in col.lower()][:2])
        if categories['date_columns']:
            query_fields.extend(categories['date_columns'][:1])
        
        # Generate routing keywords
        routing_keywords = [network, proposal_type.lower()]
        if 'referendum' in proposal_type.lower():
            routing_keywords.extend(['referendum', 'vote', 'track', 'opengov'])
        elif 'treasury' in proposal_type.lower():
            routing_keywords.extend(['treasury', 'spend', 'beneficiary', 'funding'])
        elif 'bounty' in proposal_type.lower():
            routing_keywords.extend(['bounty', 'reward', 'curator', 'community'])
        
        # Build query capabilities list (no None values)
        query_capabilities = []
        if categories['status_columns']:
            query_capabilities.append("status_filtering")
        if categories['date_columns']:
            query_capabilities.append("date_filtering")
        if categories['content_columns']:
            query_capabilities.append("content_search")
        if categories['key_columns']:
            query_capabilities.append("id_lookup")
        if categories['numeric_columns']:
            query_capabilities.append("financial_analysis")
        
        # Generate example queries based on actual column names and data
        example_queries = self.generate_example_queries(
            table_name, categories, network, proposal_type, sample_data
        )
        
        return {
            "table_name": table_name,
            "network": network,
            "proposal_type": proposal_type,
            "description": description,
            "record_count": total_items,
            "key_fields": query_fields[:5],  # Top 5 most important fields
            "routing_keywords": routing_keywords,
            "query_capabilities": query_capabilities,
            "example_queries": example_queries
        }
    
    def generate_example_queries(self, table_name: str, categories: Dict[str, List[str]], 
                                network: str, proposal_type: str, sample_data: Dict[str, Any]) -> List[str]:
        """Generate realistic example queries based on actual column names and data"""
        queries = []
        
        # Get actual column names
        status_cols = categories['status_columns'][:2]
        date_cols = categories['date_columns'][:2] 
        key_cols = categories['key_columns'][:3]
        content_cols = categories['content_columns'][:2]
        numeric_cols = categories['numeric_columns'][:3]
        
        # Get sample values for more realistic queries
        sample_status = []
        sample_ids = []
        
        for col, data in sample_data.items():
            if col in status_cols and data['sample_values']:
                sample_status.extend([str(v) for v in data['sample_values'][:2]])
            elif col in key_cols and 'id' in col.lower() and data['sample_values']:
                sample_ids.extend([str(v) for v in data['sample_values'][:2]])
        
        # Status-based queries (natural language)
        if status_cols:
            queries.extend([
                "Show me all active proposals",
                "What proposals are currently in deciding phase?",
                "How many proposals have been approved?",
                "Which proposals are still pending?",
                "Show me rejected proposals"
            ])
            if sample_status:
                queries.append(f"Find all proposals that are {sample_status[0]}")
        
        # Date-based queries (natural language)
        if date_cols:
            queries.extend([
                "What proposals were created in the last month?",
                "Show me recent proposals from this year",
                "Find proposals submitted in 2024",
                "What are the newest proposals?",
                "Show me proposals from last week"
            ])
        
        # ID/Key-based queries (natural language)
        if key_cols:
            queries.extend([
                "Find a specific proposal by ID",
                "Show me proposal details",
                "Look up a particular proposal"
            ])
            if sample_ids:
                queries.append(f"Show me proposal {sample_ids[0]}")
        
        # Content search queries (natural language)
        if content_cols:
            queries.extend([
                "Search for proposals about treasury spending",
                "Find proposals mentioning funding",
                "What proposals discuss development grants?",
                "Search for proposals about infrastructure"
            ])
        
        # Financial/Numeric queries (natural language)
        if numeric_cols:
            queries.extend([
                "What are the highest value proposals?",
                "Show me proposals with large amounts",
                "How much funding has been requested?",
                "What's the total amount of all proposals?"
            ])
        
        # Network-specific queries (natural language)
        queries.extend([
            f"Show me all proposals from {network}",
            "Compare governance between Polkadot and Kusama",
            f"What's happening in {network} governance?",
            f"How is {network} performing?"
        ])
        
        # Proposal type specific queries (natural language)
        if 'referendum' in proposal_type.lower():
            queries.extend([
                "How did people vote on recent referendums?",
                "What referendums are currently active?",
                "Show me OpenGov referendum results",
                "Which referendums passed recently?",
                "What tracks are most active?"
            ])
        elif 'treasury' in proposal_type.lower():
            queries.extend([
                "What treasury proposals are being funded?",
                "Who are the beneficiaries of treasury spending?",
                "How much money is being requested from treasury?",
                "Show me approved treasury proposals",
                "What projects got treasury funding?"
            ])
        elif 'bounty' in proposal_type.lower():
            queries.extend([
                "What bounties are currently active?",
                "Who are the bounty curators?",
                "Show me completed bounties and their rewards",
                "What bounties need curators?",
                "How much are bounty rewards?"
            ])
        
        # General governance queries (natural language)
        queries.extend([
            "What are the latest governance trends?",
            "Who are the most active proposers?",
            "What's the approval rate for proposals?",
            "Show me governance statistics",
            "How active is the community?"
        ])
        
        return queries[:20]  # Return top 20 most relevant queries
    
    def process_json_file(self, json_file_path: Path) -> Tuple[bool, str]:
        """Process a single JSON file and generate CSV + metadata"""
        try:
            logger.info(f"Processing {json_file_path.name}")
            
            # Read JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract metadata
            network = data.get('network', 'unknown')
            timestamp = data.get('timestamp', '')
            total_items = data.get('total_items', 0)
            items = data.get('items', [])
            
            if not items:
                logger.warning(f"No items found in {json_file_path.name}")
                return False, "No items found"
            
            # Extract proposal type early
            proposal_type = data.get('items', [{}])[0].get('proposalType', 'Unknown') if items else 'Unknown'
            
            # Flatten all items
            flat_records = [self.flatten_all(item) for item in items]
            
            # Create DataFrame
            df = pd.DataFrame(flat_records)
            
            # Generate CSV filename
            csv_filename = json_file_path.stem + '.csv'
            csv_path = self.csv_dir / csv_filename
            
            # Save CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_filename} with {len(df)} rows and {len(df.columns)} columns")
            
            # # Analyze CSV data for comprehensive metadata
            # csv_analysis = self.analyze_csv_for_metadata(df, network, proposal_type)
            
            # # Extract sample data for metadata
            # sample_data = self.extract_sample_data(df)
            
            # # Generate metadata
            # metadata = self.generate_metadata_with_openai(
            #     json_file_path.name, sample_data, network, proposal_type, total_items, csv_analysis
            # )
            
            # # Add minimal file-specific metadata
            # metadata.update({
            #     "csv_file": csv_filename,
            #     "processing_date": datetime.now().isoformat()
            # })
            
            # # Save metadata
            # metadata_filename = json_file_path.stem + '_metadata.json'
            # metadata_path = self.metadata_dir / metadata_filename
            
            # with open(metadata_path, 'w', encoding='utf-8') as f:
            #     json.dump(metadata, f, indent=2, default=str)
            
            # logger.info(f"Saved metadata: {metadata_filename}")
            return True, f"Successfully processed {len(df)} rows"
            
        except Exception as e:
            error_msg = f"Error processing {json_file_path.name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_all_files(self) -> Dict[str, Any]:
        """Process all JSON files in the directory"""
        print(f"DEBUG: Looking for JSON files in: {self.data_dir}")
        print(f"DEBUG: Directory exists: {self.data_dir.exists()}")
        json_files = list(self.data_dir.glob("*.json"))
        print(f"DEBUG: Found JSON files: {[f.name for f in json_files[:5]]}")  # Show first 5
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = {
            "total_files": len(json_files),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "processed_files": []
        }
        
        for json_file in json_files:
            success, message = self.process_json_file(json_file)
            
            if success:
                results["successful"] += 1
                results["processed_files"].append({
                    "file": json_file.name,
                    "status": "success",
                    "message": message
                })
            else:
                results["failed"] += 1
                results["errors"].append({
                    "file": json_file.name,
                    "error": message
                })
        
        # Save summary
        summary_path = Path(__file__).parent.parent / "data" / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Processing complete. Success: {results['successful']}, Failed: {results['failed']}")
        return results

def main():
    """Main execution function"""
    # Configuration - JSON files are in the data/onchain_data directory
    data_directory = Path(str(os.getenv("BASE_PATH")) + "/data/onchain_data")
    openai_api_key = os.getenv('OPENAI_API_KEY')  # Set this in your environment
    
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not set. Using basic metadata generation.")
    
    # Debug: Print path information
    print(f"Data directory: {data_directory}")
    print(f"Data directory exists: {data_directory.exists()}")
    print(f"Data directory absolute: {data_directory.resolve()}")
    
    # Set output directory for CSV files
    output_directory = str(os.getenv("BASE_PATH")) + "/onchain_data/onchain_first_pull"
    # Initialize flattener
    flattener = DataFlattener(str(data_directory), output_data_dir=output_directory, openai_api_key=openai_api_key)
    
    # Process all files
    results = flattener.process_all_files()
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error['file']}: {error['error']}")
    
    print(f"\nOutput directories:")
    print(f"  CSV files: {flattener.csv_dir}")
    print(f"  Metadata: {flattener.metadata_dir}")
    print(f"  Summary: {Path(__file__).parent.parent / 'data' / 'processing_summary.json'}")

if __name__ == "__main__":
    main()
