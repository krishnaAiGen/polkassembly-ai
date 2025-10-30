#!/usr/bin/env python3
"""
Insert combined governance data CSV into AWS PostgreSQL database.
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv
import time
from contextlib import contextmanager
import json
from datetime import datetime
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgresInserter:
    def __init__(self, table_name: Optional[str] = None, schema_path: Optional[str] = None):
        """Initialize with database connection parameters from environment"""
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DATABASE'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
        # Validate required environment variables
        required_vars = ['POSTGRES_HOST', 'POSTGRES_DATABASE', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.table_name = table_name or os.getenv('POSTGRES_TABLE_NAME', 'governance_data')
        self.batch_size = int(os.getenv('POSTGRES_BATCH_SIZE', '1000'))
        
        # Load schema information
        self.schema_path = schema_path
        self.schema_info = self._load_schema_info()
        
        logger.info(f"Initialized PostgreSQL inserter for table: {self.table_name}")
        logger.info(f"Target database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        logger.info(f"Loaded schema info for {len(self.schema_info)} columns")
    
    def _load_schema_info(self) -> Dict[str, Dict[str, str]]:
        """Load schema information from the specified schema_path or a default location."""
        if self.schema_path:
            schema_path = Path(self.schema_path)
        else:
            # Default path for backward compatibility
            schema_path = Path(__file__).parent.parent / "data" / "one_table" / "schema_info.json"
        
        if not schema_path.exists():
            logger.warning(f"Schema info file not found at {schema_path}. Using automatic type inference.")
            return {}
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            logger.info(f"Loaded schema information from {schema_path}")
            return schema_data
            
        except Exception as e:
            logger.warning(f"Error loading schema info: {e}. Using automatic type inference.")
            return {}
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            logger.info("Connecting to PostgreSQL database...")
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = False
            logger.info("Successfully connected to PostgreSQL")
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
                logger.info("Database connection closed")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"PostgreSQL version: {version}")
                    return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def analyze_csv_structure(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze CSV structure to determine table schema"""
        logger.info(f"Analyzing CSV structure: {csv_path.name}")
        
        # Read a sample to understand structure
        sample_df = pd.read_csv(csv_path, nrows=1000)
        
        # Get full column info
        df_info = pd.read_csv(csv_path, nrows=0)  # Just headers
        columns = list(df_info.columns)
        
        # Analyze data types
        column_info = {}
        for col in columns:
            col_data = sample_df[col]
            
            # Determine PostgreSQL data type
            pg_type = self._infer_postgres_type(col, col_data)
            
            # Check for constraints
            has_nulls = col_data.isnull().any()
            is_unique = len(col_data) == col_data.nunique() if not has_nulls else False
            
            column_info[col] = {
                'postgres_type': pg_type,
                'nullable': has_nulls,
                'unique': is_unique,
                'pandas_dtype': str(col_data.dtype)
            }
        
        # Get total row count
        total_rows = len(pd.read_csv(csv_path))
        
        analysis = {
            'total_rows': total_rows,
            'total_columns': len(columns),
            'columns': columns,
            'column_info': column_info
        }
        
        logger.info(f"CSV analysis complete: {total_rows:,} rows, {len(columns)} columns")
        return analysis
    
    def _infer_postgres_type(self, column_name: str, series: pd.Series) -> str:
        """Infer PostgreSQL data type from schema info or pandas Series"""
        
        # First, check if we have schema information for this column
        if self.schema_info and column_name in self.schema_info:
            schema_type = self.schema_info[column_name].get('data_type', '').lower()
            
            # Map schema types to PostgreSQL types
            if schema_type in ['date', 'Date']:
                return 'DATE'
            elif schema_type in ['datetime', 'timestamp']:
                return 'TIMESTAMP'
            elif schema_type in ['int64', 'integer', 'int']:
                # Use NUMERIC for all integer types to avoid BIGINT range issues
                logger.info(f"Column {column_name} (schema: {schema_type}) using NUMERIC for safety")
                return 'NUMERIC'
            elif schema_type in ['float64', 'float', 'double']:
                return 'DOUBLE PRECISION'
            elif schema_type in ['bool', 'boolean']:
                return 'BOOLEAN'
            elif schema_type in ['string', 'str', 'text']:
                # Use TEXT for all string types to avoid length issues
                return 'TEXT'
            elif schema_type == 'object':
                return 'TEXT'
        
        # Fallback to automatic inference if no schema info
        col_lower = column_name.lower()
        
        # Check pandas dtype
        if pd.api.types.is_integer_dtype(series):
            # Use NUMERIC for all integer types to avoid any range issues
            logger.info(f"Column {column_name} detected as integer, using NUMERIC for safety")
            return 'NUMERIC'
        
        elif pd.api.types.is_float_dtype(series):
            return 'DOUBLE PRECISION'
        
        elif pd.api.types.is_bool_dtype(series):
            return 'BOOLEAN'
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'TIMESTAMP'
        
        # For object types, analyze content
        elif series.dtype == 'object':
            non_null_data = series.dropna()
            
            if len(non_null_data) == 0:
                return 'TEXT'
            
            # Check if it's date strings
            if any(pattern in col_lower for pattern in ['date', 'time', 'created', 'updated']):
                return 'DATE'
            
            # Use TEXT for all object types to avoid length issues
            return 'TEXT'
        
        return 'TEXT'  # Default fallback
    
    def create_table(self, analysis: Dict[str, Any], drop_if_exists: bool = False) -> bool:
        """Create table based on CSV analysis"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Drop table if requested
                    if drop_if_exists:
                        logger.info(f"Dropping table {self.table_name} if exists...")
                        cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                    
                    # Build CREATE TABLE statement
                    columns_sql = []
                    for col in analysis['columns']:
                        col_info = analysis['column_info'][col]
                        pg_type = col_info['postgres_type']
                        
                        # Clean column name for PostgreSQL
                        clean_col = self._clean_column_name(col)
                        
                        # Add NOT NULL for source tracking columns only
                        not_null = ""
                        if col.startswith('source_'):
                            not_null = " NOT NULL"
                        
                        columns_sql.append(f'    "{clean_col}" {pg_type}{not_null}')
                    
                    # Add auto-incrementing primary key
                    columns_sql.insert(0, '    "row_id" SERIAL PRIMARY KEY')
                    
                    # Alternative: Use composite key if no auto-increment desired
                    # if 'source_file' in analysis['columns'] and 'source_row_id' in analysis['columns']:
                    #     columns_sql.append('    PRIMARY KEY ("source_file", "source_row_id")')
                    
                    # Build CREATE TABLE statement without f-string backslash issues
                    columns_part = ',\n'.join(columns_sql)
                    create_sql = f"""
CREATE TABLE {self.table_name} (
{columns_part}
);
"""
                    
                    logger.info(f"Creating table {self.table_name}...")
                    logger.debug(f"CREATE TABLE SQL:{chr(10)}{create_sql}")
                    
                    cur.execute(create_sql)
                    
                    # Create indexes for common query patterns
                    self._create_indexes(cur, analysis)
                    
                    conn.commit()
                    logger.info(f"Table {self.table_name} created successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            return False
    
    def _clean_column_name(self, column: str) -> str:
        """Clean column name for PostgreSQL compatibility"""
        # Replace problematic characters
        clean = column.replace(' ', '_').replace('-', '_').replace('.', '_')
        # Remove multiple underscores
        while '__' in clean:
            clean = clean.replace('__', '_')
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        # Ensure it starts with a letter
        if clean and clean[0].isdigit():
            clean = f'col_{clean}'
        return clean.lower()
    
    def _create_indexes(self, cursor, analysis: Dict[str, Any]):
        """Create indexes for common query patterns"""
        logger.info("Creating indexes...")
        
        # Common columns to index
        index_candidates = [
            'source_network', 'source_proposal_type', 'status', 'created_at',
            'updated_at', 'id', 'proposal_hash', 'network', 'row_index'
        ]
        
        for col in index_candidates:
            if col in analysis['columns']:
                clean_col = self._clean_column_name(col)
                index_name = f"idx_{self.table_name}_{clean_col}"
                try:
                    cursor.execute(f'CREATE INDEX {index_name} ON {self.table_name} ("{clean_col}");')
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to create index {index_name}: {e}")
        
        # Composite indexes
        composite_indexes = [
            ('source_network', 'source_proposal_type'),
            ('source_network', 'status'),
        ]
        
        for cols in composite_indexes:
            if all(col in analysis['columns'] for col in cols):
                clean_cols = [self._clean_column_name(col) for col in cols]
                index_name = f"idx_{self.table_name}_{'_'.join(clean_cols)}"
                cols_sql = ', '.join([f'"{col}"' for col in clean_cols])
                try:
                    cursor.execute(f'CREATE INDEX {index_name} ON {self.table_name} ({cols_sql});')
                    logger.info(f"Created composite index: {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to create composite index {index_name}: {e}")
    
    def insert_csv_data(self, csv_path: Path, analysis: Dict[str, Any]) -> bool:
        """Insert CSV data into PostgreSQL table in batches"""
        try:
            logger.info(f"Starting data insertion from {csv_path.name}")
            logger.info(f"Total rows to insert: {analysis['total_rows']:,}")
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Clean column names for SQL
                    clean_columns = [self._clean_column_name(col) for col in analysis['columns']]
                    columns_sql = ', '.join([f'"{col}"' for col in clean_columns])
                    
                    # Prepare INSERT statement
                    placeholders = ', '.join(['%s'] * len(clean_columns))
                    insert_sql = f"INSERT INTO {self.table_name} ({columns_sql}) VALUES ({placeholders})"
                    
                    # Process CSV in chunks
                    total_inserted = 0
                    chunk_size = self.batch_size
                    
                    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
                        # Clean and prepare data
                        chunk_data = self._prepare_data_for_insert(chunk_df, analysis)
                        
                        # Insert batch
                        execute_values(
                            cur,
                            f"INSERT INTO {self.table_name} ({columns_sql}) VALUES %s",
                            chunk_data,
                            template=None,
                            page_size=min(1000, len(chunk_data))
                        )
                        
                        total_inserted += len(chunk_data)
                        
                        # Commit periodically
                        if total_inserted % (chunk_size * 5) == 0:
                            conn.commit()
                            logger.info(f"Inserted {total_inserted:,} / {analysis['total_rows']:,} rows ({total_inserted/analysis['total_rows']*100:.1f}%)")
                    
                    # Final commit
                    conn.commit()
                    logger.info(f"Successfully inserted all {total_inserted:,} rows")
                    
                    # Verify insertion
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                    db_count = cur.fetchone()[0]
                    logger.info(f"Database verification: {db_count:,} rows in table")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False
    
    def _prepare_data_for_insert(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[tuple]:
        """Prepare DataFrame data for PostgreSQL insertion with proper type conversion"""
        df_clean = df.copy()
        
        # Convert each column based on schema information and PostgreSQL type
        for col in analysis['columns']:
            if col not in df_clean.columns:
                continue
                
            col_info = analysis['column_info'][col]
            pg_type = col_info['postgres_type']
            
            logger.debug(f"Converting column {col} to {pg_type}")
            
            try:
                if pg_type == 'DATE':
                    df_clean[col] = self._convert_to_date(df_clean[col])
                elif pg_type == 'TIMESTAMP':
                    df_clean[col] = self._convert_to_timestamp(df_clean[col])
                elif pg_type in ['BIGINT', 'INTEGER', 'SMALLINT']:
                    df_clean[col] = self._convert_to_integer(df_clean[col])
                elif pg_type in ['NUMERIC', 'DOUBLE PRECISION']:
                    df_clean[col] = self._convert_to_float(df_clean[col])
                elif pg_type == 'BOOLEAN':
                    df_clean[col] = self._convert_to_boolean(df_clean[col])
                # String types (VARCHAR, TEXT) don't need conversion
                
            except Exception as e:
                logger.error(f"Error converting column {col} to {pg_type}: {e}")
                # Keep original data if conversion fails
                pass
        
        # Convert NaN to None for PostgreSQL
        df_clean = df_clean.where(pd.notna(df_clean), None)
        
        # Also handle string 'NaN' and 'null' values
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].replace(['NaN', 'nan', 'null', 'NULL', ''], None)
        
        # Convert to list of tuples
        return [tuple(row) for row in df_clean.values]
    
    def _convert_to_date(self, series: pd.Series) -> pd.Series:
        """Convert series to date format"""
        def parse_date(value):
            if pd.isna(value) or value is None or value == '':
                return None
            
            # Handle various date formats
            try:
                # Try pandas to_datetime first
                parsed = pd.to_datetime(value, errors='coerce')
                if pd.isna(parsed):
                    return None
                return parsed.date()
            except:
                return None
        
        return series.apply(parse_date)
    
    def _convert_to_timestamp(self, series: pd.Series) -> pd.Series:
        """Convert series to timestamp format"""
        def parse_timestamp(value):
            if pd.isna(value) or value is None or value == '':
                return None
            
            try:
                # Try pandas to_datetime first
                parsed = pd.to_datetime(value, errors='coerce')
                if pd.isna(parsed):
                    return None
                return parsed
            except:
                return None
        
        return series.apply(parse_timestamp)
    
    def _convert_to_integer(self, series: pd.Series) -> pd.Series:
        """Convert series to integer format"""
        def parse_integer(value):
            if pd.isna(value) or value is None or value == '':
                return None
            
            try:
                # Handle string representations
                if isinstance(value, str):
                    value = value.strip()
                    if value == '':
                        return None
                
                # Convert to float first to handle decimal strings, then to int
                float_val = float(value)
                if np.isnan(float_val):
                    return None
                
                # Check if the value is within reasonable integer range
                int_val = int(float_val)
                
                return int_val
            except:
                return None
        
        return series.apply(parse_integer)
    
    def _convert_to_float(self, series: pd.Series) -> pd.Series:
        """Convert series to float format"""
        def parse_float(value):
            if pd.isna(value) or value is None or value == '':
                return None
            
            try:
                if isinstance(value, str):
                    value = value.strip()
                    if value == '':
                        return None
                
                float_val = float(value)
                if np.isnan(float_val):
                    return None
                return float_val
            except:
                return None
        
        return series.apply(parse_float)
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean format"""
        def parse_boolean(value):
            if pd.isna(value) or value is None or value == '':
                return None
            
            try:
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower in ['true', '1', 'yes', 'y', 't']:
                        return True
                    elif value_lower in ['false', '0', 'no', 'n', 'f']:
                        return False
                    else:
                        return None
                elif isinstance(value, (int, float)):
                    return bool(value)
                else:
                    return None
            except:
                return None
        
        return series.apply(parse_boolean)
    
    def get_table_stats(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the inserted data, if relevant columns exist."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    stats = {}
                    # Basic stats
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                    stats['total_rows'] = cur.fetchone()[0]
                    
                    # Network distribution (if column exists)
                    if 'source_network' in analysis['columns']:
                        cur.execute(f"""
                            SELECT source_network, COUNT(*) as count 
                            FROM {self.table_name} 
                            GROUP BY source_network 
                            ORDER BY count DESC;
                        """)
                        stats['network_distribution'] = dict(cur.fetchall())
                    
                    # Proposal type distribution (if column exists)
                    if 'source_proposal_type' in analysis['columns']:
                        cur.execute(f"""
                            SELECT source_proposal_type, COUNT(*) as count 
                            FROM {self.table_name} 
                            GROUP BY source_proposal_type 
                            ORDER BY count DESC;
                        """)
                        stats['proposal_type_distribution'] = dict(cur.fetchall())

                    return stats
                    
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}
    
    def run_full_import(self, csv_path: Path, drop_existing: bool = False) -> bool:
        """Run the complete import process"""
        logger.info("Starting full import process...")
        
        try:
            # Test connection
            if not self.test_connection():
                logger.error("Database connection test failed")
                return False
            
            # Analyze CSV structure
            analysis = self.analyze_csv_structure(csv_path)
            
            # Create table
            if not self.create_table(analysis, drop_if_exists=drop_existing):
                logger.error("Table creation failed")
                return False
            
            # Insert data
            if not self.insert_csv_data(csv_path, analysis):
                logger.error("Data insertion failed")
                return False
            
            # Get final stats
            stats = self.get_table_stats(analysis)
            if stats:
                logger.info("Import completed successfully!")
                logger.info(f"Final statistics:")
                logger.info(f"  Total rows: {stats.get('total_rows', 'N/A')}")
                if 'network_distribution' in stats:
                    logger.info(f"  Networks: {stats['network_distribution']}")
                if 'proposal_type_distribution' in stats:
                    logger.info(f"  Proposal types: {len(stats['proposal_type_distribution'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Import process failed: {e}")
            return False

def main():
    """Main execution function"""
    # Path to the specific CSV file to insert
    csv_file = Path(str(os.getenv("BASE_PATH")) + "/onchain_data/onchain_first_pull/one_table/filter_data/governance_data_86.csv")    
    
    if not csv_file.exists():
        logger.error(f"CSV file not found at: {csv_file}")
        logger.error("Please ensure the governance_data_86.csv file exists in the specified location.")
        return
    
    try:
        # Create inserter
        inserter = PostgresInserter()
        
        logger.info("Dropping existing table")
        # Automatically drop existing table
        drop_existing = True  # Always drop existing table
        
        # Run import
        success = inserter.run_full_import(csv_file, drop_existing=drop_existing)
        
        if success:
            logger.info("🎉 Import completed successfully!")
        else:
            logger.error("❌ Import failed!")
            
    except KeyboardInterrupt:
        logger.info("Import cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
