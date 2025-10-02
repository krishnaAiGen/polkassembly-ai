#!/usr/bin/env python3
"""
Create one giant CSV by combining all CSV files from csv_files directory.
Handles overlapping and unique columns properly.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVCombiner:
    def __init__(self, csv_directory: str, output_directory: str = None):
        self.csv_dir = Path(csv_directory)
        self.output_dir = Path(output_directory) if output_directory else self.csv_dir.parent
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def analyze_csv_files(self) -> Dict[str, Dict]:
        """Analyze all CSV files to understand their structure"""
        csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        file_info = {}
        all_columns = set()
        
        for csv_file in csv_files:
            try:
                logger.info(f"Analyzing {csv_file.name}")
                
                # Read just the header to get column info
                df_sample = pd.read_csv(csv_file, nrows=0)
                columns = list(df_sample.columns)
                
                # Get basic file info
                df_full = pd.read_csv(csv_file)
                
                file_info[csv_file.name] = {
                    'path': csv_file,
                    'columns': columns,
                    'column_count': len(columns),
                    'row_count': len(df_full),
                    'file_size_mb': csv_file.stat().st_size / (1024 * 1024)
                }
                
                # Add to global column set
                all_columns.update(columns)
                
                logger.info(f"  - {len(columns)} columns, {len(df_full)} rows, {file_info[csv_file.name]['file_size_mb']:.2f} MB")
                
            except Exception as e:
                logger.error(f"Error analyzing {csv_file.name}: {e}")
                continue
        
        logger.info(f"Total unique columns across all files: {len(all_columns)}")
        return file_info, sorted(all_columns)
    
    def find_column_overlaps(self, file_info: Dict[str, Dict]) -> Dict[str, any]:
        """Find which columns overlap between files"""
        column_frequency = {}
        column_to_files = {}
        
        for filename, info in file_info.items():
            for column in info['columns']:
                if column not in column_frequency:
                    column_frequency[column] = 0
                    column_to_files[column] = []
                
                column_frequency[column] += 1
                column_to_files[column].append(filename)
        
        # Categorize columns
        common_columns = {col: freq for col, freq in column_frequency.items() if freq > len(file_info) * 0.5}
        rare_columns = {col: freq for col, freq in column_frequency.items() if freq <= 3}
        unique_columns = {col: freq for col, freq in column_frequency.items() if freq == 1}
        
        overlap_analysis = {
            'column_frequency': column_frequency,
            'column_to_files': column_to_files,
            'common_columns': common_columns,
            'rare_columns': rare_columns,
            'unique_columns': unique_columns,
            'total_columns': len(column_frequency)
        }
        
        logger.info(f"Column overlap analysis:")
        logger.info(f"  - Common columns (>50% of files): {len(common_columns)}")
        logger.info(f"  - Rare columns (≤3 files): {len(rare_columns)}")
        logger.info(f"  - Unique columns (1 file only): {len(unique_columns)}")
        
        return overlap_analysis
    
    def add_source_tracking_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Add columns to track source file and metadata"""
        # Extract metadata from filename
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) >= 2:
            network = parts[0] if parts[0] in ['polkadot', 'kusama'] else 'unknown'
            proposal_type = parts[1] if len(parts) > 1 else 'unknown'
        else:
            network = 'unknown'
            proposal_type = 'unknown'
        
        # Add tracking columns at the beginning
        df.insert(0, 'source_file', filename)
        df.insert(1, 'source_network', network)
        df.insert(2, 'source_proposal_type', proposal_type)
        df.insert(3, 'source_row_id', range(1, len(df) + 1))
        
        return df
    
    def combine_csv_files(self, file_info: Dict[str, Dict], all_columns: List[str]) -> pd.DataFrame:
        """Combine all CSV files into one giant DataFrame"""
        logger.info("Starting to combine CSV files...")
        
        combined_dfs = []
        total_rows = 0
        
        for filename, info in file_info.items():
            try:
                logger.info(f"Processing {filename} ({info['row_count']} rows)")
                
                # Read the CSV file
                df = pd.read_csv(info['path'])
                
                # Add source tracking columns
                df = self.add_source_tracking_columns(df, filename)
                
                # Reindex to match all columns (fills missing columns with NaN)
                current_columns = list(df.columns)
                missing_columns = [col for col in all_columns if col not in current_columns and not col.startswith('source_')]
                
                # Add missing columns with NaN values
                for col in missing_columns:
                    df[col] = pd.NA
                
                # Reorder columns to match the master column order
                source_cols = ['source_file', 'source_network', 'source_proposal_type', 'source_row_id']
                other_cols = [col for col in all_columns if col not in source_cols]
                final_column_order = source_cols + other_cols
                
                # Only select columns that exist in the dataframe
                available_columns = [col for col in final_column_order if col in df.columns]
                df = df[available_columns]
                
                combined_dfs.append(df)
                total_rows += len(df)
                
                logger.info(f"  - Added {len(df)} rows, running total: {total_rows}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        if not combined_dfs:
            raise ValueError("No CSV files could be processed successfully")
        
        # Combine all DataFrames
        logger.info("Concatenating all DataFrames...")
        combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
        
        logger.info("Creating row_index column...")
        # Per user request, create a row_index from specific columns with an underscore separator.
        # Using corrected column names for robustness: source_network and onchaininfo_hash.
        cols_to_concat = ['source_network', 'source_proposal_type', 'index', 'onchaininfo_hash']
        separator = "_"

        # Ensure required columns exist to avoid errors
        for col in cols_to_concat:
            if col not in combined_df.columns:
                logger.warning(f"'{col}' column not found for row_index generation. It will be treated as empty.")
                combined_df[col] = ''

        # Create the new column by joining the specified columns with the separator
        combined_df["row_index"] = combined_df[cols_to_concat].astype(str).agg(separator.join, axis=1)
        
        logger.info(f"Combined DataFrame created: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df
    
    def generate_summary_report(self, file_info: Dict[str, Dict], overlap_analysis: Dict, 
                              combined_df: pd.DataFrame) -> Dict:
        """Generate a comprehensive summary report"""
        
        # Source file statistics
        source_stats = combined_df['source_file'].value_counts().to_dict()
        network_stats = combined_df['source_network'].value_counts().to_dict()
        proposal_type_stats = combined_df['source_proposal_type'].value_counts().to_dict()
        
        # Data quality analysis
        null_counts = combined_df.isnull().sum().sort_values(ascending=False)
        columns_with_data = (combined_df.notna().sum() > 0).sum()
        completely_empty_columns = (combined_df.isnull().all()).sum()
        
        summary = {
            'total_files_processed': len(file_info),
            'total_rows': len(combined_df),
            'total_columns': len(combined_df.columns),
            'source_file_distribution': source_stats,
            'network_distribution': network_stats,
            'proposal_type_distribution': proposal_type_stats,
            'data_quality': {
                'columns_with_data': columns_with_data,
                'completely_empty_columns': completely_empty_columns,
                'top_10_columns_with_nulls': dict(null_counts.head(10))
            },
            'column_analysis': {
                'common_columns': len(overlap_analysis['common_columns']),
                'rare_columns': len(overlap_analysis['rare_columns']),
                'unique_columns': len(overlap_analysis['unique_columns'])
            }
        }
        
        return summary
    
    def save_combined_csv(self, combined_df: pd.DataFrame, filename: str = "combined_governance_data.csv") -> Path:
        """Save the combined DataFrame to CSV"""
        output_path = self.output_dir / filename
        
        logger.info(f"Saving combined CSV to {output_path}")
        combined_df.to_csv(output_path, index=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Combined CSV saved: {file_size_mb:.2f} MB")
        
        return output_path
    
    def save_summary_report(self, summary: Dict, filename: str = "combination_summary.json") -> Path:
        """Save the summary report to JSON"""
        import json
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {output_path}")
        return output_path
    
    def run_combination(self) -> Dict[str, Path]:
        """Run the complete CSV combination process"""
        logger.info("Starting CSV combination process...")
        
        # Step 1: Analyze all CSV files
        file_info, all_columns = self.analyze_csv_files()
        
        if not file_info:
            raise ValueError("No valid CSV files found to process")
        
        # Step 2: Analyze column overlaps
        overlap_analysis = self.find_column_overlaps(file_info)
        
        # Step 3: Combine all CSV files
        combined_df = self.combine_csv_files(file_info, all_columns)
        
        # Step 4: Generate summary report
        summary = self.generate_summary_report(file_info, overlap_analysis, combined_df)
        
        # Step 5: Save results
        csv_path = self.save_combined_csv(combined_df)
        summary_path = self.save_summary_report(summary)
        
        logger.info("CSV combination process completed successfully!")
        
        return {
            'combined_csv': csv_path,
            'summary_report': summary_path,
            'summary_data': summary
        }

def main():
    """Main execution function"""
    # Configuration
    csv_directory = Path(__file__).parent.parent / "data" / "csv_files"
    output_directory = Path(__file__).parent.parent / "data" / "one_table"
    
    if not csv_directory.exists():
        logger.error(f"CSV directory not found: {csv_directory}")
        return
    
    # Initialize combiner
    combiner = CSVCombiner(str(csv_directory), str(output_directory))
    
    try:
        # Run combination process
        results = combiner.run_combination()
        
        # Print summary
        summary = results['summary_data']
        print("\n" + "="*60)
        print("CSV COMBINATION SUMMARY")
        print("="*60)
        print(f"Files processed: {summary['total_files_processed']}")
        print(f"Total rows: {summary['total_rows']:,}")
        print(f"Total columns: {summary['total_columns']}")
        print(f"Output file: {results['combined_csv']}")
        print(f"Summary report: {results['summary_report']}")
        
        print("\nNetwork distribution:")
        for network, count in summary['network_distribution'].items():
            print(f"  - {network}: {count:,} rows")
        
        print("\nProposal type distribution:")
        for prop_type, count in summary['proposal_type_distribution'].items():
            print(f"  - {prop_type}: {count:,} rows")
        
        print("\nData quality:")
        print(f"  - Columns with data: {summary['data_quality']['columns_with_data']}")
        print(f"  - Empty columns: {summary['data_quality']['completely_empty_columns']}")
        
    except Exception as e:
        logger.error(f"Error during CSV combination: {e}")
        raise

if __name__ == "__main__":
    main()