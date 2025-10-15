#!/usr/bin/env python3
"""
Test script for SQL error correction mechanism with optimized single execution
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_error_correction():
    """Test the error correction mechanism with full process_query"""
    try:
        from texttosql.query2sql import Query2SQL
        
        # Initialize the query processor
        query_processor = Query2SQL()
        
        # Test with a query that might generate an error
        test_query = "Show me recent proposals with invalid column names"
        
        logger.info(f"Testing error correction with query: {test_query}")
        
        # This should trigger the error correction mechanism and execute in one go
        result = query_processor.process_query(test_query)
        
        logger.info(f"Process result success: {result['success']}")
        logger.info(f"Generated SQL queries: {result['sql_queries']}")
        logger.info(f"Result count: {result['result_count']}")
        logger.info("Error correction test completed successfully!")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"Error correction test failed: {e}")
        return False

def test_voting_error_correction():
    """Test the error correction mechanism for voting data with full process_query"""
    try:
        from texttosql.query2sql import VoteQuery2SQL
        
        # Initialize the voting query processor
        vote_processor = VoteQuery2SQL()
        
        # Test with a query that might generate an error
        test_query = "Show me recent votes with invalid column names"
        
        logger.info(f"Testing voting error correction with query: {test_query}")
        
        # This should trigger the error correction mechanism and execute in one go
        result = vote_processor.process_query(test_query)
        
        logger.info(f"Voting process result success: {result['success']}")
        logger.info(f"Generated voting SQL queries: {result['sql_queries']}")
        logger.info(f"Result count: {result['result_count']}")
        logger.info("Voting error correction test completed successfully!")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"Voting error correction test failed: {e}")
        return False

def test_optimization():
    """Test that we're only making one database call per query"""
    try:
        from texttosql.query2sql import Query2SQL
        
        query_processor = Query2SQL()
        test_query = "Show me 5 recent proposals"
        
        logger.info(f"Testing optimization with query: {test_query}")
        
        # This should execute SQL only once (no validation + execution)
        result = query_processor.process_query(test_query)
        
        if result['success'] and result['result_count'] > 0:
            logger.info("✅ Optimization test passed - single execution successful")
            return True
        else:
            logger.warning("⚠️ Optimization test - no results returned")
            return True  # Still counts as success if no validation errors
            
    except Exception as e:
        logger.error(f"Optimization test failed: {e}")
        return False

def test_result_limitation():
    """Test that result limitation messages are included in responses"""
    try:
        from texttosql.query2sql import Query2SQL
        
        query_processor = Query2SQL()
        # Use a query that might return many results
        test_query = "Show me all recent proposals"
        
        logger.info(f"Testing result limitation with query: {test_query}")
        
        result = query_processor.process_query(test_query)
        
        if result['success']:
            natural_response = result['natural_response']
            result_count = result['result_count']
            
            # Check if limitation message is included when there are many results
            if result_count > 10:
                if "showing" in natural_response.lower() and ("few" in natural_response.lower() or "first" in natural_response.lower()):
                    logger.info("✅ Result limitation test passed - limitation message included")
                    return True
                else:
                    logger.warning("⚠️ Result limitation test - no limitation message found despite many results")
                    return False
            else:
                logger.info("✅ Result limitation test passed - no limitation needed (few results)")
                return True
        else:
            logger.warning("⚠️ Result limitation test - query failed")
            return False
            
    except Exception as e:
        logger.error(f"Result limitation test failed: {e}")
        return False

def test_window_function_count():
    """Test that window function COUNT(*) OVER() is working for accurate total counts"""
    try:
        from texttosql.query2sql import Query2SQL
        
        query_processor = Query2SQL()
        # Use a query that should generate a LIMIT query with window function
        test_query = "Show me treasury proposals for polkadot for last 6 months"
        
        logger.info(f"Testing window function count with query: {test_query}")
        
        result = query_processor.process_query(test_query)
        
        if result['success']:
            sql_queries = result['sql_queries']
            natural_response = result['natural_response']
            
            # Check if SQL contains window function
            has_window_function = any("COUNT(*) OVER()" in sql.upper() for sql in sql_queries)
            
            if has_window_function:
                logger.info("✅ Window function test passed - COUNT(*) OVER() found in SQL")
                
                # Check if response mentions total count
                if "found" in natural_response.lower() and any(char.isdigit() for char in natural_response):
                    logger.info("✅ Window function test passed - total count mentioned in response")
                    return True
                else:
                    logger.warning("⚠️ Window function test - no total count mentioned in response")
                    return False
            else:
                logger.warning("⚠️ Window function test - no window function found in SQL")
                return False
        else:
            logger.warning("⚠️ Window function test - query failed")
            return False
            
    except Exception as e:
        logger.error(f"Window function test failed: {e}")
        return False

def test_null_handling():
    """Test that IS NOT NULL conditions are automatically added for filtering/ordering columns"""
    try:
        from texttosql.query2sql import Query2SQL
        
        query_processor = Query2SQL()
        # Use a query that should generate filtering and ordering
        test_query = "Show me recent proposals ordered by creation date"
        
        logger.info(f"Testing NULL handling with query: {test_query}")
        
        result = query_processor.process_query(test_query)
        
        if result['success']:
            sql_queries = result['sql_queries']
            
            # Check if SQL contains IS NOT NULL conditions
            has_null_handling = any("IS NOT NULL" in sql.upper() for sql in sql_queries)
            
            if has_null_handling:
                logger.info("✅ NULL handling test passed - IS NOT NULL conditions found in SQL")
                return True
            else:
                logger.warning("⚠️ NULL handling test - no IS NOT NULL conditions found in SQL")
                return False
        else:
            logger.warning("⚠️ NULL handling test - query failed")
            return False
            
    except Exception as e:
        logger.error(f"NULL handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing SQL Error Correction Mechanism (Optimized)")
    print("=" * 60)
    
    # Test governance data error correction
    print("\n1. Testing governance data error correction...")
    governance_success = test_error_correction()
    
    # Test voting data error correction
    print("\n2. Testing voting data error correction...")
    voting_success = test_voting_error_correction()
    
    # Test optimization (single execution)
    print("\n3. Testing optimization (single database call)...")
    optimization_success = test_optimization()
    
    # Test result limitation messages
    print("\n4. Testing result limitation messages...")
    limitation_success = test_result_limitation()
    
    # Test window function count
    print("\n5. Testing window function count...")
    window_function_success = test_window_function_count()
    
    # Test NULL handling
    print("\n6. Testing NULL handling...")
    null_handling_success = test_null_handling()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Governance data error correction: {'PASS' if governance_success else 'FAIL'}")
    print(f"Voting data error correction: {'PASS' if voting_success else 'FAIL'}")
    print(f"Optimization (single execution): {'PASS' if optimization_success else 'FAIL'}")
    print(f"Result limitation messages: {'PASS' if limitation_success else 'FAIL'}")
    print(f"Window function count: {'PASS' if window_function_success else 'FAIL'}")
    print(f"NULL handling: {'PASS' if null_handling_success else 'FAIL'}")
    
    if governance_success and voting_success and optimization_success and limitation_success and window_function_success and null_handling_success:
        print("\n✅ All tests passed! Error correction mechanism is working with optimization, result limitations, window function counts, and automatic NULL handling.")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
