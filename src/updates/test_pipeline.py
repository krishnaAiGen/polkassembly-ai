#!/usr/bin/env python3
"""
Test script for the onchain data pipeline.
This script tests each step individually to ensure everything works.
"""

import os
import sys
from pathlib import Path
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent  # Gets us to polkassembly-ai/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from insert_onchain_to_postgres import OnchainDataPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_setup():
    """Test that the pipeline can be initialized"""
    logger.info("Testing pipeline setup...")
    
    try:
        pipeline = OnchainDataPipeline()
        logger.info("✅ Pipeline initialized successfully")
        logger.info(f"📁 Data directory: {pipeline.updates_data_dir}")
        
        # Check that directories were created
        directories = [
            pipeline.raw_data_dir,
            pipeline.csv_data_dir,
            pipeline.combined_data_dir,
            pipeline.filtered_data_dir,
            pipeline.new_records_dir
        ]
        
        for directory in directories:
            if directory.exists():
                logger.info(f"✅ Directory exists: {directory.name}")
            else:
                logger.error(f"❌ Directory missing: {directory.name}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline setup failed: {e}")
        return False

def test_environment_variables():
    """Test that required environment variables are set"""
    logger.info("Testing environment variables...")
    
    required_vars = [
        'POSTGRES_HOST',
        'POSTGRES_DATABASE', 
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            logger.info(f"✅ {var} is set")
    
    if missing_vars:
        logger.warning(f"⚠️  Missing environment variables: {missing_vars}")
        logger.warning("Pipeline will work but step 5 (database comparison) may fail")
        return True  # Not critical for testing
    else:
        logger.info("✅ All required environment variables are set")
        return True

def test_script_files():
    """Test that required script files exist"""
    logger.info("Testing script files...")
    
    scripts = [
        project_root / "src" / "data" / "onchain_data.py",
        project_root / "src" / "texttosql" / "flatten_all_data.py",
        project_root / "src" / "texttosql" / "create_one_table.py",
        project_root / "src" / "texttosql" / "filter_data.py"
    ]
    
    all_exist = True
    for script in scripts:
        if script.exists():
            logger.info(f"✅ Script exists: {script.name}")
        else:
            logger.error(f"❌ Script missing: {script}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    logger.info("🧪 Testing Onchain Data Pipeline")
    logger.info("=" * 50)
    
    tests = [
        ("Pipeline Setup", test_pipeline_setup),
        ("Environment Variables", test_environment_variables),
        ("Script Files", test_script_files)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        results[test_name] = test_func()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🏁 Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("✅ All tests passed! Pipeline is ready to run.")
        logger.info("\n📝 To run the full pipeline:")
        logger.info("   cd polkassembly-ai/src/updates")
        logger.info("   python insert_onchain_to_postgres.py")
    else:
        logger.error("❌ Some tests failed. Please fix issues before running pipeline.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
