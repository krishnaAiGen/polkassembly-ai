#!/usr/bin/env python3
"""
Test script to verify the step 3 fix works correctly.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent  # Gets us to polkassembly-ai/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_csv_combiner_methods():
    """Test that CSVCombiner has the correct methods"""
    print("🔍 Testing CSVCombiner methods...")
    
    try:
        from texttosql.create_one_table import CSVCombiner
        
        # Check available methods
        methods = [method for method in dir(CSVCombiner) if not method.startswith('_')]
        print(f"✅ CSVCombiner methods: {methods}")
        
        # Check if run_combination exists
        if hasattr(CSVCombiner, 'run_combination'):
            print("✅ run_combination method exists")
        else:
            print("❌ run_combination method NOT found")
            return False
        
        # Check if combine_all_csvs exists (this should NOT exist)
        if hasattr(CSVCombiner, 'combine_all_csvs'):
            print("⚠️  combine_all_csvs method exists (unexpected)")
        else:
            print("✅ combine_all_csvs method correctly does NOT exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CSVCombiner: {e}")
        return False

def test_data_flattener_methods():
    """Test that DataFlattener has the correct methods"""
    print("\n🔍 Testing DataFlattener methods...")
    
    try:
        from texttosql.flatten_all_data import DataFlattener
        
        # Check available methods
        methods = [method for method in dir(DataFlattener) if not method.startswith('_')]
        print(f"✅ DataFlattener methods: {methods}")
        
        # Check if process_all_files exists
        if hasattr(DataFlattener, 'process_all_files'):
            print("✅ process_all_files method exists")
        else:
            print("❌ process_all_files method NOT found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DataFlattener: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Step 3 Fix")
    print("=" * 40)
    
    # Test CSVCombiner
    csv_combiner_ok = test_csv_combiner_methods()
    
    # Test DataFlattener
    data_flattener_ok = test_data_flattener_methods()
    
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"  CSVCombiner: {'✅ OK' if csv_combiner_ok else '❌ FAIL'}")
    print(f"  DataFlattener: {'✅ OK' if data_flattener_ok else '❌ FAIL'}")
    
    if csv_combiner_ok and data_flattener_ok:
        print("\n🎉 All tests passed! Step 3 fix should work correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
