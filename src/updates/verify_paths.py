#!/usr/bin/env python3
"""
Simple script to verify that all paths are correct.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent  # Gets us to polkassembly-ai/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def verify_paths():
    """Verify all required paths exist"""
    print("🔍 Verifying paths...")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Absolute project root: {project_root.absolute()}")
    
    # Check required script files
    scripts = [
        project_root / "src" / "data" / "onchain_data.py",
        project_root / "src" / "texttosql" / "flatten_all_data.py",
        project_root / "src" / "texttosql" / "create_one_table.py",
        project_root / "src" / "texttosql" / "filter_data.py"
    ]
    
    all_exist = True
    for script in scripts:
        if script.exists():
            print(f"✅ {script.name} exists at: {script}")
        else:
            print(f"❌ {script.name} NOT FOUND at: {script}")
            all_exist = False
    
    return all_exist

def test_import():
    """Test importing the pipeline"""
    try:
        from insert_onchain_to_postgres import OnchainDataPipeline
        print("✅ Successfully imported OnchainDataPipeline")
        
        # Test initialization
        pipeline = OnchainDataPipeline()
        print(f"✅ Pipeline initialized with data dir: {pipeline.updates_data_dir}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import/initialize pipeline: {e}")
        return False

def main():
    print("🧪 Path Verification Script")
    print("=" * 40)
    
    # Test paths
    paths_ok = verify_paths()
    
    print("\n" + "=" * 40)
    
    # Test import
    import_ok = test_import()
    
    print("\n" + "=" * 40)
    print("📋 Results:")
    print(f"  Paths: {'✅ OK' if paths_ok else '❌ FAIL'}")
    print(f"  Import: {'✅ OK' if import_ok else '❌ FAIL'}")
    
    if paths_ok and import_ok:
        print("\n🎉 All verifications passed! Pipeline is ready to run.")
        return 0
    else:
        print("\n❌ Some verifications failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
