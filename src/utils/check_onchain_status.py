#!/usr/bin/env python3
"""
Utility script to check the status of onchain data and embeddings
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .onchain_data import get_onchain_data_status, check_existing_data, check_existing_embeddings

def main():
    print("=== ONCHAIN DATA STATUS CHECK ===")
    
    # Get comprehensive status
    status = get_onchain_data_status(use_multi_collection=True)
    
    print("\n📊 OVERALL STATUS:")
    print(f"Ready for use: {'✓ YES' if status['ready'] else '✗ NO'}")
    
    print("\n📁 DATA STATUS:")
    data_info = status['data']
    if data_info['exists']:
        print("✓ Data exists")
        info = data_info['info']
        print(f"  Total files: {info['total_files']}")
        
        if info.get('latest_extraction'):
            extraction = info['latest_extraction']
            print(f"  Latest extraction: {extraction['extraction_timestamp']}")
            print(f"  Total items: {extraction['total_items_fetched']}")
            print(f"  Networks: {', '.join(extraction['networks_processed'])}")
        
        print("  Files by network:")
        for network, net_info in info['networks'].items():
            print(f"    {network}: {net_info['files']} files")
    else:
        print("✗ No data found")
        print(f"  Reason: {data_info['info'].get('reason', 'Unknown')}")
    
    print("\n🔮 EMBEDDINGS STATUS:")
    embedding_info = status['embeddings']
    if embedding_info['exists']:
        print("✓ Embeddings exist")
        info = embedding_info['info']
        print(f"  Total chunks: {info.get('total_chunks', 0)}")
        print(f"  Embedding model: {info.get('embedding_model', 'Unknown')}")
        
        # Multi-collection information
        if 'static_collection' in info:
            static_info = info['static_collection']
            print(f"  Static collection: {static_info.get('chunks', 0)} chunks")
            if 'sources' in static_info:
                for source, count in static_info['sources'].items():
                    print(f"    {source}: {count}")
        
        if 'onchain_collection' in info:
            onchain_info = info['onchain_collection']
            print(f"  Onchain collection: {onchain_info.get('chunks', 0)} chunks")
            if 'sources' in onchain_info:
                for source, count in onchain_info['sources'].items():
                    print(f"    {source}: {count}")
        
        # Fallback for single collection
        if 'chunks_by_source' in info:
            print("  Chunks by source:")
            for source, count in info['chunks_by_source'].items():
                print(f"    {source}: {count}")
    else:
        print("✗ No embeddings found")
        print(f"  Reason: {embedding_info['info'].get('reason', 'Unknown')}")
    
    print("\n💡 RECOMMENDATIONS:")
    if status['ready']:
        print("✓ System is ready! You can start the server with: python run_server.py")
    else:
        if not data_info['exists']:
            print("→ Run data extraction: python -m src.rag.create_embeddings --extract-only")
        if not embedding_info['exists']:
            print("→ Create embeddings: python -m src.rag.create_embeddings --embeddings-only")
        print("→ Or run full setup: python -m src.rag.create_embeddings --onchain-only")

if __name__ == "__main__":
    main() 