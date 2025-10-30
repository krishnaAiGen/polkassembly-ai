import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_dir = os.getenv("ONCHAIN_DATA_DIR", "")

def explore_onchain_data():
    """Explore and print the first index of every JSON file in the onchain_data directory"""
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Directory does not exist: {data_dir}")
        return
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"📁 No JSON files found in: {data_dir}")
        return
    
    print(f"🔍 Exploring JSON files in: {data_dir}")
    print("=" * 80)
    
    for filename in sorted(json_files):
        file_path = os.path.join(data_dir, filename)
        
        try:
            print(f"\n📄 File: {filename}")
            print("-" * 50)
            
            # Read and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if data is a list or dict
            if isinstance(data, list):
                if len(data) > 0:
                    print(f"📊 Type: List with {len(data)} items")
                    print(f"🎯 First item:")
                    print(json.dumps(data[0], indent=2, ensure_ascii=False))
                else:
                    print("📊 Type: Empty list")
                    
            elif isinstance(data, dict):
                print(f"📊 Type: Dictionary with {len(data)} keys")
                print(f"🔑 Keys: {list(data.keys())[:10]}{'...' if len(data.keys()) > 10 else ''}")
                
                # Check if there's an 'items' array and show first item from it
                if 'items' in data and isinstance(data['items'], list) and len(data['items']) > 0:
                    print(f"🎯 First item from 'items' array (total items: {len(data['items'])}):")
                    first_item = data['items'][0]
                    print(json.dumps(first_item, indent=2, ensure_ascii=False))
                else:
                    print(f"🎯 First few key-value pairs:")
                    # Show first few items
                    for i, (key, value) in enumerate(data.items()):
                        if i >= 3:  # Show only first 3 items
                            print("    ...")
                            break
                        value_str = str(value)
                        print(f"    {key}: {value_str}")
                    
            else:
                print(f"📊 Type: {type(data).__name__}")
                print(f"🎯 Content: {str(data)}")
                
            # File size info
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            
            print(f"💾 File size: {size_str}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error in {filename}: {e}")
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    print("\n" + "=" * 80)
    print(f"✅ Exploration complete! Found {len(json_files)} JSON files.")

def show_file_summary():
    """Show a quick summary of all files"""
    if not os.path.exists(data_dir):
        print(f"❌ Directory does not exist: {data_dir}")
        return
        
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    print(f"\n📋 Quick Summary of {len(json_files)} JSON files:")
    print("-" * 60)
    
    for filename in sorted(json_files):
        file_path = os.path.join(data_dir, filename)
        try:
            file_size = os.path.getsize(file_path)
            size_str = f"{file_size / 1024:.1f} KB" if file_size > 1024 else f"{file_size} B"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                item_count = len(data)
                data_type = f"List ({item_count} items)"
            elif isinstance(data, dict):
                item_count = len(data)
                data_type = f"Dict ({item_count} keys)"
            else:
                data_type = type(data).__name__
            
            print(f"  {filename:<30} | {size_str:>8} | {data_type}")
            
        except Exception as e:
            print(f"  {filename:<30} | {'ERROR':>8} | {str(e)[:30]}")

if __name__ == "__main__":
    print("🚀 Onchain Data Explorer")
    print("=" * 80)
    
    # Show quick summary first
    show_file_summary()
    
    # Ask user if they want detailed exploration
    print(f"\n❓ Do you want detailed exploration of each file? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', '1']:
            explore_onchain_data()
        else:
            print("👋 Skipping detailed exploration.")
    except KeyboardInterrupt:
        print("\n👋 Exploration cancelled.")
    except:
        # If running in non-interactive mode, show detailed exploration
        print("Running in non-interactive mode, showing detailed exploration...")
        explore_onchain_data()
