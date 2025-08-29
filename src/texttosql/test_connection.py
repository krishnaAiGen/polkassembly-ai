#!/usr/bin/env python3
"""
Test PostgreSQL connection with the correct .env file path
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment variables from the main polkassembly-ai directory
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

print(f"Loading .env from: {env_path}")
print(f".env file exists: {env_path.exists()}")

# Check if environment variables are loaded
db_config = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DATABASE'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

print("\nDatabase Configuration:")
for key, value in db_config.items():
    if key == 'password':
        print(f"  {key}: {'*' * len(value) if value else 'None'}")
    else:
        print(f"  {key}: {value}")

# Test connection to default postgres database first
try:
    print("\n1. Testing connection to default 'postgres' database...")
    default_config = db_config.copy()
    default_config['database'] = 'postgres'
    
    conn = psycopg2.connect(**default_config)
    cursor = conn.cursor()
    
    # Test query
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✅ Connection to 'postgres' database successful!")
    print(f"PostgreSQL version: {version[0]}")
    
    # List all databases
    cursor.execute("SELECT datname FROM pg_database ORDER BY datname;")
    databases = cursor.fetchall()
    print(f"\n📋 ALL Available databases:")
    for db in databases:
        print(f"  - {db[0]}")
    
    # Check if onchain-data exists (case variations)
    cursor.execute("SELECT datname FROM pg_database WHERE datname ILIKE '%onchain%' OR datname ILIKE '%data%';")
    matching_dbs = cursor.fetchall()
    print(f"\n🔍 Databases matching 'onchain' or 'data':")
    for db in matching_dbs:
        print(f"  - {db[0]}")
    
    cursor.close()
    conn.close()
    
    # Now try connecting to the target database
    print(f"\n2. Testing connection to '{db_config['database']}' database...")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    print(f"✅ Connection to '{db_config['database']}' successful!")
    cursor.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    if "does not exist" in str(e):
        print(f"❌ Database '{db_config['database']}' does not exist")
        print("💡 Try using one of the available databases listed above")
    else:
        print(f"❌ Connection failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
