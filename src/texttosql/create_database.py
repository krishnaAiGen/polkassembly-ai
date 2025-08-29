#!/usr/bin/env python3
"""
Create the onchain-data database on AWS RDS PostgreSQL
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment variables from the main polkassembly-ai directory
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def create_database():
    """Create the onchain-data database"""
    
    # Connect to default postgres database first
    db_config = {
        'host': os.getenv('POSTGRES_HOST'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': 'postgres',  # Connect to default database
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }
    
    target_database = os.getenv('POSTGRES_DATABASE')
    
    try:
        print(f"Connecting to PostgreSQL server...")
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True  # Required for CREATE DATABASE
        cursor = conn.cursor()
        
        # Check if database already exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_database,))
        exists = cursor.fetchone()
        
        if exists:
            print(f"✅ Database '{target_database}' already exists!")
        else:
            print(f"Creating database '{target_database}'...")
            cursor.execute(f'CREATE DATABASE "{target_database}"')
            print(f"✅ Database '{target_database}' created successfully!")
        
        # Verify creation
        cursor.execute("SELECT datname FROM pg_database WHERE datname = %s", (target_database,))
        result = cursor.fetchone()
        
        if result:
            print(f"✅ Verified: Database '{target_database}' exists")
            
            # Test connection to new database
            print(f"Testing connection to '{target_database}'...")
            test_config = db_config.copy()
            test_config['database'] = target_database
            
            test_conn = psycopg2.connect(**test_config)
            test_cursor = test_conn.cursor()
            test_cursor.execute("SELECT current_database();")
            current_db = test_cursor.fetchone()[0]
            print(f"✅ Successfully connected to database: {current_db}")
            
            test_cursor.close()
            test_conn.close()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating PostgreSQL Database")
    print("=" * 50)
    
    success = create_database()
    
    if success:
        print("\n🎉 Database setup completed!")
        print("You can now run: python insert_into_postgres.py")
    else:
        print("\n❌ Database setup failed!")
