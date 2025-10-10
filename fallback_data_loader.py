"""
Fallback Data Loader - Uploads local cache data to Railway PostgreSQL
Matched to your actual database schema
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

# Railway PostgreSQL connection
RAILWAY_DB_URL = os.getenv("DATABASE_URL")
if RAILWAY_DB_URL and RAILWAY_DB_URL.startswith("postgres://"):
    RAILWAY_DB_URL = RAILWAY_DB_URL.replace("postgres://", "postgresql://", 1)

def load_sqlite_data():
    """Load data from local SQLite database"""
    if not os.path.exists('aave_risk.db'):
        print("❌ aave_risk.db not found")
        return {'reserves': [], 'positions': [], 'liquidations': []}
    
    conn = sqlite3.connect('aave_risk.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    data = {
        'reserves': [],
        'positions': [],
        'liquidations': []
    }
    
    # Load reserves
    try:
        cursor.execute("SELECT * FROM reserves ORDER BY query_time DESC LIMIT 1000")
        data['reserves'] = [dict(row) for row in cursor.fetchall()]
        print(f"✅ Loaded {len(data['reserves'])} reserves from SQLite")
    except Exception as e:
        print(f"⚠️ No reserves table: {e}")
    
    # Load positions
    try:
        cursor.execute("SELECT * FROM positions ORDER BY last_updated DESC LIMIT 1000")
        data['positions'] = [dict(row) for row in cursor.fetchall()]
        print(f"✅ Loaded {len(data['positions'])} positions from SQLite")
    except Exception as e:
        print(f"⚠️ No positions table: {e}")
    
    # Load liquidations
    try:
        cursor.execute("SELECT * FROM liquidation_history ORDER BY liquidation_date DESC LIMIT 500")
        data['liquidations'] = [dict(row) for row in cursor.fetchall()]
        print(f"✅ Loaded {len(data['liquidations'])} liquidations from SQLite")
    except Exception as e:
        print(f"⚠️ No liquidations table: {e}")
    
    conn.close()
    return data

def load_price_cache():
    """Load price data from JSON cache"""
    if not os.path.exists('price_cache.json'):
        print("⚠️ price_cache.json not found")
        return {}
    
    try:
        with open('price_cache.json', 'r') as f:
            prices = json.load(f)
        print(f"✅ Loaded {len(prices)} prices from cache")
        return prices
    except Exception as e:
        print(f"⚠️ Could not load price_cache.json: {e}")
        return {}

def upload_to_railway(data, prices):
    """Upload data to Railway PostgreSQL"""
    if not RAILWAY_DB_URL:
        print("❌ DATABASE_URL not found in environment")
        print("💡 Make sure you have .env file with DATABASE_URL")
        return False
    
    try:
        conn = psycopg2.connect(RAILWAY_DB_URL)
        cursor = conn.cursor()
        
        cache_timestamp = datetime.now(timezone.utc)
        
        # Upload reserves
        if data['reserves']:
            print(f"\n📤 Uploading {len(data['reserves'])} reserves...")
            reserves_query = """
                INSERT INTO reserves (
                    chain, token_address, token_symbol, token_name, decimals,
                    liquidity_rate, variable_borrow_rate, stable_borrow_rate,
                    supply_apy, borrow_apy, ltv, liquidation_threshold, liquidation_bonus,
                    is_active, is_frozen, borrowing_enabled, stable_borrowing_enabled,
                    liquidity_index, variable_borrow_index,
                    atoken_address, variable_debt_token_address,
                    price_usd, price_available,
                    last_update_timestamp, query_time, created_at
                ) VALUES (
                    %(chain)s, %(token_address)s, %(token_symbol)s, %(token_name)s, %(decimals)s,
                    %(liquidity_rate)s, %(variable_borrow_rate)s, %(stable_borrow_rate)s,
                    %(supply_apy)s, %(borrow_apy)s, %(ltv)s, %(liquidation_threshold)s, %(liquidation_bonus)s,
                    %(is_active)s, %(is_frozen)s, %(borrowing_enabled)s, %(stable_borrowing_enabled)s,
                    %(liquidity_index)s, %(variable_borrow_index)s,
                    %(atoken_address)s, %(variable_debt_token_address)s,
                    %(price_usd)s, %(price_available)s,
                    %(last_update_timestamp)s, %(query_time)s, %(created_at)s
                )
                ON CONFLICT (chain, token_address, query_time) DO NOTHING
            """
            
            # Convert datetime strings to datetime objects
            for reserve in data['reserves']:
                if isinstance(reserve.get('query_time'), str):
                    try:
                        reserve['query_time'] = datetime.fromisoformat(reserve['query_time'].replace('Z', '+00:00'))
                    except:
                        reserve['query_time'] = cache_timestamp
                
                if isinstance(reserve.get('created_at'), str):
                    try:
                        reserve['created_at'] = datetime.fromisoformat(reserve['created_at'].replace('Z', '+00:00'))
                    except:
                        reserve['created_at'] = cache_timestamp
            
            execute_batch(cursor, reserves_query, data['reserves'], page_size=100)
            print(f"✅ Uploaded {len(data['reserves'])} reserves")
        
        # Upload positions
        if data['positions']:
            print(f"\n📤 Uploading {len(data['positions'])} positions...")
            positions_query = """
                INSERT INTO positions (
                    borrower_address, chain, token_symbol, token_address,
                    collateral_amount, debt_amount, health_factor,
                    total_collateral_usd, total_debt_usd,
                    enhanced_health_factor, risk_category, liquidation_threshold,
                    last_updated, query_time
                ) VALUES (
                    %(borrower_address)s, %(chain)s, %(token_symbol)s, %(token_address)s,
                    %(collateral_amount)s, %(debt_amount)s, %(health_factor)s,
                    %(total_collateral_usd)s, %(total_debt_usd)s,
                    %(enhanced_health_factor)s, %(risk_category)s, %(liquidation_threshold)s,
                    %(last_updated)s, %(query_time)s
                )
                ON CONFLICT DO NOTHING
            """
            
            # Convert datetime strings
            for position in data['positions']:
                if isinstance(position.get('last_updated'), str):
                    try:
                        position['last_updated'] = datetime.fromisoformat(position['last_updated'].replace('Z', '+00:00'))
                    except:
                        position['last_updated'] = cache_timestamp
                
                if isinstance(position.get('query_time'), str):
                    try:
                        position['query_time'] = datetime.fromisoformat(position['query_time'].replace('Z', '+00:00'))
                    except:
                        position['query_time'] = cache_timestamp
            
            execute_batch(cursor, positions_query, data['positions'], page_size=100)
            print(f"✅ Uploaded {len(data['positions'])} positions")
        
        # Upload liquidations
        if data['liquidations']:
            print(f"\n📤 Uploading {len(data['liquidations'])} liquidations...")
            liquidations_query = """
                INSERT INTO liquidation_history (
                    liquidation_date, chain, borrower, liquidator,
                    collateral_symbol, debt_symbol, collateral_asset, debt_asset,
                    total_collateral_seized, total_debt_normalized,
                    liquidated_collateral_usd, liquidated_debt_usd,
                    liquidation_count, avg_debt_per_event, unique_liquidators,
                    health_factor_before, created_at, query_time
                ) VALUES (
                    %(liquidation_date)s, %(chain)s, %(borrower)s, %(liquidator)s,
                    %(collateral_symbol)s, %(debt_symbol)s, %(collateral_asset)s, %(debt_asset)s,
                    %(total_collateral_seized)s, %(total_debt_normalized)s,
                    %(liquidated_collateral_usd)s, %(liquidated_debt_usd)s,
                    %(liquidation_count)s, %(avg_debt_per_event)s, %(unique_liquidators)s,
                    %(health_factor_before)s, %(created_at)s, %(query_time)s
                )
                ON CONFLICT DO NOTHING
            """
            
            # Convert datetime strings
            for liq in data['liquidations']:
                if isinstance(liq.get('liquidation_date'), str):
                    try:
                        liq['liquidation_date'] = datetime.fromisoformat(liq['liquidation_date'].replace('Z', '+00:00'))
                    except:
                        liq['liquidation_date'] = cache_timestamp
                
                if isinstance(liq.get('created_at'), str):
                    try:
                        liq['created_at'] = datetime.fromisoformat(liq['created_at'].replace('Z', '+00:00'))
                    except:
                        liq['created_at'] = cache_timestamp
                
                if isinstance(liq.get('query_time'), str):
                    try:
                        liq['query_time'] = datetime.fromisoformat(liq['query_time'].replace('Z', '+00:00'))
                    except:
                        liq['query_time'] = cache_timestamp
            
            execute_batch(cursor, liquidations_query, data['liquidations'], page_size=100)
            print(f"✅ Uploaded {len(data['liquidations'])} liquidations")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n✅ Successfully uploaded cache data to Railway!")
        print(f"⚠️  Remember: This is cached data. Run /api/data/refresh to get fresh data.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Railway: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Loading local cache data...\n")
    
    # Load local data
    sqlite_data = load_sqlite_data()
    prices = load_price_cache()
    
    # Confirm with user
    total_records = (
        len(sqlite_data['reserves']) + 
        len(sqlite_data['positions']) + 
        len(sqlite_data['liquidations'])
    )
    
    if total_records == 0:
        print("\n❌ No data found to upload!")
        print("Make sure aave_risk.db exists and contains data")
        exit(1)
    
    print(f"\n📊 Summary:")
    print(f"  - Reserves: {len(sqlite_data['reserves'])}")
    print(f"  - Positions: {len(sqlite_data['positions'])}")
    print(f"  - Liquidations: {len(sqlite_data['liquidations'])}")
    print(f"  - Total: {total_records} records")
    
    response = input("\n❓ Upload to Railway? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = upload_to_railway(sqlite_data, prices)
        if success:
            print("\n✅ Done! Check your Railway app:")
            print("   - /startup-status")
            print("   - /api/reserves/rpc/summary")
            print("   - /api/positions")
    else:
        print("❌ Upload cancelled")