"""
Railway Data Uploader - Upload Local SQLite to Railway PostgreSQL
Save this as: railway_uploader.py
Run with: python railway_uploader.py

This is a COMPLETELY NEW script, not your fallback_data_loader.py
Key difference: Uses PUBLIC Railway URL, not .railway.internal
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

# Get Railway PostgreSQL URL (PUBLIC)
RAILWAY_DB_URL = os.getenv("DATABASE_URL_PUBLIC")

# CRITICAL FIX: Don't use .railway.internal from local machine
if RAILWAY_DB_URL and "railway.internal" in RAILWAY_DB_URL:
    print("❌ ERROR: DATABASE_URL contains 'railway.internal'")
    print("💡 You need the PUBLIC database URL from Railway dashboard:")
    print("   1. Go to Railway dashboard")
    print("   2. Click on your PostgreSQL service")
    print("   3. Go to 'Connect' tab")
    print("   4. Copy 'Postgres Connection URL' (NOT the internal one)")
    print("   5. Update your .env file with: DATABASE_URL=<public-url>")
    exit(1)

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
    
    # Load reserves (latest per token)
    try:
        cursor.execute("""
            SELECT r.* FROM reserves r
            INNER JOIN (
                SELECT chain, token_address, MAX(query_time) as max_time
                FROM reserves
                GROUP BY chain, token_address
            ) latest ON r.chain = latest.chain 
                AND r.token_address = latest.token_address 
                AND r.query_time = latest.max_time
            LIMIT 500
        """)
        data['reserves'] = [dict(row) for row in cursor.fetchall()]
        print(f"✅ Loaded {len(data['reserves'])} reserves from SQLite")
    except Exception as e:
        print(f"⚠️ No reserves table: {e}")
    
    # Load positions
    try:
        cursor.execute("SELECT * FROM positions ORDER BY last_updated DESC LIMIT 500")
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

def upload_to_railway(data):
    """Upload data to Railway PostgreSQL"""
    if not RAILWAY_DB_URL:
        print("❌ DATABASE_URL not found in environment")
        print("💡 Add your Railway PUBLIC database URL to .env file")
        return False
    
    try:
        print(f"\n🔌 Connecting to Railway PostgreSQL...")
        conn = psycopg2.connect(RAILWAY_DB_URL, connect_timeout=10)
        cursor = conn.cursor()
        print("✅ Connected successfully!")
        
        cache_timestamp = datetime.now(timezone.utc)
        
        # Upload reserves
                # Upload reserves
        if data['reserves']:
            print(f"\n📤 Uploading {len(data['reserves'])} reserves...")

            # ✅ Convert integer flags (1/0) to booleans (True/False)
            for reserve in data['reserves']:
                for key in ['is_active', 'is_frozen', 'borrowing_enabled', 'stable_borrowing_enabled', 'price_available']:

                    if key in reserve:
                        if isinstance(reserve[key], int):
                            reserve[key] = bool(reserve[key])
                        elif isinstance(reserve[key], str) and reserve[key].isdigit():
                            reserve[key] = bool(int(reserve[key]))

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
            """

            # Fix datetime fields
            for reserve in data['reserves']:
                for field in ['query_time', 'created_at']:
                    if isinstance(reserve.get(field), str):
                        try:
                            reserve[field] = datetime.fromisoformat(reserve[field].replace('Z', '+00:00'))
                        except:
                            reserve[field] = cache_timestamp
                    elif not reserve.get(field):
                        reserve[field] = cache_timestamp

            execute_batch(cursor, reserves_query, data['reserves'], page_size=50)
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
            """
            
            for position in data['positions']:
                for field in ['last_updated', 'query_time']:
                    if isinstance(position.get(field), str):
                        try:
                            position[field] = datetime.fromisoformat(position[field].replace('Z', '+00:00'))
                        except:
                            position[field] = cache_timestamp
                    elif not position.get(field):
                        position[field] = cache_timestamp
            
            execute_batch(cursor, positions_query, data['positions'], page_size=50)
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
                    liquidation_count, query_time, created_at
                ) VALUES (
                    %(liquidation_date)s, %(chain)s, %(borrower)s, %(liquidator)s,
                    %(collateral_symbol)s, %(debt_symbol)s, %(collateral_asset)s, %(debt_asset)s,
                    %(total_collateral_seized)s, %(total_debt_normalized)s,
                    %(liquidated_collateral_usd)s, %(liquidated_debt_usd)s,
                    %(liquidation_count)s, %(query_time)s, %(created_at)s
                )
            """
            
            for liq in data['liquidations']:
                for field in ['liquidation_date', 'created_at', 'query_time']:
                    if isinstance(liq.get(field), str):
                        try:
                            liq[field] = datetime.fromisoformat(liq[field].replace('Z', '+00:00'))
                        except:
                            liq[field] = cache_timestamp
                    elif not liq.get(field):
                        liq[field] = cache_timestamp
            
            execute_batch(cursor, liquidations_query, data['liquidations'], page_size=50)
            print(f"✅ Uploaded {len(data['liquidations'])} liquidations")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n✅ Successfully uploaded cache data to Railway!")
        print(f"\n📋 Next steps:")
        print(f"   1. Check your Railway app: /api/data/status")
        print(f"   2. View reserves: /api/reserves/rpc/summary")
        print(f"   3. View positions: /api/positions")
        print(f"   4. Run /api/data/refresh to get fresh data from Dune")
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection error: {e}")
        print(f"\n💡 Make sure you're using the PUBLIC database URL, not .railway.internal")
        print(f"   Get it from: Railway Dashboard > PostgreSQL > Connect tab")
        return False
    except Exception as e:
        print(f"❌ Error uploading to Railway: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Railway Data Uploader\n")
    print("=" * 50)
    
    # Validate DATABASE_URL
    if not RAILWAY_DB_URL:
        print("\n❌ DATABASE_URL not found!")
        print("\n📝 To fix this:")
        print("   1. Go to Railway dashboard")
        print("   2. Click PostgreSQL service > Connect tab")
        print("   3. Copy 'Postgres Connection URL'")
        print("   4. Add to .env file: DATABASE_URL=<that-url>")
        exit(1)
    
    print(f"✅ DATABASE_URL found")
    print(f"   Host: {RAILWAY_DB_URL.split('@')[1].split(':')[0] if '@' in RAILWAY_DB_URL else 'unknown'}")
    
    # Load local data
    print(f"\n📂 Loading local SQLite data...")
    sqlite_data = load_sqlite_data()
    
    total_records = (
        len(sqlite_data['reserves']) + 
        len(sqlite_data['positions']) + 
        len(sqlite_data['liquidations'])
    )
    
    if total_records == 0:
        print("\n❌ No data found to upload!")
        exit(1)
    
    print(f"\n📊 Data Summary:")
    print(f"   • Reserves: {len(sqlite_data['reserves'])}")
    print(f"   • Positions: {len(sqlite_data['positions'])}")
    print(f"   • Liquidations: {len(sqlite_data['liquidations'])}")
    print(f"   • Total: {total_records} records")
    
    response = input("\n❓ Upload to Railway? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        success = upload_to_railway(sqlite_data)
        if success:
            print("\n🎉 Upload complete!")
    else:
        print("❌ Upload cancelled")