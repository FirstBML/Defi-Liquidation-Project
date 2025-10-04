"""
Complete Integration Test Script
Run this after setting up all files to verify everything works
"""
import sys
from datetime import datetime

def test_imports():
    """Test 1: Verify all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Importing Modules")
    print("="*60)
    
    try:
        from app.rpc_reserve_fetcher import AaveRPCReserveFetcher
        print("✅ RPC fetcher imported")
        
        from app.db_models import Reserve, Position, SessionLocal
        print("✅ Database models imported")
        
        from app.price_fetcher import EnhancedPriceFetcher
        print("✅ Price fetcher imported")
        
        from app.scheduler import job_run, fetch_and_store_rpc_reserves
        print("✅ Scheduler imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_rpc_connection():
    """Test 2: Test RPC connection to one chain"""
    print("\n" + "="*60)
    print("TEST 2: RPC Connection (Polygon - fastest)")
    print("="*60)
    
    try:
        from app.rpc_reserve_fetcher import AaveRPCReserveFetcher
        
        fetcher = AaveRPCReserveFetcher()
        df = fetcher.fetch_chain_reserves("polygon")
        
        if df.empty:
            print("❌ No data fetched from Polygon")
            return False
        
        print(f"✅ Fetched {len(df)} reserves from Polygon")
        print("\nSample data:")
        print(df[['token_symbol', 'supply_apy', 'borrow_apy', 'ltv']].head(3))
        return True
        
    except Exception as e:
        print(f"❌ RPC connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_storage():
    """Test 3: Test database storage"""
    print("\n" + "="*60)
    print("TEST 3: Database Storage")
    print("="*60)
    
    try:
        from app.rpc_reserve_fetcher import AaveRPCReserveFetcher
        from app.db_models import Reserve, SessionLocal
        from datetime import timezone
        
        # Fetch small dataset
        fetcher = AaveRPCReserveFetcher()
        df = fetcher.fetch_chain_reserves("polygon")
        
        if df.empty:
            print("⚠️ No data to store")
            return False
        
        # Store in database
        db = SessionLocal()
        try:
            # Store first 3 reserves as test
            for _, row in df.head(3).iterrows():
                reserve = Reserve(
                    chain=row['chain'],
                    token_address=row['token_address'],
                    token_symbol=row['token_symbol'],
                    token_name=row['token_name'],
                    decimals=row['decimals'],
                    liquidity_rate=row['liquidity_rate'],
                    variable_borrow_rate=row['variable_borrow_rate'],
                    stable_borrow_rate=row['stable_borrow_rate'],
                    supply_apy=row['supply_apy'],
                    borrow_apy=row['borrow_apy'],
                    ltv=row['ltv'],
                    liquidation_threshold=row['liquidation_threshold'],
                    liquidation_bonus=row['liquidation_bonus'],
                    is_active=row['is_active'],
                    is_frozen=row['is_frozen'],
                    borrowing_enabled=row['borrowing_enabled'],
                    stable_borrowing_enabled=row['stable_borrowing_enabled'],
                    liquidity_index=row['liquidity_index'],
                    variable_borrow_index=row['variable_borrow_index'],
                    atoken_address=row['atoken_address'],
                    variable_debt_token_address=row['variable_debt_token_address'],
                    price_usd=row.get('price_usd', 0.0),
                    price_available=row.get('price_available', False),
                    last_update_timestamp=row['last_update_timestamp'],
                    query_time=row['query_time'],
                    created_at=datetime.now(timezone.utc)
                )
                db.add(reserve)
            
            db.commit()
            
            # Verify storage
            count = db.query(Reserve).filter(Reserve.chain == "polygon").count()
            print(f"✅ Stored 3 reserves, total polygon reserves in DB: {count}")
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Database storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_price_integration():
    """Test 4: Test price fetching integration"""
    print("\n" + "="*60)
    print("TEST 4: Price Integration")
    print("="*60)
    
    try:
        from app.rpc_reserve_fetcher import AaveRPCReserveFetcher
        from app.price_fetcher import EnhancedPriceFetcher
        import os
        
        api_key = os.getenv("COINGECKO_API_KEY")
        if not api_key:
            print("⚠️ No CoinGecko API key - skipping price test")
            return True
        
        price_fetcher = EnhancedPriceFetcher(api_key=api_key)
        rpc_fetcher = AaveRPCReserveFetcher(price_fetcher=price_fetcher)
        
        df = rpc_fetcher.fetch_chain_reserves("polygon")
        
        if df.empty:
            print("⚠️ No data fetched")
            return False
        
        prices_available = df['price_available'].sum()
        total = len(df)
        
        print(f"✅ Price data: {prices_available}/{total} tokens ({prices_available/total*100:.1f}%)")
        
        if prices_available > 0:
            print("\nSample prices:")
            priced = df[df['price_available']].head(3)
            for _, row in priced.iterrows():
                print(f"  {row['token_symbol']}: ${row['price_usd']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Price integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler_integration():
    """Test 5: Test scheduler integration (dry run)"""
    print("\n" + "="*60)
    print("TEST 5: Scheduler Integration (Dry Run)")
    print("="*60)
    
    try:
        from app.scheduler import fetch_and_store_rpc_reserves
        from app.price_fetcher import EnhancedPriceFetcher
        from app.db_models import SessionLocal
        import os
        
        api_key = os.getenv("COINGECKO_API_KEY")
        price_fetcher = EnhancedPriceFetcher(api_key=api_key)
        
        db = SessionLocal()
        try:
            print("Running scheduler function...")
            count = fetch_and_store_rpc_reserves(db, price_fetcher)
            print(f"✅ Scheduler completed: {count} reserves stored")
            return count > 0
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test 6: Verify API can access reserve data"""
    print("\n" + "="*60)
    print("TEST 6: API Data Access")
    print("="*60)
    
    try:
        from app.db_models import Reserve, SessionLocal
        from sqlalchemy import func, desc
        
        db = SessionLocal()
        try:
            # Query like the API endpoint does
            reserves = db.query(Reserve).filter(
                Reserve.is_active == True
            ).order_by(desc(Reserve.query_time)).limit(10).all()
            
            print(f"✅ API can access {len(reserves)} reserve records")
            
            if reserves:
                print("\nSample reserve:")
                r = reserves[0]
                print(f"  Chain: {r.chain}")
                print(f"  Token: {r.token_symbol}")
                print(f"  Supply APY: {r.supply_apy:.2f}%")
                print(f"  Borrow APY: {r.borrow_apy:.2f}%")
                print(f"  LTV: {r.ltv:.4f}")
            
            return len(reserves) > 0
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ API access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary():
    """Generate summary statistics"""
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    
    try:
        from app.db_models import Reserve, Position, SessionLocal
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            # Reserve stats
            total_reserves = db.query(Reserve).count()
            chains = db.query(Reserve.chain).distinct().count()
            latest_update = db.query(func.max(Reserve.query_time)).scalar()
            
            print(f"\n📊 Database Statistics:")
            print(f"  Total reserve records: {total_reserves}")
            print(f"  Chains with data: {chains}")
            print(f"  Latest update: {latest_update}")
            
            # Position stats
            total_positions = db.query(Position).count()
            print(f"  Total positions: {total_positions}")
            
            # Chain breakdown
            chain_counts = db.query(
                Reserve.chain,
                func.count(Reserve.id).label('count')
            ).group_by(Reserve.chain).all()
            
            if chain_counts:
                print(f"\n📈 Reserves by Chain:")
                for chain, count in chain_counts:
                    print(f"  {chain}: {count}")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"⚠️ Could not generate summary: {e}")

def main():
    """Run all tests"""
    print("="*60)
    print("AAVE RPC INTEGRATION TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.now()}")
    
    tests = [
        ("Imports", test_imports),
        ("RPC Connection", test_rpc_connection),
        ("Database Storage", test_database_storage),
        ("Price Integration", test_price_integration),
        ("Scheduler", test_scheduler_integration),
        ("API Compatibility", test_api_compatibility)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration successful!")
        generate_summary()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Start your server: uvicorn app.main:app --reload")
        print("2. Test API: http://localhost:8000/api/reserves/rpc")
        print("3. Manual refresh: curl -X POST http://localhost:8000/api/reserves/rpc/refresh")
        print("4. Check docs: http://localhost:8000/docs")
        
    else:
        print("\n⚠️ Some tests failed. Review errors above and:")
        print("1. Verify all files are in correct locations")
        print("2. Check database migration was run")
        print("3. Ensure dependencies are installed: pip install web3")
        print("4. Verify .env configuration")
    
    print(f"\nCompleted: {datetime.now()}")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)