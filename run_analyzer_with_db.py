#!/usr/bin/env python3
"""
Run AAveRiskAnalyzer using data from your database instead of Dune API - Fixed Version
"""
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def load_data_from_database():
    """Load data from your database tables"""
    try:
        from app.db_models import SessionLocal, Reserve, Position, LiquidationHistory
        
        session = SessionLocal()
        
        # Load reserves data
        reserves = session.query(Reserve).all()
        df_reserve = pd.DataFrame([{
            'token_symbol': r.token_symbol,
            'token_address': r.token_address,
            'chain': r.chain,
            'total_liquidity': r.total_liquidity,
            'available_liquidity': r.available_liquidity,
            'utilization_rate': r.utilization_rate,
            'liquidation_threshold': r.liquidation_threshold,
            'ltv': r.ltv,
            'price_usd': r.price_usd,
            'is_active': r.is_active,
            'variable_borrow_rate': r.variable_borrow_rate,
            'liquidity_rate': r.liquidity_rate
        } for r in reserves])
        
        # Load positions data
        positions = session.query(Position).all()
        df_positions = pd.DataFrame([{
            'borrower_address': p.borrower_address,
            'chain': p.chain,
            'token_symbol': p.token_symbol,
            'token_address': p.token_address,
            'collateral_amount': p.collateral_amount,
            'debt_amount': p.debt_amount,
            'health_factor': p.health_factor,
            'total_collateral_usd': p.total_collateral_usd,
            'total_debt_usd': p.total_debt_usd,
            'enhanced_health_factor': p.enhanced_health_factor,
            'risk_category': p.risk_category,
            'last_updated': p.last_updated
        } for p in positions])
        
        # Load liquidation history
        liquidations = session.query(LiquidationHistory).all()
        df_liquidations = pd.DataFrame([{
            'liquidation_date': liq.liquidation_date,
            'chain': liq.chain,
            'borrower': liq.borrower,
            'collateral_symbol': liq.collateral_symbol,
            'debt_symbol': liq.debt_symbol,
            'total_collateral_seized': liq.total_collateral_seized,
            'total_debt_normalized': liq.total_debt_normalized,
            'liquidation_count': liq.liquidation_count,
            'unique_liquidators': liq.unique_liquidators,
            'health_factor_before': liq.health_factor_before
        } for liq in liquidations])
        
        session.close()
        
        print(f"Loaded from database:")
        print(f"  - Reserves: {len(df_reserve)} records")
        print(f"  - Positions: {len(df_positions)} records") 
        print(f"  - Liquidations: {len(df_liquidations)} records")
        
        # Warn if DB is empty
        if len(df_reserve) == 0 and len(df_positions) == 0 and len(df_liquidations) == 0:
            print("\n⚠️  Database appears empty. Did you run `python sync_dune_data.py`?")
            print("   Try: stop the API server, run the sync script, then restart the API.")
            sys.exit(1)  # exit early so you don’t waste cycles

        return df_reserve, df_positions, df_liquidations
        
    except Exception as e:
        print(f"Error loading from database: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    
def run_simple_analysis():
    """Run a simple analysis without the full AAveRiskAnalyzer"""
    print("Running Simple Risk Analysis with database data...")
    print("=" * 50)
    
    # Load data from database
    df_reserve, df_positions, df_liquidations = load_data_from_database()
    
    if df_positions.empty:
        print("No position data found in database")
        return
    
    print("\n📊 SIMPLE RISK ANALYSIS RESULTS")
    print("=" * 40)
    
    # Basic statistics
    total_positions = len(df_positions)
    positions_with_debt = len(df_positions[df_positions['total_debt_usd'] > 0])
    
    print(f"Total Positions: {total_positions}")
    print(f"Positions with Debt: {positions_with_debt}")
    
    # Risk distribution
    if 'risk_category' in df_positions.columns:
        risk_dist = df_positions['risk_category'].value_counts()
        print(f"\nRisk Distribution:")
        for risk, count in risk_dist.items():
            if risk:  # Skip None values
                print(f"  {risk}: {count} positions")
    
    # Financial metrics
    total_collateral = df_positions['total_collateral_usd'].sum()
    total_debt = df_positions['total_debt_usd'].sum()
    protocol_ltv = total_debt / total_collateral if total_collateral > 0 else 0
    
    print(f"\nProtocol Metrics:")
    print(f"  Total Collateral: ${total_collateral:,.0f}")
    print(f"  Total Debt: ${total_debt:,.0f}")
    print(f"  Protocol LTV: {protocol_ltv:.2%}")
    
    # Critical positions
    critical_positions = df_positions[
        (df_positions['risk_category'] == 'CRITICAL') | 
        (df_positions['enhanced_health_factor'] < 1.1)
    ]
    
    if len(critical_positions) > 0:
        print(f"\n⚠️  CRITICAL POSITIONS ({len(critical_positions)}):")
        for _, pos in critical_positions.iterrows():
            hf = pos['enhanced_health_factor'] or pos['health_factor'] or 0
            print(f"  - {pos['borrower_address'][:10]}... ({pos['token_symbol']}) HF: {hf:.3f}")
    else:
        print(f"\n✅ No critical positions detected")
    
    # Liquidation summary
    if not df_liquidations.empty:
        recent_liquidations = len(df_liquidations)
        total_liquidated = df_liquidations['total_collateral_seized'].sum()
        
        print(f"\nLiquidation History:")
        print(f"  Recent Liquidations: {recent_liquidations}")
        print(f"  Total Value Liquidated: ${total_liquidated:,.0f}")
    
    # Reserve summary
    if not df_reserve.empty:
        active_reserves = len(df_reserve[df_reserve['is_active'] == True])
        print(f"\nReserve Status:")
        print(f"  Active Reserves: {active_reserves}/{len(df_reserve)}")
        
        if 'utilization_rate' in df_reserve.columns:
            avg_utilization = df_reserve['utilization_rate'].mean()
            if pd.notna(avg_utilization):
                print(f"  Average Utilization: {avg_utilization:.1%}")
    
    # Save simple results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"simple_analysis_results_{timestamp}.csv"
    df_positions.to_csv(results_file, index=False)
    print(f"\n💾 Results saved to: {results_file}")
    
    return df_positions

def run_enhanced_analysis():
    """Try to run the full AAveRiskAnalyzer if available"""
    try:
        # Try to import and run the full analyzer
        from app.AAveRiskAnalyze import EnhancedPriceFetcher, AAVERiskAnalyzer, ProgressTracker
        
        print("Running Enhanced AAveRiskAnalyzer...")
        print("=" * 50)
        
        # Load data
        df_reserve, df_positions, df_liquidations = load_data_from_database()
        
        if df_positions.empty:
            print("No position data found")
            return
        
        # Initialize progress tracker
        progress = ProgressTracker()
        progress.add_step("Loading Database Data")
        progress.add_step("Fetching Current Prices")  
        progress.add_step("Running Risk Analysis")
        
        progress.start_step(0)
        progress.add_detail("Database data loaded successfully")
        progress.complete_step(0, True)
        
        # Initialize price fetcher with correct parameters
        progress.start_step(1)
        try:
            # Use the correct initialization parameters for your EnhancedPriceFetcher
            price_fetcher = EnhancedPriceFetcher(
                api_key=os.getenv("COINGECKO_API_KEY"),
                cache_ttl=300,  # 5 minutes
                request_delay=1.0
            )
            
            # Get unique tokens
            unique_tokens = df_positions['token_symbol'].dropna().unique().tolist()
            progress.add_detail(f"Fetching prices for {len(unique_tokens)} tokens")
            
            current_prices = price_fetcher.get_batch_prices(unique_tokens, progress)
            progress.add_detail(f"Fetched prices for {len([p for p in current_prices.values() if p is not None])} tokens")
            progress.complete_step(1, True)
            
        except Exception as e:
            progress.complete_step(1, False)
            progress.add_detail(f"Price fetching failed: {e}")
            current_prices = {}
        
        # Run analysis
        progress.start_step(2)
        try:
            analyzer = AAVERiskAnalyzer(price_fetcher)
            
            # Clean and enhance data
            df_positions_clean = analyzer.clean_position_data(df_positions, progress)
            df_positions_enhanced = analyzer.calculate_enhanced_metrics(
                df_positions_clean, current_prices, progress
            )
            
            # Generate summary
            progress.add_detail("Generating analysis summary...")
            
            risk_dist = df_positions_enhanced['risk_category'].value_counts()
            critical_count = risk_dist.get('CRITICAL', 0) + risk_dist.get('LIQUIDATION_IMMINENT', 0)
            
            print(f"\n📊 ENHANCED ANALYSIS RESULTS:")
            print(f"Total positions analyzed: {len(df_positions_enhanced)}")
            print(f"Critical positions: {critical_count}")
            
            for risk, count in risk_dist.items():
                print(f"{risk}: {count} positions")
            
            # Save enhanced results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"enhanced_analysis_results_{timestamp}.csv"
            df_positions_enhanced.to_csv(results_file, index=False)
            progress.add_detail(f"Results saved to {results_file}")
            
            progress.complete_step(2, True)
            
            print(f"\n{progress.get_summary()}")
            
            return df_positions_enhanced
            
        except Exception as e:
            progress.complete_step(2, False)
            print(f"Enhanced analysis failed: {e}")
            print("Falling back to simple analysis...")
            return run_simple_analysis()
            
    except ImportError as e:
        print(f"Could not import AAveRiskAnalyze components: {e}")
        print("Running simple analysis instead...")
        return run_simple_analysis()
    except Exception as e:
        print(f"Enhanced analysis setup failed: {e}")
        print("Running simple analysis instead...")
        return run_simple_analysis()

if __name__ == "__main__":
    print("🚀 Database Risk Analysis")
    print("=" * 30)
    
    # Try enhanced analysis first, fall back to simple if it fails
    results = run_enhanced_analysis()
    
    if results is not None and not results.empty:
        print("\n✅ Analysis completed successfully!")
        print("Check the generated CSV file for detailed results.")
    else:
        print("\n⚠️  Analysis completed with limited results.")