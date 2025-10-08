#!/usr/bin/env python3
"""
Update database with CORRECTED health factors using borrower aggregation
This fixes the critical issue where collateral and debt rows were treated separately
"""
import sys
import os
import pandas as pd
from datetime import datetime, timezone

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def update_with_fixed_aggregation():
    """Calculate CORRECT health factors with borrower aggregation"""
    try:
        from app.db_models import SessionLocal, Position
        from app.price_fetcher import EnhancedPriceFetcher
        
        # Import the FIXED analyzer (you'll need to update your AAveRiskAnalyze.py)
        from app.AAveRiskAnalyze import AAVERiskAnalyzer, ProgressTracker
        
        print("🔧 Applying Fixed Health Factor Calculation")
        print("=" * 50)
        print("CRITICAL FIX: Aggregating by borrower before calculating HF")
        
        session = SessionLocal()
        
        # Load ALL position rows (including both collateral and debt rows)
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
            'last_updated': p.last_updated
        } for p in positions])
        
        print(f"Loaded {len(df_positions)} position rows")
        print(f"These will be aggregated by borrower...")
        
        # Initialize components
        price_fetcher = EnhancedPriceFetcher(
            api_key=os.getenv("COINGECKO_API_KEY"),
            cache_ttl=300
        )
        
        analyzer = AAVERiskAnalyzer(price_fetcher)
        progress = ProgressTracker()
        progress.add_step("Cleaning Data")
        progress.add_step("Aggregating by Borrower")
        progress.add_step("Calculating Correct Health Factors")
        progress.add_step("Updating Database")
        
        # Clean
        progress.start_step(0)
        df_clean = analyzer.clean_position_data(df_positions, progress)
        progress.complete_step(0, True)
        
        # Calculate with FIXED aggregation
        progress.start_step(1)
        df_enhanced = analyzer.calculate_enhanced_metrics(df_clean, {}, progress)
        progress.complete_step(1, True)
        
        # Clear old positions and insert aggregated ones
        progress.start_step(2)
        print(f"\nClearing old position rows...")
        session.query(Position).delete()
        
        print(f"Inserting {len(df_enhanced)} aggregated borrower positions...")
        
        for _, row in df_enhanced.iterrows():
            position = Position(
                borrower_address=row['borrower_address'],
                chain=row['chain'],
                token_symbol=row['token_symbol'],  # Display token
                token_address=None,  # Aggregated position doesn't have single address
                collateral_amount=0,  # Aggregated across multiple tokens
                debt_amount=0,  # Aggregated across multiple tokens
                health_factor=row['health_factor'],
                total_collateral_usd=float(row['total_collateral_usd']),
                total_debt_usd=float(row['total_debt_usd']),
                enhanced_health_factor=float(row['enhanced_health_factor']) if pd.notna(row['enhanced_health_factor']) and row['enhanced_health_factor'] != np.inf else None,
                risk_category=row['risk_category'],
                current_ltv=float(row['current_ltv']) if pd.notna(row['current_ltv']) else None,
                liquidation_threshold=float(row['liquidation_threshold']) if pd.notna(row['liquidation_threshold']) else None,
                current_collateral_usd=float(row['current_collateral_usd']) if pd.notna(row['current_collateral_usd']) else None,
                price_available=bool(row['price_available']) if pd.notna(row['price_available']) else False,
                last_updated=row['last_updated']
            )
            session.add(position)
        
        session.commit()
        progress.complete_step(2, True)
        
        # Summary
        print(f"\n✅ Database updated with CORRECT health factors")
        print(f"\nRisk Distribution (CORRECT):")
        risk_dist = df_enhanced['risk_category'].value_counts()
        for risk, count in risk_dist.items():
            print(f"  {risk}: {count} borrowers")
        
        critical_count = risk_dist.get('CRITICAL', 0) + risk_dist.get('LIQUIDATION_IMMINENT', 0)
        print(f"\n⚠️  {critical_count} borrowers actually at critical risk")
        print(f"(Previous count was inflated by counting collateral/debt rows separately)")
        
        session.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np  # Needed for np.inf check
    
    success = update_with_fixed_aggregation()
    
    if success:
        print("\n🎉 Fixed health factor calculation applied!")
        print("\nNext steps:")
        print("1. Restart API server")
        print("2. Check /api/positions - should show fewer positions")
        print("3. Check /api/dashboard/summary - risk distribution should be accurate")
        print("\nEach position now represents a BORROWER, not a token row")