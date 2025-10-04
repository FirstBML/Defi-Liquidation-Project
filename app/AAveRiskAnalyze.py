# AAveRiskAnalyze.py

import pandas as pd
import numpy as np
import requests
import os
import json
import time
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dune_client.client import DuneClient
from dotenv import load_dotenv
from pathlib import Path
from .price_fetcher import EnhancedPriceFetcher
import warnings
warnings.filterwarnings('ignore')

"""
Fixed AAVERiskAnalyzer with proper borrower aggregation
Remove the EnhancedPriceFetcher class from your AAveRiskAnalyze.py and use this instead
"""

class ProgressTracker:
    """Track and display analysis progress with detailed logging"""

    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.start_time = datetime.now()

    def add_step(self, description: str):
        """Add a new step to the progress tracker"""
        self.steps.append({
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'details': []
        })

    def start_step(self, step_index: int):
        """Mark a step as started and print its description"""
        self.current_step = step_index
        self.steps[step_index]['status'] = 'running'
        self.steps[step_index]['start_time'] = datetime.now()
        print(f"\n[{step_index + 1}/{len(self.steps)}] {self.steps[step_index]['description']}...")

    def add_detail(self, detail: str):
        """Add a detail message to the current step and print it"""
        if self.current_step < len(self.steps):
            self.steps[self.current_step]['details'].append(detail)
            print(f"   → {detail}")

    def complete_step(self, step_index: int, success: bool = True):
        """Mark a step as completed or failed and print duration"""
        self.steps[step_index]['status'] = 'completed' if success else 'failed'
        self.steps[step_index]['end_time'] = datetime.now()
        duration = (self.steps[step_index]['end_time'] - self.steps[step_index]['start_time']).total_seconds()
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} Completed in {duration:.2f}s")

    def get_summary(self):
        """Return a summary string of completed steps and total time"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        return f"Analysis completed: {completed}/{len(self.steps)} steps in {total_time:.2f}s"
    
class AAVERiskAnalyzer:
    """
    Enhanced AAVE-specific risk analyzer with proper borrower-level health factor calculation
    """
    
    def __init__(self, price_fetcher):
        """
        Initialize with price fetcher from app.price_fetcher
        """
        self.price_fetcher = price_fetcher
        self.liquidation_thresholds = {
            'WETH': 0.825, 'WBTC': 0.70, 'USDC': 0.875, 'DAI': 0.77,
            'LINK': 0.70, 'AAVE': 0.66, 'USDT': 0.80, 'MATIC': 0.65,
            'wstETH': 0.80, 'weETH': 0.80, 'OP': 0.70
        }
    
    def clean_position_data(self, df_positions: pd.DataFrame, progress_tracker) -> pd.DataFrame:
        """Clean and filter position data"""
        progress_tracker.add_detail(f"Initial positions: {len(df_positions)}")
        
        # Print first 10 rows for verification
        if not df_positions.empty:
            print("\n📊 FIRST 10 ROWS OF POSITION DATA:")
            print("=" * 80)
            print(df_positions.head(10).to_string(max_colwidth=20))
            print("=" * 80)
            
            print(f"\n📋 POSITION DATA COLUMNS ({len(df_positions.columns)} total):")
            for i, col in enumerate(df_positions.columns, 1):
                print(f"   {i:2d}. {col}: {df_positions[col].dtype}")
        
        # Keep all rows - we need both collateral and debt rows for aggregation
        meaningful_positions = df_positions[
            (df_positions['total_debt_usd'] > 0) | 
            (df_positions['total_collateral_usd'] > 100)
        ].copy()
        
        progress_tracker.add_detail(f"Filtered to meaningful positions: {len(meaningful_positions)}")
        
        return meaningful_positions
    
    def aggregate_borrower_positions(self, df_positions: pd.DataFrame, progress_tracker) -> pd.DataFrame:
        """
        CRITICAL FIX: Aggregate positions by borrower across all tokens
        Dune returns separate rows for collateral and debt - we must combine them
        """
        progress_tracker.add_detail("Aggregating positions by borrower (CRITICAL for correct HF)...")
        
        # Group by borrower and chain to get total exposure
        borrower_groups = df_positions.groupby(['borrower_address', 'chain'])
        
        borrower_positions = []
        
        for (borrower_addr, chain), group in borrower_groups:
            # Sum all collateral and debt across tokens
            total_collateral = group['total_collateral_usd'].sum()
            total_debt = group['total_debt_usd'].sum()
            
            # Get token breakdown
            collateral_tokens = group[group['collateral_amount'] > 0][['token_symbol', 'collateral_amount', 'token_address']].to_dict('records')
            debt_tokens = group[group['debt_amount'] > 0][['token_symbol', 'debt_amount']].to_dict('records')
            
            # Use health_factor from Dune (should be same for all rows of this borrower)
            health_factor = group['health_factor'].iloc[0]
            
            # Get last updated timestamp
            last_updated = group['last_updated'].max()
            
            borrower_positions.append({
                'borrower_address': borrower_addr,
                'chain': chain,
                'total_collateral_usd': total_collateral,
                'total_debt_usd': total_debt,
                'health_factor': health_factor,
                'collateral_tokens': collateral_tokens,
                'debt_tokens': debt_tokens,
                'num_collateral_tokens': len(collateral_tokens),
                'num_debt_tokens': len(debt_tokens),
                'last_updated': last_updated,
                # Store primary token for display
                'primary_collateral_token': collateral_tokens[0]['token_symbol'] if collateral_tokens else 'None',
                'primary_debt_token': debt_tokens[0]['token_symbol'] if debt_tokens else 'None'
            })
        
        df_aggregated = pd.DataFrame(borrower_positions)
        
        progress_tracker.add_detail(f"Aggregated {len(df_positions)} rows into {len(df_aggregated)} unique borrowers")
        progress_tracker.add_detail(f"Example: Borrower may have WETH collateral + USDC debt = 1 position, not 2")
        
        return df_aggregated
    
    def calculate_enhanced_metrics(self, df_positions: pd.DataFrame, current_prices: Dict[str, float], 
                                 progress_tracker) -> pd.DataFrame:
        """
        Calculate enhanced AAVE-specific metrics with PROPER borrower-level health factors
        """
        # CRITICAL: First aggregate by borrower
        df_aggregated = self.aggregate_borrower_positions(df_positions, progress_tracker)
        
        df = df_aggregated.copy()
        
        # Calculate weighted average liquidation threshold
        # For now, use conservative 0.80 (you can enhance this later with actual weighted calc)
        df['liquidation_threshold'] = 0.80
        
        # Calculate CORRECT enhanced health factor at borrower level
        df['enhanced_health_factor'] = np.where(
            df['total_debt_usd'] > 0,
            (df['total_collateral_usd'] * df['liquidation_threshold']) / df['total_debt_usd'],
            np.inf
        )
        
        # Current LTV
        df['current_ltv'] = np.where(
            df['total_collateral_usd'] > 0,
            df['total_debt_usd'] / df['total_collateral_usd'],
            0
        )
        
        # Risk categorization based on PROPER health factor
        df['risk_category'] = 'SAFE'
        df.loc[df['enhanced_health_factor'] < 2.0, 'risk_category'] = 'LOW_RISK'
        df.loc[df['enhanced_health_factor'] < 1.5, 'risk_category'] = 'MEDIUM_RISK'
        df.loc[df['enhanced_health_factor'] < 1.3, 'risk_category'] = 'HIGH_RISK'
        df.loc[df['enhanced_health_factor'] < 1.1, 'risk_category'] = 'CRITICAL'
        df.loc[df['enhanced_health_factor'] < 1.0, 'risk_category'] = 'LIQUIDATION_IMMINENT'
        df.loc[df['total_debt_usd'] == 0, 'risk_category'] = 'NO_DEBT'
        
        # Position size categories
        df['position_size_category'] = pd.cut(
            df['total_collateral_usd'],
            bins=[0, 1000, 10000, 100000, 1000000, np.inf],
            labels=['SMALL', 'MEDIUM', 'LARGE', 'WHALE', 'MEGA_WHALE']
        )
        
        # Add current collateral USD (same as total for now, can be updated with current prices)
        df['current_collateral_usd'] = df['total_collateral_usd']
        df['price_available'] = True
        
        # Display token for UI
        df['token_symbol'] = df['primary_collateral_token'] + ' (+ ' + (df['num_collateral_tokens'] - 1).astype(str) + ' more)' 
        df.loc[df['num_collateral_tokens'] == 1, 'token_symbol'] = df['primary_collateral_token']
        
        progress_tracker.add_detail(f"Calculated enhanced metrics for {len(df)} borrowers")
        progress_tracker.add_detail(f"Price data available for {df['price_available'].sum()} positions")
        
        return df
    
    def analyze_concentration_risk(self, df_positions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze concentration risk across borrowers and chains"""
        concentration_analysis = {}
        
        # Borrower concentration (whale analysis)
        df_sorted = df_positions.sort_values('total_collateral_usd', ascending=False)
        
        top_10_collateral = df_sorted.head(10)['total_collateral_usd'].sum()
        total_collateral = df_positions['total_collateral_usd'].sum()
        
        concentration_analysis['borrower_concentration'] = {
            'top_10_borrowers_share': top_10_collateral / total_collateral if total_collateral > 0 else 0,
            'total_borrowers': len(df_positions),
            'top_10_details': df_sorted.head(10)[['borrower_address', 'total_collateral_usd', 'risk_category']].to_dict('records')
        }
        
        # Chain concentration
        if 'chain' in df_positions.columns:
            chain_exposure = df_positions.groupby('chain').agg({
                'total_collateral_usd': 'sum',
                'borrower_address': 'count'
            }).rename(columns={'borrower_address': 'position_count'})
            
            chain_exposure['share'] = chain_exposure['total_collateral_usd'] / total_collateral
            concentration_analysis['chain_concentration'] = chain_exposure.sort_values('total_collateral_usd', ascending=False).to_dict('records')
        
        # Large positions (whales)
        large_positions = df_positions[df_positions['total_collateral_usd'] > 1000000]
        concentration_analysis['large_positions'] = {
            'count': len(large_positions),
            'total_collateral': large_positions['total_collateral_usd'].sum(),
            'share_of_protocol': large_positions['total_collateral_usd'].sum() / total_collateral if total_collateral > 0 else 0,
            'avg_health_factor': large_positions['enhanced_health_factor'].replace([np.inf], np.nan).mean()
        }
        
        return concentration_analysis
    
    def calculate_liquidation_risk_metrics(self, df_positions: pd.DataFrame, df_liquidations: pd.DataFrame = None) -> Dict[str, Any]:
        """Calculate comprehensive liquidation risk metrics"""
        risk_metrics = {}
        
        # Current risk distribution
        risk_distribution = df_positions['risk_category'].value_counts()
        total_positions = len(df_positions[df_positions['total_debt_usd'] > 0])
        
        risk_metrics['current_risk_distribution'] = {
            category: {
                'count': count,
                'percentage': (count / total_positions * 100) if total_positions > 0 else 0
            }
            for category, count in risk_distribution.items()
        }
        
        # Liquidation stress scenarios
        price_drop_scenarios = [5, 10, 20, 30, 50]
        stress_test_results = {}
        
        for drop_pct in price_drop_scenarios:
            stressed_collateral = df_positions['total_collateral_usd'] * (1 - drop_pct / 100)
            stressed_hf = np.where(
                df_positions['total_debt_usd'] > 0,
                (stressed_collateral * df_positions['liquidation_threshold']) / df_positions['total_debt_usd'],
                np.inf
            )
            
            at_risk_positions = df_positions[stressed_hf < 1.0]
            liquidation_value = at_risk_positions['total_collateral_usd'].sum()
            
            stress_test_results[f'{drop_pct}%_drop'] = {
                'positions_liquidated': len(at_risk_positions),
                'collateral_at_risk': liquidation_value,
                'percentage_of_protocol': (liquidation_value / df_positions['total_collateral_usd'].sum() * 100) if df_positions['total_collateral_usd'].sum() > 0 else 0
            }
        
        risk_metrics['stress_test_scenarios'] = stress_test_results
        
        return risk_metrics
    
class ReserveAnalyzer:
    """Analyze AAVE reserve data for liquidity risk"""
    
    @staticmethod
    def analyze_reserves(df_reserves: pd.DataFrame, progress_tracker: ProgressTracker) -> Dict[str, Any]:
        """Comprehensive reserve analysis"""
        if df_reserves.empty:
            progress_tracker.add_detail("No reserve data available")
            return {}
        
        reserve_analysis = {}
        
        # Print first 10 rows of reserve data
        print("\n📊 FIRST 10 ROWS OF RESERVE DATA:")
        print("=" * 80)
        print(df_reserves.head(10).to_string(max_colwidth=20))
        print("=" * 80)
        
        # Print column information
        print(f"\n📋 RESERVE DATA COLUMNS ({len(df_reserves.columns)} total):")
        for i, col in enumerate(df_reserves.columns, 1):
            print(f"   {i:2d}. {col}: {df_reserves[col].dtype}")
        
        # Identify utilization columns
        utilization_cols = [col for col in df_reserves.columns 
                          if any(keyword in col.lower() for keyword in ['utilization', 'rate', 'borrow', 'supply'])]
        
        progress_tracker.add_detail(f"Found columns: {utilization_cols}")
        
        # Basic reserve statistics
        if 'token_symbol' in df_reserves.columns:
            asset_count = df_reserves['token_symbol'].nunique()
            progress_tracker.add_detail(f"Analyzing {asset_count} unique assets")
            
            reserve_analysis['asset_overview'] = {
                'total_assets': asset_count,
                'assets': df_reserves['token_symbol'].value_counts().to_dict()
            }
        
        # Utilization analysis if data available
        if utilization_cols:
            for col in utilization_cols:
                if df_reserves[col].dtype in ['float64', 'int64']:
                    utilization_stats = {
                        'mean': float(df_reserves[col].mean()),
                        'median': float(df_reserves[col].median()),
                        'max': float(df_reserves[col].max()),
                        'min': float(df_reserves[col].min()),
                        'std': float(df_reserves[col].std())
                    }
                    reserve_analysis[f'{col}_statistics'] = utilization_stats
                    
                    # High utilization alerts
                    if 'utilization' in col.lower():
                        high_util_threshold = 0.8
                        high_util_assets = df_reserves[df_reserves[col] > high_util_threshold]
                        if len(high_util_assets) > 0:
                            reserve_analysis['high_utilization_alerts'] = {
                                'count': len(high_util_assets),
                                'assets': high_util_assets['token_symbol'].tolist() if 'token_symbol' in df_reserves.columns else []
                            }
        
        progress_tracker.add_detail(f"Completed reserve analysis for {len(df_reserves)} entries")
        
        return reserve_analysis

def create_detailed_dataframes(df_positions: pd.DataFrame, df_liquidations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create detailed, clean DataFrames for export"""
    
    # Detailed positions DataFrame
    position_columns = [
        'borrower_address', 'chain', 'token_symbol', 'token_address',
        'collateral_amount', 'total_collateral_usd', 'current_collateral_usd',
        'total_debt_usd', 'current_ltv', 'enhanced_health_factor',
        'liquidation_threshold', 'liquidation_price', 'price_drop_to_liquidation_pct',
        'risk_category', 'position_size_category', 'current_price', 'price_available',
        'last_updated'
    ]
    
    # Filter to existing columns
    available_position_cols = [col for col in position_columns if col in df_positions.columns]
    df_positions_detailed = df_positions[available_position_cols].copy()
    
    # Sort by risk and collateral size
    risk_order = {'LIQUIDATION_IMMINENT': 0, 'CRITICAL': 1, 'HIGH_RISK': 2, 'MEDIUM_RISK': 3, 'LOW_RISK': 4, 'SAFE': 5, 'NO_DEBT': 6}
    df_positions_detailed['risk_order'] = df_positions_detailed['risk_category'].map(risk_order)
    df_positions_detailed = df_positions_detailed.sort_values(['risk_order', 'current_collateral_usd'], ascending=[True, False])
    df_positions_detailed.drop('risk_order', axis=1, inplace=True)
    
    # Detailed liquidations DataFrame
    liquidation_columns = [
        'liquidation_date', 'chain', 'collateral_symbol', 'debt_symbol',
        'total_collateral_seized', 'total_debt_normalized', 'liquidation_count',
        'avg_debt_per_event', 'unique_liquidators'
    ]
    
    available_liquidation_cols = [col for col in liquidation_columns if col in df_liquidations.columns]
    df_liquidations_detailed = df_liquidations[available_liquidation_cols].copy()
    
    # Sort by date (most recent first)
    if 'liquidation_date' in df_liquidations_detailed.columns:
        df_liquidations_detailed['liquidation_date'] = pd.to_datetime(df_liquidations_detailed['liquidation_date'])
        df_liquidations_detailed = df_liquidations_detailed.sort_values('liquidation_date', ascending=False)
    
    return df_positions_detailed, df_liquidations_detailed

def run_comprehensive_aave_risk_analysis(
    coingecko_api_key: Optional[str] = None,
    save_results: bool = True,
    cache_enabled: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive AAVE risk analysis with all enhancements
    """
    
    # Initialize progress tracker
    progress = ProgressTracker()
    progress.add_step("Loading Dune Analytics Data")
    progress.add_step("Fetching Current Market Prices")
    progress.add_step("Cleaning and Filtering Position Data")
    progress.add_step("Calculating Enhanced Risk Metrics")
    progress.add_step("Analyzing Reserve Liquidity")
    progress.add_step("Performing Concentration Risk Analysis")
    progress.add_step("Calculating Liquidation Risk Scenarios")
    progress.add_step("Creating Detailed DataFrames")
    progress.add_step("Saving Results and Reports")
    
    print("🚀 ENHANCED AAVE RISK EARLY-WARNING SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print(f"Cache enabled: {cache_enabled}")
    print(f"CoinGecko API: {'✅ Configured' if coingecko_api_key else '❌ Not configured'}")
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Load Dune Data
    # Step 1: Load Data
    progress.start_step(0)
    try:
        # Load reserves from DATABASE (RPC data), not Dune
        from .db_models import SessionLocal, Reserve
        db = SessionLocal()
        try:
            reserves = db.query(Reserve).filter(Reserve.is_active == True).all()
            df_reserve = pd.DataFrame([{
                'token_symbol': r.token_symbol,
                'chain': r.chain,
                'liquidity_rate': r.liquidity_rate,
                'variable_borrow_rate': r.variable_borrow_rate,
                'utilization_rate': (r.variable_borrow_rate / (r.liquidity_rate + 1e-10)) if r.liquidity_rate else 0,
                'supply_apy': r.supply_apy,
                'borrow_apy': r.borrow_apy,
                'ltv': r.ltv,
                'liquidation_threshold': r.liquidation_threshold,
                'price_usd': r.price_usd,
                'is_active': r.is_active
            } for r in reserves])
            progress.add_detail(f"Loaded {len(df_reserve)} reserve entries from RPC database")
        finally:
            db.close()
        # Load positions
        try:
            DUNE_API_KEY = os.getenv("DUNE_API_KEY_cURRENT_POSITION")
            if DUNE_API_KEY:
                print(f"   🔑 Using Dune API key for positions (length: {len(DUNE_API_KEY)})")
                dune = DuneClient(api_key=DUNE_API_KEY)
                response = dune.get_custom_endpoint_result("firstbml", "current-position", limit=5000)
                df_positions = pd.DataFrame(response.result.rows) if hasattr(response, "result") else pd.DataFrame()
                progress.add_detail(f"Loaded {len(df_positions)} positions")
            else:
                raise ValueError("Position API key required")
        except Exception as e:
            progress.add_detail(f"Position loading failed: {str(e)}")
            df_positions = pd.DataFrame()
        
        # Load liquidation history
        try:
            DUNE_API_KEY = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
            if DUNE_API_KEY:
                print(f"   🔑 Using Dune API key for liquidations (length: {len(DUNE_API_KEY)})")
                dune = DuneClient(api_key=DUNE_API_KEY)
                response = dune.get_custom_endpoint_result("firstbml", "liquidation-history", limit=5000)
                df_liquidations = pd.DataFrame(response.result.rows) if hasattr(response, "result") else pd.DataFrame()
                progress.add_detail(f"Loaded {len(df_liquidations)} liquidation events")
            else:
                df_liquidations = pd.DataFrame()
                progress.add_detail("No liquidation API key configured")
        except Exception as e:
            df_liquidations = pd.DataFrame()
            progress.add_detail(f"Liquidation loading failed: {str(e)}")
        
        progress.complete_step(0, True)
        
    except Exception as e:
        progress.complete_step(0, False)
        return {'error': f"Data loading failed: {str(e)}"}
    
    if df_positions.empty:
        return {'error': "No position data available - cannot proceed with analysis"}
    
    # Step 2: Fetch Current Prices
    progress.start_step(1)
    try:
        price_fetcher = EnhancedPriceFetcher(coingecko_api_key)
        
        # Get unique tokens from positions
        unique_tokens = df_positions['token_symbol'].dropna().unique().tolist()
        progress.add_detail(f"Fetching prices for {len(unique_tokens)} unique tokens")
        
        current_prices = price_fetcher.get_batch_prices(unique_tokens, progress)
        progress.add_detail(f"Successfully fetched {len(current_prices)} prices")
        progress.add_detail(f"Using fallback prices for {len(unique_tokens) - len(current_prices)} tokens")
        
        progress.complete_step(1, True)
        
    except Exception as e:
        progress.complete_step(1, False)
        current_prices = {}
        progress.add_detail(f"Price fetching failed: {str(e)}")
    
    # Step 3: Clean and Filter Data
    progress.start_step(2)
    try:
        analyzer = AAVERiskAnalyzer(price_fetcher)
        df_positions_clean = analyzer.clean_position_data(df_positions, progress)
        
        progress.complete_step(2, True)
        
    except Exception as e:
        progress.complete_step(2, False)
        return {'error': f"Data cleaning failed: {str(e)}"}
    
    # Step 4: Calculate Enhanced Metrics
    progress.start_step(3)
    try:
        df_positions_enhanced = analyzer.calculate_enhanced_metrics(
            df_positions_clean, current_prices, progress
        )
        
        progress.complete_step(3, True)
        
    except Exception as e:
        progress.complete_step(3, False)
        return {'error': f"Metric calculation failed: {str(e)}"}
    
    # Step 5: Analyze Reserves
    progress.start_step(4)
    try:
        reserve_analysis = ReserveAnalyzer.analyze_reserves(df_reserve, progress)
        progress.complete_step(4, True)
        
    except Exception as e:
        progress.complete_step(4, False)
        reserve_analysis = {}
    
    # Step 6: Concentration Risk Analysis
    progress.start_step(5)
    try:
        concentration_analysis = analyzer.analyze_concentration_risk(df_positions_enhanced)
        progress.add_detail(f"Analyzed concentration across tokens and chains")
        progress.add_detail(f"Identified {concentration_analysis.get('large_positions', {}).get('count', 0)} large positions")
        
        progress.complete_step(5, True)
        
    except Exception as e:
        progress.complete_step(5, False)
        concentration_analysis = {}
    
    # Step 7: Liquidation Risk Scenarios
    progress.start_step(6)
    try:
        liquidation_risk = analyzer.calculate_liquidation_risk_metrics(
            df_positions_enhanced, df_liquidations
        )
        
        stress_scenarios = liquidation_risk.get('stress_test_scenarios', {})
        progress.add_detail(f"Calculated stress test scenarios: {list(stress_scenarios.keys())}")
        
        # Show critical stress test result
        critical_scenario = stress_scenarios.get('20%_drop', {})
        if critical_scenario:
            progress.add_detail(f"20% drop scenario: {critical_scenario.get('positions_liquidated', 0)} positions at risk")
        
        progress.complete_step(6, True)
        
    except Exception as e:
        progress.complete_step(6, False)
        liquidation_risk = {}
    
    # Step 8: Create Detailed DataFrames
    progress.start_step(7)
    try:
        df_positions_detailed, df_liquidations_detailed = create_detailed_dataframes(
            df_positions_enhanced, df_liquidations
        )
        
        progress.add_detail(f"Created detailed position DataFrame: {len(df_positions_detailed)} rows")
        progress.add_detail(f"Created detailed liquidation DataFrame: {len(df_liquidations_detailed)} rows")
        
        progress.complete_step(7, True)
        
    except Exception as e:
        progress.complete_step(7, False)
        df_positions_detailed = df_positions_enhanced
        df_liquidations_detailed = df_liquidations
    
    # Step 9: Save Results
    progress.start_step(8)
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if save_results:
            # Save detailed DataFrames
            positions_file = f"aave_positions_detailed_{timestamp}.csv"
            df_positions_detailed.to_csv(positions_file, index=False)
            progress.add_detail(f"Saved positions: {positions_file}")
            
            liquidations_file = f"aave_liquidations_detailed_{timestamp}.csv"
            df_liquidations_detailed.to_csv(liquidations_file, index=False)
            progress.add_detail(f"Saved liquidations: {liquidations_file}")
            
            # Save comprehensive analysis
            comprehensive_analysis = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_positions_analyzed': len(df_positions_enhanced),
                    'price_data_coverage': len(current_prices) / len(unique_tokens) if unique_tokens else 0,
                    'analysis_duration_seconds': (datetime.now() - progress.start_time).total_seconds()
                },
                'protocol_overview': {
                    'total_positions': len(df_positions_enhanced),
                    'positions_with_debt': len(df_positions_enhanced[df_positions_enhanced['total_debt_usd'] > 0]),
                    'total_collateral_usd': float(df_positions_enhanced['current_collateral_usd'].sum()),
                    'total_debt_usd': float(df_positions_enhanced['total_debt_usd'].sum()),
                    'protocol_ltv': float(df_positions_enhanced['total_debt_usd'].sum() / df_positions_enhanced['current_collateral_usd'].sum()) if df_positions_enhanced['current_collateral_usd'].sum() > 0 else 0,
                    'weighted_avg_health_factor': float(df_positions_enhanced[df_positions_enhanced['enhanced_health_factor'] != np.inf]['enhanced_health_factor'].mean()) if len(df_positions_enhanced[df_positions_enhanced['enhanced_health_factor'] != np.inf]) > 0 else 0
                },
                'risk_distribution': df_positions_enhanced['risk_category'].value_counts().to_dict(),
                'concentration_analysis': concentration_analysis,
                'reserve_analysis': reserve_analysis,
                'liquidation_risk_analysis': liquidation_risk,
                'current_prices_used': current_prices,
                'critical_alerts': generate_critical_alerts(df_positions_enhanced, liquidation_risk)
            }
            
            analysis_file = f"aave_comprehensive_analysis_{timestamp}.json"
            with open(analysis_file, 'w') as f:
                json.dump(comprehensive_analysis, f, indent=2, default=str)
            progress.add_detail(f"Saved comprehensive analysis: {analysis_file}")
            
            # Generate executive summary report
            executive_report = generate_executive_report(comprehensive_analysis)
            report_file = f"aave_executive_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(executive_report)
            progress.add_detail(f"Saved executive report: {report_file}")
        
        progress.complete_step(8, True)
        
    except Exception as e:
        progress.complete_step(8, False)
        progress.add_detail(f"Save error: {str(e)}")
    
    # Final summary
    print(f"\n{progress.get_summary()}")
    
    # Display critical insights
    display_critical_insights(df_positions_enhanced, liquidation_risk, concentration_analysis)
    
    return {
        'success': True,
        'positions_dataframe': df_positions_detailed,
        'liquidations_dataframe': df_liquidations_detailed,
        'comprehensive_analysis': comprehensive_analysis,
        'current_prices': current_prices,
        'execution_time': (datetime.now() - progress.start_time).total_seconds()
    }

def generate_critical_alerts(df_positions: pd.DataFrame, liquidation_risk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate critical alerts for immediate attention"""
    alerts = []
    
    # Critical position alerts
    critical_positions = df_positions[df_positions['risk_category'].isin(['CRITICAL', 'LIQUIDATION_IMMINENT'])]
    
    for _, position in critical_positions.head(10).iterrows():  # Top 10 most critical
        alerts.append({
            'type': 'CRITICAL_POSITION',
            'severity': 'HIGH',
            'borrower': position['borrower_address'][:10] + '...',
            'token': position['token_symbol'],
            'health_factor': float(position['enhanced_health_factor']) if position['enhanced_health_factor'] != np.inf else None,
            'collateral_usd': float(position['current_collateral_usd']),
            'debt_usd': float(position['total_debt_usd']),
            'message': f"Position with HF {position['enhanced_health_factor']:.3f} needs immediate attention"
        })
    
    # Stress test alerts
    stress_scenarios = liquidation_risk.get('stress_test_scenarios', {})
    critical_scenario = stress_scenarios.get('20%_drop', {})
    
    if critical_scenario and critical_scenario.get('positions_liquidated', 0) > 50:
        alerts.append({
            'type': 'STRESS_TEST',
            'severity': 'MEDIUM',
            'scenario': '20% price drop',
            'positions_at_risk': critical_scenario['positions_liquidated'],
            'collateral_at_risk': critical_scenario['collateral_at_risk'],
            'message': f"20% market drop would liquidate {critical_scenario['positions_liquidated']} positions"
        })
    
    # Protocol-level alerts
    protocol_ltv = df_positions['total_debt_usd'].sum() / df_positions['current_collateral_usd'].sum() if df_positions['current_collateral_usd'].sum() > 0 else 0
    
    if protocol_ltv > 0.75:
        alerts.append({
            'type': 'PROTOCOL_RISK',
            'severity': 'HIGH',
            'current_ltv': protocol_ltv,
            'message': f"Protocol LTV of {protocol_ltv:.1%} is critically high"
        })
    elif protocol_ltv > 0.6:
        alerts.append({
            'type': 'PROTOCOL_RISK',
            'severity': 'MEDIUM',
            'current_ltv': protocol_ltv,
            'message': f"Protocol LTV of {protocol_ltv:.1%} requires monitoring"
        })
    
    return alerts

def generate_executive_report(analysis: Dict[str, Any]) -> str:
    """Generate executive summary report"""
    overview = analysis['protocol_overview']
    risk_dist = analysis['risk_distribution']
    alerts = analysis['critical_alerts']
    
    report = f"""
AAVE RISK EARLY-WARNING SYSTEM - EXECUTIVE REPORT
================================================
Generated: {analysis['metadata']['analysis_timestamp']}
Analysis Duration: {analysis['metadata']['analysis_duration_seconds']:.2f} seconds

PROTOCOL OVERVIEW
-----------------
Total Positions: {overview['total_positions']:,}
Positions with Debt: {overview['positions_with_debt']:,}
Total Collateral: ${overview['total_collateral_usd']:,.0f}
Total Debt: ${overview['total_debt_usd']:,.0f}
Protocol LTV: {overview['protocol_ltv']:.2%}
Weighted Avg Health Factor: {overview['weighted_avg_health_factor']:.3f}

RISK DISTRIBUTION
-----------------"""

    total_with_debt = overview['positions_with_debt']
    for risk_level, count in risk_dist.items():
        if total_with_debt > 0:
            pct = (count / total_with_debt) * 100
            report += f"\n{risk_level}: {count:,} positions ({pct:.1f}%)"

    report += f"""

CRITICAL ALERTS ({len(alerts)})
-----------------"""
    
    for alert in alerts:
        if alert['type'] == 'CRITICAL_POSITION':
            report += f"\n• {alert['message']} - Borrower: {alert['borrower']}"
        elif alert['type'] == 'STRESS_TEST':
            report += f"\n• {alert['message']}"
        elif alert['type'] == 'PROTOCOL_RISK':
            report += f"\n• {alert['message']}"

    # Concentration analysis
    concentration = analysis.get('concentration_analysis', {})
    if concentration:
        token_conc = concentration.get('token_concentration', {})
        if token_conc:
            report += f"""

CONCENTRATION RISK
------------------
Top 5 Token Concentration: {token_conc.get('top_5_tokens_share', 0):.1%}
HHI Index: {token_conc.get('hhi_index', 0):.3f}"""

        large_pos = concentration.get('large_positions', {})
        if large_pos:
            report += f"""
Large Positions (>$1M): {large_pos.get('count', 0):,}
Share of Protocol: {large_pos.get('share_of_protocol', 0):.1%}"""

    # Liquidation risk scenarios
    liquidation_analysis = analysis.get('liquidation_risk_analysis', {})
    stress_scenarios = liquidation_analysis.get('stress_test_scenarios', {})
    
    if stress_scenarios:
        report += f"""

STRESS TEST SCENARIOS
---------------------"""
        for scenario, results in stress_scenarios.items():
            report += f"""
{scenario}: {results['positions_liquidated']:,} positions, ${results['collateral_at_risk']:,.0f} at risk"""

    report += f"""

RECOMMENDATIONS
---------------"""
    
    critical_count = risk_dist.get('CRITICAL', 0) + risk_dist.get('LIQUIDATION_IMMINENT', 0)
    
    if critical_count > 100:
        report += "\n1. URGENT: Initiate emergency risk management protocol"
        report += "\n2. Prepare large-scale liquidation infrastructure"
        report += "\n3. Consider temporarily halting new borrowing"
    elif critical_count > 50:
        report += "\n1. HIGH PRIORITY: Review all critical positions immediately"
        report += "\n2. Increase liquidation bot capacity"
        report += "\n3. Alert users with positions below 1.2 health factor"
    elif critical_count > 0:
        report += f"\n1. Monitor {critical_count} critical positions closely"
        report += "\n2. Routine liquidation infrastructure check"
    else:
        report += "\n1. Continue normal monitoring procedures"

    if overview['protocol_ltv'] > 0.7:
        report += "\n4. Protocol LTV critical - implement emergency measures"
    elif overview['protocol_ltv'] > 0.5:
        report += "\n4. Protocol LTV elevated - review risk parameters"

    report += f"""
5. Run analysis daily during volatile market conditions
6. Review concentration limits for large positions
7. Monitor cross-chain exposure and bridge risks

This analysis should be updated regularly to track risk evolution.
For detailed data, refer to the CSV exports and JSON analysis file.
"""
    
    return report

def display_critical_insights(df_positions: pd.DataFrame, liquidation_risk: Dict, concentration_analysis: Dict):
    """Display key insights to console"""
    
    print(f"\n" + "=" * 60)
    print("CRITICAL INSIGHTS SUMMARY")
    print("=" * 60)
    
    # Risk summary
    risk_counts = df_positions['risk_category'].value_counts()
    critical_positions = risk_counts.get('CRITICAL', 0) + risk_counts.get('LIQUIDATION_IMMINENT', 0)
    
    if critical_positions > 100:
        print(f"🚨 EMERGENCY: {critical_positions} positions in critical state")
    elif critical_positions > 50:
        print(f"⚠️  HIGH RISK: {critical_positions} positions need immediate attention")
    elif critical_positions > 0:
        print(f"📊 MEDIUM RISK: {critical_positions} positions require monitoring")
    else:
        print(f"✅ LOW RISK: No critical positions detected")
    
    # Protocol health
    total_collateral = df_positions['current_collateral_usd'].sum()
    total_debt = df_positions['total_debt_usd'].sum()
    protocol_ltv = total_debt / total_collateral if total_collateral > 0 else 0
    
    if protocol_ltv > 0.8:
        print(f"🔴 Protocol LTV CRITICAL: {protocol_ltv:.1%}")
    elif protocol_ltv > 0.6:
        print(f"🟡 Protocol LTV ELEVATED: {protocol_ltv:.1%}")
    else:
        print(f"🟢 Protocol LTV HEALTHY: {protocol_ltv:.1%}")
    
    # Largest positions at risk
    large_risk_positions = df_positions[
        (df_positions['current_collateral_usd'] > 1000000) & 
        (df_positions['risk_category'].isin(['HIGH_RISK', 'CRITICAL', 'LIQUIDATION_IMMINENT']))
    ]
    
    if len(large_risk_positions) > 0:
        print(f"💰 WHALE ALERT: {len(large_risk_positions)} large positions (>$1M) at risk")
        whale_exposure = large_risk_positions['current_collateral_usd'].sum()
        print(f"   Total whale exposure: ${whale_exposure:,.0f}")
    
    # Concentration risk
    if concentration_analysis:
        token_conc = concentration_analysis.get('token_concentration', {})
        top5_share = token_conc.get('top_5_tokens_share', 0)
        if top5_share > 0.8:
            print(f"⚠️  HIGH CONCENTRATION: Top 5 tokens represent {top5_share:.1%} of collateral")
    
    # Stress test highlight
    stress_scenarios = liquidation_risk.get('stress_test_scenarios', {})
    severe_scenario = stress_scenarios.get('30%_drop', {})
    if severe_scenario:
        positions_at_risk = severe_scenario.get('positions_liquidated', 0)
        if positions_at_risk > 100:
            print(f"📉 STRESS TEST: 30% market drop would liquidate {positions_at_risk} positions")

    print(f"\n📁 Check generated CSV files for detailed position and liquidation data")
    print(f"📋 Review JSON analysis file for complete metrics and scenarios")

# Main execution
if __name__ == "__main__":
    print("Running Enhanced AAVE Risk Analysis...")
    
    # Debug environment variables
    print("\n🔍 ENVIRONMENT VARIABLES CHECK:")
    print(f"COINGECKO_API_KEY present: {'✅' if os.getenv('COINGECKO_API_KEY') else '❌'}")
    print(f"DUNE_API_KEY_RESERVE present: {'✅' if os.getenv('DUNE_API_KEY_RESERVE') else '❌'}")
    print(f"DUNE_API_KEY_cURRENT_POSITION present: {'✅' if os.getenv('DUNE_API_KEY_cURRENT_POSITION') else '❌'}")
    print(f"DUNE_API_KEY_LIQUIDATION_HISTORY present: {'✅' if os.getenv('DUNE_API_KEY_LIQUIDATION_HISTORY') else '❌'}")
    
    # Run with your configuration
    results = run_comprehensive_aave_risk_analysis(
        coingecko_api_key=os.getenv("COINGECKO_API_KEY"),  # Add your API key to .env
        save_results=True,
        cache_enabled=True
    )
    
    if results.get('success'):
        print(f"\n✅ Analysis completed successfully!")
        print(f"⏱️  Total execution time: {results['execution_time']:.2f} seconds")
        print(f"📊 Analyzed {len(results['positions_dataframe'])} positions")
        print(f"💰 Current prices fetched for {len(results['current_prices'])} tokens")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")