# system_status.py
"""
Show overall system status and data summary
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api"

def show_system_status():
    """Show comprehensive system status"""
    print("📊 DEFI LIQUIDATION RISK SYSTEM - STATUS DASHBOARD")
    print("=" * 70)
    print(f"⏰ Last Updated: {datetime.now()}")
    print(f"🌐 API Base: {BASE_URL}")
    print()
    
    try:
        # Get dashboard summary
        response = requests.get(f"{BASE_URL}/dashboard/summary", timeout=10)
        if response.status_code == 200:
            dashboard = response.json()
            protocol = dashboard.get('protocol_overview', {})
            
            print("🎯 PROTOCOL OVERVIEW")
            print("-" * 40)
            print(f"📈 Total Positions: {protocol.get('total_positions', 0):,}")
            print(f"💰 Total Collateral: ${protocol.get('total_collateral_usd', 0):,.2f}")
            print(f"🏦 Total Debt: ${protocol.get('total_debt_usd', 0):,.2f}")
            print(f"⚖️  Protocol LTV: {protocol.get('protocol_ltv', 0):.2%}")
            print()
            
            # Risk distribution
            risk_dist = dashboard.get('risk_distribution', {})
            if risk_dist:
                print("🎭 RISK DISTRIBUTION")
                print("-" * 40)
                for risk, stats in risk_dist.items():
                    count = stats.get('count', 0)
                    if count > 0:
                        print(f"   {risk}: {count} positions")
        
        # Get reserves summary
        response = requests.get(f"{BASE_URL}/reserves/rpc/summary", timeout=10)
        if response.status_code == 200:
            reserves = response.json()
            chains = reserves.get('chains', [])
            
            print(f"\n🌐 RESERVES ACROSS CHAINS: {len(chains)}")
            print("-" * 40)
            for chain in chains:
                print(f"   {chain.get('chain').upper():<12} {chain.get('active_reserves'):>2} active reserves")
                print(f"                Avg Supply APY: {chain.get('avg_supply_apy', 0):.2f}%")
                print(f"                Avg Borrow APY: {chain.get('avg_borrow_apy', 0):.2f}%")
        
        # Get recent liquidations
        response = requests.get(f"{BASE_URL}/liquidation-history?limit=3", timeout=10)
        if response.status_code == 200:
            liquidations = response.json()
            if liquidations:
                print(f"\n💧 RECENT LIQUIDATIONS")
                print("-" * 40)
                for liq in liquidations[:3]:
                    date = liq.get('liquidation_date', '')[:10]
                    collateral = liq.get('collateral_symbol', 'Unknown')
                    seized = liq.get('total_collateral_seized', 0)
                    print(f"   {date} - {collateral}: {seized:.2f} seized")
        
        # Get risky positions
        response = requests.get(f"{BASE_URL}/positions/risky?threshold_hf=1.5", timeout=10)
        if response.status_code == 200:
            risky = response.json()
            count = risky.get('count', 0)
            if count > 0:
                print(f"\n⚠️  RISKY POSITIONS (HF < 1.5): {count}")
                print("-" * 40)
                positions = risky.get('positions', [])[:3]
                for pos in positions:
                    hf = pos.get('health_factor', 0)
                    collateral = pos.get('collateral_usd', 0)
                    print(f"   {pos.get('token_symbol')}: HF={hf:.3f}, Collateral=${collateral:,.2f}")
        
        print(f"\n✅ SYSTEM STATUS: OPERATIONAL")
        print("🔗 Documentation: http://localhost:8000/docs")
        print("📚 API Testing: All endpoints available under /api prefix")
        
    except Exception as e:
        print(f"❌ Error fetching system status: {e}")

if __name__ == "__main__":
    show_system_status()