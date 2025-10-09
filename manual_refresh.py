#!/usr/bin/env python3
"""
Manual Data Refresh Script
Run this locally or via cron to refresh all data
"""
import requests
import argparse
import sys
from datetime import datetime

# Your deployed API URL
API_URL = "https://web-production-bac5f.up.railway.app/api/data/refresh"

def refresh_all():
    """Refresh everything"""
    print("\n🔄 Starting full data refresh...")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    response = requests.post(
        API_URL,
        json={
            "refresh_reserves": True,
            "refresh_positions": True,
            "refresh_liquidations": True,
            "prices_only": False
        },
        timeout=600  # 10 min timeout
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print(f"Reserves: {result['summary']['reserves_updated']}")
        print(f"Positions: {result['summary']['positions_updated']}")
        print(f"Liquidations: {result['summary']['liquidations_updated']}")
        print(f"Prices: {result['summary']['prices_updated']}")
        return True
    else:
        print(f"\n❌ FAILED: {response.status_code}")
        print(response.text)
        return False

def refresh_prices_only():
    """Quick price refresh"""
    print("\n💰 Refreshing prices only...")
    
    response = requests.post(
        API_URL,
        json={"prices_only": True},
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Updated {result['summary']['prices_updated']} prices")
        return True
    else:
        print(f"❌ FAILED: {response.status_code}")
        return False

def refresh_specific(chains: list):
    """Refresh specific chains only"""
    print(f"\n🔄 Refreshing chains: {', '.join(chains)}")
    
    response = requests.post(
        API_URL,
        json={
            "chains": chains,
            "refresh_reserves": True,
            "refresh_positions": True,
            "refresh_liquidations": True
        },
        timeout=600
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Refresh completed for {chains}")
        print(f"Summary: {result['summary']}")
        return True
    else:
        print(f"❌ FAILED: {response.status_code}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual data refresh")
    parser.add_argument("--full", action="store_true", help="Full refresh (all data)")
    parser.add_argument("--prices", action="store_true", help="Prices only (fast)")
    parser.add_argument("--chains", nargs="+", help="Specific chains: --chains ethereum polygon")
    
    args = parser.parse_args()
    
    if args.prices:
        success = refresh_prices_only()
    elif args.chains:
        success = refresh_specific(args.chains)
    elif args.full:
        success = refresh_all()
    else:
        print("Usage:")
        print("  python manual_refresh.py --full              # Full refresh")
        print("  python manual_refresh.py --prices            # Prices only")
        print("  python manual_refresh.py --chains ethereum   # Specific chains")
        sys.exit(1)
    
    sys.exit(0 if success else 1)