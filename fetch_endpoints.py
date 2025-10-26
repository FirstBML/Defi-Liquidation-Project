"""
Ultra Simple Endpoints Check - GUARANTEED OUTPUT
"""

import requests
import json
import os
from datetime import datetime

def ultra_simple_check():
    base_url = "https://easygoing-charm-production-707b.up.railway.app"
    
    print("🚀 ULTRA SIMPLE ENDPOINTS CHECK")
    print("=" * 50)
    
    endpoints = [
        "/", "/health", "/startup-status", "/docs", 
        "/openapi.json", "/api/v1/reserves", "/api/v2/reserves"
    ]
    
    results = []
    
    for endpoint in endpoints:
        full_url = base_url + endpoint
        try:
            print(f"Testing: {endpoint}")
            response = requests.get(full_url, timeout=10)
            
            result = {
                "endpoint": endpoint,
                "status": response.status_code,
                "success": response.status_code == 200,
                "time": datetime.now().strftime("%H:%M:%S")
            }
            results.append(result)
            print(f"  → Status: {response.status_code}")
            
        except Exception as e:
            result = {
                "endpoint": endpoint,
                "status": "ERROR",
                "success": False,
                "error": str(e),
                "time": datetime.now().strftime("%H:%M:%S")
            }
            results.append(result)
            print(f"  → ERROR: {e}")
    
    # SAVE FILE - GUARANTEED
    filename = "SIMPLE_RESULTS.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"✅ FILE SAVED: {filename}")
    print(f"📍 Location: {os.getcwd()}/{filename}")
    
    # Count results
    success_count = sum(1 for r in results if r['success'])
    print(f"📊 Results: {success_count}/{len(results)} successful")
    
    return results

# RUN IMMEDIATELY
if __name__ == "__main__":
    ultra_simple_check()