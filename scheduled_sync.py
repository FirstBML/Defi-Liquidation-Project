#!/usr/bin/env python3
"""
Scheduled data sync and analysis system
Runs every 24 hours (1440 minutes) and restarts API server
"""
import sys
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import subprocess
import signal
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Path to API entrypoint
API_MODULE = "app.main:app"
API_HOST = "127.0.0.1"
API_PORT = "8000"

scheduler = BackgroundScheduler()
uvicorn_process = None

def restart_api_server():
    """Restart the uvicorn API server after sync"""
    global uvicorn_process

    # Kill existing process if running
    if uvicorn_process and uvicorn_process.poll() is None:
        print("\n🔄 Stopping old API server...")
        uvicorn_process.send_signal(signal.SIGTERM)
        try:
            uvicorn_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            uvicorn_process.kill()
            print("⚠️ Forced kill of old API server")

    # Start a new server
    print("🚀 Starting fresh API server...")
    uvicorn_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        API_MODULE, "--host", API_HOST, "--port", API_PORT, "--reload"
    ])
    print(f"✅ API server restarted at http://{API_HOST}:{API_PORT}")

def sync_and_analyze():
    """Complete sync and analysis job"""
    try:
        print(f"\n{'='*60}")
        print(f"SCHEDULED JOB STARTED: {datetime.now()}")
        print(f"{'='*60}")
        
        # Step 1: Sync Dune data
        print("\n1. Syncing Dune data...")
        subprocess.run([sys.executable, "sync_dune_data.py"], check=True)
        
        # Step 2: Update risk metrics
        print("\n2. Updating risk metrics...")
        subprocess.run([sys.executable, "update_risk_metrics.py"], check=True)
        
        # Step 3: Deduplicate positions
        print("\n3. Deduplicating positions...")
        subprocess.run([sys.executable, "deduplicate_positions.py"], check=True)
        
        # Step 4: Check and send alerts
        print("\n4. Checking for alerts...")
        subprocess.run([sys.executable, "automated_alerts.py"], check=True)
        
        # Step 5: Restart API server
        restart_api_server()
        
        print(f"\n{'='*60}")
        print(f"SCHEDULED JOB COMPLETED: {datetime.now()}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Scheduled job failed: {e}")
        import traceback
        traceback.print_exc()

def start_scheduler():
    """Start the scheduler with 24-hour interval"""
    scheduler.add_job(
        sync_and_analyze,
        trigger=IntervalTrigger(minutes=1440),
        id='sync_and_analyze_job',
        name='Sync Dune Data and Analyze',
        replace_existing=True
    )
    scheduler.start()
    print("Scheduler started - Running every 24 hours (1440 minutes)")
    print(f"Next run scheduled at: {scheduler.get_jobs()[0].next_run_time}")
    
    # Run once immediately
    print("\nRunning initial sync...")
    sync_and_analyze()

def stop_scheduler():
    """Stop the scheduler"""
    scheduler.shutdown()
    print("Scheduler stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scheduled sync system")
    parser.add_argument("--run-once", action="store_true", help="Run once and exit")
    parser.add_argument("--start", action="store_true", help="Start scheduler")
    args = parser.parse_args()
    
    if args.run_once:
        sync_and_analyze()
    elif args.start:
        start_scheduler()
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            stop_scheduler()
    else:
        print("Usage:")
        print("  python scheduled_sync.py --run-once    # Run once and exit")
        print("  python scheduled_sync.py --start       # Start scheduler (runs continuously)")
