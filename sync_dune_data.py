"""
Sync real Dune Analytics data to your database
Now handles only Position and Liquidation data (Reserve logic removed)
"""
import sys
import os
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests

# Load environment variables FIRST
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


# ---------------------- POSITION DATA ---------------------- #
def fetch_dune_position_data() -> pd.DataFrame:
    """Fetch position data from Dune"""
    try:
        from dune_client.client import DuneClient

        DUNE_API_KEY = os.getenv("DUNE_API_KEY_CURRENT_POSITION")
        if not DUNE_API_KEY:
            print("⚠️ DUNE_API_KEY_CURRENT_POSITION not found in environment")
            return pd.DataFrame()

        print("📡 Fetching position data from Dune...")
        dune = DuneClient(api_key=DUNE_API_KEY)
        response = dune.get_custom_endpoint_result("firstbml", "current-position", limit=5000)

        if hasattr(response, "result") and response.result:
            df = pd.DataFrame(response.result.rows)
            print(f"✅ Fetched {len(df)} position records from Dune")
            return df
        else:
            print("❌ No position data returned from Dune")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Failed to fetch position data: {e}")
        return pd.DataFrame()


def sync_positions_to_db(df_positions: pd.DataFrame) -> int:
    """Sync position data to database"""
    if df_positions.empty:
        return 0

    try:
        from app.db_models import SessionLocal, Position

        session = SessionLocal()
        session.query(Position).delete()

        inserted_count = 0
        for _, row in df_positions.iterrows():
            position = Position(
                borrower_address=row.get('borrower_address'),
                chain=row.get('chain'),
                token_symbol=row.get('token_symbol'),
                token_address=row.get('token_address'),
                collateral_amount=row.get('collateral_amount'),
                debt_amount=row.get('debt_amount'),
                health_factor=row.get('health_factor'),
                total_collateral_usd=row.get('total_collateral_usd'),
                total_debt_usd=row.get('total_debt_usd'),
                enhanced_health_factor=row.get('enhanced_health_factor'),
                risk_category=row.get('risk_category'),
                last_updated=datetime.now(timezone.utc),
                query_time=datetime.now(timezone.utc)
            )
            session.add(position)
            inserted_count += 1

        session.commit()
        session.close()

        print(f"✅ Synced {inserted_count} positions to database")
        return inserted_count

    except Exception as e:
        print(f"❌ Failed to sync positions: {e}")
        session.rollback()
        session.close()
        return 0


# ---------------------- LIQUIDATION DATA ---------------------- #
def fetch_dune_liquidation_data() -> pd.DataFrame:
    """Fetch liquidation history from Dune"""
    try:
        from dune_client.client import DuneClient

        DUNE_API_KEY = os.getenv("DUNE_API_KEY_LIQUIDATION_HISTORY")
        if not DUNE_API_KEY:
            print("⚠️ DUNE_API_KEY_LIQUIDATION_HISTORY not found in environment")
            return pd.DataFrame()

        print("📡 Fetching liquidation data from Dune...")
        dune = DuneClient(api_key=DUNE_API_KEY)
        response = dune.get_custom_endpoint_result("firstbml", "liquidation-history", limit=5000)

        if hasattr(response, "result") and response.result:
            df = pd.DataFrame(response.result.rows)
            print(f"✅ Fetched {len(df)} liquidation records from Dune")
            return df
        else:
            print("❌ No liquidation data returned from Dune")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Failed to fetch liquidation data: {e}")
        return pd.DataFrame()


def sync_liquidations_to_db(df_liquidations: pd.DataFrame) -> int:
    """Sync liquidation data to database with USD value calculation"""
    if df_liquidations.empty:
        return 0

    try:
        from app.db_models import SessionLocal, LiquidationHistory
        from app.price_fetcher import EnhancedPriceFetcher

        session = SessionLocal()
        session.query(LiquidationHistory).delete()

        price_fetcher = EnhancedPriceFetcher(api_key=os.getenv("COINGECKO_API_KEY"))
        inserted_count = 0

        for _, row in df_liquidations.iterrows():
            collateral_seized = row.get('total_collateral_seized', 0) or 0
            collateral_symbol = row.get('collateral_symbol')
            chain = row.get('chain')

            liquidated_usd = 0.0
            if collateral_seized > 0 and collateral_symbol and chain:
                price_data = price_fetcher.get_batch_prices([{
                    'symbol': collateral_symbol,
                    'address': row.get('collateral_token'),
                    'chain': chain
                }])
                if collateral_symbol in price_data:
                    price = price_data[collateral_symbol].get('price', 0)
                    liquidated_usd = collateral_seized * price
                    print(f"💰 {collateral_symbol}: {collateral_seized} × ${price} = ${liquidated_usd}")

            liquidation = LiquidationHistory(
                liquidation_date=pd.to_datetime(row.get('liquidation_date')) if row.get('liquidation_date') else datetime.now(timezone.utc),
                chain=chain,
                borrower=row.get('borrower'),
                collateral_symbol=collateral_symbol,
                debt_symbol=row.get('debt_symbol'),
                collateral_asset=row.get('collateral_token'),
                debt_asset=row.get('debt_token'),
                total_collateral_seized=collateral_seized,
                total_debt_normalized=row.get('total_debt_normalized'),
                liquidated_collateral_usd=liquidated_usd,
                liquidation_count=row.get('liquidation_count'),
                avg_debt_per_event=row.get('avg_debt_per_event'),
                unique_liquidators=row.get('unique_liquidators'),
                created_at=datetime.now(timezone.utc),
                query_time=datetime.now(timezone.utc)
            )
            session.add(liquidation)
            inserted_count += 1

        session.commit()
        session.close()

        print(f"✅ Synced {inserted_count} liquidations with USD values to database")
        return inserted_count

    except Exception as e:
        print(f"❌ Failed to sync liquidations: {e}")
        session.rollback()
        session.close()
        return 0


# ---------------------- RUNNER ---------------------- #
def run_full_sync():
    print("\n1. Syncing Position Data...")
    df_positions = fetch_dune_position_data()
    sync_positions_to_db(df_positions)

    print("\n2. Syncing Liquidation Data...")
    df_liquidations = fetch_dune_liquidation_data()
    sync_liquidations_to_db(df_liquidations)

    print("\n✅ Full sync completed (positions + liquidations).")


def setup_scheduled_sync():
    """Set up automatic syncing (for future implementation)"""
    print("\n📅 Scheduled Sync Setup")
    print("-" * 30)
    print("To set up automatic syncing, you can:")
    print("1. Use a cron job to run this script periodically")
    print("2. Add scheduling to your FastAPI app with APScheduler")
    print("3. Use a task queue like Celery")
    print(f"\nExample cron job (runs every 30 minutes):")
    print(f"*/30 * * * * cd {project_root} && python sync_dune_data.py")


# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync Dune data to local database (positions + liquidations)")
    parser.add_argument("--positions-only", action="store_true", help="Sync only position data")
    parser.add_argument("--liquidations-only", action="store_true", help="Sync only liquidation data")
    parser.add_argument("--setup-cron", action="store_true", help="Show cron setup instructions")

    args = parser.parse_args()

    if args.setup_cron:
        setup_scheduled_sync()
    elif args.positions_only:
        df_positions = fetch_dune_position_data()
        sync_positions_to_db(df_positions)
    elif args.liquidations_only:
        df_liquidations = fetch_dune_liquidation_data()
        sync_liquidations_to_db(df_liquidations)
    else:
        run_full_sync()
