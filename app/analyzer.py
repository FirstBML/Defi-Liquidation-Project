# app/analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from .db_models import SessionLocal, Position, AnalysisSnapshot
from .alerts import notify_critical_alerts

def categorize_risk(hf: float, hf_threshold: float = 1.1) -> str:
    if pd.isna(hf):
        return "unknown"
    if hf < 1.0:
        return "liquidation"
    if hf < hf_threshold:
        return "high"
    if hf < 2.0:
        return "medium"
    return "low"

class AAVERiskAnalyzer:
    def __init__(self, price_fetcher, settings):
        self.fetcher = price_fetcher
        self.settings = settings

    def clean_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # ensure numeric
        for c in ["collateral_amount", "debt_amount", "health_factor", "liquidation_threshold_pct", "token_decimals"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # parse datetimes
        for d in ["query_time", "last_updated"]:
            if d in df.columns:
                df[d] = pd.to_datetime(df[d], errors="coerce")
        # ensure token symbol/address present
        if "token_symbol" not in df.columns and "token" in df.columns:
            df["token_symbol"] = df["token"]
        df = df.dropna(subset=["borrower_address", "token_symbol", "token_address"])
        return df

    def compute_enhanced_metrics(self, df: pd.DataFrame, current_prices: Dict[str, float]) -> pd.DataFrame:
        df = df.copy()
        # map price by token_symbol (some entries in current_prices may be keyed like 'SYM|addr|chain')
        def lookup_price(row):
            key_sym = row.get("token_symbol")
            # direct symbol
            if key_sym in current_prices:
                return current_prices.get(key_sym)
            # try fullname keys (symbol|address|chain)
            key2 = f"{row.get('token_symbol')}|{row.get('token_address')}|{row.get('chain')}"
            return current_prices.get(key2) or current_prices.get((row.get('token_address') or "").lower())
        df["price_usd"] = df.apply(lookup_price, axis=1)
        # fill via total_collateral_usd / collateral_amount if possible
        if "total_collateral_usd" in df.columns:
            df["current_collateral_usd"] = df.apply(
                lambda r: r["collateral_amount"] * r["price_usd"] if pd.notna(r.get("price_usd")) else r.get("total_collateral_usd"),
                axis=1
            )
        else:
            df["current_collateral_usd"] = df["collateral_amount"] * df["price_usd"]
        if "total_debt_usd" in df.columns:
            df["current_debt_usd"] = df.apply(
                lambda r: r["debt_amount"] * r["price_usd"] if pd.notna(r.get("price_usd")) else r.get("total_debt_usd"),
                axis=1
            )
        else:
            df["current_debt_usd"] = df["debt_amount"] * df["price_usd"]

        # enhanced health factor: (collateral * liquidation_threshold_pct) / debt
        def calc_enhanced(r):
            debt = r.get("current_debt_usd") or 0.0
            coll = r.get("current_collateral_usd") or 0.0
            thr = r.get("liquidation_threshold_pct") or 1.0
            if debt == 0 or pd.isna(debt):
                return float("inf")
            try:
                return (coll * thr) / debt
            except Exception:
                return np.nan

        df["enhanced_health_factor"] = df.apply(calc_enhanced, axis=1)
        df["risk_category"] = df["enhanced_health_factor"].apply(lambda x: categorize_risk(x, self.settings.ALERT_HEALTH_FACTOR_THRESHOLD))
        return df

    def run_stress_scenarios(self, df: pd.DataFrame, scenarios: Dict[str, float]) -> Dict[str, Any]:
        out = {}
        for name, drop in scenarios.items():
            if df.empty:
                out[name] = {"positions_at_risk": 0, "total_debt_at_risk_usd": 0.0, "positions": []}
                continue
            tmp = df.copy()
            tmp["collateral_after"] = tmp["current_collateral_usd"] * (1 - drop)
            tmp["hf_after"] = tmp.apply(lambda r: ((r.get("collateral_after") or 0.0) * (r.get("liquidation_threshold_pct") or 1.0) / (r.get("current_debt_usd") or 1e12)) if (r.get("current_debt_usd") or 0) > 0 else float("inf"), axis=1)
            at_risk = tmp[tmp["hf_after"] < 1.0]
            out[name] = {
                "positions_at_risk": int(len(at_risk)),
                "total_debt_at_risk_usd": float(at_risk["current_debt_usd"].sum()) if not at_risk.empty else 0.0,
                "positions": at_risk[["borrower_address", "token_symbol", "current_collateral_usd", "current_debt_usd", "hf_after"]].to_dict(orient="records")
            }
        return out

    def generate_alerts(self, df_enh: pd.DataFrame, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        alerts = {"critical": [], "warning": []}
        # whales
        if not df_enh.empty:
            whales = df_enh[df_enh["current_debt_usd"] >= self.settings.ALERT_WHALE_MIN_USD]
            for _, r in whales.iterrows():
                alerts["warning"].append({"type": "whale", "borrower": r["borrower_address"], "debt_usd": float(r["current_debt_usd"]), "hf": float(r["enhanced_health_factor"])})
        # scenario criticals (example: 20% drop)
        for name, result in scenario_results.items():
            if name == "20%_drop" and result.get("positions_at_risk", 0) > 0:
                alerts["critical"].append({"scenario": name, "positions_at_risk": result["positions_at_risk"]})
        # immediate liquidation
        if not df_enh.empty:
            immed = df_enh[df_enh["enhanced_health_factor"] < 1.0]
            for _, r in immed.iterrows():
                alerts["critical"].append({"type": "liquidation_now", "borrower": r["borrower_address"], "hf": float(r["enhanced_health_factor"]), "debt_usd": float(r["current_debt_usd"] or 0)})
        # notify
        if alerts["critical"]:
            notify_critical_alerts(alerts)
        return alerts

    def persist_results(self, df_enh: pd.DataFrame, prices_map: dict, snapshot: dict):
        session = SessionLocal()
        try:
            # simple upsert approach: delete and insert (for demo)
            session.query(Position).delete()
            for _, r in df_enh.iterrows():
                pos = Position(
                    borrower_address=r.get("borrower_address"),
                    chain=r.get("chain"),
                    token_symbol=r.get("token_symbol"),
                    token_address=r.get("token_address"),
                    collateral_amount=float(r.get("collateral_amount") or 0),
                    debt_amount=float(r.get("debt_amount") or 0),
                    health_factor=float(r.get("health_factor") or 0),
                    total_collateral_usd=float(r.get("current_collateral_usd") or 0),
                    total_debt_usd=float(r.get("current_debt_usd") or 0),
                    enhanced_health_factor=float(r.get("enhanced_health_factor") or 0),
                    risk_category=r.get("risk_category"),
                    last_updated=r.get("last_updated"),
                    query_time=r.get("query_time")
                )
                session.add(pos)
            snap = AnalysisSnapshot(summary=snapshot)
            session.add(snap)
            session.commit()
        finally:
            session.close()

def analyze_positions(db_session=None, price_fetcher=None, settings=None, positions_data=None):
    """
    Main function to analyze positions for liquidation risk
    """
    try:
        # If only db_session is provided, create other dependencies
        if db_session and price_fetcher is None:
            from .price_fetcher import EnhancedPriceFetcher
            price_fetcher = EnhancedPriceFetcher()
        
        if db_session and settings is None:
            from .config import get_settings
            settings = get_settings()
        
        if db_session and positions_data is None:
            # You might want to load positions data from the database here
            positions_data = None  # Or implement database loading
        
        analyzer = AAVERiskAnalyzer(price_fetcher, settings)
        
        # Your existing analysis logic...
        # [Keep the rest of your function]
        
            
        # Clean positions data - handle Dune Analytics column names
        df_cleaned = analyzer.clean_positions(positions_data)
        
        if df_cleaned.empty:
            return {"error": "No positions data available", "timestamp": datetime.now().isoformat()}
        
        # Map Dune column names to expected column names if needed
        column_mapping = {
            # Add any necessary column mappings here
            # Example: 'dune_column_name': 'expected_column_name'
        }
        
        if column_mapping:
            df_cleaned = df_cleaned.rename(columns=column_mapping)
        
        # Get current prices
        current_prices = price_fetcher.get_current_prices()
        
        # Compute enhanced metrics
        df_enhanced = analyzer.compute_enhanced_metrics(df_cleaned, current_prices)
        
        # Run stress scenarios
        scenarios = {
            "10%_drop": 0.10,
            "20%_drop": 0.20,
            "30%_drop": 0.30
        }
        scenario_results = analyzer.run_stress_scenarios(df_enhanced, scenarios)
        
        # Generate alerts
        alerts = analyzer.generate_alerts(df_enhanced, scenario_results)
        
        # Create snapshot
        snapshot = {
            "total_positions": len(df_enhanced),
            "high_risk_positions": len(df_enhanced[df_enhanced["risk_category"] == "high"]),
            "medium_risk_positions": len(df_enhanced[df_enhanced["risk_category"] == "medium"]),
            "low_risk_positions": len(df_enhanced[df_enhanced["risk_category"] == "low"]),
            "liquidation_positions": len(df_enhanced[df_enhanced["risk_category"] == "liquidation"]),
            "total_collateral_usd": df_enhanced["current_collateral_usd"].sum(),
            "total_debt_usd": df_enhanced["current_debt_usd"].sum(),
            "scenario_results": scenario_results,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
            "data_source": "dune_analytics"
        }
        
        # Persist results
        analyzer.persist_results(df_enhanced, current_prices, snapshot)
        
        return snapshot
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}", "timestamp": datetime.now().isoformat()}
    