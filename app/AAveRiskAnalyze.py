# AAveRiskAnalyze.py
"""
Enhanced Risk Early-Warning System for AAVE Lending Protocol
Features:
- Progress tracking and detailed logging
- Smart caching with joblib
- Data filtering for meaningful analysis
- Comprehensive AAVE-specific metrics
- API rate limiting and error handling
- Detailed position and liquidation DataFrames
"""

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

load_dotenv()


# Configure caching
CACHE_DIR = Path("risk_cache")
CACHE_DIR.mkdir(exist_ok=True)
PRICE_CACHE_TTL = 7200  # 5 minutes for current prices
HISTORICAL_CACHE_TTL = 7200  # 1 hour for historical data
RESERVE_CACHE_TTL = 7200  # 15 minutes for reserve data

class ProgressTracker:
    """Track and display analysis progress"""
    def __init__(self):
        self.steps = []
        self.current_step = 0
        self.start_time = datetime.now()
    
    def add_step(self, description: str):
        self.steps.append({
            'description': description,
            'status': 'pending',
            'start_time': None,
            'end_time': None,
            'details': []
        })
    
    def start_step(self, step_index: int):
        self.current_step = step_index
        self.steps[step_index]['status'] = 'running'
        self.steps[step_index]['start_time'] = datetime.now()
        print(f"\n[{step_index + 1}/{len(self.steps)}] {self.steps[step_index]['description']}...")
    
    def add_detail(self, detail: str):
        if self.current_step < len(self.steps):
            self.steps[self.current_step]['details'].append(detail)
            print(f"   → {detail}")
    
    def complete_step(self, step_index: int, success: bool = True):
        self.steps[step_index]['status'] = 'completed' if success else 'failed'
        self.steps[step_index]['end_time'] = datetime.now()
        duration = (self.steps[step_index]['end_time'] - self.steps[step_index]['start_time']).total_seconds()
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} Completed in {duration:.2f}s")
    
    def get_summary(self):
        total_time = (datetime.now() - self.start_time).total_seconds()
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        return f"Analysis completed: {completed}/{len(self.steps)} steps in {total_time:.2f}s"

class SmartCache:
    """Intelligent caching system with TTL support"""
    
    @staticmethod
    def get_cache_key(prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "_".join(key_parts).replace("/", "_").replace(":", "_")
    
    @staticmethod
    def is_cache_valid(cache_file: Path, ttl_seconds: int) -> bool:
        """Check if cache file is still valid"""
        if not cache_file.exists():
            return False
        
        file_age = time.time() - cache_file.stat().st_mtime
        return file_age < ttl_seconds
    
    @classmethod
    def get_cached_data(cls, cache_key: str, ttl_seconds: int) -> Optional[Any]:
        """Get cached data if valid"""
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if cls.is_cache_valid(cache_file, ttl_seconds):
            try:
                return joblib.load(cache_file)
            except Exception as e:
                print(f"   ⚠️ Cache read error: {e}")
                return None
        return None
    
    @classmethod
    def save_cached_data(cls, cache_key: str, data: Any) -> bool:
        """Save data to cache"""
        try:
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            joblib.dump(data, cache_file)
            return True
        except Exception as e:
            print(f"   ⚠️ Cache save error: {e}")
            return False

# ---------- Configuration ----------
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY") or os.getenv("GECKO_API_KEY")
COINGECKO_BASE = "https://pro-api.coingecko.com/api/v3" if COINGECKO_API_KEY else "https://api.coingecko.com/api/v3"

DEFAULT_PRICE_CACHE_FILE = "price_cache.json"
DEFAULT_ID_CACHE_FILE = "coin_id_cache.json"

# A reasonable chunk size for CoinGecko simple price requests
COINGECKO_BATCH_SIZE = 80

# Stablecoins allowed for fallback $1
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FRAX", "LUSD", "USDC.E", "USDT.E", "DAI.E"}

# Hardcoded token overrides (expand as needed)
TOKEN_ID_OVERRIDES = {
    "BTC": "bitcoin", "WBTC": "wrapped-bitcoin",
    "ETH": "ethereum", "WETH": "weth",
    "USDT": "tether", "USDC": "usd-coin", "DAI": "dai", "BUSD": "binance-usd",
    "MATIC": "matic-network", "AVAX": "avalanche-2", "ARB": "arbitrum",
    "OP": "optimism", "LINK": "chainlink", "UNI": "uniswap", "AAVE": "aave"
}

# Map common Aave / EVM chains to CoinGecko platform slugs
CHAIN_TO_PLATFORM = {
    "ethereum": "ethereum",
    "arbitrum": "arbitrum-one",
    "optimism": "optimism",
    "polygon": "polygon-pos",
    "base": "base",
    "avalanche": "avalanche",
    "fantom": "fantom",
    "bnb": "binance-smart-chain",
    "bsc": "binance-smart-chain",
    "gnosis": "xdai",
    "celo": "celo",
    "metis": "metis-andromeda",
    "scroll": "scroll",
    "zksync": "zksync-era",
    "tron": "tron",
    "near": "near",
    # add more if needed
}

# ---------- Helper functions ----------
def _save_json_safe(path: str, data: Any):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def _load_json_safe(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _normalize_symbol(sym: str) -> str:
    """Normalize token symbol like 'DAI.e' -> 'DAI' for basic mapping."""
    if not sym:
        return ""
    # split on '.' or '-' and take first part (common suffix patterns)
    return str(sym).split(".")[0].split("-")[0].upper()

# ---------- Class ----------
class EnhancedPriceFetcher:
    """
    Holistic CoinGecko price fetcher with:
     - symbol overrides
     - contract lookup (chain-aware)
     - search fallback
     - batch price calls
     - caching (memory + disk) with TTL
     - optional historical lookup around a timestamp
     - optional progress tracker (progress.add_detail())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        price_cache_file: str = DEFAULT_PRICE_CACHE_FILE,
        id_cache_file: str = DEFAULT_ID_CACHE_FILE,
        cache_ttl: int = 300,  # seconds
        request_delay: Optional[float] = None,
    ):
        self.api_key = api_key or COINGECKO_API_KEY
        self.base = COINGECKO_BASE if (api_key or COINGECKO_API_KEY) else "https://api.coingecko.com/api/v3"
        self.headers = {"accept": "application/json"}
        if self.api_key:
            # pro header name accepted by Coingecko
            self.headers["X-Cg-Pro-Api-Key"] = self.api_key

        # rate limiting
        self.last_request_ts = 0.0
        if request_delay is not None:
            self.request_delay = request_delay
        else:
            # allow faster if pro key present
            self.request_delay = 0.45 if self.api_key else 1.25

        # caches
        self.cache_ttl = cache_ttl
        self.price_cache_file = price_cache_file
        self.id_cache_file = id_cache_file

        # In-memory caches: key -> (value, timestamp)
        # price_cache keys: (coin_id, date_iso) -> float
        # id_cache keys: (chain_normalized, symbol_normalized, address_lower) -> coin_id
        self.price_cache: Dict[str, List] = _load_json_safe(self.price_cache_file) or {}
        self.id_cache: Dict[str, str] = _load_json_safe(self.id_cache_file) or {}

        # internal markers
        self.failed_tokens = set()

        print(f"EnhancedPriceFetcher initialized. Coingecko Pro: {'YES' if self.api_key else 'NO'}")
        print(f"Base URL: {self.base}, cache_ttl={self.cache_ttl}s, request_delay={self.request_delay}s")

    # ---------- I/O / cache helpers ----------
    def _cache_key_price(self, coin_id: str, date_key: Optional[str] = None) -> str:
        if date_key is None:
            date_key = datetime.utcnow().date().isoformat()
        return f"p::{coin_id}::{date_key}"

    def _cache_key_id(self, chain: Optional[str], symbol: Optional[str], address: Optional[str]) -> str:
        chain_k = (chain or "").lower()
        sym_k = (symbol or "").upper()
        addr_k = (address or "").lower()
        return f"id::{chain_k}::{sym_k}::{addr_k}"

    def _get_cached_price(self, coin_id: str, date_key: Optional[str] = None) -> Optional[float]:
        k = self._cache_key_price(coin_id, date_key)
        rec = self.price_cache.get(k)
        if not rec:
            return None
        price, ts = rec if isinstance(rec, list) else (rec, time.time())
        # stored ts optional; if stored as [price, ts], use ts; else fallback to now
        if isinstance(rec, list):
            store_ts = float(rec[1])
        else:
            store_ts = time.time()
        if (time.time() - store_ts) < self.cache_ttl:
            return float(price)
        return None

    def _set_cached_price(self, coin_id: str, price: float, date_key: Optional[str] = None):
        k = self._cache_key_price(coin_id, date_key)
        self.price_cache[k] = [price, time.time()]
        _save_json_safe(self.price_cache_file, self.price_cache)

    def _get_cached_id(self, chain: Optional[str], symbol: Optional[str], address: Optional[str]) -> Optional[str]:
        k = self._cache_key_id(chain, symbol, address)
        return self.id_cache.get(k)

    def _set_cached_id(self, chain: Optional[str], symbol: Optional[str], address: Optional[str], coin_id: str):
        k = self._cache_key_id(chain, symbol, address)
        self.id_cache[k] = coin_id
        _save_json_safe(self.id_cache_file, self.id_cache)

    # ---------- network / rate limit ----------
    def _sleep_rate_limit(self):
        delta = time.time() - self.last_request_ts
        if delta < self.request_delay:
            time.sleep(self.request_delay - delta)
        self.last_request_ts = time.time()

    def _request_with_retry(self, url: str, params: dict = None, max_retries: int = 3) -> Optional[requests.Response]:
        """Perform GET with rate limiting and backoff on 429"""
        attempt = 0
        while attempt <= max_retries:
            try:
                self._sleep_rate_limit()
                resp = requests.get(url, params=params, headers=self.headers, timeout=15)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    attempt += 1
                    time.sleep(wait)
                    continue
                # non-200 & non-429 -> return None
                return resp
            except requests.RequestException as e:
                attempt += 1
                time.sleep(1 + attempt)
                if attempt > max_retries:
                    return None
        return None

    # ---------- coin id resolution ----------
    def _resolve_coin_id(self, symbol: str, chain: Optional[str] = None, address: Optional[str] = None) -> Optional[str]:
        """
        Resolve CoinGecko coin id using:
         1) overrides
         2) cached id
         3) contract lookup for chain (if platform known)
         4) search endpoint fallback
        """
        if not symbol:
            return None

        sym_norm = _normalize_symbol(symbol)

        # 1. Overrides
        if sym_norm in TOKEN_ID_OVERRIDES:
            coin_id = TOKEN_ID_OVERRIDES[sym_norm]
            # cache it
            self._set_cached_id(chain, symbol, address, coin_id)
            return coin_id

        # 2. cached id
        cached = self._get_cached_id(chain, symbol, address)
        if cached:
            return cached

        # 3. contract lookup (chain-aware)
        if address and chain:
            platform = CHAIN_TO_PLATFORM.get(chain.lower())
            if platform:
                url = f"{self.base}/coins/{platform}/contract/{address}"
                resp = self._request_with_retry(url)
                if resp and resp.status_code == 200:
                    try:
                        data = resp.json()
                        coin_id = data.get("id")
                        if coin_id:
                            self._set_cached_id(chain, symbol, address, coin_id)
                            return coin_id
                    except Exception:
                        pass

        # 4. search fallback by symbol (CoinGecko returns candidates)
        url = f"{self.base}/search"
        params = {"query": sym_norm}
        resp = self._request_with_retry(url, params=params)
        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                coins = data.get("coins")
                if coins and len(coins) > 0:
                    coin_id = coins[0].get("id")
                    if coin_id:
                        self._set_cached_id(chain, symbol, address, coin_id)
                        return coin_id
            except Exception:
                pass

        # no id found
        self.failed_tokens.add((chain, symbol, address))
        return None

    # ---------- price fetch helpers ----------
    def _chunk(self, seq: List[Any], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    def _fetch_prices_by_ids(self, coin_ids: List[str], vs_currency: str = "usd") -> Dict[str, float]:
        """Batch fetch simple prices for a list of CoinGecko IDs. Returns {coin_id: price}"""
        results: Dict[str, float] = {}
        if not coin_ids:
            return results

        for chunk in self._chunk(list(dict.fromkeys(coin_ids)), COINGECKO_BATCH_SIZE):
            ids_str = ",".join(chunk)
            url = f"{self.base}/simple/price"
            params = {"ids": ids_str, "vs_currencies": vs_currency}
            resp = self._request_with_retry(url, params=params, max_retries=3)
            if not resp:
                # skip this chunk
                continue
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    for cid, payload_val in payload.items():
                        if isinstance(payload_val, dict) and vs_currency in payload_val:
                            results[cid] = float(payload_val[vs_currency])
                except Exception:
                    continue
            else:
                # Log non-200
                try:
                    text = resp.text[:200]
                except Exception:
                    text = "<no response text>"
                # continue; results for these ids will be missing
        return results

    def _fetch_historical_price(self, coin_id: str, ts: datetime, window_days: int = 1, vs_currency: str = "usd") -> Optional[float]:
        """Fetch historical prices around ts using market_chart/range and return nearest price"""
        from_dt = int((ts - timedelta(days=window_days)).timestamp())
        to_dt = int((ts + timedelta(days=window_days)).timestamp())
        url = f"{self.base}/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": vs_currency, "from": from_dt, "to": to_dt}
        resp = self._request_with_retry(url, params=params, max_retries=3)
        if not resp or resp.status_code != 200:
            return None
        try:
            payload = resp.json()
            prices = payload.get("prices", [])
            if not prices:
                return None
            target_ms = int(ts.timestamp() * 1000)
            # find closest
            closest = min(prices, key=lambda x: abs(int(x[0]) - target_ms))
            return float(closest[1])
        except Exception:
            return None

    # ---------- public API ----------
    def get_current_price(
        self,
        symbol: str,
        address: Optional[str] = None,
        chain: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        vs_currency: str = "usd",
    ) -> Optional[float]:
        """
        Get current or near-time price for a single token.
        - symbol: token symbol (string)
        - address: contract address (optional, chain-specific)
        - chain: blockchain name used to resolve contract (optional)
        - timestamp: if provided, will try historical price near timestamp
        Returns float price in vs_currency or None.
        """
        if not symbol:
            return None

        # Normalized symbol for stablecoin logic
        sym_norm = _normalize_symbol(symbol)

        # Stablecoin shortcut (explicit)
        if sym_norm in STABLECOINS:
            return 1.0

        # Resolve coin id
        coin_id = self._resolve_coin_id(symbol, chain=chain, address=address)
        if not coin_id:
            return None

        # Use cache for current-day
        date_key = timestamp.date().isoformat() if timestamp else datetime.utcnow().date().isoformat()
        cached = self._get_cached_price(coin_id, date_key=date_key)
        if cached is not None:
            return cached

        # If historical timestamp provided, fetch market_chart/range
        if timestamp:
            price = self._fetch_historical_price(coin_id, timestamp, window_days=1, vs_currency=vs_currency)
            if price is not None:
                self._set_cached_price(coin_id, price, date_key=date_key)
            return price

        # Otherwise fetch via simple price
        fetched = self._fetch_prices_by_ids([coin_id], vs_currency=vs_currency)
        if coin_id in fetched:
            price = fetched[coin_id]
            self._set_cached_price(coin_id, price, date_key=date_key)
            return price

        # If not fetched and symbol is stable -> fallback $1
        if sym_norm in STABLECOINS:
            self._set_cached_price(coin_id, 1.0, date_key=date_key)
            return 1.0

        return None

    def get_batch_prices(
        self,
        tokens: List[Union[str, Dict[str, Optional[str]]]],
        progress: Optional[Any] = None,
        vs_currency: str = "usd"
    ) -> Dict[str, Optional[float]]:
        """
        Fetch prices for a batch of tokens.
        tokens can be:
         - list of symbol strings, e.g. ["ETH","USDC"]
         - list of dicts: {"symbol": "DAI", "address": "0x..", "chain": "avalanche"}
        Returns mapping keyed by the original token input string/dict-key to price (or None).
        For string inputs, the returned key is the original symbol string.
        """
        # Normalize tokens into entries
        entries = []
        for t in tokens:
            if isinstance(t, str):
                entries.append({"key": t, "symbol": t, "address": None, "chain": None})
            elif isinstance(t, dict):
                # dict expected to contain symbol/address/chain or similar
                s = t.get("symbol") or t.get("token_symbol") or ""
                addr = t.get("address") or t.get("token_address")
                chain = t.get("chain")
                key = t.get("key") or f"{s}|{addr or ''}|{chain or ''}"
                entries.append({"key": key, "symbol": s, "address": addr, "chain": chain})
            else:
                # tuple-like (symbol,address,chain)
                try:
                    s, addr, chain = t
                    key = f"{s}|{addr or ''}|{chain or ''}"
                    entries.append({"key": key, "symbol": s, "address": addr, "chain": chain})
                except Exception:
                    # fallback to str()
                    entries.append({"key": str(t), "symbol": str(t), "address": None, "chain": None})

        results: Dict[str, Optional[float]] = {}
        # First pass: try to satisfy from cache or stablecoin rule or overrides
        unresolved = []
        coinid_map = {}  # coin_id -> list of entry keys

        for e in entries:
            sym = e["symbol"]
            addr = e["address"]
            chain = e["chain"]
            key = e["key"]

            norm_sym = _normalize_symbol(sym)
            if norm_sym in STABLECOINS:
                results[key] = 1.0
                continue

            # Try cached id -> then cached price
            coin_id_cached = self._get_cached_id(chain, sym, addr)
            if coin_id_cached:
                # try price cache
                cached_price = self._get_cached_price(coin_id_cached)
                if cached_price is not None:
                    results[key] = cached_price
                    continue
                # mark coin_id for batch fetch
                coinid_map.setdefault(coin_id_cached, []).append(key)
                continue

            # If no cached coin id, we'll need to resolve
            unresolved.append(e)

        # Resolve coin ids for unresolved entries
        for e in unresolved:
            sym = e["symbol"]
            addr = e["address"]
            chain = e["chain"]
            key = e["key"]

            coin_id = self._resolve_coin_id(sym, chain=chain, address=addr)
            if coin_id:
                coinid_map.setdefault(coin_id, []).append(key)
            else:
                # If not resolved AND normalized symbol is a stablecoin -> fallback $1
                if _normalize_symbol(sym) in STABLECOINS:
                    results[key] = 1.0
                else:
                    # leave as None (no fake fallback)
                    results[key] = None
                    if progress:
                        progress.add_detail(f"⚠️ No CoinGecko ID for {sym} ({addr}) on {chain}")

        # Batch fetch prices for all coin_ids we have
        all_coin_ids = list(coinid_map.keys())
        if all_coin_ids:
            fetched_prices = self._fetch_prices_by_ids(all_coin_ids, vs_currency=vs_currency)
            # map back to keys
            for cid, keys in coinid_map.items():
                price_val = fetched_prices.get(cid)
                if price_val is not None:
                    # cache and assign
                    self._set_cached_price(cid, price_val)
                    for k in keys:
                        results[k] = price_val
                        if progress:
                            progress.add_detail(f"Price for {k}: ${price_val:,.6f}")
                else:
                    # price not returned for this coin id
                    for k in keys:
                        sym_guess = k.split("|")[0] if "|" in k else k
                        if _normalize_symbol(sym_guess) in STABLECOINS:
                            results[k] = 1.0
                            if progress:
                                progress.add_detail(f"Using stable fallback for {k}")
                        else:
                            results[k] = None
                            if progress:
                                progress.add_detail(f"⚠️ Missing price for {k} (coin id: {cid})")

        # final: return mapping keyed by simple symbol where possible (compatibility)
        out: Dict[str, Optional[float]] = {}
        for e in entries:
            orig_key = e["key"]
            # if input was plain symbol string, return under that symbol
            if isinstance(orig_key, str) and "|" not in orig_key:
                out[orig_key] = results.get(orig_key)
            else:
                out[orig_key] = results.get(orig_key)

        return out

    # ---------- persistence / cleanup ----------
    def save_caches(self):
        _save_json_safe(self.price_cache_file, self.price_cache)
        _save_json_safe(self.id_cache_file, self.id_cache)

    def clear_caches(self):
        self.price_cache = {}
        self.id_cache = {}
        try:
            if os.path.exists(self.price_cache_file):
                os.remove(self.price_cache_file)
            if os.path.exists(self.id_cache_file):
                os.remove(self.id_cache_file)
        except Exception:
            pass

# ---------- usage example ----------
if __name__ == "__main__":
    # quick demo (replace with your real tokens from positions)
    fetcher = EnhancedPriceFetcher()
    samples = ["USDC", "WETH", "BTC", {"symbol": "DAI.e", "address": "0xd586e7f844cea2f... (example)", "chain": "avalanche"}]
    prices = fetcher.get_batch_prices(samples, progress=None)
    print("Prices:", prices)

class AAVERiskAnalyzer:
    """Enhanced AAVE-specific risk analyzer"""
    
    def __init__(self, price_fetcher: EnhancedPriceFetcher):
        self.price_fetcher = price_fetcher
        self.liquidation_thresholds = {
            'WETH': 0.825, 'WBTC': 0.7, 'USDC': 0.875, 'DAI': 0.77,
            'LINK': 0.7, 'AAVE': 0.66, 'USDT': 0.8, 'MATIC': 0.65
        }
    
    def clean_position_data(self, df_positions: pd.DataFrame, progress_tracker: ProgressTracker) -> pd.DataFrame:
        """Clean and filter position data"""
        progress_tracker.add_detail(f"Initial positions: {len(df_positions)}")
        
        # Print first 10 rows for verification
        if not df_positions.empty:
            print("\n📊 FIRST 10 ROWS OF POSITION DATA:")
            print("=" * 80)
            print(df_positions.head(10).to_string(max_colwidth=20))
            print("=" * 80)
            
            # Print column information
            print(f"\n📋 POSITION DATA COLUMNS ({len(df_positions.columns)} total):")
            for i, col in enumerate(df_positions.columns, 1):
                print(f"   {i:2d}. {col}: {df_positions[col].dtype}")
        
        # Remove positions with both zero debt and zero collateral
        meaningful_positions = df_positions[
            (df_positions['total_debt_usd'] > 0) | 
            (df_positions['total_collateral_usd'] > 100)  # Keep collateral > $100
        ].copy()
        
        progress_tracker.add_detail(f"Filtered to meaningful positions: {len(meaningful_positions)}")
        
        # Clean extreme outliers
        if len(meaningful_positions) > 0:
            # Cap debt at 10x max collateral (likely data errors)
            max_collateral = meaningful_positions['total_collateral_usd'].max()
            debt_cap = max_collateral * 10
            
            outliers = meaningful_positions[meaningful_positions['total_debt_usd'] > debt_cap]
            if len(outliers) > 0:
                progress_tracker.add_detail(f"Capped {len(outliers)} extreme debt outliers")
                meaningful_positions['total_debt_usd'] = meaningful_positions['total_debt_usd'].clip(upper=debt_cap)
        
        return meaningful_positions
    
    def calculate_enhanced_metrics(self, df_positions: pd.DataFrame, current_prices: Dict[str, float], 
                                 progress_tracker: ProgressTracker) -> pd.DataFrame:
        """Calculate enhanced AAVE-specific metrics"""
        df = df_positions.copy()
        
        # Update collateral values with current prices
        df['current_price'] = df['token_symbol'].map(current_prices)
        df['price_available'] = df['current_price'].notna()
        
        # Calculate current collateral value
        df['current_collateral_usd'] = np.where(
            df['price_available'],
            df['collateral_amount'] * df['current_price'],
            df['total_collateral_usd']
        )
        
        # Enhanced health factor calculation
        df['liquidation_threshold'] = df['token_symbol'].map(self.liquidation_thresholds).fillna(0.75)
        
        df['enhanced_health_factor'] = np.where(
            df['total_debt_usd'] > 0,
            (df['current_collateral_usd'] * df['liquidation_threshold']) / df['total_debt_usd'],
            np.inf
        )
        
        # Current LTV
        df['current_ltv'] = np.where(
            df['current_collateral_usd'] > 0,
            df['total_debt_usd'] / df['current_collateral_usd'],
            0
        )
        
        # Liquidation price calculation
        df['liquidation_price'] = np.where(
            (df['collateral_amount'] > 0) & (df['total_debt_usd'] > 0),
            df['total_debt_usd'] / (df['collateral_amount'] * df['liquidation_threshold']),
            0
        )
        
        # Price drop to liquidation
        df['price_drop_to_liquidation_pct'] = np.where(
            (df['current_price'] > 0) & (df['liquidation_price'] > 0),
            ((df['current_price'] - df['liquidation_price']) / df['current_price']) * 100,
            100
        )
        
        # Risk categorization
        df['risk_category'] = 'SAFE'
        df.loc[df['enhanced_health_factor'] < 2.0, 'risk_category'] = 'LOW_RISK'
        df.loc[df['enhanced_health_factor'] < 1.5, 'risk_category'] = 'MEDIUM_RISK'
        df.loc[df['enhanced_health_factor'] < 1.3, 'risk_category'] = 'HIGH_RISK'
        df.loc[df['enhanced_health_factor'] < 1.1, 'risk_category'] = 'CRITICAL'
        df.loc[df['enhanced_health_factor'] < 1.0, 'risk_category'] = 'LIQUIDATION_IMMINENT'
        df.loc[df['total_debt_usd'] == 0, 'risk_category'] = 'NO_DEBT'
        
        # Position size categories
        df['position_size_category'] = pd.cut(
            df['current_collateral_usd'],
            bins=[0, 1000, 10000, 100000, 1000000, np.inf],
            labels=['SMALL', 'MEDIUM', 'LARGE', 'WHALE', 'MEGA_WHALE']
        )
        
        progress_tracker.add_detail(f"Calculated enhanced metrics for {len(df)} positions")
        progress_tracker.add_detail(f"Price data available for {df['price_available'].sum()} positions")
        
        return df
    
    def analyze_concentration_risk(self, df_positions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze concentration risk across tokens, chains, and large positions"""
        concentration_analysis = {}
        
        # Token concentration
        token_exposure = df_positions.groupby('token_symbol').agg({
            'current_collateral_usd': 'sum',
            'total_debt_usd': 'sum',
            'borrower_address': 'count'
        }).rename(columns={'borrower_address': 'position_count'})
        
        token_exposure['collateral_share'] = token_exposure['current_collateral_usd'] / token_exposure['current_collateral_usd'].sum()
        token_exposure = token_exposure.sort_values('current_collateral_usd', ascending=False)
        
        concentration_analysis['token_concentration'] = {
            'top_5_tokens_share': token_exposure['collateral_share'].head(5).sum(),
            'hhi_index': (token_exposure['collateral_share'] ** 2).sum(),  # Herfindahl-Hirschman Index
            'token_breakdown': token_exposure.head(10).to_dict('records')
        }
        
        # Chain concentration
        if 'chain' in df_positions.columns:
            chain_exposure = df_positions.groupby('chain').agg({
                'current_collateral_usd': 'sum',
                'borrower_address': 'count'
            }).rename(columns={'borrower_address': 'position_count'})
            
            chain_exposure['share'] = chain_exposure['current_collateral_usd'] / chain_exposure['current_collateral_usd'].sum()
            concentration_analysis['chain_concentration'] = chain_exposure.sort_values('current_collateral_usd', ascending=False).to_dict('records')
        
        # Large position analysis
        large_positions = df_positions[df_positions['current_collateral_usd'] > 1000000]  # > $1M
        concentration_analysis['large_positions'] = {
            'count': len(large_positions),
            'total_collateral': large_positions['current_collateral_usd'].sum(),
            'share_of_protocol': large_positions['current_collateral_usd'].sum() / df_positions['current_collateral_usd'].sum(),
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
        price_drop_scenarios = [5, 10, 20, 30, 50]  # % price drops
        stress_test_results = {}
        
        for drop_pct in price_drop_scenarios:
            # Simulate price drop
            stressed_collateral = df_positions['current_collateral_usd'] * (1 - drop_pct / 100)
            stressed_hf = np.where(
                df_positions['total_debt_usd'] > 0,
                (stressed_collateral * df_positions['liquidation_threshold']) / df_positions['total_debt_usd'],
                np.inf
            )
            
            # Count positions at risk
            at_risk_positions = df_positions[stressed_hf < 1.0]
            liquidation_value = at_risk_positions['current_collateral_usd'].sum()
            
            stress_test_results[f'{drop_pct}%_drop'] = {
                'positions_liquidated': len(at_risk_positions),
                'collateral_at_risk': liquidation_value,
                'percentage_of_protocol': (liquidation_value / df_positions['current_collateral_usd'].sum() * 100) if df_positions['current_collateral_usd'].sum() > 0 else 0
            }
        
        risk_metrics['stress_test_scenarios'] = stress_test_results
        
        # Historical liquidation analysis if available
        if df_liquidations is not None and len(df_liquidations) > 0:
            df_liq = df_liquidations.copy()
            df_liq['liquidation_date'] = pd.to_datetime(df_liq['liquidation_date'])
            
            # Print first 10 rows of liquidation data
            print("\n📊 FIRST 10 ROWS OF LIQUIDATION DATA:")
            print("=" * 80)
            print(df_liq.head(10).to_string(max_colwidth=20))
            print("=" * 80)
            
            # Recent trends (last 30 days)
            recent_date = df_liq['liquidation_date'].max() - timedelta(days=30)
            recent_liquidations = df_liq[df_liq['liquidation_date'] >= recent_date]
            
            risk_metrics['historical_liquidation_analysis'] = {
                'total_liquidations_30d': len(recent_liquidations),
                'total_volume_30d': recent_liquidations['total_collateral_seized'].sum(),
                'avg_liquidation_size': recent_liquidations['total_collateral_seized'].mean(),
                'largest_liquidation_24h': df_liq[df_liq['liquidation_date'] >= df_liq['liquidation_date'].max() - timedelta(days=1)]['total_collateral_seized'].max(),
                'liquidation_by_chain': df_liq.groupby('chain')['total_collateral_seized'].sum().to_dict()
            }
        
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
    progress.start_step(0)
    try:
        # Load reserves
        try:
            DUNE_API_KEY = os.getenv("DUNE_API_KEY_RESERVE")
            if DUNE_API_KEY:
                print(f"   🔑 Using Dune API key for reserves (length: {len(DUNE_API_KEY)})")
                dune = DuneClient(api_key=DUNE_API_KEY)
                response = dune.get_custom_endpoint_result("firstbml", "current-reserve", limit=5000)
                df_reserve = pd.DataFrame(response.result.rows) if hasattr(response, "result") else pd.DataFrame()
                progress.add_detail(f"Loaded {len(df_reserve)} reserve entries")
            else:
                df_reserve = pd.DataFrame()
                progress.add_detail("No reserve API key configured")
        except Exception as e:
            df_reserve = pd.DataFrame()
            progress.add_detail(f"Reserve loading failed: {str(e)}")
        
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