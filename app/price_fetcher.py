# enhanced_risk_early_warning_system.py
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
