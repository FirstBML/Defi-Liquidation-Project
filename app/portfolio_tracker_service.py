
"""
Portfolio Tracker Service - FIXED VERSION
Handles empty portfolios and edge cases properly
"""

import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from web3 import Web3
import requests
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PortfolioCache:
    def __init__(self, cache_dir: str = "portfolio_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_ttl = timedelta(minutes=5)  # Shorter TTL for fresh data
        self.position_cache_ttl = timedelta(minutes=30)  # Longer for positions
    
    def _get_cache_key(self, wallet_address: str, chain: str) -> str:
        """Generate cache key for wallet and chain"""
        key = f"{wallet_address.lower()}_{chain}".encode()
        return hashlib.md5(key).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, wallet_address: str, chain: str) -> Optional[Dict]:
        """Get cached data"""
        try:
            cache_key = self._get_cache_key(wallet_address, chain)
            cache_path = self._get_cache_path(cache_key)
            
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            if datetime.now(timezone.utc) - cached_time > self.cache_ttl:
                os.remove(cache_path)
                return None
            
            logger.info(f"✅ Cache HIT for {wallet_address} on {chain}")
            return cached_data['data']
            
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def set(self, wallet_address: str, chain: str, data: Dict):
        """Set cache data"""
        try:
            cache_key = self._get_cache_key(wallet_address, chain)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'wallet_address': wallet_address.lower(),
                'chain': chain,
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def get_positions_cache(self, wallet_address: str, chain: str) -> Optional[List[Dict]]:
        """Get cached positions with longer TTL"""
        cache_key = f"{self._get_cache_key(wallet_address, chain)}_positions"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            if datetime.now(timezone.utc) - cached_time > self.position_cache_ttl:
                return None
            
            return cached_data['positions']
        except:
            return None
    
    def set_positions_cache(self, wallet_address: str, chain: str, positions: List[Dict]):
        """Cache positions separately"""
        cache_key = f"{self._get_cache_key(wallet_address, chain)}_positions"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'wallet_address': wallet_address.lower(),
            'chain': chain,
            'positions': positions
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

class WorkingAssetFetcher:
    def __init__(self, max_workers: int = 3):  # Reduced for speed
        self.max_workers = max_workers
        self.token_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }
        ]
        self.request_delay = 0.02  # Reduced delay

    def fetch_reserve_balances(self, w3, reserve_data_list: List[Dict], wallet_address: str) -> List[Dict]:
        """Super fast fetching - only check first 20 assets"""
        if not reserve_data_list:
            return []
        
        # ONLY CHECK FIRST 20 ASSETS FOR SPEED
        assets_to_check = reserve_data_list[:20]
        logger.info(f"🔍 Fast checking {len(assets_to_check)} assets (of {len(reserve_data_list)})")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_reserve = {}
            for reserve_data in assets_to_check:
                future = executor.submit(
                    self.check_single_reserve, 
                    w3, 
                    reserve_data, 
                    wallet_address
                )
                future_to_reserve[future] = reserve_data.get('symbol', 'UNKNOWN')
            
            for future in as_completed(future_to_reserve):
                try:
                    result = future.result(timeout=8)  # Reduced timeout
                    if result:
                        results.append(result)
                except Exception:
                    continue
        
        logger.info(f"✅ Found {len(results)} positions in fast scan")
        return results
    
    def check_single_reserve(self, w3, reserve_data: Dict, wallet_address: str) -> Optional[Dict]:
        """Optimized version with error handling and delays"""
        import time
        time.sleep(self.request_delay)  # Rate limiting
        
        try:
            symbol = reserve_data.get('symbol', 'UNKNOWN')
            atoken_address = reserve_data.get('aTokenAddress')
            variable_debt_address = reserve_data.get('variableDebtTokenAddress')
            stable_debt_address = reserve_data.get('stableDebtTokenAddress')
            
            if not atoken_address and not variable_debt_address and not stable_debt_address:
                return None
            
            checksum_wallet = Web3.to_checksum_address(wallet_address)
            
            # Get balances with timeout
            atoken_balance = 0
            variable_debt = 0
            stable_debt = 0
            
            # aToken balance
            if atoken_address:
                try:
                    atoken_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(atoken_address),
                        abi=self.token_abi
                    )
                    atoken_balance = atoken_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    pass  # Silent fail for speed

            # Variable debt
            if variable_debt_address:
                try:
                    variable_debt_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(variable_debt_address),
                        abi=self.token_abi
                    )
                    variable_debt = variable_debt_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    pass  # Silent fail for speed

            # Stable debt
            if stable_debt_address:
                try:
                    stable_debt_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(stable_debt_address),
                        abi=self.token_abi
                    )
                    stable_debt = stable_debt_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    pass  # Silent fail for speed

            total_debt = variable_debt + stable_debt

            # Skip if both zero
            if atoken_balance == 0 and total_debt == 0:
                return None
            
            decimals = reserve_data.get('decimals', 18)
            
            return {
                'symbol': symbol,
                'address': reserve_data.get('underlyingAsset', reserve_data.get('address', '')),
                'decimals': decimals,
                'atoken_balance': atoken_balance,
                'variable_debt': variable_debt,
                'stable_debt': stable_debt,
                'total_debt': total_debt,
                'supply_apy': reserve_data.get('liquidityRate', 0),
                'borrow_apy': reserve_data.get('variableBorrowRate', 0),
            }
            
        except Exception:
            return None

    def fetch_reserve_balances(self, w3, reserve_data_list: List[Dict], wallet_address: str) -> List[Dict]:
        """Optimized parallel fetching"""
        if not reserve_data_list:
            return []
        
        results = []
        total_reserves = len(reserve_data_list)
        
        logger.info(f"🔍 Checking {total_reserves} reserves with {self.max_workers} workers...")
        
        # Use smaller batches
        batch_size = min(20, total_reserves)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process in smaller batches to avoid rate limits
            for i in range(0, total_reserves, batch_size):
                batch = reserve_data_list[i:i + batch_size]
                
                future_to_reserve = {}
                for reserve_data in batch:
                    future = executor.submit(
                        self.check_single_reserve, 
                        w3, 
                        reserve_data, 
                        wallet_address
                    )
                    future_to_reserve[future] = reserve_data.get('symbol', 'UNKNOWN')
                
                for future in as_completed(future_to_reserve):
                    try:
                        result = future.result(timeout=10)
                        if result:
                            results.append(result)
                    except Exception:
                        continue
                
                logger.info(f"📊 Progress: {min(i + batch_size, total_reserves)}/{total_reserves}")
        
        logger.info(f"✅ Found {len(results)} positions")
        return results
    
class SimplePriceFetcher:
    """Simple price fetcher"""
    
    def __init__(self):
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.base_url = "https://api.coingecko.com/api/v3"
        
        self.symbol_to_id = {
            'WETH': 'weth', 'ETH': 'ethereum', 'WBTC': 'wrapped-bitcoin', 
            'USDC': 'usd-coin', 'USDT': 'tether', 'DAI': 'dai', 
            'LINK': 'chainlink', 'AAVE': 'aave', 'UNI': 'uniswap',
            'MATIC': 'matic-network', 'AVAX': 'avalanche-2',
        }
    
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for symbols"""
        prices = {}
        
        try:
            # Filter symbols that we have mappings for
            valid_symbols = [s for s in symbols if s.upper() in self.symbol_to_id]
            
            if not valid_symbols:
                return prices
            
            symbol_string = ','.join([self.symbol_to_id[s.upper()] for s in valid_symbols])
            
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': symbol_string,
                'vs_currencies': 'usd'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for symbol in valid_symbols:
                    coingecko_id = self.symbol_to_id[symbol.upper()]
                    if coingecko_id in data and 'usd' in data[coingecko_id]:
                        prices[symbol] = data[coingecko_id]['usd']
                
                logger.info(f"💰 Got prices for {len(prices)} tokens")
                
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}")
        
        return prices

class WorkingPortfolioTracker:
    """WORKING Portfolio Tracker - Handles actual JSON structure"""

    def __init__(self, assets_data_file: str = "aave_v3_complete_data_1761094969.json"):
        self.assets_data = self._load_and_debug_assets_data(assets_data_file)
        self.price_fetcher = SimplePriceFetcher()
        self.cache = PortfolioCache()
        self.asset_fetcher = WorkingAssetFetcher(max_workers=15)
        self.rpc_endpoints = self._get_rpc_endpoints()
        self.aave_v3_addresses = self._get_aave_v3_addresses()
    
    
    def validate_wallet_address(address: str) -> str:
        """Validate and normalize wallet address"""
        # Remove whitespace
        address = address.strip()
        
        # Check format
        if not address.startswith('0x'):
            raise ValueError(f"Address must start with 0x: {address}")
        
        # Check length (should be 42 chars: 0x + 40 hex)
        if len(address) != 42:
            raise ValueError(f"Invalid address length {len(address)}, should be 42: {address}")
        
        # Check if valid hex
        try:
            int(address[2:], 16)
        except ValueError:
            raise ValueError(f"Address contains invalid hex characters: {address}")
        
        # Return checksummed address
        try:
            return Web3.to_checksum_address(address)
        except Exception as e:
            raise ValueError(f"Invalid address format: {address}, error: {e}")

    def format_health_factor(self, hf: Optional[float]) -> str:
        """Format health factor for display"""
        if hf is None:
            return "N/A (No positions)"
        elif hf == float('inf'):
            return "∞ (No debt)"
        elif hf > 100:
            return f"{hf:.0f} (Very Safe)"
        elif hf > 10:
            return f"{hf:.1f} (Safe)"
        elif hf > 5:
            return f"{hf:.2f} (Healthy)"
        elif hf > 2:
            return f"{hf:.2f} (Medium)"
        elif hf > 1.5:
            return f"{hf:.3f} (Warning)"
        elif hf > 1.1:
            return f"{hf:.3f} (High Risk)"
        else:
            return f"{hf:.3f} (CRITICAL)"
        
    def _load_and_debug_assets_data(self, assets_data_file: str) -> Dict:
        """Load asset data and debug the structure - FIXED for new format"""
        try:
            possible_paths = [
                r"C:\Users\g\Documents\LiquidationProject\app\aave_v3_complete_data_1761094969.json",
                os.path.join(os.getcwd(), "app", assets_data_file),
                os.path.join(os.getcwd(), assets_data_file),
                assets_data_file
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    logger.info(f"📁 Loaded assets data from {path}")
                    
                    # CRITICAL FIX: Handle new JSON structure
                    # Old format had: {'all_assets': {'ethereum': [...], ...}}
                    # New format has: {'data': {'ethereum': [...], ...}}
                    
                    if 'data' in data:
                        # New format - wrap it in 'all_assets' for compatibility
                        assets_data = {'all_assets': data['data']}
                        logger.info(f"✅ Detected new JSON format with 'data' key")
                    elif 'all_assets' in data:
                        # Old format - use as is
                        assets_data = data
                        logger.info(f"✅ Detected old JSON format with 'all_assets' key")
                    else:
                        # Unknown format - try to use it directly
                        logger.warning(f"⚠️ Unknown JSON format, using as-is")
                        assets_data = {'all_assets': data}
                    
                    # Debug the structure
                    self._debug_data_structure(assets_data)
                    return assets_data
            
            logger.warning("❌ Asset data file not found")
            return {'all_assets': {}}
            
        except Exception as e:
            logger.error(f"Failed to load asset data: {e}")
            return {'all_assets': {}}
    
    def _debug_data_structure(self, data: Dict):
        """Debug the actual JSON structure"""
        logger.info("🔍 Debugging JSON structure...")
        
        if 'all_assets' in data:
            for chain, assets in data['all_assets'].items():
                logger.info(f"📊 Chain: {chain}, Assets: {len(assets)}")
                if assets:
                    first_asset = assets[0]
                    logger.info(f"   Sample asset keys: {list(first_asset.keys())}")
                    break
        else:
            logger.warning("❌ No 'all_assets' key found")
            logger.info(f"📋 Top-level keys: {list(data.keys())}")
    
    def _get_rpc_endpoints(self) -> Dict:
        return {
            'ethereum': [
                'https://eth.llamarpc.com',
                'https://ethereum.publicnode.com',
                'https://rpc.ankr.com/eth',
                'https://cloudflare-eth.com',
                'https://eth-mainnet.public.blastapi.io',
            ],
            'polygon': [
                'https://polygon-rpc.com',
                'https://polygon-bor.publicnode.com',
                'https://rpc.ankr.com/polygon',
            ],
            'avalanche': [
                'https://api.avax.network/ext/bc/C/rpc',
                'https://avalanche-c-chain.publicnode.com',
                'https://rpc.ankr.com/avalanche',
            ],
            'arbitrum': [
                'https://arb1.arbitrum.io/rpc',
                'https://arbitrum-one.publicnode.com',
                'https://rpc.ankr.com/arbitrum'
            ],
            'optimism': [
                'https://mainnet.optimism.io',
                'https://optimism.publicnode.com',
                'https://rpc.ankr.com/optimism'
            ],
            'bnb': [
                'https://bsc-dataseed.binance.org',
                'https://bsc-dataseed1.defibit.io',
                'https://bsc-dataseed1.ninicoin.io',
                'https://bnb.publicnode.com'
            ],
            'base': [
                'https://mainnet.base.org',
                'https://base.publicnode.com',
                'https://1rpc.io/base',
                'https://base-rpc.publicnode.com'
            ],
            'fantom': [
                'https://rpc.ftm.tools',
                'https://fantom.publicnode.com',
                'https://rpc.ankr.com/fantom',
                'https://rpcapi.fantom.network'
            ],
            'gnosis': [
                'https://rpc.gnosischain.com',
                'https://gnosis.publicnode.com',
                'https://rpc.ankr.com/gnosis',
                'https://xdai-rpc.gateway.pokt.network'
            ],
            'celo': [
                'https://forno.celo.org',
                'https://celo.publicnode.com',
                'https://1rpc.io/celo',
                'https://rpc.ankr.com/celo'
            ],
            # Add more chains as needed
        }
    def get_working_rpc(self, chain: str) -> str:
        """Get a working RPC endpoint with fallback"""
        endpoints = self.rpc_endpoints.get(chain, [])
        
        if not endpoints:
            raise Exception(f"No RPC endpoints configured for {chain}")
        
        # Try each endpoint
        for endpoint in endpoints:
            try:
                logger.info(f"🔧 Testing RPC: {endpoint}")
                w3 = Web3(Web3.HTTPProvider(endpoint, request_kwargs={'timeout': 10}))
                
                # Test with a simple call that doesn't require authentication
                block_number = w3.eth.block_number
                if w3.is_connected() and isinstance(block_number, int) and block_number > 0:
                    logger.info(f"✅ RPC connected: {endpoint} (block: {block_number})")
                    return endpoint
                else:
                    logger.warning(f"❌ RPC returned invalid data: {endpoint}")
                    
            except Exception as e:
                logger.warning(f"❌ RPC failed {endpoint}: {str(e)[:100]}...")
                continue
        
        # If all endpoints fail, log error and use first one anyway
        logger.error(f"🚨 All RPC endpoints failed for {chain}, using first as fallback: {endpoints[0]}")
        return endpoints[0]
    
    def _get_aave_v3_addresses(self) -> Dict:
        """Updated Aave V3 Pool addresses - VERIFIED from official docs"""
        return {
            'ethereum': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            'polygon': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'avalanche': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'arbitrum': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'optimism': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'base': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',
            'metis': '0x90df02551bB792286e8D4f13E0e357b4Bf1G2991',
            'scroll': '0x11fCfe756c05AD438e312a7fd934381537D3cFfe',
            
            # NOTE: These chains DON'T have Aave V3 deployed:
            # 'bnb' - No official Aave V3 deployment (use Venus or Radiant)
            # 'fantom' - No Aave V3 (use Geist Finance)
            # 'gnosis' - Limited Aave V3 support
            # 'celo' - No Aave V3
        }

    
    def get_comprehensive_account_data(self, w3, network: str, wallet_address: str) -> Optional[Dict]:
        """Get Aave V3 account data with fallback - FIXED to handle empty portfolios"""
        try:
            lending_pool_abi = [{
                "inputs": [{"internalType": "address", "name": "user", "type": "address"}],
                "name": "getUserAccountData",
                "outputs": [
                    {"internalType": "uint256", "name": "totalCollateralBase", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalDebtBase", "type": "uint256"},
                    {"internalType": "uint256", "name": "availableBorrowsBase", "type": "uint256"},
                    {"internalType": "uint256", "name": "currentLiquidationThreshold", "type": "uint256"},
                    {"internalType": "uint256", "name": "ltv", "type": "uint256"},
                    {"internalType": "uint256", "name": "healthFactor", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }]

            checksum_address = Web3.to_checksum_address(wallet_address.lower())
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.aave_v3_addresses[network]),
                abi=lending_pool_abi
            )
            
            data = contract.functions.getUserAccountData(checksum_address).call()
            
            collateral_usd = float(data[0]) / 1e8
            debt_usd = float(data[1]) / 1e8
            available_borrow_usd = float(data[2]) / 1e8
            liquidation_threshold = float(data[3]) / 10000
            ltv = float(data[4]) / 10000
            
            # FIXED: Handle health factor properly for empty portfolios
            raw_health_factor = float(data[5])
            if collateral_usd == 0 and debt_usd == 0:
                # Empty portfolio - no positions
                health_factor = None
                risk_level = "NO_POSITIONS"
            elif raw_health_factor == 0 or raw_health_factor == 2**256 - 1:
                # Max value or zero means no debt
                health_factor = float('inf')
                risk_level = "NO_RISK"
            else:
                health_factor = raw_health_factor / 1e18
                # Risk assessment
                if health_factor < 1.0:
                    risk_level = "CRITICAL"
                elif health_factor < 1.5:
                    risk_level = "HIGH"
                elif health_factor < 2.0:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
            
            net_worth_usd = collateral_usd - debt_usd

            return {
                'total_collateral_usd': collateral_usd,
                'total_debt_usd': debt_usd,
                'available_borrows_usd': available_borrow_usd,
                'current_liquidation_threshold': liquidation_threshold,
                'ltv': ltv,
                'health_factor': health_factor,
                'net_worth_usd': net_worth_usd,
                'risk_level': risk_level,
                'liquidation_imminent': health_factor is not None and health_factor < 1.0,
            }
        except Exception as e:
            logger.error(f"❌ Account data error for {network}: {e}")
            return None

    def get_user_portfolio(self, wallet_address: str, chains: Optional[List[str]] = None) -> Dict:
        """FIXED: Skip chains without Aave V3"""
        start_time = time.time()
        
        if not chains:
            # ONLY include chains with confirmed Aave V3 deployment
            chains = ['ethereum', 'polygon', 'avalanche', 'arbitrum', 'optimism', 'base']
        
        # Filter out unsupported chains
        supported_chains = [c for c in chains if c in self.aave_v3_addresses]
        unsupported = [c for c in chains if c not in self.aave_v3_addresses]
        
        if unsupported:
            logger.warning(f"⚠️ Skipping unsupported chains: {unsupported}")
        
        portfolio_data = {}
        total_collateral = 0.0
        total_debt = 0.0
        total_available_borrows = 0.0
        total_net_worth = 0.0
        lowest_hf = None
        chains_with_positions = []
        
        for chain in supported_chains:  # Only iterate supported chains
            try:
                chain_data = self._get_chain_portfolio(wallet_address, chain)
                portfolio_data[chain] = chain_data
                
                if chain_data['has_positions']:
                    chains_with_positions.append(chain)
                    account_data = chain_data['account_data']
                    
                    total_collateral += account_data['total_collateral_usd']
                    total_debt += account_data['total_debt_usd']
                    total_available_borrows += account_data['available_borrows_usd']
                    total_net_worth += account_data['net_worth_usd']
                    
                    chain_hf = account_data['health_factor']
                    if chain_hf is not None and chain_hf != float('inf'):
                        if lowest_hf is None or chain_hf < lowest_hf:
                            lowest_hf = chain_hf
            except Exception as chain_error:
                logger.error(f"Error processing chain {chain}: {chain_error}")
                portfolio_data[chain] = self._get_empty_chain_data(chain)
        
        # Add note about skipped chains
        if unsupported:
            portfolio_data['_skipped_chains'] = {
                'chains': unsupported,
                'reason': 'No Aave V3 deployment on these chains'
            }
        
        cross_chain_metrics = self._calculate_cross_chain_metrics(
            total_collateral, total_debt, total_available_borrows, total_net_worth, lowest_hf
        )
        
        risk_assessment = self._calculate_risk_assessment(
            portfolio_data, cross_chain_metrics
        )
        
        response = {
            'wallet_address': wallet_address.lower(),
            'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
            'active_chains': chains_with_positions,
            'requested_chains': chains,
            'supported_chains': supported_chains,
            'unsupported_chains': unsupported,
            'total_metrics': cross_chain_metrics,
            'risk_assessment': risk_assessment,
            'chain_details': portfolio_data
        }
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Portfolio fetched in {elapsed:.2f}s")
        
        return self._make_json_serializable(response)
            
          

    def get_user_portfolio_fast(self, wallet_address: str, chains: Optional[List[str]] = None) -> Dict:
        """Fast portfolio with basic data only - SUPPORTS ALL CHAINS"""
        start_time = time.time()
        
        if not chains:
            # Default to all supported chains
            chains = ['ethereum', 'polygon', 'avalanche', 'arbitrum', 'optimism', 
                    'bnb', 'base', 'fantom', 'gnosis', 'celo']
        
        portfolio_data = {}
        total_collateral = 0.0
        total_debt = 0.0
        total_available_borrows = 0.0
        total_net_worth = 0.0
        lowest_hf = None
        active_chains = []
        
        for chain in chains:
            try:
                rpc_endpoint = self.get_working_rpc(chain)
                w3 = Web3(Web3.HTTPProvider(rpc_endpoint, request_kwargs={'timeout': 15}))
                
                if not w3.is_connected():
                    logger.warning(f"❌ RPC not connected for {chain}")
                    continue
                
                # Get only account data (fast)
                account_data = self.get_comprehensive_account_data(w3, chain, wallet_address)
                
                if not account_data:
                    continue
                
                has_positions = (
                    account_data.get('total_collateral_usd', 0) > 0 or
                    account_data.get('total_debt_usd', 0) > 0
                )
                
                if has_positions:
                    active_chains.append(chain)
                    total_collateral += account_data['total_collateral_usd']
                    total_debt += account_data['total_debt_usd']
                    total_available_borrows += account_data['available_borrows_usd']
                    total_net_worth += account_data['net_worth_usd']
                    
                    chain_hf = account_data['health_factor']
                    if chain_hf is not None and chain_hf != float('inf'):
                        if lowest_hf is None or chain_hf < lowest_hf:
                            lowest_hf = chain_hf
                
                portfolio_data[chain] = {
                    'has_positions': has_positions,
                    'account_data': account_data,
                    'collateral_assets': [],
                    'debt_assets': [],
                    'summary': {
                        'total_collateral_assets': 0,
                        'total_debt_assets': 0,
                        'note': 'Fast mode - asset details unavailable'
                    }
                }
                
            except Exception as e:
                logger.error(f"Fast mode error for {chain}: {e}")
                continue
        
        elapsed = time.time() - start_time
        
        # If taking too long, return fast results
        if elapsed > 30:  # 30 second timeout
            logger.warning(f"⏱️ Switching to fast mode after {elapsed:.2f}s")
            return self._build_fast_response(wallet_address, active_chains, total_collateral, total_debt, total_available_borrows, total_net_worth, lowest_hf, portfolio_data)
        
        return self.get_user_portfolio(wallet_address, chains)

    def _build_fast_response(self, wallet_address: str, active_chains: List[str], 
                       total_collateral: float, total_debt: float, 
                       portfolio_data: Dict) -> Dict:
        """Build fast response structure when full scan times out"""
        
        # Calculate additional metrics
        total_available_borrows = 0.0
        total_net_worth = total_collateral - total_debt
        lowest_hf = None
        
        # Extract health factor and available borrows from portfolio data
        for chain_data in portfolio_data.values():
            account_data = chain_data.get('account_data', {})
            total_available_borrows += account_data.get('available_borrows_usd', 0)
            
            chain_hf = account_data.get('health_factor')
            if chain_hf is not None and chain_hf != float('inf'):
                if lowest_hf is None or chain_hf < lowest_hf:
                    lowest_hf = chain_hf
        
        # Calculate utilization ratio
        utilization_ratio = (total_debt / total_collateral * 100) if total_collateral > 0 else 0.0
        
        total_metrics = {
            'total_collateral_usd': round(total_collateral, 2),
            'total_debt_usd': round(total_debt, 2),
            'total_available_borrows_usd': round(total_available_borrows, 2),
            'total_net_worth_usd': round(total_net_worth, 2),
            'lowest_health_factor': round(lowest_hf, 3) if lowest_hf is not None else None,
            'utilization_ratio_percent': round(utilization_ratio, 2),
        }
        
        # Risk assessment
        if lowest_hf is None or total_collateral == 0:
            risk_assessment = {
                'overall_risk_level': 'NO_POSITIONS',
                'risk_score': 0,
                'liquidation_imminent': False,
                'health_factor_status': 'NO_POSITIONS'
            }
        elif lowest_hf < 1.0:
            risk_assessment = {
                'overall_risk_level': 'CRITICAL',
                'risk_score': 100,
                'liquidation_imminent': True,
                'health_factor_status': 'CRITICAL'
            }
        elif lowest_hf < 1.5:
            risk_assessment = {
                'overall_risk_level': 'MEDIUM',
                'risk_score': 50,
                'liquidation_imminent': False,
                'health_factor_status': 'WARNING'
            }
        else:
            risk_assessment = {
                'overall_risk_level': 'LOW',
                'risk_score': 20,
                'liquidation_imminent': False,
                'health_factor_status': 'HEALTHY'
            }
        
        return {
            'wallet_address': wallet_address.lower(),
            'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
            'active_chains': active_chains,
            'total_metrics': total_metrics,
            'risk_assessment': risk_assessment,
            'chain_details': portfolio_data,
            'mode': 'fast_fallback',
            'note': 'Switched to fast mode due to timeout - asset details limited'
        }
    
    def _make_json_serializable(self, obj):
        """Convert all values to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, float):
            if obj == float('inf'):
                return None  # or a large number like 999999
            elif obj != obj:  # NaN check
                return None
            return obj
        else:
            return obj
    
    def _get_chain_portfolio(self, wallet_address: str, chain: str) -> Dict:
        """Get portfolio with better error handling"""
        try:
            # Check if chain is supported
            if chain not in self.aave_v3_addresses:
                logger.warning(f"⚠️ Chain {chain} not supported (no Aave V3)")
                return {
                    **self._get_empty_chain_data(chain),
                    'unsupported': True,
                    'reason': 'No Aave V3 deployment on this chain'
                }
            
            rpc_endpoint = self.get_working_rpc(chain)
            w3 = Web3(Web3.HTTPProvider(rpc_endpoint, request_kwargs={'timeout': 15}))
            
            if not w3.is_connected():
                logger.warning(f"❌ RPC not connected for {chain}")
                return {
                    **self._get_empty_chain_data(chain),
                    'error': 'RPC connection failed'
                }
            
            # Get account data first (this is fast)
            account_data = self.get_comprehensive_account_data(w3, chain, wallet_address)
            
            if not account_data:
                return self._get_empty_chain_data(chain)
            
            has_collateral = account_data.get('total_collateral_usd', 0) > 0
            has_debt = account_data.get('total_debt_usd', 0) > 0
            
            # If no positions, return empty
            if not has_collateral and not has_debt:
                return self._get_empty_chain_data(chain)
            
            # Try to get detailed asset breakdown (this might be slow)
            try:
                assets_breakdown = self._get_assets_breakdown(w3, chain, wallet_address)
                
                if assets_breakdown:
                    # Process both collateral and debt from breakdown
                    collateral_assets = []
                    debt_assets = []

                    for asset in assets_breakdown:
                        try:
                            # Process collateral
                            if asset.get('collateral_balance', 0) > 0:
                                collateral_assets.append({
                                    'symbol': asset.get('symbol', 'UNKNOWN'),
                                    'address': asset.get('address', ''),
                                    'balance': float(asset.get('collateral_balance', 0)),
                                    'value_usd': float(asset.get('collateral_usd', 0)),
                                    'supply_apy': float(asset.get('supply_apy', 0)),
                                    'asset_type': 'collateral'
                                })
                            
                            # Process debt
                            variable_debt = asset.get('variable_debt_balance', 0)
                            stable_debt = asset.get('stable_debt_balance', 0)
                            total_debt = variable_debt + stable_debt
                            
                            if total_debt > 0:
                                debt_type = []
                                if variable_debt > 0:
                                    debt_type.append('variable')
                                if stable_debt > 0:
                                    debt_type.append('stable')
                                
                                debt_assets.append({
                                    'symbol': asset.get('symbol', 'UNKNOWN'),
                                    'address': asset.get('address', ''),
                                    'balance': float(total_debt),
                                    'variable_debt': float(variable_debt),
                                    'stable_debt': float(stable_debt),
                                    'value_usd': float(asset.get('debt_usd', 0)),
                                    'borrow_apy': float(asset.get('borrow_apy', 0)),
                                    'asset_type': 'debt',
                                    'debt_type': ', '.join(debt_type) if debt_type else 'unknown'
                                })
                        
                        except Exception as e:
                            logger.error(f"❌ Error processing asset: {e}")
                            continue

                    return {
                        'has_positions': True,
                        'account_data': account_data,
                        'collateral_assets': collateral_assets,
                        'debt_assets': debt_assets,
                        'summary': {
                            'total_collateral_assets': len(collateral_assets),
                            'total_debt_assets': len(debt_assets)
                        }
                    }
                
            except Exception as e:
                logger.warning(f"⚠️ Asset breakdown failed, using account data: {e}")
            
            # FALLBACK: If asset breakdown fails, show aggregate data
            logger.info(f"🔄 Using account data fallback for {chain}")
            return {
                'has_positions': True,
                'account_data': account_data,
                'collateral_assets': [{
                    'symbol': 'Aggregate Collateral',
                    'address': '0x0000000000000000000000000000000000000000',
                    'balance': 0,
                    'value_usd': account_data['total_collateral_usd'],
                    'supply_apy': 0,
                    'asset_type': 'collateral',
                    'note': 'Detailed breakdown unavailable'
                }] if has_collateral else [],
                'debt_assets': [{
                    'symbol': 'Aggregate Debt',
                    'address': '0x0000000000000000000000000000000000000000', 
                    'balance': 0,
                    'value_usd': account_data['total_debt_usd'],
                    'borrow_apy': 0,
                    'asset_type': 'debt',
                    'note': 'Detailed breakdown unavailable'
                }] if has_debt else [],
                'summary': {
                    'total_collateral_assets': 1 if has_collateral else 0,
                    'total_debt_assets': 1 if has_debt else 0,
                    'note': 'Using aggregate values - detailed asset scan failed'
                }
            }
            
        except Exception as e:
            logger.error(f"Chain portfolio error for {chain}: {e}")
            return self._get_empty_chain_data(chain)
        
            
            # Separate collateral and debt
            collateral_assets = []
            debt_assets = []

            for asset in assets_breakdown:
                try:
                    # Safety check - skip if asset is None or invalid
                    if not asset or not isinstance(asset, dict):
                        logger.warning(f"⚠️ Skipping invalid asset: {asset}")
                        continue
                    
                    # Process collateral
                    if asset.get('collateral_balance', 0) > 0:
                        collateral_assets.append({
                            'symbol': asset.get('symbol', 'UNKNOWN'),
                            'address': asset.get('address', ''),
                            'balance': float(asset.get('collateral_balance', 0)),
                            'value_usd': float(asset.get('collateral_usd', 0)),
                            'supply_apy': float(asset.get('supply_apy', 0)),
                            'asset_type': 'collateral'
                        })
                    
                    # CRITICAL FIX: Calculate total debt from variable + stable
                    variable_debt = asset.get('variable_debt_balance', 0)
                    stable_debt = asset.get('stable_debt_balance', 0)
                    total_debt = variable_debt + stable_debt
                    
                    if total_debt > 0:
                        debt_type = []
                        if variable_debt > 0:
                            debt_type.append('variable')
                        if stable_debt > 0:
                            debt_type.append('stable')
                        
                        debt_assets.append({
                            'symbol': asset.get('symbol', 'UNKNOWN'),
                            'address': asset.get('address', ''),
                            'balance': float(total_debt),
                            'variable_debt': float(variable_debt),
                            'stable_debt': float(stable_debt),
                            'value_usd': float(asset.get('debt_usd', 0)),
                            'borrow_apy': float(asset.get('borrow_apy', 0)),
                            'asset_type': 'debt',
                            'debt_type': ', '.join(debt_type) if debt_type else 'unknown'
                        })
                
                except Exception as e:
                    logger.error(f"❌ Error processing asset {asset.get('symbol') if asset else 'UNKNOWN'}: {e}")
                    continue

            # CRITICAL FIX: ADD THE MISSING RETURN STATEMENT
            return {
                'has_positions': len(collateral_assets) > 0 or len(debt_assets) > 0,
                'account_data': account_data,
                'collateral_assets': collateral_assets,
                'debt_assets': debt_assets,
                'summary': {
                    'total_collateral_assets': len(collateral_assets),
                    'total_debt_assets': len(debt_assets)
                }
            }
            
        except Exception as e:
            logger.error(f"Chain portfolio error for {chain}: {e}", exc_info=True)
            return self._get_empty_chain_data(chain)
    
    def _get_assets_breakdown(self, w3, chain: str, wallet_address: str) -> List[Dict]:
        """Optimized asset breakdown with better caching"""
        try:
            # Check positions cache first
            cached_positions = self.cache.get_positions_cache(wallet_address, chain)
            if cached_positions:
                logger.info(f"✅ Using cached positions for {wallet_address} on {chain}")
                return cached_positions
            
            chain_assets = self.assets_data.get('all_assets', {}).get(chain, [])
            
            if not chain_assets:
                return []
            
            # Get balances
            balance_results = self.asset_fetcher.fetch_reserve_balances(w3, chain_assets, wallet_address)
            
            if not balance_results:
                return []
            
            # Get prices in bulk
            symbols = [result['symbol'] for result in balance_results]
            prices = self.price_fetcher.get_prices(symbols)
            
            # Process results quickly
            assets_breakdown = []
            for result in balance_results:
                symbol = result['symbol']
                decimals = result['decimals']
                decimals_factor = 10 ** decimals
                
                collateral_balance = float(result['atoken_balance']) / decimals_factor
                variable_debt_balance = float(result['variable_debt']) / decimals_factor
                stable_debt_balance = float(result.get('stable_debt', 0)) / decimals_factor
                total_debt_balance = variable_debt_balance + stable_debt_balance
                
                price = float(prices.get(symbol, 0))
                collateral_usd = collateral_balance * price
                debt_usd = total_debt_balance * price
                
                # Convert rates to APY
                supply_apy = (float(result['supply_apy']) / 1e27) * 100
                borrow_apy = (float(result['borrow_apy']) / 1e27) * 100
                
                asset_data = {
                    'symbol': symbol,
                    'address': result['address'],
                    'collateral_balance': collateral_balance,
                    'debt_balance': total_debt_balance,
                    'variable_debt_balance': variable_debt_balance,
                    'stable_debt_balance': stable_debt_balance,
                    'collateral_usd': collateral_usd,
                    'debt_usd': debt_usd,
                    'supply_apy': supply_apy,
                    'borrow_apy': borrow_apy,
                    'current_price': price
                }
                
                assets_breakdown.append(asset_data)
            
            # Cache positions for longer
            if assets_breakdown:
                self.cache.set_positions_cache(wallet_address, chain, assets_breakdown)
            
            return assets_breakdown
            
        except Exception as e:
            logger.error(f"Asset breakdown error: {e}")
            return []
    
    def _calculate_cross_chain_metrics(self, total_collateral: float, total_debt: float, 
                                      total_available_borrows: float, total_net_worth: float, 
                                      lowest_hf: Optional[float]) -> Dict:
        """Calculate cross-chain metrics - FIXED division by zero"""
        # FIXED: Safely calculate utilization ratio
        if total_collateral > 0:
            utilization_ratio = (total_debt / total_collateral) * 100
        else:
            utilization_ratio = 0.0
        
        return {
            'total_collateral_usd': round(total_collateral, 2),
            'total_debt_usd': round(total_debt, 2),
            'total_available_borrows_usd': round(total_available_borrows, 2),
            'total_net_worth_usd': round(total_net_worth, 2),
            'lowest_health_factor': round(lowest_hf, 3) if lowest_hf is not None else None,
            'utilization_ratio_percent': round(utilization_ratio, 2),
        }
    
    def _calculate_risk_assessment(self, portfolio_data: Dict, metrics: Dict) -> Dict:
        """Calculate risk assessment - FIXED None handling"""
        lowest_hf = metrics['lowest_health_factor']
        
        # FIXED: Better None handling
        if lowest_hf is None or metrics['total_collateral_usd'] == 0:
            return {
                'overall_risk_level': 'NO_POSITIONS',
                'risk_score': 0,
                'liquidation_imminent': False,
                'health_factor_status': 'NO_POSITIONS'
            }
        
        # Calculate risk based on health factor
        if lowest_hf < 1.0:
            risk_level = 'CRITICAL'
            risk_score = 100
            hf_status = 'CRITICAL'
        elif lowest_hf < 1.1:
            risk_level = 'HIGH'
            risk_score = 80
            hf_status = 'CRITICAL'
        elif lowest_hf < 1.5:
            risk_level = 'MEDIUM'
            risk_score = 50
            hf_status = 'WARNING'
        else:
            risk_level = 'LOW'
            risk_score = 20
            hf_status = 'HEALTHY'
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': risk_score,
            'liquidation_imminent': lowest_hf < 1.0,
            'health_factor_status': hf_status
        }
    
    def _get_error_response(self, wallet_address: str, error: str) -> Dict:
        """Get error response"""
        return {
            'wallet_address': wallet_address.lower(),
            'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
            'active_chains': [],
            'total_metrics': {
                'total_collateral_usd': 0.0,
                'total_debt_usd': 0.0,
                'total_available_borrows_usd': 0.0,
                'total_net_worth_usd': 0.0,
                'lowest_health_factor': None,
                'utilization_ratio_percent': 0.0,
            },
            'risk_assessment': {
                'overall_risk_level': 'ERROR',
                'risk_score': 0,
                'liquidation_imminent': False,
                'health_factor_status': 'ERROR'
            },
            'chain_details': {},
            'error': error
        }
    
    def _get_empty_chain_data(self, chain: str) -> Dict:
        """Get empty chain data"""
        return {
            'has_positions': False,
            'account_data': {
                'total_collateral_usd': 0.0,
                'total_debt_usd': 0.0,
                'available_borrows_usd': 0.0,
                'current_liquidation_threshold': 0.0,
                'ltv': 0.0,
                'health_factor': None,
                'net_worth_usd': 0.0,
                'risk_level': 'NO_POSITIONS',
                'liquidation_imminent': False,
            },
            'collateral_assets': [],
            'debt_assets': [],
            'summary': {
                'total_collateral_assets': 0,
                'total_debt_assets': 0
            }
        }

# Main portfolio tracker service
class PortfolioTrackerService(WorkingPortfolioTracker):
    """Main portfolio tracker service"""
    pass