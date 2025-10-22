
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
    """Simple file-based cache for portfolio data"""
    
    def __init__(self, cache_dir: str = "portfolio_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_ttl = timedelta(minutes=10)
    
    def _get_cache_key(self, wallet_address: str, chain: str) -> str:
        key = f"{wallet_address.lower()}_{chain}".encode()
        return hashlib.md5(key).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, wallet_address: str, chain: str) -> Optional[Dict]:
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

class WorkingAssetFetcher:
    """Working version that handles actual JSON structure"""
    
    def __init__(self, max_workers: int = 5):
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
    
    def check_single_reserve(self, w3, reserve_data: Dict, wallet_address: str) -> Optional[Dict]:
        """Check a single reserve for user positions - FIXED debt detection"""
        try:
            atoken_address = reserve_data.get('aTokenAddress')
            variable_debt_address = reserve_data.get('variableDebtTokenAddress')
            stable_debt_address = reserve_data.get('stableDebtTokenAddress')
            
            if not atoken_address and not variable_debt_address and not stable_debt_address:
                return None
            
            checksum_wallet = Web3.to_checksum_address(wallet_address)
            
            # Get aToken balance (collateral)
            atoken_balance = 0
            if atoken_address:
                try:
                    atoken_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(atoken_address),
                        abi=self.token_abi
                    )
                    atoken_balance = atoken_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    logger.debug(f"aToken balance failed for {reserve_data.get('symbol')}: {e}")

            # Check variable debt tokens
            variable_debt = 0
            if variable_debt_address:
                try:
                    variable_debt_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(variable_debt_address),
                        abi=self.token_abi
                    )
                    variable_debt = variable_debt_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    logger.debug(f"Variable debt balance failed for {reserve_data.get('symbol')}: {e}")

            # Check stable debt tokens
            stable_debt = 0
            if stable_debt_address:
                try:
                    stable_debt_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(stable_debt_address),
                        abi=self.token_abi
                    )
                    stable_debt = stable_debt_contract.functions.balanceOf(checksum_wallet).call()
                except Exception as e:
                    logger.debug(f"Stable debt balance failed for {reserve_data.get('symbol')}: {e}")

            # Calculate total debt
            total_debt = variable_debt + stable_debt

            # Skip ONLY if BOTH collateral and debt are zero
            if atoken_balance == 0 and total_debt == 0:
                return None
            
            symbol = reserve_data.get('symbol', 'UNKNOWN')
            decimals = reserve_data.get('decimals', 18)
            
            logger.info(f"✅ Found position: {symbol} - Supply: {atoken_balance}, Debt: {total_debt} (variable: {variable_debt}, stable: {stable_debt})")
            
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
                'stable_borrow_apy': reserve_data.get('stableBorrowRate', 0)
            }
            
        except Exception as e:
            symbol = reserve_data.get('symbol', 'UNKNOWN')
            logger.debug(f"Balance check failed for {symbol}: {e}")
            return None
    
    def fetch_reserve_balances(self, w3, reserve_data_list: List[Dict], wallet_address: str) -> List[Dict]:
        """Fetch balances for reserves in parallel"""
        
        if not reserve_data_list:
            return []
        
        results = []
        total_reserves = len(reserve_data_list)
        
        logger.info(f"🔍 Checking {total_reserves} reserves...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_reserve = {}
            for reserve_data in reserve_data_list:
                future = executor.submit(
                    self.check_single_reserve, 
                    w3, 
                    reserve_data, 
                    wallet_address
                )
                future_to_reserve[future] = reserve_data.get('symbol', 'UNKNOWN')
            
            completed = 0
            for future in as_completed(future_to_reserve):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"📊 Progress: {completed}/{total_reserves}")
                
                try:
                    result = future.result(timeout=10)
                    if result:
                        results.append(result)
                except Exception as e:
                    symbol = future_to_reserve[future]
                    logger.debug(f"Failed to check {symbol}: {e}")
        
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
            'ethereum': 'https://eth.llamarpc.com',
            'polygon': 'https://polygon-rpc.com',
            'avalanche': 'https://api.avax.network/ext/bc/C/rpc',
            'arbitrum': 'https://arb1.arbitrum.io/rpc',
            'optimism': 'https://mainnet.optimism.io',
        }
    
    def _get_aave_v3_addresses(self) -> Dict:
        return {
            'ethereum': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            'polygon': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'avalanche': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'arbitrum': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
            'optimism': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
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
        """WORKING: Get portfolio with proper error handling - FIXED active_chains"""
        start_time = time.time()
        try:
            if not chains:
                chains = ['ethereum']  # Default to ethereum only
            
            portfolio_data = {}
            total_collateral = 0.0
            total_debt = 0.0
            total_available_borrows = 0.0
            total_net_worth = 0.0
            lowest_hf = None
            chains_with_positions = []
            
            for chain in chains:
                try:
                    chain_data = self._get_chain_portfolio(wallet_address, chain)
                    portfolio_data[chain] = chain_data
                    
                    # CRITICAL FIX: Check if chain has positions properly
                    if chain_data['has_positions']:
                        chains_with_positions.append(chain)
                        account_data = chain_data['account_data']
                        
                        total_collateral += account_data['total_collateral_usd']
                        total_debt += account_data['total_debt_usd']
                        total_available_borrows += account_data['available_borrows_usd']
                        total_net_worth += account_data['net_worth_usd']
                        
                        chain_hf = account_data['health_factor']
                        # FIXED: Properly handle None and inf health factors
                        if chain_hf is not None and chain_hf != float('inf'):
                            if lowest_hf is None or chain_hf < lowest_hf:
                                lowest_hf = chain_hf
                except Exception as chain_error:
                    logger.error(f"Error processing chain {chain}: {chain_error}")
                    # Add empty chain data on error
                    portfolio_data[chain] = self._get_empty_chain_data(chain)
            
            # Calculate metrics - FIXED to handle empty portfolios
            cross_chain_metrics = self._calculate_cross_chain_metrics(
                total_collateral, total_debt, total_available_borrows, total_net_worth, lowest_hf
            )
            
            risk_assessment = self._calculate_risk_assessment(
                portfolio_data, cross_chain_metrics
            )
            
            response = {
                'wallet_address': wallet_address.lower(),
                'fetch_timestamp': datetime.now(timezone.utc).isoformat(),
                'active_chains': chains_with_positions,  # FIXED: Now properly populated
                'total_metrics': cross_chain_metrics,
                'risk_assessment': risk_assessment,
                'chain_details': portfolio_data
            }
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Portfolio fetched in {elapsed:.2f}s")
            
            # FIXED: Ensure response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            logger.error(f"❌ Portfolio fetch failed: {e}", exc_info=True)
            return self._get_error_response(wallet_address, str(e))
    
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
        """Get portfolio for a specific chain - FIXED to handle empty asset breakdown"""
        try:
            w3 = Web3(Web3.HTTPProvider(self.rpc_endpoints.get(chain)))
            if not w3.is_connected():
                logger.warning(f"❌ RPC not connected for {chain}")
                return self._get_empty_chain_data(chain)
            
            account_data = self.get_comprehensive_account_data(w3, chain, wallet_address)
            
            # CRITICAL FIX: Check if account_data shows positions
            if not account_data:
                return self._get_empty_chain_data(chain)
            
            has_collateral = account_data.get('total_collateral_usd', 0) > 0
            has_debt = account_data.get('total_debt_usd', 0) > 0
            
            # If no collateral and no debt, return empty
            if not has_collateral and not has_debt:
                return self._get_empty_chain_data(chain)
            
            # Try to get asset breakdown
            assets_breakdown = self._get_assets_breakdown(w3, chain, wallet_address)
            
            # CRITICAL FIX: If asset breakdown is empty but account data shows positions,
            # return account data with a note about unavailable breakdown
            if not assets_breakdown and (has_collateral or has_debt):
                logger.warning(f"⚠️ Account data shows positions but asset breakdown unavailable for {chain}")
                
                return {
                    'has_positions': True,  # FIXED: Should be True since account_data shows positions
                    'account_data': account_data,
                    'collateral_assets': [{
                        'symbol': 'Multiple Assets',
                        'address': '0x0000000000000000000000000000000000000000',
                        'balance': 0,
                        'value_usd': account_data['total_collateral_usd'],
                        'supply_apy': 0,
                        'asset_type': 'collateral',
                        'note': 'asset breakdown avail soon'
                    }] if has_collateral else [],
                    'debt_assets': [{
                        'symbol': 'Multiple Assets',
                        'address': '0x0000000000000000000000000000000000000000',
                        'balance': 0,
                        'value_usd': account_data['total_debt_usd'],
                        'borrow_apy': 0,
                        'asset_type': 'debt',
                        'note': 'asset breakdown avail soon'
                    }] if has_debt else [],
                    'summary': {
                        'total_collateral_assets': 1 if has_collateral else 0,
                        'total_debt_assets': 1 if has_debt else 0,
                        'note': 'Detailed asset breakdown unavailable - showing aggregate values only'
                    }
                }
            
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
        """Get asset breakdown with actual JSON data - FIXED for debt"""
        try:
            # Check cache
            cached_data = self.cache.get(wallet_address, chain)
            if cached_data:
                return cached_data
            
            chain_assets = self.assets_data.get('all_assets', {}).get(chain, [])
            
            if not chain_assets:
                logger.warning(f"⚠️ No assets found for chain {chain}")
                return []
            
            logger.info(f"🔍 Checking {len(chain_assets)} assets on {chain}")
            
            # Get balances
            balance_results = self.asset_fetcher.fetch_reserve_balances(w3, chain_assets, wallet_address)
            
            if not balance_results:
                logger.info(f"✅ No active positions found (checked {len(chain_assets)} assets)")
                return []
            
            # Get prices
            symbols = [result['symbol'] for result in balance_results]
            prices = self.price_fetcher.get_prices(symbols)
            
            # Process results
            assets_breakdown = []
            for result in balance_results:
                symbol = result['symbol']
                decimals = result['decimals']
                decimals_factor = 10 ** decimals
                
                collateral_balance = float(result['atoken_balance']) / decimals_factor
                variable_debt_balance = float(result['variable_debt']) / decimals_factor
                stable_debt_balance = float(result.get('stable_debt', 0)) / decimals_factor  # ADD THIS
                total_debt_balance = variable_debt_balance + stable_debt_balance  # ADD THIS
                
                price = float(prices.get(symbol, 0))
                collateral_usd = collateral_balance * price
                debt_usd = total_debt_balance * price  # Use total debt
                
                # Convert rates to APY
                supply_apy = (float(result['supply_apy']) / 1e27) * 100
                borrow_apy = (float(result['borrow_apy']) / 1e27) * 100
                
                asset_data = {
                    'symbol': symbol,
                    'address': result['address'],
                    'collateral_balance': collateral_balance,
                    'debt_balance': total_debt_balance,  # Use total debt
                    'variable_debt_balance': variable_debt_balance,  # Keep separate for info
                    'stable_debt_balance': stable_debt_balance,      # Keep separate for info
                    'collateral_usd': collateral_usd,
                    'debt_usd': debt_usd,
                    'supply_apy': supply_apy,
                    'borrow_apy': borrow_apy,
                    'current_price': price
                }
                
                assets_breakdown.append(asset_data)
                logger.info(f"  💰 {symbol}: ${price:.2f}, Supply: ${collateral_usd:.2f}, Debt: ${debt_usd:.2f} (variable: ${variable_debt_balance * price:.2f}, stable: ${stable_debt_balance * price:.2f})")
            
            # Cache results
            if assets_breakdown:
                self.cache.set(wallet_address, chain, assets_breakdown)
            
            return assets_breakdown
            
        except Exception as e:
            logger.error(f"Asset breakdown error: {e}", exc_info=True)
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