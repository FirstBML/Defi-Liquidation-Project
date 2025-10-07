# rpc_reserve_fetcher.py

"""
RPC-based Aave V3 Reserve Data Fetcher
Now supports 8 chains: Ethereum, Polygon, Arbitrum, Optimism, Avalanche, BNB, Base, Fantom
"""
from web3 import Web3
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class AaveRPCReserveFetcher:
    """Fetch Aave V3 reserve data directly from blockchain RPCs"""
    
    # Primary RPC endpoints (with fallbacks for new chains)
    RPC_ENDPOINTS = {
        'ethereum': 'https://eth.llamarpc.com',
        'polygon': 'https://polygon-rpc.com',
        'arbitrum': 'https://arb1.arbitrum.io/rpc',
        'optimism': 'https://mainnet.optimism.io',
        'avalanche': 'https://api.avax.network/ext/bc/C/rpc',
        'bnb': 'https://bsc-dataseed.binance.org',
        'base': 'https://mainnet.base.org',
        'fantom': 'https://fantom-mainnet.public.blastapi.io'
    }
    
    # Fallback RPCs for reliability
    FALLBACK_RPCS = {
        'bnb': ['https://bsc-dataseed1.defibit.io', 'https://bsc-dataseed1.ninicoin.io'],
        'base': ['https://base-rpc.publicnode.com', 'https://rpc.ankr.com/base'],
        'fantom': ['https://fantom.publicnode.com', 'https://rpc.fantom.network']
    }
    
    POOL_ADDRESSES = {
        'ethereum': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
        'polygon': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'arbitrum': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'optimism': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'avalanche': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'bnb': '0x6807dc923806fE8Fd134338EABCA509979a7e0cB',
        'base': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',
        'fantom': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'
    }
    
    POOL_ABI = [
        {
            "inputs": [],
            "name": "getReservesList",
            "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
            "name": "getReserveData",
            "outputs": [
                {"internalType": "uint256", "name": "configuration", "type": "uint256"},
                {"internalType": "uint128", "name": "liquidityIndex", "type": "uint128"},
                {"internalType": "uint128", "name": "variableBorrowIndex", "type": "uint128"},
                {"internalType": "uint128", "name": "currentLiquidityRate", "type": "uint128"},
                {"internalType": "uint128", "name": "currentVariableBorrowRate", "type": "uint128"},
                {"internalType": "uint128", "name": "currentStableBorrowRate", "type": "uint128"},
                {"internalType": "uint40", "name": "lastUpdateTimestamp", "type": "uint40"},
                {"internalType": "uint16", "name": "id", "type": "uint16"},
                {"internalType": "address", "name": "aTokenAddress", "type": "address"},
                {"internalType": "address", "name": "stableDebtTokenAddress", "type": "address"},
                {"internalType": "address", "name": "variableDebtTokenAddress", "type": "address"},
                {"internalType": "address", "name": "interestRateStrategyAddress", "type": "address"},
                {"internalType": "uint128", "name": "accruedToTreasury", "type": "uint128"},
                {"internalType": "uint128", "name": "unbacked", "type": "uint128"},
                {"internalType": "uint128", "name": "isolationModeTotalDebt", "type": "uint128"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    ERC20_ABI = [
        {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
        {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
        {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"}
    ]
    
    def __init__(self, price_fetcher=None):
        """Initialize with optional price fetcher for USD values"""
        self.price_fetcher = price_fetcher
        self.connection_cache = {}
    
    def _get_web3_connection(self, chain: str) -> Optional[Web3]:
        """Get or create Web3 connection with fallback support"""
        if chain in self.connection_cache:
            w3 = self.connection_cache[chain]
            if w3.is_connected():
                return w3
        
        # Try primary endpoint
        endpoints = [self.RPC_ENDPOINTS[chain]]
        
        # Add fallbacks if available
        if chain in self.FALLBACK_RPCS:
            endpoints.extend(self.FALLBACK_RPCS[chain])
        
        for endpoint in endpoints:
            try:
                w3 = Web3(Web3.HTTPProvider(endpoint, request_kwargs={'timeout': 60}))
                if w3.is_connected():
                    block_number = w3.eth.block_number
                    self.connection_cache[chain] = w3
                    logger.info(f"Connected to {chain} at block {block_number}")
                    return w3
            except Exception as e:
                logger.warning(f"Failed {endpoint}: {str(e)[:80]}")
                continue
        
        logger.error(f"All RPC endpoints failed for {chain}")
        return None
    
    def _get_token_info(self, w3: Web3, address: str) -> Dict[str, any]:
        """Get token symbol, decimals, and name (handles bytes32 tokens)"""
        try:
            token_contract = w3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=self.ERC20_ABI
            )
            
            try:
                symbol = token_contract.functions.symbol().call()
            except:
                # Handle bytes32 tokens like MKR
                symbol_bytes = w3.eth.call({'to': address, 'data': '0x95d89b41'})
                symbol = symbol_bytes.decode('utf-8').rstrip('\x00') if symbol_bytes else 'UNKNOWN'
            
            return {
                'symbol': symbol if symbol else 'UNKNOWN',
                'decimals': token_contract.functions.decimals().call(),
                'name': token_contract.functions.name().call()
            }
        except Exception as e:
            logger.warning(f"Failed to get token info for {address}: {str(e)[:100]}")
            return {'symbol': 'UNKNOWN', 'decimals': 18, 'name': 'Unknown'}
    
    def _decode_configuration(self, configuration: int) -> Dict[str, any]:
        """Decode Aave V3 reserve configuration bitmask"""
        return {
            'ltv': ((configuration & ((1 << 16) - 1)) / 10000),
            'liquidation_threshold': (((configuration >> 16) & ((1 << 16) - 1)) / 10000),
            'liquidation_bonus': (((configuration >> 32) & ((1 << 16) - 1)) / 10000),
            'is_active': ((configuration >> 56) & 1) == 1,
            'is_frozen': ((configuration >> 57) & 1) == 1,
            'borrowing_enabled': ((configuration >> 58) & 1) == 1,
            'stable_borrowing_enabled': ((configuration >> 59) & 1) == 1
        }
    
    def fetch_chain_reserves(self, chain: str) -> pd.DataFrame:
        """Fetch all reserves for a specific chain"""
        if chain not in self.RPC_ENDPOINTS:
            logger.error(f"Chain {chain} not supported. Supported: {list(self.RPC_ENDPOINTS.keys())}")
            return pd.DataFrame()
        
        w3 = self._get_web3_connection(chain)
        if not w3:
            return pd.DataFrame()
        
        try:
            pool_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.POOL_ADDRESSES[chain]),
                abi=self.POOL_ABI
            )
            
            reserves_list = pool_contract.functions.getReservesList().call()
            logger.info(f"{chain}: Found {len(reserves_list)} reserves")
            
            reserves_data = []
            query_time = datetime.now(timezone.utc)
            
            for i, reserve_address in enumerate(reserves_list):
                try:
                    reserve_data = pool_contract.functions.getReserveData(reserve_address).call()
                    token_info = self._get_token_info(w3, reserve_address)
                    config = self._decode_configuration(reserve_data[0])
                    
                    reserve_info = {
                        'chain': chain,
                        'token_address': reserve_address.lower(),
                        'token_symbol': token_info['symbol'],
                        'token_name': token_info['name'],
                        'decimals': token_info['decimals'],
                        'liquidity_rate': reserve_data[3] / 1e27,
                        'variable_borrow_rate': reserve_data[4] / 1e27,
                        'stable_borrow_rate': reserve_data[5] / 1e27,
                        'ltv': config['ltv'],
                        'liquidation_threshold': config['liquidation_threshold'],
                        'liquidation_bonus': config['liquidation_bonus'],
                        'is_active': config['is_active'],
                        'is_frozen': config['is_frozen'],
                        'borrowing_enabled': config['borrowing_enabled'],
                        'stable_borrowing_enabled': config['stable_borrowing_enabled'],
                        'liquidity_index': reserve_data[1] / 1e27,
                        'variable_borrow_index': reserve_data[2] / 1e27,
                        'atoken_address': reserve_data[8].lower(),
                        'variable_debt_token_address': reserve_data[10].lower(),
                        'last_update_timestamp': reserve_data[6],
                        'query_time': query_time,
                        'supply_apy': (reserve_data[3] / 1e27) * 100,
                        'borrow_apy': (reserve_data[4] / 1e27) * 100,
                    }
                    
                    reserves_data.append(reserve_info)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"{chain}: Processed {i + 1}/{len(reserves_list)} reserves")
                
                except Exception as e:
                    logger.warning(f"{chain}: Failed reserve {reserve_address}: {e}")
                    continue
            
            df = pd.DataFrame(reserves_data)
            logger.info(f"{chain}: Successfully fetched {len(df)} reserves")
            return df
            
        except Exception as e:
            logger.error(f"{chain}: RPC fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_all_chains(self, chains: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch reserves from all or specified chains"""
        if chains is None:
            chains = list(self.RPC_ENDPOINTS.keys())
        
        all_data = {}
        
        for chain in chains:
            logger.info(f"Fetching {chain} reserves...")
            df = self.fetch_chain_reserves(chain)
            
            if not df.empty:
                # Add prices if fetcher available
                if self.price_fetcher:
                    df = self._enrich_with_prices(df, chain)
                
                all_data[chain] = df
        
        return all_data
    
    def _enrich_with_prices(self, df: pd.DataFrame, chain: str) -> pd.DataFrame:
        """Enrich reserve data with current USD prices"""
        df['price_usd'] = 0.0
        df['price_available'] = False
        
        if not self.price_fetcher:
            logger.warning(f"{chain}: No price fetcher available")
            return df
        
        try:
            import time
            time.sleep(1)
            
            tokens = [
                {'symbol': row['token_symbol'], 'address': row['token_address'], 'chain': chain}
                for _, row in df.iterrows()
            ]
            
            logger.info(f"{chain}: Fetching prices for {len(tokens)} tokens...")
            prices = self.price_fetcher.get_batch_prices(tokens, progress=None)
            logger.info(f"{chain}: Received {len(prices)} price quotes")
            
            for idx, row in df.iterrows():
                # Use composite key matching price_fetcher format
                composite_key = f"{row['token_symbol']}|{row['token_address']}|{chain}"
                price_data = prices.get(composite_key)
                
                # Fallback to symbol only
                if price_data is None:
                    price_data = prices.get(row['token_symbol'])
                
                # Extract price
                if isinstance(price_data, (int, float)):
                    price = float(price_data)
                elif isinstance(price_data, dict):
                    price = float(price_data.get('price', 0.0))
                else:
                    price = 0.0
                
                df.at[idx, 'price_usd'] = price
                df.at[idx, 'price_available'] = price > 0
            
            success_count = df['price_available'].sum()
            logger.info(f"{chain}: Successfully priced {success_count}/{len(df)} tokens")
            
        except Exception as e:
            logger.error(f"{chain}: Price enrichment failed: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """Fetch all chains and return single combined DataFrame"""
        all_data = self.fetch_all_chains()
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data.values(), ignore_index=True)
        logger.info(f"Combined data: {len(combined)} total reserves across {len(all_data)} chains")
        
        return combined