"""
Add BNB, Base, and Fantom chains to RPC Reserve Fetcher
Fetches data and stores in database
"""
from web3 import Web3
import pandas as pd
from datetime import datetime, timezone
import time
from app.db_models import SessionLocal, Reserve
from app.price_fetcher import EnhancedPriceFetcher
import os
from dotenv import load_dotenv

load_dotenv()

class ExtendedAaveRPCFetcher:
    """Fetch Aave V3 reserves from additional chains"""
    
    # Multiple RPC endpoints for fallback
    RPC_ENDPOINTS = {
        'bnb': [
            'https://bsc-dataseed.binance.org',
            'https://bsc-dataseed1.defibit.io',
            'https://bsc-dataseed1.ninicoin.io'
        ],
        'base': [
            'https://mainnet.base.org',
            'https://base-rpc.publicnode.com',
            'https://rpc.ankr.com/base'
        ],
        'fantom': [
            'https://fantom-mainnet.public.blastapi.io',
            'https://fantom.publicnode.com',
            'https://rpc.fantom.network'
        ]
    }
    
    POOL_ADDRESSES = {
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
        self.price_fetcher = price_fetcher
    
    def connect_to_chain(self, chain: str):
        """Try multiple RPC endpoints for reliability"""
        for rpc_url in self.RPC_ENDPOINTS[chain]:
            try:
                print(f"  Trying {rpc_url.split('//')[1].split('/')[0]}...")
                w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 60}))
                
                block_number = w3.eth.block_number
                if block_number > 0:
                    print(f"  Connected at block {block_number}")
                    return w3
            except Exception as e:
                print(f"  Failed: {str(e)[:80]}")
                continue
        return None
    
    def get_token_info(self, w3, address):
        """Get token metadata"""
        try:
            token_contract = w3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=self.ERC20_ABI
            )
            
            try:
                symbol = token_contract.functions.symbol().call()
            except:
                # Handle bytes32 tokens
                symbol_bytes = w3.eth.call({'to': address, 'data': '0x95d89b41'})
                symbol = symbol_bytes.decode('utf-8').rstrip('\x00') if symbol_bytes else 'UNKNOWN'
            
            return {
                'symbol': symbol if symbol else 'UNKNOWN',
                'decimals': token_contract.functions.decimals().call(),
                'name': token_contract.functions.name().call()
            }
        except Exception as e:
            print(f"    Token info failed: {str(e)[:50]}")
            return {'symbol': 'UNKNOWN', 'decimals': 18, 'name': 'Unknown'}
    
    def decode_configuration(self, configuration):
        """Decode Aave configuration bitmask"""
        return {
            'ltv': ((configuration & ((1 << 16) - 1)) / 10000),
            'liquidation_threshold': (((configuration >> 16) & ((1 << 16) - 1)) / 10000),
            'liquidation_bonus': (((configuration >> 32) & ((1 << 16) - 1)) / 10000),
            'is_active': ((configuration >> 56) & 1) == 1,
            'is_frozen': ((configuration >> 57) & 1) == 1,
            'borrowing_enabled': ((configuration >> 58) & 1) == 1,
            'stable_borrowing_enabled': ((configuration >> 59) & 1) == 1
        }
    
    def fetch_chain_reserves(self, chain: str):
        """Fetch reserves for one chain"""
        print(f"\nFetching {chain.upper()}...")
        
        w3 = self.connect_to_chain(chain)
        if not w3:
            print(f"  Failed to connect")
            return pd.DataFrame()
        
        try:
            pool_contract = w3.eth.contract(
                address=Web3.to_checksum_address(self.POOL_ADDRESSES[chain]),
                abi=self.POOL_ABI
            )
            
            reserves_list = pool_contract.functions.getReservesList().call()
            print(f"  Found {len(reserves_list)} reserves")
            
            reserves_data = []
            query_time = datetime.now(timezone.utc)
            
            for i, reserve_address in enumerate(reserves_list):
                try:
                    if (i + 1) % 5 == 0:
                        print(f"  Processing {i+1}/{len(reserves_list)}...")
                    
                    reserve_data = pool_contract.functions.getReserveData(reserve_address).call()
                    token_info = self.get_token_info(w3, reserve_address)
                    config = self.decode_configuration(reserve_data[0])
                    
                    reserves_data.append({
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
                    })
                    
                except Exception as e:
                    print(f"    Failed reserve {reserve_address[:10]}: {str(e)[:60]}")
                    continue
            
            df = pd.DataFrame(reserves_data)
            print(f"  Successfully fetched {len(df)} reserves")
            return df
            
        except Exception as e:
            print(f"  RPC fetch failed: {e}")
            return pd.DataFrame()
    
    def enrich_with_prices(self, df, chain):
        """Add USD prices from CoinGecko"""
        df['price_usd'] = 0.0
        df['price_available'] = False
        
        if not self.price_fetcher or df.empty:
            return df
        
        try:
            print(f"  Fetching prices...")
            
            tokens = [
                {'symbol': row['token_symbol'], 'address': row['token_address'], 'chain': chain}
                for _, row in df.iterrows()
            ]
            
            prices = self.price_fetcher.get_batch_prices(tokens, progress=None)
            
            for idx, row in df.iterrows():
                # Use composite key
                composite_key = f"{row['token_symbol']}|{row['token_address']}|{chain}"
                price_data = prices.get(composite_key) or prices.get(row['token_symbol'])
                
                if isinstance(price_data, (int, float)):
                    price = float(price_data)
                elif isinstance(price_data, dict):
                    price = float(price_data.get('price', 0.0))
                else:
                    price = 0.0
                
                df.at[idx, 'price_usd'] = price
                df.at[idx, 'price_available'] = price > 0
            
            success = df['price_available'].sum()
            print(f"  Priced {success}/{len(df)} tokens")
            
        except Exception as e:
            print(f"  Price fetch failed: {e}")
        
        return df


def main():
    print("="*70)
    print("ADDING BNB, BASE, AND FANTOM CHAINS TO AAVE RESERVE DATA")
    print("="*70)
    
    # Initialize
    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        print("\nWarning: No COINGECKO_API_KEY found. Prices will be 0.")
        price_fetcher = None
    else:
        price_fetcher = EnhancedPriceFetcher(api_key=api_key)
    
    fetcher = ExtendedAaveRPCFetcher(price_fetcher=price_fetcher)
    
    chains = ['bnb', 'base', 'fantom']
    all_data = {}
    
    # Fetch from each chain
    for chain in chains:
        df = fetcher.fetch_chain_reserves(chain)
        
        if not df.empty:
            # Add prices
            df = fetcher.enrich_with_prices(df, chain)
            all_data[chain] = df
        
        time.sleep(3)  # Rate limit between chains
    
    if not all_data:
        print("\nFailed to fetch any data")
        return
    
    # Store in database
    print("\n" + "="*70)
    print("STORING IN DATABASE")
    print("="*70)
    
    db = SessionLocal()
    total_stored = 0
    
    try:
        for chain, df in all_data.items():
            print(f"\nStoring {chain} ({len(df)} reserves)...")
            
            for _, row in df.iterrows():
                reserve = Reserve(
                    chain=row['chain'],
                    token_address=row['token_address'],
                    token_symbol=row['token_symbol'],
                    token_name=row['token_name'],
                    decimals=row['decimals'],
                    liquidity_rate=row['liquidity_rate'],
                    variable_borrow_rate=row['variable_borrow_rate'],
                    stable_borrow_rate=row['stable_borrow_rate'],
                    supply_apy=row['supply_apy'],
                    borrow_apy=row['borrow_apy'],
                    ltv=row['ltv'],
                    liquidation_threshold=row['liquidation_threshold'],
                    liquidation_bonus=row['liquidation_bonus'],
                    is_active=row['is_active'],
                    is_frozen=row['is_frozen'],
                    borrowing_enabled=row['borrowing_enabled'],
                    stable_borrowing_enabled=row['stable_borrowing_enabled'],
                    liquidity_index=row['liquidity_index'],
                    variable_borrow_index=row['variable_borrow_index'],
                    atoken_address=row['atoken_address'],
                    variable_debt_token_address=row['variable_debt_token_address'],
                    price_usd=row.get('price_usd', 0.0),
                    price_available=row.get('price_available', False),
                    last_update_timestamp=row['last_update_timestamp'],
                    query_time=row['query_time']
                )
                db.add(reserve)
                total_stored += 1
            
            db.commit()
            print(f"  Committed {len(df)} reserves")
    
    finally:
        db.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for chain, df in all_data.items():
        tokens = df['token_symbol'].tolist()
        priced = df['price_available'].sum()
        print(f"\n{chain.upper()}:")
        print(f"  Reserves: {len(df)}")
        print(f"  Tokens: {', '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}")
        print(f"  Prices: {priced}/{len(df)}")
    
    print(f"\nTotal stored: {total_stored} reserves")
    print("\nNext steps:")
    print("  1. Run: python force_price_refresh_fixed.py  # To refresh prices")
    print("  2. Restart server: uvicorn app.main:app --reload")
    print("  3. Test: curl http://localhost:8000/api/chains/available")

if __name__ == "__main__":
    main()