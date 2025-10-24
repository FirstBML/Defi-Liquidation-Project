"""
Script to Fetch COMPLETE Aave V3 Reserve Data
Run this to create a proper JSON file with all aToken addresses
"""

from web3 import Web3
import json
from datetime import datetime
import os

# Aave V3 Pool contract ABI (just the getReservesList and getReserveData functions)
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
            {
                "components": [
                    {"internalType": "uint256", "name": "configuration", "type": "uint256"},
                    {"internalType": "uint128", "name": "liquidityIndex", "type": "uint128"},
                    {"internalType": "uint128", "name": "currentLiquidityRate", "type": "uint128"},
                    {"internalType": "uint128", "name": "variableBorrowIndex", "type": "uint128"},
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
                "internalType": "struct DataTypes.ReserveData",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Get configuration (for LTV, liquidation threshold, etc.)
POOL_DATA_PROVIDER_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getReserveConfigurationData",
        "outputs": [
            {"internalType": "uint256", "name": "decimals", "type": "uint256"},
            {"internalType": "uint256", "name": "ltv", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidationThreshold", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidationBonus", "type": "uint256"},
            {"internalType": "uint256", "name": "reserveFactor", "type": "uint256"},
            {"internalType": "bool", "name": "usageAsCollateralEnabled", "type": "bool"},
            {"internalType": "bool", "name": "borrowingEnabled", "type": "bool"},
            {"internalType": "bool", "name": "stableBorrowRateEnabled", "type": "bool"},
            {"internalType": "bool", "name": "isActive", "type": "bool"},
            {"internalType": "bool", "name": "isFrozen", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# ERC20 for getting symbol/name
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

# Aave V3 addresses for ALL chains
AAVE_ADDRESSES = {
    'ethereum': {
        'pool': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
        'pool_data_provider': '0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3',
        'rpc': 'https://eth.llamarpc.com'
    },
    'polygon': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://polygon-rpc.com'
    },
    'avalanche': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://api.avax.network/ext/bc/C/rpc'
    },
    'arbitrum': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://arb1.arbitrum.io/rpc'
    },
    'optimism': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://mainnet.optimism.io'
    },
    'bnb': {
        'pool': '0x6807dc923806fE8Fd134338EABCA509979a7e0cB',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://bsc-dataseed.binance.org'
    },
    'base': {
        'pool': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://mainnet.base.org'
    },
    'fantom': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://rpc.ftm.tools'
    },
    'gnosis': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://rpc.gnosischain.com'
    },
    'celo': {
        'pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
        'pool_data_provider': '0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654',
        'rpc': 'https://forno.celo.org'
    }
}

def fetch_complete_reserve_data(chain_name, chain_config):
    """Fetch complete reserve data for a chain"""
    print(f"\n📊 Fetching {chain_name}...")
    
    try:
        # Connect to Web3
        w3 = Web3(Web3.HTTPProvider(chain_config['rpc']))
        if not w3.is_connected():
            print(f"❌ Failed to connect to {chain_name}")
            return []
        
        # Get contracts
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(chain_config['pool']),
            abi=POOL_ABI
        )
        
        pool_data_provider = w3.eth.contract(
            address=Web3.to_checksum_address(chain_config['pool_data_provider']),
            abi=POOL_DATA_PROVIDER_ABI
        )
        
        # Get list of all reserves
        reserve_addresses = pool.functions.getReservesList().call()
        print(f"   Found {len(reserve_addresses)} reserves")
        
        reserves = []
        
        for i, address in enumerate(reserve_addresses):
            try:
                print(f"   Processing {i+1}/{len(reserve_addresses)}: {address[:10]}...")
                
                # Get basic token info
                token = w3.eth.contract(address=address, abi=ERC20_ABI)
                try:
                    symbol = token.functions.symbol().call()
                    name = token.functions.name().call()
                    decimals = token.functions.decimals().call()
                except:
                    symbol = "UNKNOWN"
                    name = "Unknown Token"
                    decimals = 18
                
                # Get reserve data (includes aToken addresses)
                reserve_data = pool.functions.getReserveData(address).call()
                
                # Get configuration (LTV, liquidation threshold, etc.)
                config = pool_data_provider.functions.getReserveConfigurationData(address).call()
                
                reserve_info = {
                    'chain': chain_name,
                    'address': address.lower(),
                    'symbol': symbol,
                    'name': name,
                    'decimals': decimals,
                    # CRITICAL: These are the missing fields!
                    'aTokenAddress': reserve_data[8].lower(),
                    'stableDebtTokenAddress': reserve_data[9].lower(),
                    'variableDebtTokenAddress': reserve_data[10].lower(),
                    'liquidityRate': reserve_data[2],
                    'variableBorrowRate': reserve_data[4],
                    'stableBorrowRate': reserve_data[5],
                    'liquidityIndex': reserve_data[1],
                    'variableBorrowIndex': reserve_data[3],
                    'lastUpdateTimestamp': reserve_data[6],
                    # Configuration
                    'ltv': config[1] / 10000,  # Convert basis points to decimal
                    'liquidationThreshold': config[2] / 10000,
                    'liquidationBonus': config[3] / 10000,
                    'usageAsCollateralEnabled': config[5],
                    'borrowingEnabled': config[6],
                    'stableBorrowRateEnabled': config[7],
                    'isActive': config[8],
                    'isFrozen': config[9]
                }
                
                reserves.append(reserve_info)
                print(f"      ✅ {symbol}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        return reserves
        
    except Exception as e:
        print(f"❌ Chain error: {e}")
        return []

def main():
    """Fetch complete data for all chains"""
    print("🚀 Fetching COMPLETE Aave V3 Reserve Data for ALL Chains...")
    print("=" * 60)
    
    all_data = {}
    total_reserves = 0
    
    for chain_name, chain_config in AAVE_ADDRESSES.items():
        reserves = fetch_complete_reserve_data(chain_name, chain_config)
        all_data[chain_name] = reserves
        total_reserves += len(reserves)
        print(f"✅ {chain_name}: {len(reserves)} reserves")
    
    # Save to the specific file path
    output_path = r"C:\Users\g\Documents\LiquidationProject\app\aave_v3_complete_data_1761094969.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        'fetch_timestamp': int(datetime.now().timestamp()),
        'total_chains': len(AAVE_ADDRESSES),
        'total_reserves': total_reserves,
        'data': all_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Saved to: {output_path}")
    print(f"📊 Total chains: {len(AAVE_ADDRESSES)}")
    print(f"📊 Total reserves: {total_reserves}")
    print("\nChains processed:")
    for chain_name in AAVE_ADDRESSES.keys():
        reserve_count = len(all_data.get(chain_name, []))
        print(f"   • {chain_name}: {reserve_count} reserves")
    
    print("\nNow update your portfolio_tracker_service.py to use this file!")

if __name__ == "__main__":
    main()