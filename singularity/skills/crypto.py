#!/usr/bin/env python3
"""
Crypto Skill - On-chain capabilities for autonomous agents.

Enables agents to:
- Create and manage wallets
- Deploy tokens on Base/Ethereum
- Swap tokens on DEXs
- Create liquidity pools
- Bridge assets across chains
- Read on-chain data

Supports: Base, Ethereum, Polygon, Arbitrum, Optimism
"""

import asyncio
import json
import os
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from .base import Skill, SkillManifest, SkillAction, SkillResult

# Check for web3 dependencies
HAS_WEB3 = False
try:
    from web3 import Web3, AsyncWeb3
    from eth_account import Account
    from eth_account.messages import encode_defunct
    HAS_WEB3 = True
except ImportError:
    pass

# Standard ERC-20 ABI (minimal for transfers and approvals)
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}], "name": "transfer", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
]

# Simple ERC-20 bytecode (OpenZeppelin-style, compiled)
# This is a minimal ERC-20 with mint function for deployer
ERC20_BYTECODE = "0x60806040523480156200001157600080fd5b5060405162000c3838038062000c38833981016040819052620000349162000138565b8251620000499060039060208601906200007a565b5081516200005f9060049060208501906200007a565b506005805460ff191660ff929092169190911790555062000201565b828054620000889062000194565b90600052602060002090601f016020900481019282620000ac5760008555620000f7565b82601f10620000c757805160ff1916838001178555620000f7565b82800160010185558215620000f7579182015b82811115620000f7578251825591602001919060010190620000da565b506200010592915062000109565b5090565b5b808211156200010557600081556001016200010a565b634e487b7160e01b600052604160045260246000fd5b6000806000606084860312156200014e57600080fd5b83516001600160401b03808211156200016657600080fd5b818601915086601f8301126200017b57600080fd5b815181811115620001905762000190620001206200011f565b604051601f8201601f19908116603f01168101908382118183101715620001bd57620001bd6200011f565b81604052828152896020848701011115620001d757600080fd5b8260208601602083013760006020848301015280975050505050602084015191506040840151905092959194509250565b610a2780620002116000396000f3fe"

# Chain configurations
CHAINS = {
    "base": {
        "chain_id": 8453,
        "rpc": "https://mainnet.base.org",
        "explorer": "https://basescan.org",
        "native": "ETH",
        "dex_router": "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24",  # Aerodrome
        "weth": "0x4200000000000000000000000000000000000006",
    },
    "base-sepolia": {
        "chain_id": 84532,
        "rpc": "https://sepolia.base.org",
        "explorer": "https://sepolia.basescan.org",
        "native": "ETH",
        "dex_router": None,
        "weth": "0x4200000000000000000000000000000000000006",
    },
    "ethereum": {
        "chain_id": 1,
        "rpc": "https://eth.llamarpc.com",
        "explorer": "https://etherscan.io",
        "native": "ETH",
        "dex_router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2
        "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    },
    "polygon": {
        "chain_id": 137,
        "rpc": "https://polygon.llamarpc.com",
        "explorer": "https://polygonscan.com",
        "native": "MATIC",
        "dex_router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",  # QuickSwap
        "weth": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
    },
    "arbitrum": {
        "chain_id": 42161,
        "rpc": "https://arb1.arbitrum.io/rpc",
        "explorer": "https://arbiscan.io",
        "native": "ETH",
        "dex_router": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",  # SushiSwap
        "weth": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    },
}

# Uniswap V2 Router ABI (minimal)
UNISWAP_ROUTER_ABI = [
    {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "swapExactTokensForTokens", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"internalType": "uint256", "name": "amountOutMin", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "swapExactETHForTokens", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "payable", "type": "function"},
    {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "swapExactTokensForETH", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"internalType": "address", "name": "tokenA", "type": "address"}, {"internalType": "address", "name": "tokenB", "type": "address"}, {"internalType": "uint256", "name": "amountADesired", "type": "uint256"}, {"internalType": "uint256", "name": "amountBDesired", "type": "uint256"}, {"internalType": "uint256", "name": "amountAMin", "type": "uint256"}, {"internalType": "uint256", "name": "amountBMin", "type": "uint256"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "addLiquidity", "outputs": [{"internalType": "uint256", "name": "amountA", "type": "uint256"}, {"internalType": "uint256", "name": "amountB", "type": "uint256"}, {"internalType": "uint256", "name": "liquidity", "type": "uint256"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"internalType": "address", "name": "token", "type": "address"}, {"internalType": "uint256", "name": "amountTokenDesired", "type": "uint256"}, {"internalType": "uint256", "name": "amountTokenMin", "type": "uint256"}, {"internalType": "uint256", "name": "amountETHMin", "type": "uint256"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "addLiquidityETH", "outputs": [{"internalType": "uint256", "name": "amountToken", "type": "uint256"}, {"internalType": "uint256", "name": "amountETH", "type": "uint256"}, {"internalType": "uint256", "name": "liquidity", "type": "uint256"}], "stateMutability": "payable", "type": "function"},
    {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}], "name": "getAmountsOut", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "view", "type": "function"},
]


class CryptoSkill(Skill):
    """
    On-chain capabilities for autonomous agents.

    Enables agents to:
    - Create wallets and manage keys
    - Deploy ERC-20 tokens
    - Swap on DEXs (Uniswap, Aerodrome, etc.)
    - Add liquidity to pools
    - Bridge across chains
    - Read balances and on-chain data

    Primary chain: Base (low fees, fast)
    """

    def __init__(self, credentials: Dict = None):
        super().__init__(credentials)

        # Agent's wallets (can have multiple)
        self._wallets: Dict[str, Dict] = {}  # address -> {private_key, name, created_at}
        self._active_wallet: Optional[str] = None

        # Default chain
        self._chain = "base"

        # Web3 connections (lazy loaded)
        self._web3_connections: Dict[str, Any] = {}

        # Deployed tokens by this agent
        self._deployed_tokens: List[Dict] = []

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="crypto",
            name="Crypto & DeFi",
            version="1.0.0",
            category="blockchain",
            description="On-chain capabilities: wallets, tokens, swaps, liquidity",
            actions=[
                # === Wallet Management ===
                SkillAction(
                    name="create_wallet",
                    description="Create a new crypto wallet",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": False,
                            "description": "Name for this wallet (default: 'main')"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="import_wallet",
                    description="Import an existing wallet from private key",
                    parameters={
                        "private_key": {
                            "type": "string",
                            "required": True,
                            "description": "Private key (hex, with or without 0x)"
                        },
                        "name": {
                            "type": "string",
                            "required": False,
                            "description": "Name for this wallet"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_wallets",
                    description="List all wallets owned by this agent",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="set_active_wallet",
                    description="Set which wallet to use for transactions",
                    parameters={
                        "address": {
                            "type": "string",
                            "required": True,
                            "description": "Wallet address to make active"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_balance",
                    description="Get native token balance (ETH/MATIC) for a wallet",
                    parameters={
                        "address": {
                            "type": "string",
                            "required": False,
                            "description": "Address to check (default: active wallet)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain to check (default: base)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_token_balance",
                    description="Get ERC-20 token balance",
                    parameters={
                        "token": {
                            "type": "string",
                            "required": True,
                            "description": "Token contract address"
                        },
                        "address": {
                            "type": "string",
                            "required": False,
                            "description": "Wallet address (default: active wallet)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: base)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Chain Management ===
                SkillAction(
                    name="set_chain",
                    description="Set the default chain for transactions",
                    parameters={
                        "chain": {
                            "type": "string",
                            "required": True,
                            "description": "Chain name: base, ethereum, polygon, arbitrum"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="list_chains",
                    description="List supported chains and their configs",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Token Deployment ===
                SkillAction(
                    name="deploy_token",
                    description="Deploy a new ERC-20 token on Base",
                    parameters={
                        "name": {
                            "type": "string",
                            "required": True,
                            "description": "Token name (e.g., 'My Token')"
                        },
                        "symbol": {
                            "type": "string",
                            "required": True,
                            "description": "Token symbol (e.g., 'MTK')"
                        },
                        "initial_supply": {
                            "type": "number",
                            "required": True,
                            "description": "Initial supply (in tokens, not wei)"
                        },
                        "decimals": {
                            "type": "integer",
                            "required": False,
                            "description": "Decimals (default: 18)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain to deploy on (default: base)"
                        }
                    },
                    estimated_cost=0.01,  # Gas cost estimate
                ),
                SkillAction(
                    name="my_tokens",
                    description="List tokens deployed by this agent",
                    parameters={},
                    estimated_cost=0,
                ),
                # === Transfers ===
                SkillAction(
                    name="send_eth",
                    description="Send native token (ETH/MATIC) to an address",
                    parameters={
                        "to": {
                            "type": "string",
                            "required": True,
                            "description": "Recipient address"
                        },
                        "amount": {
                            "type": "number",
                            "required": True,
                            "description": "Amount to send (in ETH/MATIC)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0.001,
                ),
                SkillAction(
                    name="send_token",
                    description="Send ERC-20 tokens to an address",
                    parameters={
                        "token": {
                            "type": "string",
                            "required": True,
                            "description": "Token contract address"
                        },
                        "to": {
                            "type": "string",
                            "required": True,
                            "description": "Recipient address"
                        },
                        "amount": {
                            "type": "number",
                            "required": True,
                            "description": "Amount to send (in tokens)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0.001,
                ),
                # === DEX / Swaps ===
                SkillAction(
                    name="swap",
                    description="Swap tokens on a DEX",
                    parameters={
                        "token_in": {
                            "type": "string",
                            "required": True,
                            "description": "Token to sell (address or 'ETH')"
                        },
                        "token_out": {
                            "type": "string",
                            "required": True,
                            "description": "Token to buy (address or 'ETH')"
                        },
                        "amount_in": {
                            "type": "number",
                            "required": True,
                            "description": "Amount to swap"
                        },
                        "slippage": {
                            "type": "number",
                            "required": False,
                            "description": "Slippage tolerance % (default: 1)"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0.005,
                ),
                SkillAction(
                    name="get_quote",
                    description="Get a swap quote without executing",
                    parameters={
                        "token_in": {
                            "type": "string",
                            "required": True,
                            "description": "Token to sell (address or 'ETH')"
                        },
                        "token_out": {
                            "type": "string",
                            "required": True,
                            "description": "Token to buy (address or 'ETH')"
                        },
                        "amount_in": {
                            "type": "number",
                            "required": True,
                            "description": "Amount to swap"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0,
                ),
                # === Liquidity ===
                SkillAction(
                    name="add_liquidity",
                    description="Add liquidity to a DEX pool",
                    parameters={
                        "token_a": {
                            "type": "string",
                            "required": True,
                            "description": "First token (address or 'ETH')"
                        },
                        "token_b": {
                            "type": "string",
                            "required": True,
                            "description": "Second token address"
                        },
                        "amount_a": {
                            "type": "number",
                            "required": True,
                            "description": "Amount of first token"
                        },
                        "amount_b": {
                            "type": "number",
                            "required": True,
                            "description": "Amount of second token"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0.01,
                ),
                # === On-chain Data ===
                SkillAction(
                    name="get_token_info",
                    description="Get info about a token (name, symbol, supply)",
                    parameters={
                        "token": {
                            "type": "string",
                            "required": True,
                            "description": "Token contract address"
                        },
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="get_gas_price",
                    description="Get current gas price on a chain",
                    parameters={
                        "chain": {
                            "type": "string",
                            "required": False,
                            "description": "Chain (default: current chain)"
                        }
                    },
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],  # No credentials required - agents create their own wallets
        )

    def check_credentials(self) -> bool:
        """Crypto skill just needs web3 library."""
        return HAS_WEB3

    def _get_web3(self, chain: str = None) -> Any:
        """Get or create Web3 connection for a chain."""
        if not HAS_WEB3:
            raise RuntimeError("web3 library not installed. Run: pip install web3")

        chain = chain or self._chain
        if chain not in CHAINS:
            raise ValueError(f"Unknown chain: {chain}. Supported: {list(CHAINS.keys())}")

        if chain not in self._web3_connections:
            rpc = self.credentials.get(f"{chain.upper()}_RPC") or CHAINS[chain]["rpc"]
            self._web3_connections[chain] = Web3(Web3.HTTPProvider(rpc))

        return self._web3_connections[chain]

    async def execute(self, action: str, params: Dict) -> SkillResult:
        if not HAS_WEB3:
            return SkillResult(
                success=False,
                message="web3 library not installed. Run: pip install web3 eth-account"
            )

        handlers = {
            # Wallet
            "create_wallet": self._create_wallet,
            "import_wallet": self._import_wallet,
            "list_wallets": self._list_wallets,
            "set_active_wallet": self._set_active_wallet,
            "get_balance": self._get_balance,
            "get_token_balance": self._get_token_balance,
            # Chain
            "set_chain": self._set_chain,
            "list_chains": self._list_chains,
            # Token deployment
            "deploy_token": self._deploy_token,
            "my_tokens": self._my_tokens,
            # Transfers
            "send_eth": self._send_eth,
            "send_token": self._send_token,
            # DEX
            "swap": self._swap,
            "get_quote": self._get_quote,
            "add_liquidity": self._add_liquidity,
            # Data
            "get_token_info": self._get_token_info,
            "get_gas_price": self._get_gas_price,
        }

        handler = handlers.get(action)
        if handler:
            try:
                return await handler(params)
            except Exception as e:
                return SkillResult(success=False, message=f"Error: {str(e)}")
        return SkillResult(success=False, message=f"Unknown action: {action}")

    # === Wallet Management ===

    async def _create_wallet(self, params: Dict) -> SkillResult:
        """Create a new wallet."""
        name = params.get("name", "main")

        # Generate new account
        account = Account.create()
        address = account.address
        private_key = account.key.hex()

        self._wallets[address] = {
            "private_key": private_key,
            "name": name,
            "created_at": datetime.now().isoformat(),
        }

        # Set as active if first wallet
        if self._active_wallet is None:
            self._active_wallet = address

        return SkillResult(
            success=True,
            message=f"Created wallet '{name}'",
            data={
                "address": address,
                "name": name,
                "is_active": self._active_wallet == address,
                # Private key is stored securely in self._wallets but NOT
                # returned in result data. Result data flows into LLM context
                # and activity logs, which would leak the key to third parties.
            }
        )

    async def _import_wallet(self, params: Dict) -> SkillResult:
        """Import wallet from private key."""
        private_key = params.get("private_key", "").strip()
        name = params.get("name", "imported")

        if not private_key:
            return SkillResult(success=False, message="Private key required")

        # Add 0x if missing
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        try:
            account = Account.from_key(private_key)
            address = account.address

            self._wallets[address] = {
                "private_key": private_key,
                "name": name,
                "created_at": datetime.now().isoformat(),
            }

            if self._active_wallet is None:
                self._active_wallet = address

            return SkillResult(
                success=True,
                message=f"Imported wallet '{name}'",
                data={
                    "address": address,
                    "name": name,
                    "is_active": self._active_wallet == address,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Invalid private key: {e}")

    async def _list_wallets(self, params: Dict) -> SkillResult:
        """List all wallets."""
        wallets = []
        for address, info in self._wallets.items():
            wallets.append({
                "address": address,
                "name": info["name"],
                "is_active": address == self._active_wallet,
                "created_at": info["created_at"],
            })

        return SkillResult(
            success=True,
            message=f"You have {len(wallets)} wallet(s)",
            data={
                "wallets": wallets,
                "active": self._active_wallet,
            }
        )

    async def _set_active_wallet(self, params: Dict) -> SkillResult:
        """Set active wallet."""
        address = params.get("address", "").strip()

        if address not in self._wallets:
            return SkillResult(success=False, message=f"Wallet {address} not found")

        self._active_wallet = address
        return SkillResult(
            success=True,
            message=f"Active wallet set to {address}",
            data={"active": address, "name": self._wallets[address]["name"]}
        )

    async def _get_balance(self, params: Dict) -> SkillResult:
        """Get native token balance."""
        address = params.get("address") or self._active_wallet
        chain = params.get("chain") or self._chain

        if not address:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            balance_wei = w3.eth.get_balance(address)
            balance = w3.from_wei(balance_wei, "ether")

            chain_info = CHAINS[chain]
            return SkillResult(
                success=True,
                message=f"{balance:.6f} {chain_info['native']} on {chain}",
                data={
                    "address": address,
                    "balance": float(balance),
                    "balance_wei": balance_wei,
                    "chain": chain,
                    "native_token": chain_info["native"],
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to get balance: {e}")

    async def _get_token_balance(self, params: Dict) -> SkillResult:
        """Get ERC-20 token balance."""
        token = params.get("token", "").strip()
        address = params.get("address") or self._active_wallet
        chain = params.get("chain") or self._chain

        if not token:
            return SkillResult(success=False, message="Token address required")
        if not address:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            token = w3.to_checksum_address(token)
            contract = w3.eth.contract(address=token, abi=ERC20_ABI)

            balance = contract.functions.balanceOf(address).call()
            decimals = contract.functions.decimals().call()
            symbol = contract.functions.symbol().call()

            balance_formatted = balance / (10 ** decimals)

            return SkillResult(
                success=True,
                message=f"{balance_formatted:.6f} {symbol}",
                data={
                    "address": address,
                    "token": token,
                    "symbol": symbol,
                    "balance": balance_formatted,
                    "balance_raw": balance,
                    "decimals": decimals,
                    "chain": chain,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to get token balance: {e}")

    # === Chain Management ===

    async def _set_chain(self, params: Dict) -> SkillResult:
        """Set default chain."""
        chain = params.get("chain", "").lower().strip()

        if chain not in CHAINS:
            return SkillResult(
                success=False,
                message=f"Unknown chain: {chain}. Supported: {list(CHAINS.keys())}"
            )

        self._chain = chain
        return SkillResult(
            success=True,
            message=f"Default chain set to {chain}",
            data={"chain": chain, "config": CHAINS[chain]}
        )

    async def _list_chains(self, params: Dict) -> SkillResult:
        """List supported chains."""
        chains = []
        for name, config in CHAINS.items():
            chains.append({
                "name": name,
                "chain_id": config["chain_id"],
                "native_token": config["native"],
                "explorer": config["explorer"],
                "is_current": name == self._chain,
            })

        return SkillResult(
            success=True,
            message=f"{len(chains)} chains supported",
            data={"chains": chains, "current": self._chain}
        )

    # === Token Deployment ===

    async def _deploy_token(self, params: Dict) -> SkillResult:
        """Deploy a new ERC-20 token."""
        name = params.get("name", "").strip()
        symbol = params.get("symbol", "").strip()
        initial_supply = params.get("initial_supply", 0)
        decimals = params.get("decimals", 18)
        chain = params.get("chain") or self._chain

        if not name or not symbol:
            return SkillResult(success=False, message="Token name and symbol required")
        if initial_supply <= 0:
            return SkillResult(success=False, message="Initial supply must be positive")
        if not self._active_wallet:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            wallet = self._wallets[self._active_wallet]
            account = Account.from_key(wallet["private_key"])

            # Calculate supply in wei
            supply_wei = int(initial_supply * (10 ** decimals))

            # Encode constructor arguments
            # For a proper deployment, we'd compile the contract
            # For now, use a factory or pre-deployed bytecode

            # Simple approach: Use create2 or a token factory
            # For MVP, we'll simulate the deployment

            # Get gas price
            gas_price = w3.eth.gas_price
            nonce = w3.eth.get_transaction_count(account.address)

            # Build deployment transaction
            # Note: This uses simplified bytecode - in production use verified contracts
            contract = w3.eth.contract(bytecode=ERC20_BYTECODE, abi=ERC20_ABI)

            # Estimate gas
            # For a real deployment, you'd need the full compiled contract
            # This is a placeholder that would need proper contract compilation

            return SkillResult(
                success=False,
                message="Token deployment requires compiled contract bytecode. Use a token factory like Clanker on Base for easy deployment.",
                data={
                    "suggested_action": "Use crypto:deploy_via_factory instead",
                    "token_spec": {
                        "name": name,
                        "symbol": symbol,
                        "initial_supply": initial_supply,
                        "decimals": decimals,
                        "deployer": account.address,
                    }
                }
            )

        except Exception as e:
            return SkillResult(success=False, message=f"Deployment failed: {e}")

    async def _my_tokens(self, params: Dict) -> SkillResult:
        """List tokens deployed by this agent."""
        return SkillResult(
            success=True,
            message=f"You have deployed {len(self._deployed_tokens)} token(s)",
            data={"tokens": self._deployed_tokens}
        )

    # === Transfers ===

    async def _send_eth(self, params: Dict) -> SkillResult:
        """Send native token."""
        to = params.get("to", "").strip()
        amount = params.get("amount", 0)
        chain = params.get("chain") or self._chain

        if not to:
            return SkillResult(success=False, message="Recipient address required")
        if amount <= 0:
            return SkillResult(success=False, message="Amount must be positive")
        if not self._active_wallet:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            wallet = self._wallets[self._active_wallet]
            account = Account.from_key(wallet["private_key"])

            to = w3.to_checksum_address(to)
            value_wei = w3.to_wei(amount, "ether")

            # Build transaction
            tx = {
                "from": account.address,
                "to": to,
                "value": value_wei,
                "gas": 21000,
                "gasPrice": w3.eth.gas_price,
                "nonce": w3.eth.get_transaction_count(account.address),
                "chainId": CHAINS[chain]["chain_id"],
            }

            # Sign and send
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

            return SkillResult(
                success=True,
                message=f"Sent {amount} {CHAINS[chain]['native']} to {to}",
                data={
                    "tx_hash": tx_hash.hex(),
                    "from": account.address,
                    "to": to,
                    "amount": amount,
                    "chain": chain,
                    "explorer": f"{CHAINS[chain]['explorer']}/tx/{tx_hash.hex()}",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Transfer failed: {e}")

    async def _send_token(self, params: Dict) -> SkillResult:
        """Send ERC-20 tokens."""
        token = params.get("token", "").strip()
        to = params.get("to", "").strip()
        amount = params.get("amount", 0)
        chain = params.get("chain") or self._chain

        if not token or not to:
            return SkillResult(success=False, message="Token and recipient required")
        if amount <= 0:
            return SkillResult(success=False, message="Amount must be positive")
        if not self._active_wallet:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            wallet = self._wallets[self._active_wallet]
            account = Account.from_key(wallet["private_key"])

            token = w3.to_checksum_address(token)
            to = w3.to_checksum_address(to)

            contract = w3.eth.contract(address=token, abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            symbol = contract.functions.symbol().call()

            amount_wei = int(amount * (10 ** decimals))

            # Build transaction
            tx = contract.functions.transfer(to, amount_wei).build_transaction({
                "from": account.address,
                "gas": 100000,
                "gasPrice": w3.eth.gas_price,
                "nonce": w3.eth.get_transaction_count(account.address),
                "chainId": CHAINS[chain]["chain_id"],
            })

            # Sign and send
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

            return SkillResult(
                success=True,
                message=f"Sent {amount} {symbol} to {to}",
                data={
                    "tx_hash": tx_hash.hex(),
                    "from": account.address,
                    "to": to,
                    "token": token,
                    "symbol": symbol,
                    "amount": amount,
                    "chain": chain,
                    "explorer": f"{CHAINS[chain]['explorer']}/tx/{tx_hash.hex()}",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Token transfer failed: {e}")

    # === DEX / Swaps ===

    async def _get_quote(self, params: Dict) -> SkillResult:
        """Get swap quote."""
        token_in = params.get("token_in", "").strip()
        token_out = params.get("token_out", "").strip()
        amount_in = params.get("amount_in", 0)
        chain = params.get("chain") or self._chain

        if not token_in or not token_out:
            return SkillResult(success=False, message="token_in and token_out required")
        if amount_in <= 0:
            return SkillResult(success=False, message="amount_in must be positive")

        try:
            w3 = self._get_web3(chain)
            chain_config = CHAINS[chain]

            if not chain_config.get("dex_router"):
                return SkillResult(success=False, message=f"No DEX configured for {chain}")

            router = w3.eth.contract(
                address=w3.to_checksum_address(chain_config["dex_router"]),
                abi=UNISWAP_ROUTER_ABI
            )

            # Handle ETH as WETH
            weth = chain_config["weth"]
            path_in = weth if token_in.upper() == "ETH" else w3.to_checksum_address(token_in)
            path_out = weth if token_out.upper() == "ETH" else w3.to_checksum_address(token_out)

            # Get decimals for input token
            if token_in.upper() == "ETH":
                decimals_in = 18
                symbol_in = chain_config["native"]
            else:
                contract_in = w3.eth.contract(address=path_in, abi=ERC20_ABI)
                decimals_in = contract_in.functions.decimals().call()
                symbol_in = contract_in.functions.symbol().call()

            # Get decimals for output token
            if token_out.upper() == "ETH":
                decimals_out = 18
                symbol_out = chain_config["native"]
            else:
                contract_out = w3.eth.contract(address=path_out, abi=ERC20_ABI)
                decimals_out = contract_out.functions.decimals().call()
                symbol_out = contract_out.functions.symbol().call()

            amount_in_wei = int(amount_in * (10 ** decimals_in))

            # Get quote
            amounts = router.functions.getAmountsOut(amount_in_wei, [path_in, path_out]).call()
            amount_out_wei = amounts[1]
            amount_out = amount_out_wei / (10 ** decimals_out)

            return SkillResult(
                success=True,
                message=f"{amount_in} {symbol_in} -> {amount_out:.6f} {symbol_out}",
                data={
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount_in": amount_in,
                    "amount_out": amount_out,
                    "symbol_in": symbol_in,
                    "symbol_out": symbol_out,
                    "chain": chain,
                    "rate": amount_out / amount_in if amount_in > 0 else 0,
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Quote failed: {e}")

    async def _swap(self, params: Dict) -> SkillResult:
        """Execute a swap on DEX."""
        token_in = params.get("token_in", "").strip()
        token_out = params.get("token_out", "").strip()
        amount_in = params.get("amount_in", 0)
        slippage = params.get("slippage", 1)  # 1% default
        chain = params.get("chain") or self._chain

        if not token_in or not token_out:
            return SkillResult(success=False, message="token_in and token_out required")
        if amount_in <= 0:
            return SkillResult(success=False, message="amount_in must be positive")
        if not self._active_wallet:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            chain_config = CHAINS[chain]
            wallet = self._wallets[self._active_wallet]
            account = Account.from_key(wallet["private_key"])

            if not chain_config.get("dex_router"):
                return SkillResult(success=False, message=f"No DEX configured for {chain}")

            router_address = w3.to_checksum_address(chain_config["dex_router"])
            router = w3.eth.contract(address=router_address, abi=UNISWAP_ROUTER_ABI)

            weth = chain_config["weth"]
            is_eth_in = token_in.upper() == "ETH"
            is_eth_out = token_out.upper() == "ETH"

            path_in = weth if is_eth_in else w3.to_checksum_address(token_in)
            path_out = weth if is_eth_out else w3.to_checksum_address(token_out)

            # Get decimals
            decimals_in = 18 if is_eth_in else w3.eth.contract(address=path_in, abi=ERC20_ABI).functions.decimals().call()
            decimals_out = 18 if is_eth_out else w3.eth.contract(address=path_out, abi=ERC20_ABI).functions.decimals().call()

            amount_in_wei = int(amount_in * (10 ** decimals_in))

            # Get quote for min output
            amounts = router.functions.getAmountsOut(amount_in_wei, [path_in, path_out]).call()
            amount_out_min = int(amounts[1] * (100 - slippage) / 100)

            deadline = w3.eth.get_block("latest")["timestamp"] + 300  # 5 min

            # Build appropriate swap tx
            if is_eth_in:
                tx = router.functions.swapExactETHForTokens(
                    amount_out_min,
                    [path_in, path_out],
                    account.address,
                    deadline
                ).build_transaction({
                    "from": account.address,
                    "value": amount_in_wei,
                    "gas": 300000,
                    "gasPrice": w3.eth.gas_price,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": chain_config["chain_id"],
                })
            elif is_eth_out:
                # Approve first
                token_contract = w3.eth.contract(address=path_in, abi=ERC20_ABI)
                allowance = token_contract.functions.allowance(account.address, router_address).call()

                if allowance < amount_in_wei:
                    approve_tx = token_contract.functions.approve(
                        router_address,
                        2**256 - 1  # Max approval
                    ).build_transaction({
                        "from": account.address,
                        "gas": 100000,
                        "gasPrice": w3.eth.gas_price,
                        "nonce": w3.eth.get_transaction_count(account.address),
                        "chainId": chain_config["chain_id"],
                    })
                    signed_approve = account.sign_transaction(approve_tx)
                    w3.eth.send_raw_transaction(signed_approve.raw_transaction)
                    # Wait for approval
                    await asyncio.sleep(2)

                tx = router.functions.swapExactTokensForETH(
                    amount_in_wei,
                    amount_out_min,
                    [path_in, path_out],
                    account.address,
                    deadline
                ).build_transaction({
                    "from": account.address,
                    "gas": 300000,
                    "gasPrice": w3.eth.gas_price,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": chain_config["chain_id"],
                })
            else:
                # Token to token - need approval
                token_contract = w3.eth.contract(address=path_in, abi=ERC20_ABI)
                allowance = token_contract.functions.allowance(account.address, router_address).call()

                if allowance < amount_in_wei:
                    approve_tx = token_contract.functions.approve(
                        router_address,
                        2**256 - 1
                    ).build_transaction({
                        "from": account.address,
                        "gas": 100000,
                        "gasPrice": w3.eth.gas_price,
                        "nonce": w3.eth.get_transaction_count(account.address),
                        "chainId": chain_config["chain_id"],
                    })
                    signed_approve = account.sign_transaction(approve_tx)
                    w3.eth.send_raw_transaction(signed_approve.raw_transaction)
                    await asyncio.sleep(2)

                tx = router.functions.swapExactTokensForTokens(
                    amount_in_wei,
                    amount_out_min,
                    [path_in, path_out],
                    account.address,
                    deadline
                ).build_transaction({
                    "from": account.address,
                    "gas": 300000,
                    "gasPrice": w3.eth.gas_price,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": chain_config["chain_id"],
                })

            # Sign and send
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

            amount_out = amounts[1] / (10 ** decimals_out)

            return SkillResult(
                success=True,
                message=f"Swapped {amount_in} {token_in} for ~{amount_out:.6f} {token_out}",
                data={
                    "tx_hash": tx_hash.hex(),
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount_in": amount_in,
                    "expected_out": amount_out,
                    "slippage": slippage,
                    "chain": chain,
                    "explorer": f"{chain_config['explorer']}/tx/{tx_hash.hex()}",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Swap failed: {e}")

    async def _add_liquidity(self, params: Dict) -> SkillResult:
        """Add liquidity to DEX pool."""
        token_a = params.get("token_a", "").strip()
        token_b = params.get("token_b", "").strip()
        amount_a = params.get("amount_a", 0)
        amount_b = params.get("amount_b", 0)
        chain = params.get("chain") or self._chain

        if not token_a or not token_b:
            return SkillResult(success=False, message="token_a and token_b required")
        if amount_a <= 0 or amount_b <= 0:
            return SkillResult(success=False, message="Amounts must be positive")
        if not self._active_wallet:
            return SkillResult(success=False, message="No wallet. Create one first.")

        try:
            w3 = self._get_web3(chain)
            chain_config = CHAINS[chain]
            wallet = self._wallets[self._active_wallet]
            account = Account.from_key(wallet["private_key"])

            if not chain_config.get("dex_router"):
                return SkillResult(success=False, message=f"No DEX configured for {chain}")

            router_address = w3.to_checksum_address(chain_config["dex_router"])
            router = w3.eth.contract(address=router_address, abi=UNISWAP_ROUTER_ABI)

            is_eth_a = token_a.upper() == "ETH"
            weth = chain_config["weth"]

            if is_eth_a:
                # addLiquidityETH
                token_b_addr = w3.to_checksum_address(token_b)
                token_b_contract = w3.eth.contract(address=token_b_addr, abi=ERC20_ABI)
                decimals_b = token_b_contract.functions.decimals().call()

                amount_eth_wei = w3.to_wei(amount_a, "ether")
                amount_b_wei = int(amount_b * (10 ** decimals_b))

                # Approve token B
                allowance = token_b_contract.functions.allowance(account.address, router_address).call()
                if allowance < amount_b_wei:
                    approve_tx = token_b_contract.functions.approve(
                        router_address, 2**256 - 1
                    ).build_transaction({
                        "from": account.address,
                        "gas": 100000,
                        "gasPrice": w3.eth.gas_price,
                        "nonce": w3.eth.get_transaction_count(account.address),
                        "chainId": chain_config["chain_id"],
                    })
                    signed_approve = account.sign_transaction(approve_tx)
                    w3.eth.send_raw_transaction(signed_approve.raw_transaction)
                    await asyncio.sleep(2)

                deadline = w3.eth.get_block("latest")["timestamp"] + 300

                tx = router.functions.addLiquidityETH(
                    token_b_addr,
                    amount_b_wei,
                    int(amount_b_wei * 0.95),  # 5% slippage
                    int(amount_eth_wei * 0.95),
                    account.address,
                    deadline
                ).build_transaction({
                    "from": account.address,
                    "value": amount_eth_wei,
                    "gas": 500000,
                    "gasPrice": w3.eth.gas_price,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": chain_config["chain_id"],
                })
            else:
                # addLiquidity (token/token)
                token_a_addr = w3.to_checksum_address(token_a)
                token_b_addr = w3.to_checksum_address(token_b)

                token_a_contract = w3.eth.contract(address=token_a_addr, abi=ERC20_ABI)
                token_b_contract = w3.eth.contract(address=token_b_addr, abi=ERC20_ABI)

                decimals_a = token_a_contract.functions.decimals().call()
                decimals_b = token_b_contract.functions.decimals().call()

                amount_a_wei = int(amount_a * (10 ** decimals_a))
                amount_b_wei = int(amount_b * (10 ** decimals_b))

                # Approve both tokens
                for token_contract, amount_wei in [(token_a_contract, amount_a_wei), (token_b_contract, amount_b_wei)]:
                    allowance = token_contract.functions.allowance(account.address, router_address).call()
                    if allowance < amount_wei:
                        approve_tx = token_contract.functions.approve(
                            router_address, 2**256 - 1
                        ).build_transaction({
                            "from": account.address,
                            "gas": 100000,
                            "gasPrice": w3.eth.gas_price,
                            "nonce": w3.eth.get_transaction_count(account.address),
                            "chainId": chain_config["chain_id"],
                        })
                        signed_approve = account.sign_transaction(approve_tx)
                        w3.eth.send_raw_transaction(signed_approve.raw_transaction)
                        await asyncio.sleep(2)

                deadline = w3.eth.get_block("latest")["timestamp"] + 300

                tx = router.functions.addLiquidity(
                    token_a_addr,
                    token_b_addr,
                    amount_a_wei,
                    amount_b_wei,
                    int(amount_a_wei * 0.95),
                    int(amount_b_wei * 0.95),
                    account.address,
                    deadline
                ).build_transaction({
                    "from": account.address,
                    "gas": 500000,
                    "gasPrice": w3.eth.gas_price,
                    "nonce": w3.eth.get_transaction_count(account.address),
                    "chainId": chain_config["chain_id"],
                })

            # Sign and send
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

            return SkillResult(
                success=True,
                message=f"Added liquidity: {amount_a} {token_a} + {amount_b} {token_b}",
                data={
                    "tx_hash": tx_hash.hex(),
                    "token_a": token_a,
                    "token_b": token_b,
                    "amount_a": amount_a,
                    "amount_b": amount_b,
                    "chain": chain,
                    "explorer": f"{chain_config['explorer']}/tx/{tx_hash.hex()}",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Add liquidity failed: {e}")

    # === On-chain Data ===

    async def _get_token_info(self, params: Dict) -> SkillResult:
        """Get token information."""
        token = params.get("token", "").strip()
        chain = params.get("chain") or self._chain

        if not token:
            return SkillResult(success=False, message="Token address required")

        try:
            w3 = self._get_web3(chain)
            token = w3.to_checksum_address(token)
            contract = w3.eth.contract(address=token, abi=ERC20_ABI)

            name = contract.functions.name().call()
            symbol = contract.functions.symbol().call()
            decimals = contract.functions.decimals().call()
            total_supply = contract.functions.totalSupply().call()

            supply_formatted = total_supply / (10 ** decimals)

            return SkillResult(
                success=True,
                message=f"{name} ({symbol})",
                data={
                    "address": token,
                    "name": name,
                    "symbol": symbol,
                    "decimals": decimals,
                    "total_supply": supply_formatted,
                    "total_supply_raw": total_supply,
                    "chain": chain,
                    "explorer": f"{CHAINS[chain]['explorer']}/token/{token}",
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to get token info: {e}")

    async def _get_gas_price(self, params: Dict) -> SkillResult:
        """Get current gas price."""
        chain = params.get("chain") or self._chain

        try:
            w3 = self._get_web3(chain)
            gas_price = w3.eth.gas_price
            gas_gwei = w3.from_wei(gas_price, "gwei")

            return SkillResult(
                success=True,
                message=f"Gas price on {chain}: {gas_gwei:.2f} gwei",
                data={
                    "chain": chain,
                    "gas_price_wei": gas_price,
                    "gas_price_gwei": float(gas_gwei),
                }
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to get gas price: {e}")
