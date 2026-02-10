"""Prediction Markets Skill — trade on Polymarket and Kalshi."""

import os
import asyncio
from typing import Dict, List
from singularity.skills.base import Skill, SkillResult, SkillManifest, SkillAction


def _a(name, desc, params=None, prob=0.85, dur=5):
    return SkillAction(name=name, description=desc, parameters=params or {},
                       estimated_cost=0, estimated_duration_seconds=dur, success_probability=prob)


class PredictionMarketsSkill(Skill):
    """Trade on prediction markets (Polymarket + Kalshi)."""

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._poly = None
        self._kalshi = None

    def _init_polymarket(self):
        if self._poly is not None:
            return self._poly
        pk = self.credentials.get("POLYMARKET_PRIVATE_KEY") or os.environ.get("POLYMARKET_PRIVATE_KEY")
        funder = self.credentials.get("POLYMARKET_FUNDER_ADDRESS") or os.environ.get("POLYMARKET_FUNDER_ADDRESS")
        if pk and funder:
            try:
                from .polymarket import PolymarketClient
                self._poly = PolymarketClient(pk, funder)
            except ImportError:
                print("[predictions] py-clob-client not installed, Polymarket disabled")
        return self._poly

    def _init_kalshi(self):
        if self._kalshi is not None:
            return self._kalshi
        api_key = self.credentials.get("KALSHI_API_KEY") or os.environ.get("KALSHI_API_KEY")
        pem = self.credentials.get("KALSHI_PRIVATE_KEY_PEM") or os.environ.get("KALSHI_PRIVATE_KEY_PEM")
        if api_key and pem:
            try:
                from .kalshi import KalshiClient
                demo = (os.environ.get("KALSHI_DEMO", "false").lower() == "true")
                self._kalshi = KalshiClient(api_key, pem, demo=demo)
            except ImportError:
                print("[predictions] kalshi-python not installed, Kalshi disabled")
        return self._kalshi

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="predictions",
            name="Prediction Markets",
            version="1.0.0",
            category="trading",
            description="Trade on prediction markets (Polymarket, Kalshi). Search markets, place bets, check positions.",
            required_credentials=[],
            install_cost=0,
            author="system",
            actions=[
                _a("search", "Search prediction markets by keyword", {
                    "query": {"type": "string", "required": True, "description": "Search query (e.g. 'bitcoin', 'election', 'AI')"},
                    "platform": {"type": "string", "required": False, "description": "Platform: 'polymarket', 'kalshi', or 'all' (default: all)"},
                    "limit": {"type": "integer", "required": False, "description": "Max results per platform (default: 5)"},
                }, prob=0.9, dur=5),

                _a("orderbook", "Get orderbook/prices for a specific market", {
                    "platform": {"type": "string", "required": True, "description": "'polymarket' or 'kalshi'"},
                    "market_id": {"type": "string", "required": True, "description": "Token ID (Polymarket) or ticker (Kalshi)"},
                }, prob=0.9, dur=3),

                _a("buy", "Buy YES or NO shares on a prediction market", {
                    "platform": {"type": "string", "required": True, "description": "'polymarket' or 'kalshi'"},
                    "market_id": {"type": "string", "required": True, "description": "Token ID (Polymarket) or ticker (Kalshi)"},
                    "side": {"type": "string", "required": True, "description": "'yes' or 'no'"},
                    "amount": {"type": "number", "required": True, "description": "Amount in USD (Polymarket) or number of contracts (Kalshi)"},
                    "price": {"type": "number", "required": False, "description": "Limit price 0-1 (probability). Omit for market order."},
                }, prob=0.8, dur=10),

                _a("sell", "Sell an existing prediction market position", {
                    "platform": {"type": "string", "required": True, "description": "'polymarket' or 'kalshi'"},
                    "market_id": {"type": "string", "required": True, "description": "Token ID (Polymarket) or ticker (Kalshi)"},
                    "side": {"type": "string", "required": True, "description": "'yes' or 'no'"},
                    "amount": {"type": "number", "required": True, "description": "Amount to sell"},
                    "price": {"type": "number", "required": False, "description": "Limit price 0-1. Omit for market order."},
                }, prob=0.8, dur=10),

                _a("positions", "Check current open positions", {
                    "platform": {"type": "string", "required": False, "description": "'polymarket', 'kalshi', or 'all' (default: all)"},
                }, prob=0.9, dur=5),

                _a("balance", "Check trading account balance", {
                    "platform": {"type": "string", "required": False, "description": "'polymarket', 'kalshi', or 'all' (default: all)"},
                }, prob=0.9, dur=3),
            ],
        )

    def check_credentials(self) -> bool:
        """At least one platform must be configured."""
        poly_ok = bool(
            (self.credentials.get("POLYMARKET_PRIVATE_KEY") or os.environ.get("POLYMARKET_PRIVATE_KEY"))
            and (self.credentials.get("POLYMARKET_FUNDER_ADDRESS") or os.environ.get("POLYMARKET_FUNDER_ADDRESS"))
        )
        kalshi_ok = bool(
            (self.credentials.get("KALSHI_API_KEY") or os.environ.get("KALSHI_API_KEY"))
            and (self.credentials.get("KALSHI_PRIVATE_KEY_PEM") or os.environ.get("KALSHI_PRIVATE_KEY_PEM"))
        )
        return poly_ok or kalshi_ok

    def _available_platforms(self) -> List[str]:
        platforms = []
        if self._init_polymarket():
            platforms.append("polymarket")
        if self._init_kalshi():
            platforms.append("kalshi")
        return platforms

    async def execute(self, action: str, params: Dict) -> SkillResult:
        try:
            if action == "search":
                return await self._search(params)
            elif action == "orderbook":
                return await self._orderbook(params)
            elif action == "buy":
                return await self._buy(params)
            elif action == "sell":
                return await self._sell(params)
            elif action == "positions":
                return await self._positions(params)
            elif action == "balance":
                return await self._balance(params)
            else:
                return SkillResult(success=False, message=f"Unknown action: {action}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error: {e}")

    async def _search(self, params: Dict) -> SkillResult:
        query = params.get("query", "")
        if not query:
            return SkillResult(success=False, message="query is required")

        platform = params.get("platform", "all").lower()
        limit = int(params.get("limit", 5))
        results = []

        if platform in ("all", "polymarket") and self._init_polymarket():
            try:
                results.extend(await self._poly.search_markets(query, limit))
            except Exception as e:
                results.append({"platform": "polymarket", "error": str(e)})

        if platform in ("all", "kalshi") and self._init_kalshi():
            try:
                results.extend(await self._kalshi.search_markets(query, limit))
            except Exception as e:
                results.append({"platform": "kalshi", "error": str(e)})

        if not results:
            available = self._available_platforms()
            if not available:
                return SkillResult(success=False, message="No prediction market platforms configured. Set POLYMARKET_PRIVATE_KEY/POLYMARKET_FUNDER_ADDRESS or KALSHI_API_KEY/KALSHI_PRIVATE_KEY_PEM.")
            return SkillResult(success=True, message=f"No markets found for '{query}'", data={"results": []})

        summary = "\n".join(
            f"- [{r.get('platform')}] {r.get('question', r.get('ticker', ''))}"
            + (f" (YES: {r['yes_price']:.0%})" if r.get('yes_price') else "")
            + (f" | id: {r.get('ticker') or r.get('yes_token', '')}" if r.get('ticker') or r.get('yes_token') else "")
            for r in results if not r.get("error")
        )
        return SkillResult(success=True, message=f"Found {len(results)} markets:\n{summary}", data={"results": results})

    async def _orderbook(self, params: Dict) -> SkillResult:
        platform = params.get("platform", "").lower()
        market_id = params.get("market_id", "")
        if not platform or not market_id:
            return SkillResult(success=False, message="platform and market_id are required")

        if platform == "polymarket" and self._init_polymarket():
            book = await self._poly.get_orderbook(market_id)
        elif platform == "kalshi" and self._init_kalshi():
            book = await self._kalshi.get_orderbook(market_id)
        else:
            return SkillResult(success=False, message=f"Platform '{platform}' not available")

        return SkillResult(success=True, message=f"Orderbook for {market_id}", data=book)

    async def _buy(self, params: Dict) -> SkillResult:
        platform = params.get("platform", "").lower()
        market_id = params.get("market_id", "")
        side = params.get("side", "yes").lower()
        amount = float(params.get("amount", 0))
        price = params.get("price")
        if price is not None:
            price = float(price)

        if not platform or not market_id or amount <= 0:
            return SkillResult(success=False, message="platform, market_id, and amount > 0 are required")
        if side not in ("yes", "no"):
            return SkillResult(success=False, message="side must be 'yes' or 'no'")

        if platform == "polymarket" and self._init_polymarket():
            result = self._poly.buy(market_id, side, amount, price)
        elif platform == "kalshi" and self._init_kalshi():
            kalshi_price = int(price * 100) if price is not None else None
            result = self._kalshi.buy(market_id, side, int(amount), kalshi_price)
        else:
            return SkillResult(success=False, message=f"Platform '{platform}' not available")

        return SkillResult(
            success=True,
            message=f"Placed {side.upper()} buy on {platform}: {amount} @ {price or 'market'}",
            data=result,
            cost=amount if platform == "polymarket" else (amount * (price or 0.5)),
        )

    async def _sell(self, params: Dict) -> SkillResult:
        platform = params.get("platform", "").lower()
        market_id = params.get("market_id", "")
        side = params.get("side", "yes").lower()
        amount = float(params.get("amount", 0))
        price = params.get("price")
        if price is not None:
            price = float(price)

        if not platform or not market_id or amount <= 0:
            return SkillResult(success=False, message="platform, market_id, and amount > 0 are required")

        if platform == "polymarket" and self._init_polymarket():
            result = self._poly.sell(market_id, side, amount, price)
        elif platform == "kalshi" and self._init_kalshi():
            kalshi_price = int(price * 100) if price is not None else None
            result = self._kalshi.sell(market_id, side, int(amount), kalshi_price)
        else:
            return SkillResult(success=False, message=f"Platform '{platform}' not available")

        return SkillResult(
            success=True,
            message=f"Placed {side.upper()} sell on {platform}: {amount} @ {price or 'market'}",
            data=result,
            revenue=amount if platform == "polymarket" else (amount * (price or 0.5)),
        )

    async def _positions(self, params: Dict) -> SkillResult:
        platform = params.get("platform", "all").lower()
        positions = []

        if platform in ("all", "polymarket") and self._init_polymarket():
            try:
                positions.extend(await self._poly.get_positions())
            except Exception as e:
                positions.append({"platform": "polymarket", "error": str(e)})

        if platform in ("all", "kalshi") and self._init_kalshi():
            try:
                positions.extend(self._kalshi.get_positions())
            except Exception as e:
                positions.append({"platform": "kalshi", "error": str(e)})

        if not positions:
            return SkillResult(success=True, message="No open positions", data={"positions": []})

        summary = "\n".join(
            f"- [{p.get('platform')}] {p.get('market', p.get('ticker', ''))}: {p.get('size', p.get('yes_count', ''))}"
            for p in positions if not p.get("error")
        )
        return SkillResult(success=True, message=f"Positions:\n{summary}", data={"positions": positions})

    async def _balance(self, params: Dict) -> SkillResult:
        platform = params.get("platform", "all").lower()
        balances = []

        if platform in ("all", "polymarket") and self._init_polymarket():
            try:
                balances.append(await self._poly.get_balance())
            except Exception as e:
                balances.append({"platform": "polymarket", "error": str(e)})

        if platform in ("all", "kalshi") and self._init_kalshi():
            try:
                balances.append(self._kalshi.get_balance())
            except Exception as e:
                balances.append({"platform": "kalshi", "error": str(e)})

        return SkillResult(success=True, message="Balances retrieved", data={"balances": balances})
