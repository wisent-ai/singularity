"""Polymarket client wrapper using py-clob-client and Gamma API."""

import httpx
from typing import Dict, List, Optional

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


class PolymarketClient:
    """Wrapper around Polymarket CLOB + Gamma APIs."""

    def __init__(self, private_key: str, funder_address: str):
        self.private_key = private_key
        self.funder_address = funder_address
        self._clob = None

    def _get_clob(self):
        """Lazy-init the CLOB client (requires py-clob-client)."""
        if self._clob is None:
            from py_clob_client.client import ClobClient
            self._clob = ClobClient(
                CLOB_HOST,
                key=self.private_key,
                chain_id=CHAIN_ID,
                signature_type=0,
                funder=self.funder_address,
            )
            self._clob.set_api_creds(self._clob.create_or_derive_api_creds())
        return self._clob

    async def search_markets(self, query: str, limit: int = 10) -> List[Dict]:
        """Search markets via Gamma API."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{GAMMA_API}/markets", params={
                "limit": limit, "active": "true",
            })
            resp.raise_for_status()
            markets = resp.json()

        q = query.lower()
        results = []
        for m in markets:
            title = (m.get("question") or m.get("title") or "").lower()
            desc = (m.get("description") or "").lower()
            if q in title or q in desc:
                tokens = m.get("clobTokenIds") or m.get("tokens", [])
                yes_token = tokens[0] if isinstance(tokens, list) and tokens else None
                no_token = tokens[1] if isinstance(tokens, list) and len(tokens) > 1 else None
                results.append({
                    "platform": "polymarket",
                    "id": m.get("conditionId") or m.get("id"),
                    "question": m.get("question") or m.get("title"),
                    "yes_token": yes_token,
                    "no_token": no_token,
                    "slug": m.get("slug"),
                    "volume": m.get("volume"),
                    "liquidity": m.get("liquidity"),
                    "end_date": m.get("endDate"),
                    "yes_price": None,
                    "no_price": None,
                })
        return results[:limit]

    async def get_orderbook(self, token_id: str) -> Dict:
        """Get orderbook for a token from CLOB API."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{CLOB_HOST}/book", params={"token_id": token_id})
            resp.raise_for_status()
            book = resp.json()

        midpoint = None
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{CLOB_HOST}/midpoint", params={"token_id": token_id})
                if resp.status_code == 200:
                    midpoint = float(resp.json().get("mid", 0))
        except Exception:
            pass

        return {
            "platform": "polymarket",
            "token_id": token_id,
            "midpoint": midpoint,
            "bids": book.get("bids", [])[:5],
            "asks": book.get("asks", [])[:5],
        }

    def buy(self, token_id: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place a buy order. side='YES'/'NO', amount in USDC, price 0-1."""
        from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        clob = self._get_clob()

        if price is not None:
            order = OrderArgs(token_id=token_id, price=price, size=amount, side=BUY)
            signed = clob.create_order(order)
            resp = clob.post_order(signed, OrderType.GTC)
        else:
            order = MarketOrderArgs(token_id=token_id, amount=amount, side=BUY)
            signed = clob.create_market_order(order)
            resp = clob.post_order(signed, OrderType.FOK)

        return {
            "platform": "polymarket",
            "order_id": resp.get("orderID") or resp.get("id"),
            "status": resp.get("status", "submitted"),
            "token_id": token_id,
            "side": side,
            "amount": amount,
            "price": price,
        }

    def sell(self, token_id: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place a sell order."""
        from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        clob = self._get_clob()

        if price is not None:
            order = OrderArgs(token_id=token_id, price=price, size=amount, side=SELL)
            signed = clob.create_order(order)
            resp = clob.post_order(signed, OrderType.GTC)
        else:
            order = MarketOrderArgs(token_id=token_id, amount=amount, side=SELL)
            signed = clob.create_market_order(order)
            resp = clob.post_order(signed, OrderType.FOK)

        return {
            "platform": "polymarket",
            "order_id": resp.get("orderID") or resp.get("id"),
            "status": resp.get("status", "submitted"),
            "token_id": token_id,
            "side": side,
            "amount": amount,
            "price": price,
        }

    async def get_positions(self) -> List[Dict]:
        """Get current positions from Data API."""
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{DATA_API}/positions", params={
                "user": self.funder_address,
                "sizeThreshold": 0.01,
            })
            resp.raise_for_status()
            data = resp.json()

        positions = []
        for p in (data if isinstance(data, list) else data.get("positions", [])):
            positions.append({
                "platform": "polymarket",
                "market": p.get("title") or p.get("market", ""),
                "token_id": p.get("asset"),
                "size": p.get("size"),
                "avg_price": p.get("avgPrice"),
                "current_value": p.get("currentValue"),
                "pnl": p.get("pnl"),
            })
        return positions

    async def get_balance(self) -> Dict:
        """Get USDC balance info."""
        try:
            clob = self._get_clob()
            # The CLOB client doesn't have a direct balance method,
            # but we can check via the allowances/collateral endpoint
            return {"platform": "polymarket", "address": self.funder_address, "note": "Check USDC balance on Polygon"}
        except Exception as e:
            return {"platform": "polymarket", "error": str(e)}
