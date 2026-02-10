"""Kalshi client wrapper using kalshi-python SDK."""

from typing import Dict, List, Optional


PROD_HOST = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_HOST = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiClient:
    """Wrapper around the Kalshi trading API."""

    def __init__(self, api_key: str, private_key_pem: str, demo: bool = False):
        self.api_key = api_key
        self.private_key_pem = private_key_pem
        self.host = DEMO_HOST if demo else PROD_HOST
        self._client = None

    def _get_client(self):
        """Lazy-init the Kalshi SDK client."""
        if self._client is None:
            from kalshi_python import Configuration, KalshiClient as KC
            config = Configuration(host=self.host)
            config.api_key_id = self.api_key
            config.private_key_pem = self.private_key_pem
            self._client = KC(config)
        return self._client

    async def search_markets(self, query: str, limit: int = 10) -> List[Dict]:
        """Search open markets by keyword."""
        import httpx

        # Use REST directly for search since SDK may not support text search
        headers = {}
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{self.host}/markets", params={
                "limit": limit, "status": "open",
            }, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        q = query.lower()
        results = []
        for m in data.get("markets", []):
            title = (m.get("title") or m.get("subtitle") or "").lower()
            event = (m.get("event_ticker") or "").lower()
            if q in title or q in event:
                yes_price = m.get("yes_bid") or m.get("last_price")
                no_price = m.get("no_bid")
                results.append({
                    "platform": "kalshi",
                    "ticker": m.get("ticker"),
                    "question": m.get("title") or m.get("subtitle"),
                    "event_ticker": m.get("event_ticker"),
                    "yes_price": yes_price / 100 if yes_price else None,
                    "no_price": no_price / 100 if no_price else None,
                    "volume": m.get("volume"),
                    "open_interest": m.get("open_interest"),
                    "close_time": m.get("close_time"),
                })
        return results[:limit]

    async def get_orderbook(self, ticker: str, depth: int = 10) -> Dict:
        """Get orderbook for a market."""
        import httpx
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{self.host}/markets/{ticker}/orderbook", params={"depth": depth})
            resp.raise_for_status()
            book = resp.json().get("orderbook", {})

        # Convert cents to probability (0-1)
        def convert_levels(levels):
            return [{"price": l[0] / 100, "quantity": l[1]} for l in levels] if levels else []

        return {
            "platform": "kalshi",
            "ticker": ticker,
            "yes_bids": convert_levels(book.get("yes", [])),
            "no_bids": convert_levels(book.get("no", [])),
        }

    def buy(self, ticker: str, side: str, count: int, price: Optional[int] = None) -> Dict:
        """Place a buy order. side='yes'/'no', price in cents (1-99), count=number of contracts."""
        from kalshi_python.models import CreateOrderRequest

        client = self._get_client()
        order_type = "limit" if price is not None else "market"
        req = CreateOrderRequest(
            ticker=ticker,
            side=side.lower(),
            action="buy",
            count=count,
            type=order_type,
        )
        if price is not None:
            if side.lower() == "yes":
                req.yes_price = price
            else:
                req.no_price = price

        resp = client.create_order(req)
        order = resp.order if hasattr(resp, "order") else resp

        return {
            "platform": "kalshi",
            "order_id": getattr(order, "order_id", None),
            "status": getattr(order, "status", "submitted"),
            "ticker": ticker,
            "side": side,
            "count": count,
            "price_cents": price,
            "price": price / 100 if price else None,
        }

    def sell(self, ticker: str, side: str, count: int, price: Optional[int] = None) -> Dict:
        """Sell a position. side='yes'/'no', price in cents."""
        from kalshi_python.models import CreateOrderRequest

        client = self._get_client()
        order_type = "limit" if price is not None else "market"
        req = CreateOrderRequest(
            ticker=ticker,
            side=side.lower(),
            action="sell",
            count=count,
            type=order_type,
        )
        if price is not None:
            if side.lower() == "yes":
                req.yes_price = price
            else:
                req.no_price = price

        resp = client.create_order(req)
        order = resp.order if hasattr(resp, "order") else resp

        return {
            "platform": "kalshi",
            "order_id": getattr(order, "order_id", None),
            "status": getattr(order, "status", "submitted"),
            "ticker": ticker,
            "side": side,
            "count": count,
            "price_cents": price,
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        client = self._get_client()
        resp = client.get_positions(count_filter="position")
        positions_data = resp.market_positions if hasattr(resp, "market_positions") else []

        positions = []
        for p in positions_data:
            positions.append({
                "platform": "kalshi",
                "ticker": getattr(p, "ticker", ""),
                "market": getattr(p, "market_title", ""),
                "yes_count": getattr(p, "position", 0),
                "avg_price": getattr(p, "market_exposure", 0),
                "realized_pnl": getattr(p, "realized_pnl", 0),
            })
        return positions

    def get_balance(self) -> Dict:
        """Get account balance."""
        client = self._get_client()
        resp = client.get_balance()
        balance = resp.balance if hasattr(resp, "balance") else resp
        return {
            "platform": "kalshi",
            "balance_cents": balance,
            "balance_usd": balance / 100 if isinstance(balance, (int, float)) else None,
        }
