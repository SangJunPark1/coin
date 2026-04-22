from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .models import Candle


class MarketDataSource(Protocol):
    def get_recent_candles(self, market: str, count: int) -> list[Candle]:
        ...


class UpbitPublicDataSource:
    """Public Upbit candle reader. It never authenticates or places orders."""

    def __init__(self, unit_minutes: int = 1, timeout_seconds: int = 10) -> None:
        self.unit_minutes = unit_minutes
        self.timeout_seconds = timeout_seconds

    def get_recent_candles(self, market: str, count: int) -> list[Candle]:
        params = urllib.parse.urlencode({"market": market, "count": count})
        url = f"https://api.upbit.com/v1/candles/minutes/{self.unit_minutes}?{params}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        candles = []
        for row in payload:
            timestamp = datetime.fromisoformat(row["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
            candles.append(
                Candle(
                    market=market,
                    timestamp=timestamp,
                    open=float(row["opening_price"]),
                    high=float(row["high_price"]),
                    low=float(row["low_price"]),
                    close=float(row["trade_price"]),
                    volume=float(row["candle_acc_trade_volume"]),
                )
            )
        return list(reversed(candles))

    def get_top_krw_markets(self, count: int) -> list[str]:
        markets = self._get_krw_markets()
        tickers = []
        for chunk in chunks(markets, 80):
            params = urllib.parse.urlencode({"markets": ",".join(chunk)})
            url = f"https://api.upbit.com/v1/ticker?{params}"
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                tickers.extend(json.loads(response.read().decode("utf-8")))
        tickers.sort(key=lambda row: float(row.get("acc_trade_price_24h", 0.0)), reverse=True)
        return [str(row["market"]) for row in tickers[:count]]

    def _get_krw_markets(self) -> list[str]:
        url = "https://api.upbit.com/v1/market/all?isDetails=false"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted(str(row["market"]) for row in payload if str(row["market"]).startswith("KRW-"))


class SampleMarketDataSource:
    """Deterministic local data source for smoke tests and offline learning."""

    def __init__(self) -> None:
        self.tick = 0

    def get_recent_candles(self, market: str, count: int) -> list[Candle]:
        self.tick += 1
        base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        latest_index = self.tick + count
        candles = []
        for offset in range(count):
            idx = latest_index - count + offset
            trend = idx * 30_000
            cycle = math.sin(idx / 4.0) * 450_000
            price = 60_000_000 + trend + cycle
            open_price = price - 80_000
            high = price + 140_000
            low = price - 160_000
            volume = 1.0 + abs(math.sin(idx / 5.0)) * 2.0
            candles.append(
                Candle(
                    market=market,
                    timestamp=base_time - timedelta(minutes=count - offset),
                    open=open_price,
                    high=high,
                    low=low,
                    close=price,
                    volume=volume,
                )
            )
        return candles


def sleep_between_ticks(seconds: int, source_name: str) -> None:
    if source_name == "sample":
        return
    time.sleep(seconds)


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]
