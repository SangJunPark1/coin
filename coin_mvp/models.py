from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Candle:
    market: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    side: Side
    reason: str
    price: float
    confidence: float = 0.0
    size_fraction: float = 1.0


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0
    peak_price: float = 0.0
    partial_exit_taken: bool = False

    @property
    def is_open(self) -> bool:
        return self.qty > 0


@dataclass(frozen=True)
class OrderbookSnapshot:
    market: str
    timestamp: datetime
    best_bid_price: float
    best_bid_size: float
    best_ask_price: float
    best_ask_size: float
    total_bid_size: float
    total_ask_size: float

    @property
    def spread_bps(self) -> float:
        if self.best_bid_price <= 0 or self.best_ask_price <= 0:
            return 0.0
        mid = (self.best_bid_price + self.best_ask_price) / 2.0
        if mid <= 0:
            return 0.0
        return ((self.best_ask_price - self.best_bid_price) / mid) * 10_000.0

    @property
    def imbalance_ratio(self) -> float:
        if self.total_ask_size <= 0:
            return 999.0
        return self.total_bid_size / self.total_ask_size


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    market: str
    side: Side
    price: float
    qty: float
    fee: float
    cash_after: float
    position_qty_after: float
    realized_pnl: float
    reason: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
