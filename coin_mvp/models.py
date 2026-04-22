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


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.qty > 0


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
