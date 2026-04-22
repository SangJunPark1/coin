from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .config import StrategyConfig
from .data import UpbitPublicDataSource
from .models import Candle
from .strategy import recent_volatility_pct


@dataclass(frozen=True)
class DecisionContext:
    allows_entries: bool
    reason: str
    score_multiplier: float
    btc_momentum_pct: float
    btc_volatility_pct: float
    fear_greed_value: int | None = None
    fear_greed_label: str = "unknown"
    onchain_tx_count: int | None = None
    global_market_cap_change_pct: float | None = None
    btc_dominance_pct: float | None = None
    binance_btcusdt_change_pct: float | None = None
    binance_btcusdt_quote_volume: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_decision_context(data_source: UpbitPublicDataSource, config: StrategyConfig) -> DecisionContext:
    btc_candles = data_source.get_recent_candles("KRW-BTC", max(config.btc_long_window + 5, 30))
    btc_momentum = candle_momentum_pct(btc_candles, lookback=5)
    btc_volatility = recent_volatility_pct(btc_candles, lookback=20)
    fear_value, fear_label = fetch_fear_greed()
    tx_count = fetch_blockchain_tx_count()
    global_change, btc_dominance = fetch_coingecko_global()
    binance_change, binance_quote_volume = fetch_binance_btcusdt_24h()

    allows = True
    reasons = [f"BTC momentum {btc_momentum:.2f}%", f"BTC volatility {btc_volatility:.2f}%"]
    multiplier = 1.0

    if btc_momentum < config.min_btc_momentum_pct:
        allows = False
        multiplier *= 0.7
        reasons.append("BTC trend weak")
    if fear_value is not None:
        reasons.append(f"fear-greed {fear_value} {fear_label}")
        if fear_value >= 85:
            multiplier *= 0.85
            reasons.append("market greed overheated")
        elif fear_value <= 15:
            multiplier *= 0.8
            reasons.append("market fear stressed")
    if tx_count is not None:
        reasons.append(f"BTC tx {tx_count}")
    if global_change is not None:
        reasons.append(f"global cap 24h {global_change:.2f}%")
        if global_change < -3.0:
            multiplier *= 0.85
            reasons.append("global crypto market weak")
    if btc_dominance is not None:
        reasons.append(f"BTC dominance {btc_dominance:.2f}%")
    if binance_change is not None:
        reasons.append(f"Binance BTCUSDT 24h {binance_change:.2f}%")
        if binance_change < -2.0:
            multiplier *= 0.9
            reasons.append("global BTC pair weak")

    return DecisionContext(
        allows_entries=allows,
        reason="; ".join(reasons),
        score_multiplier=multiplier,
        btc_momentum_pct=btc_momentum,
        btc_volatility_pct=btc_volatility,
        fear_greed_value=fear_value,
        fear_greed_label=fear_label,
        onchain_tx_count=tx_count,
        global_market_cap_change_pct=global_change,
        btc_dominance_pct=btc_dominance,
        binance_btcusdt_change_pct=binance_change,
        binance_btcusdt_quote_volume=binance_quote_volume,
    )


def candle_momentum_pct(candles: list[Candle], lookback: int) -> float:
    if len(candles) <= lookback or candles[-1 - lookback].close <= 0:
        return 0.0
    return (candles[-1].close / candles[-1 - lookback].close - 1.0) * 100.0


def fetch_fear_greed() -> tuple[int | None, str]:
    try:
        payload = fetch_json("https://api.alternative.me/fng/?limit=1", timeout_seconds=4)
        first = payload.get("data", [{}])[0]
        return int(first["value"]), str(first.get("value_classification", "unknown"))
    except Exception:
        return None, "unavailable"


def fetch_blockchain_tx_count() -> int | None:
    try:
        payload = fetch_json("https://api.blockchain.info/stats", timeout_seconds=4)
        value = payload.get("n_tx")
        return int(value) if value is not None else None
    except Exception:
        return None


def fetch_coingecko_global() -> tuple[float | None, float | None]:
    try:
        payload = fetch_json("https://api.coingecko.com/api/v3/global", timeout_seconds=4)
        data = payload.get("data", {})
        change = data.get("market_cap_change_percentage_24h_usd")
        dominance = data.get("market_cap_percentage", {}).get("btc")
        return maybe_float(change), maybe_float(dominance)
    except Exception:
        return None, None


def fetch_binance_btcusdt_24h() -> tuple[float | None, float | None]:
    try:
        payload = fetch_json("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout_seconds=4)
        return maybe_float(payload.get("priceChangePercent")), maybe_float(payload.get("quoteVolume"))
    except Exception:
        return None, None


def fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "coin-paper-simulation/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
