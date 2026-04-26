from __future__ import annotations

import math

from .config import StrategyConfig
from .models import Candle, Position, Side, Signal


class MovingAverageStrategy:
    """Small explainable strategy for MVP paper trading.

    Buy only when the short moving average is above the long moving average
    and the latest price is above the long moving average. Sell by take-profit,
    stop-loss, or trend break.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def generate(self, candles: list[Candle], position: Position) -> Signal:
        if len(candles) < self.config.long_window:
            latest_price = candles[-1].close if candles else 0.0
            return Signal(Side.HOLD, "not enough candles", latest_price)

        closes = [c.close for c in candles]
        latest_price = closes[-1]
        short_ma = mean(closes[-self.config.short_window :])
        long_ma = mean(closes[-self.config.long_window :])

        if position.is_open:
            pnl_pct = (latest_price / position.avg_price - 1.0) * 100.0
            if pnl_pct >= self.config.take_profit_pct:
                return Signal(Side.SELL, f"take profit reached: {pnl_pct:.2f}%", latest_price, 0.8)
            if pnl_pct <= -self.config.stop_loss_pct:
                return Signal(Side.SELL, f"stop loss reached: {pnl_pct:.2f}%", latest_price, 0.9)
            if short_ma < long_ma and latest_price < long_ma:
                return Signal(Side.SELL, "trend break", latest_price, 0.6)
            return Signal(Side.HOLD, "position open, no exit condition", latest_price, 0.2)

        entry_ok, entry_reason, confidence = self._entry_quality(candles, short_ma, long_ma)
        if short_ma > long_ma and latest_price > long_ma and entry_ok:
            return Signal(Side.BUY, entry_reason, latest_price, confidence)
        range_signal, range_reason = self._range_rebound_signal(candles, short_ma, long_ma)
        if range_signal is not None:
            return range_signal
        if short_ma <= long_ma and latest_price <= long_ma:
            return Signal(
                Side.HOLD,
                self._combine_hold_reasons(
                    f"ma alignment failed: short {short_ma:.3f} <= long {long_ma:.3f}; price below long MA {latest_price:.3f} <= {long_ma:.3f}",
                    range_reason,
                ),
                latest_price,
                0.1,
            )
        if short_ma <= long_ma:
            return Signal(
                Side.HOLD,
                self._combine_hold_reasons(
                    f"ma alignment failed: short {short_ma:.3f} <= long {long_ma:.3f}",
                    range_reason,
                ),
                latest_price,
                0.1,
            )
        if latest_price <= long_ma:
            return Signal(
                Side.HOLD,
                self._combine_hold_reasons(
                    f"price below long MA: {latest_price:.3f} <= {long_ma:.3f}",
                    range_reason,
                ),
                latest_price,
                0.1,
            )
        return Signal(Side.HOLD, self._combine_hold_reasons(entry_reason, range_reason), latest_price, 0.1)

    def _entry_quality(self, candles: list[Candle], short_ma: float, long_ma: float) -> tuple[bool, str, float]:
        closes = [c.close for c in candles]
        latest_price = closes[-1]
        lookback = min(5, len(closes) - 1)
        recent_momentum_pct = ((latest_price / closes[-1 - lookback]) - 1.0) * 100.0 if lookback and closes[-1 - lookback] else 0.0
        volume_ratio = latest_volume_ratio(candles, lookback=10)
        ma_distance_pct = ((latest_price / long_ma) - 1.0) * 100.0 if long_ma else 0.0
        rsi = calculate_rsi(closes, self.config.rsi_period)
        long_trend_ema = calculate_ema(closes, self.config.long_trend_ema_window)

        if self.config.long_trend_ema_window and long_trend_ema is None:
            return False, "long trend filter blocked: not enough candles", 0.2
        if long_trend_ema is not None and latest_price < long_trend_ema:
            return False, f"long trend filter blocked: price below EMA{self.config.long_trend_ema_window}", 0.2

        if recent_momentum_pct < self.config.min_recent_momentum_pct:
            return False, f"weak recent momentum: {recent_momentum_pct:.2f}%", 0.2
        if recent_momentum_pct > self.config.max_recent_momentum_pct or ma_distance_pct > self.config.max_ma_distance_pct:
            return False, f"overextended: momentum {recent_momentum_pct:.2f}%, distance {ma_distance_pct:.2f}%", 0.2
        expected_upside_pct = estimate_expected_upside_pct(candles, self.config.target_upside_pct)
        if expected_upside_pct < self.config.min_expected_upside_pct:
            return False, f"insufficient upside: expected {expected_upside_pct:.2f}%", 0.2
        if rsi is not None and rsi > self.config.max_entry_rsi:
            return False, f"overextended: RSI {rsi:.1f}", 0.2
        if volume_ratio < self.config.min_volume_ratio:
            return False, f"thin volume: {volume_ratio:.2f}x", 0.2

        trend_strength = ((short_ma / long_ma) - 1.0) * 100.0 if long_ma else 0.0
        confidence = min(0.95, 0.55 + (trend_strength / 20.0) + min(volume_ratio - 1.0, 0.3) + min(expected_upside_pct / 20.0, 0.1))
        rsi_text = f"; RSI {rsi:.1f}" if rsi is not None else ""
        ema_text = f"; above EMA{self.config.long_trend_ema_window}" if long_trend_ema is not None else ""
        return True, f"uptrend filter passed; momentum {recent_momentum_pct:.2f}%; volume {volume_ratio:.2f}x; expected upside {expected_upside_pct:.2f}%{rsi_text}{ema_text}", confidence

    def _range_rebound_signal(self, candles: list[Candle], short_ma: float, long_ma: float) -> tuple[Signal | None, str]:
        if not self.config.enable_range_rebound:
            return None, "range rebound disabled"
        closes = [c.close for c in candles]
        latest_price = closes[-1]
        previous_price = closes[-2] if len(closes) >= 2 else latest_price
        lookback = min(self.config.range_rebound_lookback, len(candles))
        recent = candles[-lookback:]
        recent_low = min(candle.low for candle in recent)
        if recent_low <= 0:
            return None, "range rebound blocked: invalid recent low"
        bounce_pct = ((latest_price / previous_price) - 1.0) * 100.0 if previous_price > 0 else 0.0
        distance_from_low_pct = ((latest_price / recent_low) - 1.0) * 100.0
        volume_ratio = latest_volume_ratio(candles, lookback=10)
        expected_upside_pct = estimate_expected_upside_pct(candles, self.config.target_upside_pct)
        rsi = calculate_rsi(closes, self.config.rsi_period)
        long_trend_ema = calculate_ema(closes, self.config.long_trend_ema_window)
        ema_gap_pct = ((long_trend_ema / latest_price) - 1.0) * 100.0 if long_trend_ema and latest_price > 0 else 0.0

        if bounce_pct < self.config.range_rebound_min_bounce_pct:
            return None, f"range rebound blocked: weak bounce {bounce_pct:.2f}%"
        if distance_from_low_pct > self.config.range_rebound_max_distance_from_low_pct:
            return None, f"range rebound blocked: too far from low {distance_from_low_pct:.2f}%"
        if volume_ratio < self.config.range_rebound_min_volume_ratio:
            return None, f"range rebound blocked: thin rebound volume {volume_ratio:.2f}x"
        if expected_upside_pct < self.config.range_rebound_min_expected_upside_pct:
            return None, f"range rebound blocked: upside {expected_upside_pct:.2f}%"
        if rsi is None or rsi < self.config.range_rebound_min_rsi or rsi > self.config.range_rebound_max_entry_rsi:
            if rsi is None:
                return None, "range rebound blocked: RSI unavailable"
            return None, f"range rebound blocked: RSI {rsi:.1f}"
        if long_trend_ema is not None and ema_gap_pct > self.config.range_rebound_max_ema_gap_pct:
            return None, f"range rebound blocked: EMA gap {ema_gap_pct:.2f}%"
        if latest_price < short_ma:
            return None, f"range rebound blocked: price below short MA {latest_price:.3f} < {short_ma:.3f}"
        confidence = min(
            0.78,
            0.48
            + min(bounce_pct / 2.0, 0.08)
            + min(volume_ratio - 0.9, 0.12)
            + min(expected_upside_pct / 25.0, 0.08),
        )
        ema_text = f"; EMA gap {ema_gap_pct:.2f}%" if long_trend_ema is not None else ""
        rsi_text = f"; RSI {rsi:.1f}" if rsi is not None else ""
        return Signal(
            Side.BUY,
            f"range rebound setup: bounce {bounce_pct:.2f}%; distance from low {distance_from_low_pct:.2f}%; volume {volume_ratio:.2f}x; expected upside {expected_upside_pct:.2f}%{rsi_text}{ema_text}",
            latest_price,
            confidence,
        ), ""

    def _combine_hold_reasons(self, primary_reason: str, secondary_reason: str) -> str:
        if not secondary_reason:
            return primary_reason
        if secondary_reason in primary_reason:
            return primary_reason
        return f"{primary_reason}; {secondary_reason}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def latest_volume_ratio(candles: list[Candle], lookback: int) -> float:
    if len(candles) < 2:
        return 1.0
    history = candles[-(lookback + 1) : -1]
    if not history:
        return 1.0
    average_volume = mean([c.volume for c in history])
    if average_volume <= 0:
        return 1.0
    return candles[-1].volume / average_volume


def calculate_rsi(closes: list[float], period: int) -> float | None:
    if len(closes) <= period:
        return None
    changes = [closes[index] - closes[index - 1] for index in range(len(closes) - period, len(closes))]
    gains = [change for change in changes if change > 0]
    losses = [-change for change in changes if change < 0]
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def calculate_ema(closes: list[float], period: int) -> float | None:
    if period <= 0:
        return None
    if len(closes) < period:
        return None
    window = closes[-period:]
    multiplier = 2.0 / (period + 1.0)
    ema = window[0]
    for close in window[1:]:
        ema = (close * multiplier) + (ema * (1.0 - multiplier))
    return ema


def recent_volatility_pct(candles: list[Candle], lookback: int = 20) -> float:
    closes = [c.close for c in candles[-(lookback + 1) :]]
    if len(closes) < 3:
        return 0.0
    returns = [math.log(closes[index] / closes[index - 1]) for index in range(1, len(closes)) if closes[index - 1] > 0]
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(len(returns)) * 100.0


def estimate_expected_upside_pct(candles: list[Candle], target_upside_pct: float, lookback: int = 30) -> float:
    recent = candles[-lookback:]
    if not recent:
        return 0.0
    latest = recent[-1].close
    if latest <= 0:
        return 0.0
    recent_high = max(candle.high for candle in recent)
    recent_low = min(candle.low for candle in recent)
    upside_to_high = max(0.0, (recent_high / latest - 1.0) * 100.0)
    recent_range = max(0.0, (recent_high / recent_low - 1.0) * 100.0) if recent_low > 0 else 0.0
    volatility_budget = recent_volatility_pct(candles, lookback=min(20, max(3, len(recent) - 1))) * 1.5
    expected = max(upside_to_high, min(recent_range * 0.5, volatility_budget))
    return min(target_upside_pct, expected)


def estimate_expected_downside_pct(candles: list[Candle], stop_loss_pct: float, volatility_multiplier: float = 1.1, lookback: int = 20) -> float:
    recent = candles[-lookback:]
    if not recent:
        return stop_loss_pct
    latest = recent[-1].close
    if latest <= 0:
        return stop_loss_pct
    recent_low = min(candle.low for candle in recent)
    pullback_to_low = max(0.0, (latest / recent_low - 1.0) * 100.0) if recent_low > 0 else stop_loss_pct
    volatility_budget = recent_volatility_pct(candles, lookback=min(lookback, max(3, len(recent) - 1))) * max(0.5, volatility_multiplier)
    downside = min(max(stop_loss_pct, volatility_budget), max(stop_loss_pct, pullback_to_low))
    return max(stop_loss_pct, downside)


def market_breadth_ratio(candles_by_market: dict[str, list[Candle]], short_window: int, long_window: int, ema_window: int) -> float:
    if not candles_by_market:
        return 0.0
    passing = 0
    total = 0
    for candles in candles_by_market.values():
        if len(candles) < max(long_window, ema_window, short_window):
            continue
        closes = [c.close for c in candles]
        latest = closes[-1]
        short_ma = mean(closes[-short_window:])
        long_ma = mean(closes[-long_window:])
        ema = calculate_ema(closes, ema_window)
        total += 1
        if short_ma > long_ma and latest > long_ma and (ema is None or latest > ema):
            passing += 1
    if total == 0:
        return 0.0
    return passing / total


def volatility_adjusted_position_fraction(candles: list[Candle], config: StrategyConfig) -> float:
    realized_volatility = recent_volatility_pct(candles)
    if realized_volatility <= 0:
        return config.position_fraction
    multiplier = min(1.0, config.target_recent_volatility_pct / realized_volatility)
    multiplier = max(config.min_volatility_position_fraction, multiplier)
    return config.position_fraction * multiplier


def btc_regime_allows_entries(candles: list[Candle], config: StrategyConfig) -> tuple[bool, str, float]:
    required = max(config.btc_long_window, config.btc_short_window) + 1
    if len(candles) < required:
        return False, "btc regime blocked: not enough candles", 0.0

    closes = [c.close for c in candles]
    latest_price = closes[-1]
    short_ma = mean(closes[-config.btc_short_window :])
    long_ma = mean(closes[-config.btc_long_window :])
    momentum_pct = ((latest_price / closes[-1 - min(5, len(closes) - 1)]) - 1.0) * 100.0

    if short_ma < long_ma and momentum_pct < config.min_btc_momentum_pct:
        return False, f"btc regime blocked: trend weak {momentum_pct:.2f}%", momentum_pct
    return True, f"btc regime ok: momentum {momentum_pct:.2f}%", momentum_pct


def required_candle_count(config: StrategyConfig) -> int:
    return max(
        config.long_window + 5,
        config.rsi_period + 1,
        21,
        config.long_trend_ema_window,
        config.btc_long_window + 5,
    )
