from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    short_window: int
    long_window: int
    take_profit_pct: float
    stop_loss_pct: float
    position_fraction: float
    min_recent_momentum_pct: float = 0.05
    max_recent_momentum_pct: float = 4.0
    min_volume_ratio: float = 1.05
    max_ma_distance_pct: float = 6.0
    rsi_period: int = 14
    max_entry_rsi: float = 72.0
    target_recent_volatility_pct: float = 1.5
    min_volatility_position_fraction: float = 0.4
    long_trend_ema_window: int = 200
    time_stop_ticks: int = 12
    time_stop_min_pnl_pct: float = 0.0
    btc_short_window: int = 5
    btc_long_window: int = 20
    min_btc_momentum_pct: float = -0.7
    min_expected_upside_pct: float = 0.8
    target_upside_pct: float = 3.0
    blocked_entry_hours_kst: tuple[int, ...] = ()
    reentry_cooldown_ticks: int = 6
    stop_volatility_multiplier: float = 1.1
    breakeven_trigger_pct: float = 0.8
    partial_take_profit_pct: float = 1.0
    partial_take_profit_fraction: float = 0.5
    trailing_stop_pct: float = 0.8
    min_orderbook_imbalance: float = 0.95
    max_orderbook_spread_bps: float = 12.0
    min_market_breadth_ratio: float = 0.35
    min_price_krw: float = 100.0
    max_recent_stopouts_per_market: int = 2
    stopout_lookback_ticks: int = 144
    five_minute_short_window: int = 3
    five_minute_long_window: int = 6
    min_five_minute_momentum_pct: float = 0.0
    five_minute_trend_tolerance_pct: float = 0.2
    enable_range_rebound: bool = True
    range_rebound_lookback: int = 12
    range_rebound_max_distance_from_low_pct: float = 3.0
    range_rebound_max_ema_gap_pct: float = 3.5
    range_rebound_min_bounce_pct: float = 0.15
    range_rebound_min_volume_ratio: float = 0.9
    range_rebound_min_expected_upside_pct: float = 1.0
    range_rebound_min_rsi: float = 35.0
    range_rebound_max_entry_rsi: float = 55.0
    range_rebound_trend_break_grace_ticks: int = 2


@dataclass(frozen=True)
class RiskConfig:
    daily_profit_target_pct: float
    daily_loss_limit_pct: float
    max_entries_per_day: int
    max_position_fraction: float
    max_consecutive_losses: int
    halt_cooldown_ticks: int = 6
    consecutive_loss_cooldown_ticks: int = 12
    max_expected_downside_to_upside_ratio: float = 1.0


@dataclass(frozen=True)
class AiDecisionConfig:
    enabled: bool = True
    provider: str = "local"
    min_confidence: float = 0.55
    openai_model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class PathConfig:
    trade_journal: Path
    event_log: Path
    state_file: Path


@dataclass(frozen=True)
class AppConfig:
    mode: str
    market: str
    poll_seconds: int
    starting_cash: float
    fee_rate: float
    slippage_bps: float
    strategy: StrategyConfig
    risk: RiskConfig
    ai_decision: AiDecisionConfig
    paths: PathConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    strategy = _require_mapping(raw, "strategy")
    risk = _require_mapping(raw, "risk")
    ai_decision = raw.get("ai_decision", {})
    if not isinstance(ai_decision, dict):
        raise ValueError("ai_decision must be an object when provided.")
    paths = _require_mapping(raw, "paths")

    app = AppConfig(
        mode=str(raw.get("mode", "paper")).lower(),
        market=str(raw["market"]),
        poll_seconds=int(raw.get("poll_seconds", 15)),
        starting_cash=float(raw["starting_cash"]),
        fee_rate=float(raw.get("fee_rate", 0.0005)),
        slippage_bps=float(raw.get("slippage_bps", 5)),
        strategy=StrategyConfig(
            short_window=int(strategy["short_window"]),
            long_window=int(strategy["long_window"]),
            take_profit_pct=float(strategy["take_profit_pct"]),
            stop_loss_pct=float(strategy["stop_loss_pct"]),
            position_fraction=float(strategy["position_fraction"]),
            min_recent_momentum_pct=float(strategy.get("min_recent_momentum_pct", 0.05)),
            max_recent_momentum_pct=float(strategy.get("max_recent_momentum_pct", 4.0)),
            min_volume_ratio=float(strategy.get("min_volume_ratio", 1.05)),
            max_ma_distance_pct=float(strategy.get("max_ma_distance_pct", 6.0)),
            rsi_period=int(strategy.get("rsi_period", 14)),
            max_entry_rsi=float(strategy.get("max_entry_rsi", 72.0)),
            target_recent_volatility_pct=float(strategy.get("target_recent_volatility_pct", 1.5)),
            min_volatility_position_fraction=float(strategy.get("min_volatility_position_fraction", 0.4)),
            long_trend_ema_window=int(strategy.get("long_trend_ema_window", 200)),
            time_stop_ticks=int(strategy.get("time_stop_ticks", 12)),
            time_stop_min_pnl_pct=float(strategy.get("time_stop_min_pnl_pct", 0.0)),
            btc_short_window=int(strategy.get("btc_short_window", strategy.get("short_window", 5))),
            btc_long_window=int(strategy.get("btc_long_window", strategy.get("long_window", 20))),
            min_btc_momentum_pct=float(strategy.get("min_btc_momentum_pct", -0.7)),
            min_expected_upside_pct=float(strategy.get("min_expected_upside_pct", 0.8)),
            target_upside_pct=float(strategy.get("target_upside_pct", risk.get("daily_profit_target_pct", 3.0))),
            blocked_entry_hours_kst=tuple(int(value) for value in strategy.get("blocked_entry_hours_kst", [])),
            reentry_cooldown_ticks=int(strategy.get("reentry_cooldown_ticks", 6)),
            stop_volatility_multiplier=float(strategy.get("stop_volatility_multiplier", 1.1)),
            breakeven_trigger_pct=float(strategy.get("breakeven_trigger_pct", 0.8)),
            partial_take_profit_pct=float(strategy.get("partial_take_profit_pct", 1.0)),
            partial_take_profit_fraction=float(strategy.get("partial_take_profit_fraction", 0.5)),
            trailing_stop_pct=float(strategy.get("trailing_stop_pct", 0.8)),
            min_orderbook_imbalance=float(strategy.get("min_orderbook_imbalance", 0.95)),
            max_orderbook_spread_bps=float(strategy.get("max_orderbook_spread_bps", 12.0)),
            min_market_breadth_ratio=float(strategy.get("min_market_breadth_ratio", 0.35)),
            min_price_krw=float(strategy.get("min_price_krw", 100.0)),
            max_recent_stopouts_per_market=int(strategy.get("max_recent_stopouts_per_market", 2)),
            stopout_lookback_ticks=int(strategy.get("stopout_lookback_ticks", 144)),
            five_minute_short_window=int(strategy.get("five_minute_short_window", 3)),
            five_minute_long_window=int(strategy.get("five_minute_long_window", 6)),
            min_five_minute_momentum_pct=float(strategy.get("min_five_minute_momentum_pct", 0.0)),
            five_minute_trend_tolerance_pct=float(strategy.get("five_minute_trend_tolerance_pct", 0.2)),
            enable_range_rebound=bool(strategy.get("enable_range_rebound", True)),
            range_rebound_lookback=int(strategy.get("range_rebound_lookback", 12)),
            range_rebound_max_distance_from_low_pct=float(strategy.get("range_rebound_max_distance_from_low_pct", 3.0)),
            range_rebound_max_ema_gap_pct=float(strategy.get("range_rebound_max_ema_gap_pct", 3.5)),
            range_rebound_min_bounce_pct=float(strategy.get("range_rebound_min_bounce_pct", 0.15)),
            range_rebound_min_volume_ratio=float(strategy.get("range_rebound_min_volume_ratio", 0.9)),
            range_rebound_min_expected_upside_pct=float(strategy.get("range_rebound_min_expected_upside_pct", 1.0)),
            range_rebound_min_rsi=float(strategy.get("range_rebound_min_rsi", 35.0)),
            range_rebound_max_entry_rsi=float(strategy.get("range_rebound_max_entry_rsi", 55.0)),
            range_rebound_trend_break_grace_ticks=int(strategy.get("range_rebound_trend_break_grace_ticks", 2)),
        ),
        risk=RiskConfig(
            daily_profit_target_pct=float(risk["daily_profit_target_pct"]),
            daily_loss_limit_pct=float(risk["daily_loss_limit_pct"]),
            max_entries_per_day=int(risk.get("max_entries_per_day", risk.get("max_trades_per_day", 3))),
            max_position_fraction=float(risk["max_position_fraction"]),
            max_consecutive_losses=int(risk["max_consecutive_losses"]),
            halt_cooldown_ticks=int(risk.get("halt_cooldown_ticks", 6)),
            consecutive_loss_cooldown_ticks=int(risk.get("consecutive_loss_cooldown_ticks", risk.get("halt_cooldown_ticks", 6))),
            max_expected_downside_to_upside_ratio=float(risk.get("max_expected_downside_to_upside_ratio", 1.0)),
        ),
        ai_decision=AiDecisionConfig(
            enabled=bool(ai_decision.get("enabled", True)),
            provider=str(ai_decision.get("provider", "local")),
            min_confidence=float(ai_decision.get("min_confidence", 0.55)),
            openai_model=str(ai_decision.get("openai_model", "gpt-5.4-mini")),
            api_key_env=str(ai_decision.get("api_key_env", "OPENAI_API_KEY")),
        ),
        paths=PathConfig(
            trade_journal=_resolve_path(base_dir, paths["trade_journal"]),
            event_log=_resolve_path(base_dir, paths["event_log"]),
            state_file=_resolve_path(base_dir, paths["state_file"]),
        ),
    )
    _validate_config(app)
    return app


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object field: {key}")
    return value


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _validate_config(config: AppConfig) -> None:
    if config.mode != "paper":
        raise ValueError("Only paper mode is implemented in this MVP.")
    if config.starting_cash <= 0:
        raise ValueError("starting_cash must be positive.")
    if config.strategy.short_window <= 0:
        raise ValueError("short_window must be positive.")
    if config.strategy.long_window <= config.strategy.short_window:
        raise ValueError("long_window must be greater than short_window.")
    if not 0 < config.strategy.position_fraction <= 1:
        raise ValueError("position_fraction must be between 0 and 1.")
    if config.strategy.btc_long_window <= config.strategy.btc_short_window:
        raise ValueError("btc_long_window must be greater than btc_short_window.")
    if config.strategy.min_volume_ratio <= 0:
        raise ValueError("min_volume_ratio must be positive.")
    if config.strategy.max_recent_momentum_pct <= config.strategy.min_recent_momentum_pct:
        raise ValueError("max_recent_momentum_pct must be greater than min_recent_momentum_pct.")
    if config.strategy.max_ma_distance_pct <= 0:
        raise ValueError("max_ma_distance_pct must be positive.")
    if config.strategy.rsi_period < 2:
        raise ValueError("rsi_period must be at least 2.")
    if not 0 < config.strategy.max_entry_rsi <= 100:
        raise ValueError("max_entry_rsi must be between 0 and 100.")
    if config.strategy.target_recent_volatility_pct <= 0:
        raise ValueError("target_recent_volatility_pct must be positive.")
    if not 0 < config.strategy.min_volatility_position_fraction <= 1:
        raise ValueError("min_volatility_position_fraction must be between 0 and 1.")
    if config.strategy.long_trend_ema_window < 0:
        raise ValueError("long_trend_ema_window must not be negative.")
    if config.strategy.long_trend_ema_window > 200:
        raise ValueError("long_trend_ema_window must be 200 or lower for the Upbit candle API.")
    if config.strategy.time_stop_ticks < 0:
        raise ValueError("time_stop_ticks must not be negative.")
    if config.strategy.min_expected_upside_pct < 0:
        raise ValueError("min_expected_upside_pct must not be negative.")
    if config.strategy.target_upside_pct <= 0:
        raise ValueError("target_upside_pct must be positive.")
    if any(hour < 0 or hour > 23 for hour in config.strategy.blocked_entry_hours_kst):
        raise ValueError("blocked_entry_hours_kst must contain hours between 0 and 23.")
    if config.strategy.reentry_cooldown_ticks < 0:
        raise ValueError("reentry_cooldown_ticks must not be negative.")
    if config.strategy.stop_volatility_multiplier <= 0:
        raise ValueError("stop_volatility_multiplier must be positive.")
    if config.strategy.breakeven_trigger_pct < 0:
        raise ValueError("breakeven_trigger_pct must not be negative.")
    if config.strategy.partial_take_profit_pct < 0:
        raise ValueError("partial_take_profit_pct must not be negative.")
    if not 0 < config.strategy.partial_take_profit_fraction <= 1:
        raise ValueError("partial_take_profit_fraction must be between 0 and 1.")
    if config.strategy.trailing_stop_pct < 0:
        raise ValueError("trailing_stop_pct must not be negative.")
    if config.strategy.min_orderbook_imbalance <= 0:
        raise ValueError("min_orderbook_imbalance must be positive.")
    if config.strategy.max_orderbook_spread_bps < 0:
        raise ValueError("max_orderbook_spread_bps must not be negative.")
    if not 0 <= config.strategy.min_market_breadth_ratio <= 1:
        raise ValueError("min_market_breadth_ratio must be between 0 and 1.")
    if config.strategy.min_price_krw < 0:
        raise ValueError("min_price_krw must not be negative.")
    if config.strategy.max_recent_stopouts_per_market < 0:
        raise ValueError("max_recent_stopouts_per_market must not be negative.")
    if config.strategy.stopout_lookback_ticks < 0:
        raise ValueError("stopout_lookback_ticks must not be negative.")
    if config.strategy.five_minute_short_window <= 0:
        raise ValueError("five_minute_short_window must be positive.")
    if config.strategy.five_minute_long_window <= config.strategy.five_minute_short_window:
        raise ValueError("five_minute_long_window must be greater than five_minute_short_window.")
    if config.strategy.five_minute_trend_tolerance_pct < 0:
        raise ValueError("five_minute_trend_tolerance_pct must not be negative.")
    if config.strategy.range_rebound_lookback < 3:
        raise ValueError("range_rebound_lookback must be at least 3.")
    if config.strategy.range_rebound_max_distance_from_low_pct < 0:
        raise ValueError("range_rebound_max_distance_from_low_pct must not be negative.")
    if config.strategy.range_rebound_max_ema_gap_pct < 0:
        raise ValueError("range_rebound_max_ema_gap_pct must not be negative.")
    if config.strategy.range_rebound_min_volume_ratio <= 0:
        raise ValueError("range_rebound_min_volume_ratio must be positive.")
    if config.strategy.range_rebound_min_expected_upside_pct < 0:
        raise ValueError("range_rebound_min_expected_upside_pct must not be negative.")
    if not 0 <= config.strategy.range_rebound_min_rsi <= 100:
        raise ValueError("range_rebound_min_rsi must be between 0 and 100.")
    if not 0 <= config.strategy.range_rebound_max_entry_rsi <= 100:
        raise ValueError("range_rebound_max_entry_rsi must be between 0 and 100.")
    if config.strategy.range_rebound_trend_break_grace_ticks < 0:
        raise ValueError("range_rebound_trend_break_grace_ticks must not be negative.")
    if not 0 < config.risk.max_position_fraction <= 1:
        raise ValueError("max_position_fraction must be between 0 and 1.")
    if config.risk.max_entries_per_day < 1:
        raise ValueError("max_entries_per_day must be at least 1.")
    if config.risk.halt_cooldown_ticks < 0:
        raise ValueError("halt_cooldown_ticks must not be negative.")
    if config.risk.consecutive_loss_cooldown_ticks < 0:
        raise ValueError("consecutive_loss_cooldown_ticks must not be negative.")
    if config.risk.max_expected_downside_to_upside_ratio <= 0:
        raise ValueError("max_expected_downside_to_upside_ratio must be positive.")
    if not 0 <= config.ai_decision.min_confidence <= 1:
        raise ValueError("ai_decision.min_confidence must be between 0 and 1.")
