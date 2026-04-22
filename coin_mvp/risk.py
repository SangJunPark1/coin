from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import RiskConfig
from .models import Fill, Side, Signal


@dataclass
class RiskState:
    starting_equity: float
    day_key: str = ""
    entries_today: int = 0
    exits_today: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    period_started_at: str = ""
    halt_started_tick: int | None = None
    halt_until_tick: int | None = None


class RiskManager:
    def __init__(self, config: RiskConfig, starting_equity: float) -> None:
        self.config = config
        self.state = RiskState(starting_equity=starting_equity)

    def ensure_trading_day(self, timestamp: datetime, current_equity: float) -> None:
        timestamp = to_korea_time(timestamp)
        period_started_at = parse_state_time(self.state.period_started_at)
        if self.state.day_key == "" or period_started_at is None:
            self.state.day_key = korea_day_key(timestamp)
            self.state.period_started_at = timestamp.isoformat(timespec="seconds")
            return
        if timestamp < period_started_at + timedelta(hours=24):
            return

        self.state.day_key = korea_day_key(timestamp)
        self.state.period_started_at = timestamp.isoformat(timespec="seconds")
        self.state.starting_equity = current_equity
        self.state.entries_today = 0
        self.state.exits_today = 0
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halt_started_tick = None
        self.state.halt_until_tick = None

    def approve(self, signal: Signal, current_equity: float, position_fraction: float, tick: int | None = None) -> tuple[bool, str]:
        self._release_expired_cooldown(tick)
        self._update_halt_from_equity(current_equity, tick)
        if signal.side == Side.SELL:
            if self.state.halted:
                return True, f"approved risk-reducing exit: {self.state.halt_reason}"
            return True, "approved risk-reducing exit"
        if self.state.halted:
            return False, self.state.halt_reason
        if signal.side == Side.HOLD:
            return False, "hold signal"
        if self.state.entries_today >= self.config.max_entries_per_day:
            return False, "max daily entries reached"
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._halt("max consecutive losses reached", tick, self.config.consecutive_loss_cooldown_ticks)
            return False, self.state.halt_reason
        if signal.side == Side.BUY and position_fraction > self.config.max_position_fraction:
            return False, "position fraction exceeds risk limit"
        return True, "approved"

    def record_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY:
            self.state.entries_today += 1
        elif fill.side == Side.SELL:
            self.state.exits_today += 1
            if fill.realized_pnl < 0:
                self.state.consecutive_losses += 1
            elif fill.realized_pnl > 0:
                self.state.consecutive_losses = 0

    def _update_halt_from_equity(self, current_equity: float, tick: int | None) -> None:
        pnl_pct = (current_equity / self.state.starting_equity - 1.0) * 100.0
        if pnl_pct >= self.config.daily_profit_target_pct:
            self._halt(f"daily profit target reached: {pnl_pct:.2f}%", tick, 0)
        elif pnl_pct <= -self.config.daily_loss_limit_pct:
            self._halt(f"daily loss limit reached: {pnl_pct:.2f}%", tick, self.config.halt_cooldown_ticks)

    def _halt(self, reason: str, tick: int | None, cooldown_ticks: int) -> None:
        if self.state.halted and self.state.halt_reason == reason:
            return
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_started_tick = tick
        self.state.halt_until_tick = None if tick is None or cooldown_ticks <= 0 else tick + cooldown_ticks

    def _release_expired_cooldown(self, tick: int | None) -> None:
        if not self.state.halted or tick is None or self.state.halt_until_tick is None:
            return
        if tick < self.state.halt_until_tick:
            return
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halt_started_tick = None
        self.state.halt_until_tick = None


def korea_day_key(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (timestamp.astimezone(timezone.utc) + timedelta(hours=9)).date().isoformat()


def to_korea_time(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone(timedelta(hours=9)))


def parse_state_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return to_korea_time(parsed)
