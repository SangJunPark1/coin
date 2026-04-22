from __future__ import annotations

from dataclasses import asdict

from .broker import PaperBroker
from .config import AppConfig
from .data import MarketDataSource, sleep_between_ticks
from .journal import Journal
from .models import Side, Signal
from .risk import RiskManager
from .strategy import MovingAverageStrategy, required_candle_count, volatility_adjusted_position_fraction


class TradingApp:
    def __init__(self, config: AppConfig, data_source: MarketDataSource, source_name: str) -> None:
        self.config = config
        self.data_source = data_source
        self.source_name = source_name
        self.broker = PaperBroker(
            market=config.market,
            starting_cash=config.starting_cash,
            fee_rate=config.fee_rate,
            slippage_bps=config.slippage_bps,
        )
        self.strategy = MovingAverageStrategy(config.strategy)
        self.risk = RiskManager(config.risk, starting_equity=config.starting_cash)
        self.journal = Journal(config.paths.trade_journal, config.paths.event_log)
        self.position_entry_tick: int | None = None

    def run(self, ticks: int) -> None:
        self.journal.event(
            "bot_started",
            {
                "mode": self.config.mode,
                "market": self.config.market,
                "source": self.source_name,
                "ticks": ticks,
            },
        )

        for tick in range(1, ticks + 1):
            try:
                self._run_tick(tick)
            except Exception as exc:  # Log and stop; hidden errors are dangerous in trading systems.
                self.journal.event("bot_error", {"tick": tick, "error": repr(exc)})
                raise
            if self.risk.state.halted:
                break
            sleep_between_ticks(self.config.poll_seconds, self.source_name)

        self.journal.event(
            "bot_finished",
            {
                "cash": self.broker.cash,
                "position": asdict(self.broker.position),
                "risk": asdict(self.risk.state),
            },
        )

    def _run_tick(self, tick: int) -> None:
        candles = self.data_source.get_recent_candles(self.config.market, required_candle_count(self.config.strategy))
        latest_price = candles[-1].close
        equity = self.broker.equity(latest_price)
        self.risk.ensure_trading_day(candles[-1].timestamp, equity)
        signal = self.strategy.generate(candles, self.broker.position)
        signal = self._apply_time_stop(tick, latest_price, signal)
        position_fraction = volatility_adjusted_position_fraction(candles, self.config.strategy)
        approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)

        self.journal.event(
            "tick",
            {
                "tick": tick,
                "price": latest_price,
                "equity": equity,
                "signal": signal,
                "approved": approved,
                "risk_reason": risk_reason,
                "risk": self.risk.state,
            },
        )

        if not approved:
            if self.risk.state.halted and self.broker.position.is_open:
                fill = self.broker.sell_all(latest_price, f"forced exit: {self.risk.state.halt_reason}")
                if fill is not None:
                    self.risk.record_fill(fill)
                    self.journal.trade(fill)
                    self.journal.event("forced_exit", {"tick": tick, "fill": fill, "risk": self.risk.state})
                    self.position_entry_tick = None
            return

        fill = None
        if signal.side == Side.BUY:
            cash_to_use = min(
                self.broker.cash * position_fraction,
                equity * self.config.risk.max_position_fraction,
            )
            fill = self.broker.buy(signal.price, cash_to_use, signal.reason)
        elif signal.side == Side.SELL:
            fill = self.broker.sell_all(signal.price, signal.reason)

        if fill is None:
            self.journal.event("fill_skipped", {"tick": tick, "signal": signal})
            return

        self.risk.record_fill(fill)
        self.journal.trade(fill)
        self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
        if fill.side == Side.BUY:
            self.position_entry_tick = tick
        elif fill.side == Side.SELL:
            self.position_entry_tick = None

    def _apply_time_stop(self, tick: int, latest_price: float, signal: Signal) -> Signal:
        if signal.side == Side.SELL or not self.broker.position.is_open:
            return signal
        if self.position_entry_tick is None:
            return signal
        max_ticks = self.config.strategy.time_stop_ticks
        if max_ticks <= 0:
            return signal
        held_ticks = tick - self.position_entry_tick
        pnl_pct = (latest_price / self.broker.position.avg_price - 1.0) * 100.0
        if held_ticks >= max_ticks and pnl_pct <= self.config.strategy.time_stop_min_pnl_pct:
            return Signal(
                Side.SELL,
                f"time stop reached: held {held_ticks} ticks, pnl {pnl_pct:.2f}%",
                latest_price,
                0.7,
            )
        return signal
