from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from .ai_decision import review_entry_candidate
from .broker import PaperBroker
from .config import AppConfig, StrategyConfig, load_config
from .data import UpbitPublicDataSource, sleep_between_ticks
from .journal import Journal
from .market_context import collect_decision_context
from .models import Candle, Side, Signal
from .risk import RiskManager
from .strategy import MovingAverageStrategy, btc_regime_allows_entries, estimate_expected_upside_pct, required_candle_count, volatility_adjusted_position_fraction
from .watch import refresh_report


class MultiMarketTradingApp:
    def __init__(self, config: AppConfig, data_source: UpbitPublicDataSource, markets: list[str], request_delay: float) -> None:
        self.config = config
        self.data_source = data_source
        self.markets = markets
        self.request_delay = request_delay
        self.broker = PaperBroker(
            market=config.market,
            starting_cash=config.starting_cash,
            fee_rate=config.fee_rate,
            slippage_bps=config.slippage_bps,
        )
        self.strategy = MovingAverageStrategy(config.strategy)
        self.risk = RiskManager(config.risk, starting_equity=config.starting_cash)
        self.journal = Journal(config.paths.trade_journal, config.paths.event_log)
        self.current_market: str | None = None
        self.position_entry_tick: int | None = None

    def run_tick(self, tick: int) -> None:
        if self.broker.position.is_open:
            self._manage_open_position(tick)
            return
        self._scan_and_enter(tick)

    def _manage_open_position(self, tick: int) -> None:
        market = self.current_market or self.broker.market
        candles = self.data_source.get_recent_candles(market, required_candle_count(self.config.strategy))
        latest_price = candles[-1].close
        equity = self.broker.equity(latest_price)
        self.risk.ensure_trading_day(candles[-1].timestamp, equity)
        signal = self.strategy.generate(candles, self.broker.position)
        signal = self._apply_time_stop(tick, latest_price, signal)
        position_fraction = volatility_adjusted_position_fraction(candles, self.config.strategy)
        approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)
        self._log_tick(tick, market, latest_price, equity, signal, approved, risk_reason)
        if not approved:
            if self.risk.state.halted:
                self._force_exit_if_needed(tick, latest_price)
            return
        if signal.side == Side.SELL:
            fill = self.broker.sell_all(signal.price, signal.reason)
            if fill is not None:
                self.risk.record_fill(fill)
                self.journal.trade(fill)
                self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
                self.current_market = None
                self.position_entry_tick = None

    def _scan_and_enter(self, tick: int) -> None:
        context = collect_decision_context(self.data_source, self.config.strategy)
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        if not context.allows_entries:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": 0,
                    "candidates": 0,
                    "reason": context.reason,
                    "decision_context": context.to_dict(),
                    "risk": self.risk.state,
                },
            )
            return

        candidates = []
        blocked_reasons: dict[str, int] = {}
        blocked_samples: list[dict[str, object]] = []
        first_candles: list[Candle] | None = None
        for market in self.markets:
            try:
                candles = self.data_source.get_recent_candles(market, required_candle_count(self.config.strategy))
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
            except Exception as exc:
                self.journal.event("market_scan_error", {"tick": tick, "market": market, "error": repr(exc)})
                continue
            if first_candles is None:
                first_candles = candles
            signal = self.strategy.generate(candles, self.broker.position)
            if signal.side == Side.BUY:
                candidates.append((candidate_score(candles, signal, self.config.strategy) * context.score_multiplier, market, candles, signal))
            else:
                blocked_reasons[signal.reason] = blocked_reasons.get(signal.reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append(
                        {
                            "market": market,
                            "reason": signal.reason,
                            "price": candles[-1].close,
                        }
                    )

        if first_candles is None:
            raise RuntimeError("No market data was available during multi-market scan.")

        equity = self.broker.equity(first_candles[-1].close)
        self.risk.ensure_trading_day(first_candles[-1].timestamp, equity)
        if not candidates:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": len(self.markets),
                    "candidates": 0,
                    "reason": "no entry condition",
                    "decision_context": context.to_dict(),
                    "blocked_reasons": blocked_reasons,
                    "blocked_samples": blocked_samples,
                    "risk": self.risk.state,
                },
            )
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, market, candles, signal = candidates[0]
        latest_price = candles[-1].close
        equity = self.broker.equity(latest_price)
        position_fraction = volatility_adjusted_position_fraction(candles, self.config.strategy)
        decision_review = review_entry_candidate(signal, candles, context, self.config.strategy, self.config.ai_decision)
        if decision_review.action != "buy":
            self._log_tick(
                tick,
                market,
                latest_price,
                equity,
                signal,
                False,
                f"ai decision blocked: {decision_review.action}",
                score=score,
                candidates=len(candidates),
                btc_regime=context.reason,
                decision_context=context.to_dict(),
                ai_decision=decision_review.to_dict(),
                blocked_reasons=blocked_reasons,
                blocked_samples=blocked_samples,
            )
            return
        approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)
        self._log_tick(
            tick,
            market,
            latest_price,
            equity,
            signal,
            approved,
            risk_reason,
            score=score,
            candidates=len(candidates),
            btc_regime=context.reason,
            decision_context=context.to_dict(),
            ai_decision=decision_review.to_dict(),
            blocked_reasons=blocked_reasons,
            blocked_samples=blocked_samples,
        )
        if not approved:
            return

        self.broker.market = market
        self.current_market = market
        cash_to_use = min(
            self.broker.cash * position_fraction,
            equity * self.config.risk.max_position_fraction,
        )
        fill = self.broker.buy(signal.price, cash_to_use, f"{signal.reason}; {context.reason}; selected from top-volume scan")
        if fill is None:
            self.journal.event("fill_skipped", {"tick": tick, "signal": signal, "market": market})
            return
        self.risk.record_fill(fill)
        self.journal.trade(fill)
        self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
        self.position_entry_tick = tick

    def _force_exit_if_needed(self, tick: int, latest_price: float) -> None:
        if not self.broker.position.is_open:
            return
        fill = self.broker.sell_all(latest_price, f"forced exit: {self.risk.state.halt_reason}")
        if fill is not None:
            self.risk.record_fill(fill)
            self.journal.trade(fill)
            self.journal.event("forced_exit", {"tick": tick, "fill": fill, "risk": self.risk.state})
            self.current_market = None
            self.position_entry_tick = None

    def _btc_regime(self) -> tuple[bool, str, float]:
        candles = self.data_source.get_recent_candles("KRW-BTC", required_candle_count(self.config.strategy))
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        return btc_regime_allows_entries(candles, self.config.strategy)

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

    def _log_tick(
        self,
        tick: int,
        market: str,
        price: float,
        equity: float,
        signal: Signal,
        approved: bool,
        risk_reason: str,
        score: float | None = None,
        candidates: int | None = None,
        btc_regime: str | None = None,
        decision_context: dict[str, object] | None = None,
        ai_decision: dict[str, object] | None = None,
        blocked_reasons: dict[str, int] | None = None,
        blocked_samples: list[dict[str, object]] | None = None,
    ) -> None:
        payload = {
            "tick": tick,
            "market": market,
            "price": price,
            "equity": equity,
            "signal": signal,
            "approved": approved,
            "risk_reason": risk_reason,
            "risk": self.risk.state,
        }
        if score is not None:
            payload["candidate_score"] = score
        if candidates is not None:
            payload["candidate_count"] = candidates
        if btc_regime is not None:
            payload["btc_regime"] = btc_regime
        if decision_context is not None:
            payload["decision_context"] = decision_context
        if ai_decision is not None:
            payload["ai_decision"] = ai_decision
        if blocked_reasons is not None:
            payload["blocked_reasons"] = blocked_reasons
        if blocked_samples is not None:
            payload["blocked_samples"] = blocked_samples
        self.journal.event("tick", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-market Upbit paper observation.")
    parser.add_argument("--config", default="config.lowload.json")
    parser.add_argument("--top-markets", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=864)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--output", default="reports/latest_report.html")
    parser.add_argument("--request-delay", type=float, default=0.18, help="Delay between Upbit candle requests during a scan.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(args.top_markets)
    app = MultiMarketTradingApp(config, data_source, markets, request_delay=args.request_delay)
    output = Path(args.output)

    app.journal.event(
        "watch_started",
        {
            "mode": config.mode,
            "source": "upbit",
            "market_mode": "top_krw_markets",
            "markets": markets,
            "request_delay": args.request_delay,
            "ticks": args.ticks,
            "report_every": args.report_every,
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)

    for tick in range(1, args.ticks + 1):
        try:
            app.run_tick(tick)
            if tick % args.report_every == 0:
                refresh_report(config.paths.trade_journal, config.paths.event_log, output)
                print(f"Report refreshed at tick {tick}: {output}")
        except Exception as exc:
            app.journal.event("watch_error", {"tick": tick, "error": repr(exc)})
            refresh_report(config.paths.trade_journal, config.paths.event_log, output)
            raise
        sleep_between_ticks(config.poll_seconds, "upbit")

    app.journal.event(
        "watch_finished",
        {
            "cash": app.broker.cash,
            "position": asdict(app.broker.position),
            "risk": asdict(app.risk.state),
            "markets": markets,
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)


def candidate_score(candles: list[Candle], signal: Signal, config: StrategyConfig) -> float:
    closes = [candle.close for candle in candles]
    momentum = (closes[-1] / closes[-5] - 1.0) if len(closes) >= 5 and closes[-5] else 0.0
    volume_value = candles[-1].close * candles[-1].volume
    pullback_risk = max(0.0, (max(closes[-5:]) / closes[-1] - 1.0)) if len(closes) >= 5 and closes[-1] else 0.0
    expected_upside = estimate_expected_upside_pct(candles, target_upside_pct=config.target_upside_pct) / 100.0
    return signal.confidence + momentum + expected_upside - pullback_risk + min(volume_value / 1_000_000_000_000.0, 0.25)


if __name__ == "__main__":
    main()
