from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from datetime import timedelta, timezone
from pathlib import Path

from .ai_decision import review_entry_candidate
from .broker import PaperBroker
from .config import AppConfig, StrategyConfig, load_config
from .data import UpbitPublicDataSource, sleep_between_ticks
from .journal import Journal
from .market_context import collect_decision_context
from .models import Candle, Side, Signal
from .risk import RiskManager
from .strategy import (
    MovingAverageStrategy,
    btc_regime_allows_entries,
    calculate_ema,
    estimate_expected_downside_pct,
    estimate_expected_upside_pct,
    market_breadth_ratio,
    mean,
    required_candle_count,
    volatility_adjusted_position_fraction,
)
from .watch import refresh_report

KST = timezone(timedelta(hours=9))


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
        self.position_entry_strategy: str | None = None
        self.market_reentry_until_tick: dict[str, int] = {}
        self.market_stopout_ticks: dict[str, list[int]] = {}

    def run_tick(self, tick: int) -> None:
        if self.broker.position.is_open:
            self._manage_open_position(tick)
            return
        self._scan_and_enter(tick)

    def _manage_open_position(self, tick: int) -> None:
        market = self.current_market or self.broker.market
        candles = self.data_source.get_recent_candles(market, required_candle_count(self.config.strategy))
        latest_price = candles[-1].close
        self.broker.mark_peak(latest_price)
        equity = self.broker.equity(latest_price)
        self.risk.ensure_trading_day(candles[-1].timestamp, equity)
        signal = self._position_management_signal(tick, market, candles, latest_price)
        if signal.side == Side.HOLD:
            signal = self.strategy.generate(candles, self.broker.position)
        signal = self._apply_range_rebound_exit_grace(tick, latest_price, signal)
        signal = self._apply_time_stop(tick, latest_price, signal)
        position_fraction = volatility_adjusted_position_fraction(candles, self.config.strategy)
        approved, risk_reason = self.risk.approve(signal, equity, position_fraction, tick=tick)
        self._log_tick(tick, market, latest_price, equity, signal, approved, risk_reason)
        if not approved:
            if self.risk.state.halted:
                self._force_exit_if_needed(tick, latest_price)
            return
        if signal.side == Side.SELL:
            fill = self.broker.sell_fraction(signal.price, signal.size_fraction, signal.reason)
            if fill is not None:
                self.risk.record_fill(fill)
                self.journal.trade(fill)
                self.journal.event("fill", {"tick": tick, "fill": fill, "risk": self.risk.state})
                if signal.reason.startswith("stop loss"):
                    self._register_stopout(market, tick)
                    self.market_reentry_until_tick[market] = tick + self.config.strategy.reentry_cooldown_ticks
                if not self.broker.position.is_open:
                    self.current_market = None
                    self.position_entry_tick = None
                    self.position_entry_strategy = None

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
        candles_by_market: dict[str, list[Candle]] = {}
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
            candles_by_market[market] = candles
            trend_screen_reason, trend_screen_penalty = self._universe_trend_signal(candles)
            blocked, blocked_reason = self._entry_time_block(candles[-1].timestamp)
            if blocked:
                blocked_reasons[blocked_reason] = blocked_reasons.get(blocked_reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append(
                        {
                            "market": market,
                            "reason": blocked_reason,
                            "price": candles[-1].close,
                        }
                    )
                continue
            blocked, blocked_reason, filter_penalty = self._entry_market_filters(tick, market, candles)
            if blocked:
                blocked_reasons[blocked_reason] = blocked_reasons.get(blocked_reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append({"market": market, "reason": blocked_reason, "price": candles[-1].close})
                continue
            signal = self.strategy.generate(candles, self.broker.position)
            if signal.side == Side.BUY:
                penalty = self._recent_stopout_penalty(market, tick) + filter_penalty + trend_screen_penalty
                candidates.append((candidate_score(candles, signal, self.config.strategy, penalty=penalty) * context.score_multiplier, market, candles, signal))
            else:
                reason = signal.reason
                if trend_screen_reason != "universe trend bonus":
                    reason = f"{reason}; {trend_screen_reason}"
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                if len(blocked_samples) < 20:
                    blocked_samples.append(
                        {
                            "market": market,
                            "reason": reason,
                            "price": candles[-1].close,
                        }
                    )

        if first_candles is None:
            raise RuntimeError("No market data was available during multi-market scan.")

        equity = self.broker.equity(first_candles[-1].close)
        self.risk.ensure_trading_day(first_candles[-1].timestamp, equity)
        breadth_ratio = market_breadth_ratio(
            candles_by_market,
            self.config.strategy.short_window,
            self.config.strategy.long_window,
            self.config.strategy.long_trend_ema_window or self.config.strategy.long_window,
        )
        breadth_penalty = 0.0
        if breadth_ratio < self.config.strategy.min_market_breadth_ratio:
            shortage = self.config.strategy.min_market_breadth_ratio - breadth_ratio
            breadth_penalty = min(0.35, shortage * 0.8)
        if not candidates:
            self.journal.event(
                "market_scan",
                {
                    "tick": tick,
                    "markets_scanned": len(self.markets),
                    "candidates": 0,
                    "reason": "no entry condition" if breadth_penalty <= 0 else f"no entry condition; breadth penalty {breadth_penalty:.2f}",
                    "decision_context": context.to_dict(),
                    "blocked_reasons": blocked_reasons,
                    "blocked_samples": blocked_samples,
                    "market_breadth_ratio": breadth_ratio,
                    "market_breadth_penalty": breadth_penalty,
                    "risk": self.risk.state,
                },
            )
            return

        candidates.sort(key=lambda item: item[0], reverse=True)
        adjusted_candidates = [
            (score - breadth_penalty, market, candles, signal)
            for score, market, candles, signal in candidates
        ]
        adjusted_candidates.sort(key=lambda item: item[0], reverse=True)
        score, market, candles, signal = adjusted_candidates[0]
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
                market_breadth_ratio=breadth_ratio,
                breadth_penalty=breadth_penalty,
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
            market_breadth_ratio=breadth_ratio,
            breadth_penalty=breadth_penalty,
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
        self.position_entry_strategy = self._entry_strategy_name(signal)

    def _entry_time_block(self, timestamp) -> tuple[bool, str]:
        blocked_hours = set(self.config.strategy.blocked_entry_hours_kst)
        if not blocked_hours:
            return False, ""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        hour = timestamp.astimezone(KST).hour
        if hour in blocked_hours:
            return True, f"blocked entry hour: {hour:02d} KST"
        return False, ""

    def _entry_market_filters(self, tick: int, market: str, candles: list[Candle]) -> tuple[bool, str, float]:
        latest_price = candles[-1].close
        if latest_price < self.config.strategy.min_price_krw:
            return True, f"price below floor: {latest_price:.2f} KRW", 0.0
        reentry_until = self.market_reentry_until_tick.get(market)
        if reentry_until is not None and tick < reentry_until:
            return True, f"reentry cooldown active until tick {reentry_until}", 0.0
        self._prune_stopouts(market, tick)
        if len(self.market_stopout_ticks.get(market, [])) >= self.config.strategy.max_recent_stopouts_per_market:
            return True, f"recent stopouts limit reached: {market}", 0.0
        orderbook = self.data_source.get_orderbook_snapshot(market)
        spread_limit = self.config.strategy.max_orderbook_spread_bps
        if orderbook.spread_bps > spread_limit * 2.0:
            return True, f"wide spread: {orderbook.spread_bps:.1f}bps", 0.0
        spread_penalty = 0.0
        if orderbook.spread_bps > spread_limit:
            spread_penalty = orderbook_spread_penalty(orderbook.spread_bps, spread_limit)
        imbalance_limit = self.config.strategy.min_orderbook_imbalance
        if orderbook.imbalance_ratio < max(0.0, imbalance_limit - 0.25):
            return True, f"weak orderbook imbalance: {orderbook.imbalance_ratio:.2f}", 0.0
        imbalance_penalty = 0.0
        if orderbook.imbalance_ratio < imbalance_limit:
            imbalance_penalty = orderbook_imbalance_penalty(orderbook.imbalance_ratio, imbalance_limit)
        mtf_ok, mtf_reason, mtf_penalty = self._five_minute_trend_ok(market)
        if not mtf_ok:
            return True, mtf_reason, 0.0
        total_penalty = spread_penalty + imbalance_penalty + mtf_penalty
        reasons: list[str] = []
        if spread_penalty > 0.0:
            reasons.append(f"spread penalty: {orderbook.spread_bps:.1f}bps")
        if imbalance_penalty > 0.0:
            reasons.append(f"imbalance penalty: {orderbook.imbalance_ratio:.2f}")
        if mtf_reason != "5m trend ok":
            reasons.append(mtf_reason)
        reason = "; ".join(reasons) if reasons else mtf_reason
        return False, reason, total_penalty

    def _universe_trend_signal(self, candles: list[Candle]) -> tuple[str, float]:
        closes = [candle.close for candle in candles]
        if len(closes) < self.config.strategy.long_window:
            return "universe trend penalty: not enough candles", 0.18
        latest_price = closes[-1]
        short_ma = mean(closes[-self.config.strategy.short_window :])
        long_ma = mean(closes[-self.config.strategy.long_window :])
        long_trend_ema = calculate_ema(closes, self.config.strategy.long_trend_ema_window)
        if long_trend_ema is not None and latest_price < long_trend_ema:
            return f"universe trend penalty: below EMA{self.config.strategy.long_trend_ema_window}", 0.16
        if short_ma <= long_ma:
            return f"universe trend penalty: short {short_ma:.3f} <= long {long_ma:.3f}", 0.14
        if latest_price <= long_ma:
            return f"universe trend penalty: price {latest_price:.3f} <= long {long_ma:.3f}", 0.12
        return "universe trend bonus", -0.05

    def _five_minute_trend_ok(self, market: str) -> tuple[bool, str, float]:
        candles = self.data_source.get_recent_candles(
            market,
            max(self.config.strategy.five_minute_long_window + 3, self.config.strategy.long_trend_ema_window // 5 if self.config.strategy.long_trend_ema_window else 0),
            unit_minutes=5,
        )
        closes = [candle.close for candle in candles]
        if len(closes) < self.config.strategy.five_minute_long_window:
            return False, "5m trend unavailable", 0.0
        latest_price = closes[-1]
        short_ma = mean(closes[-self.config.strategy.five_minute_short_window :])
        long_ma = mean(closes[-self.config.strategy.five_minute_long_window :])
        tolerance = self.config.strategy.five_minute_trend_tolerance_pct / 100.0
        min_short_ma = long_ma * (1.0 - tolerance)
        min_latest_price = short_ma * (1.0 - tolerance)
        momentum_pct = ((latest_price / closes[-min(4, len(closes) - 1) - 1]) - 1.0) * 100.0 if len(closes) > 4 else 0.0
        short_ma_shortfall = max(0.0, (min_short_ma - short_ma) / long_ma) if long_ma > 0 else 0.0
        latest_price_shortfall = max(0.0, (min_latest_price - latest_price) / short_ma) if short_ma > 0 else 0.0
        trend_shortfall = max(short_ma_shortfall, latest_price_shortfall)
        if trend_shortfall > 0.0:
            if trend_shortfall >= 0.01:
                return False, "5m trend weak", 0.0
            penalty = five_minute_trend_penalty(trend_shortfall)
            return True, f"5m trend penalty: {trend_shortfall * 100.0:.2f}%", penalty
        if momentum_pct < self.config.strategy.min_five_minute_momentum_pct:
            penalty = five_minute_momentum_penalty(momentum_pct, self.config.strategy.min_five_minute_momentum_pct)
            return True, f"5m momentum penalty: {momentum_pct:.2f}%", penalty
        return True, "5m trend ok", 0.0

    def _position_management_signal(self, tick: int, market: str, candles: list[Candle], latest_price: float) -> Signal:
        if not self.broker.position.is_open:
            return Signal(Side.HOLD, "no position", latest_price, 0.0)
        avg_price = self.broker.position.avg_price
        pnl_pct = (latest_price / avg_price - 1.0) * 100.0 if avg_price > 0 else 0.0
        dynamic_stop_loss_pct = estimate_expected_downside_pct(candles, self.config.strategy.stop_loss_pct, self.config.strategy.stop_volatility_multiplier)
        if pnl_pct <= -dynamic_stop_loss_pct:
            return Signal(Side.SELL, f"stop loss reached: {pnl_pct:.2f}%", latest_price, 0.95)
        if not self.broker.position.partial_exit_taken and pnl_pct >= self.config.strategy.partial_take_profit_pct:
            return Signal(
                Side.SELL,
                f"partial take profit reached: {pnl_pct:.2f}%",
                latest_price,
                0.85,
                size_fraction=self.config.strategy.partial_take_profit_fraction,
            )
        if pnl_pct >= self.config.strategy.breakeven_trigger_pct and latest_price <= avg_price:
            return Signal(Side.SELL, f"breakeven stop reached: {pnl_pct:.2f}%", latest_price, 0.8)
        peak_price = max(self.broker.position.peak_price, latest_price)
        peak_drawdown_pct = (latest_price / peak_price - 1.0) * 100.0 if peak_price > 0 else 0.0
        if pnl_pct > 0 and peak_drawdown_pct <= -self.config.strategy.trailing_stop_pct:
            return Signal(Side.SELL, f"trailing stop reached: {peak_drawdown_pct:.2f}%", latest_price, 0.75)
        return Signal(Side.HOLD, "position managed", latest_price, 0.2)

    def _register_stopout(self, market: str, tick: int) -> None:
        ticks = self.market_stopout_ticks.setdefault(market, [])
        ticks.append(tick)
        self._prune_stopouts(market, tick)

    def _prune_stopouts(self, market: str, tick: int) -> None:
        lookback = self.config.strategy.stopout_lookback_ticks
        ticks = self.market_stopout_ticks.get(market, [])
        if not ticks:
            return
        self.market_stopout_ticks[market] = [value for value in ticks if tick - value <= lookback]

    def _recent_stopout_penalty(self, market: str, tick: int) -> float:
        self._prune_stopouts(market, tick)
        count = len(self.market_stopout_ticks.get(market, []))
        return min(0.3, count * 0.08)

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
            self.position_entry_strategy = None

    def _entry_strategy_name(self, signal: Signal) -> str:
        if "range rebound setup" in signal.reason:
            return "range_rebound"
        return "trend"

    def _apply_range_rebound_exit_grace(self, tick: int, latest_price: float, signal: Signal) -> Signal:
        if signal.side != Side.SELL or signal.reason != "trend break":
            return signal
        if self.position_entry_strategy != "range_rebound":
            return signal
        if self.position_entry_tick is None:
            return signal
        grace_ticks = self.config.strategy.range_rebound_trend_break_grace_ticks
        if grace_ticks <= 0:
            return signal
        held_ticks = tick - self.position_entry_tick
        if held_ticks <= grace_ticks:
            return Signal(
                Side.HOLD,
                f"range rebound grace active: held {held_ticks} ticks, suppress trend break",
                latest_price,
                0.2,
            )
        return signal

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
        market_breadth_ratio: float | None = None,
        breadth_penalty: float | None = None,
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
        if market_breadth_ratio is not None:
            payload["market_breadth_ratio"] = market_breadth_ratio
        if breadth_penalty is not None:
            payload["market_breadth_penalty"] = breadth_penalty
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
    markets = data_source.get_top_krw_markets(args.top_markets, min_trade_price_krw=config.strategy.min_price_krw)
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


def candidate_score(candles: list[Candle], signal: Signal, config: StrategyConfig, penalty: float = 0.0) -> float:
    closes = [candle.close for candle in candles]
    momentum = (closes[-1] / closes[-5] - 1.0) if len(closes) >= 5 and closes[-5] else 0.0
    volume_value = candles[-1].close * candles[-1].volume
    pullback_risk = max(0.0, (max(closes[-5:]) / closes[-1] - 1.0)) if len(closes) >= 5 and closes[-1] else 0.0
    expected_upside = estimate_expected_upside_pct(candles, target_upside_pct=config.target_upside_pct) / 100.0
    return signal.confidence + momentum + expected_upside - pullback_risk + min(volume_value / 1_000_000_000_000.0, 0.25) - penalty


def five_minute_momentum_penalty(momentum_pct: float, min_required_pct: float) -> float:
    shortfall = max(0.0, min_required_pct - momentum_pct)
    return min(0.18, 0.06 + shortfall * 0.6)


def five_minute_trend_penalty(shortfall_ratio: float) -> float:
    return min(0.2, 0.06 + shortfall_ratio * 8.0)


def orderbook_spread_penalty(spread_bps: float, max_spread_bps: float) -> float:
    excess = max(0.0, spread_bps - max_spread_bps)
    return min(0.22, 0.05 + excess / max(1.0, max_spread_bps) * 0.18)


def orderbook_imbalance_penalty(imbalance_ratio: float, min_imbalance_ratio: float) -> float:
    shortfall = max(0.0, min_imbalance_ratio - imbalance_ratio)
    return min(0.2, 0.05 + shortfall / max(0.1, min_imbalance_ratio) * 0.2)


if __name__ == "__main__":
    main()
