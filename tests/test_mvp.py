import unittest
from datetime import datetime, timezone
from pathlib import Path

from coin_mvp.broker import PaperBroker
from coin_mvp.ai_decision import extract_openai_json, review_entry_candidate
from coin_mvp.config import AiDecisionConfig, AppConfig, PathConfig, RiskConfig, StrategyConfig
from coin_mvp.market_context import DecisionContext, maybe_float
from coin_mvp.models import Candle, OrderbookSnapshot, Position, Side, Signal
from coin_mvp.report import calculate_max_consecutive_losses, calculate_max_drawdown
from coin_mvp.risk import RiskManager
from coin_mvp.strategy import (
    MovingAverageStrategy,
    btc_regime_allows_entries,
    calculate_ema,
    estimate_expected_downside_pct,
    estimate_expected_upside_pct,
    market_breadth_ratio,
    volatility_adjusted_position_fraction,
)
from coin_mvp.watch_multi import (
    MultiMarketTradingApp,
    five_minute_momentum_penalty,
    five_minute_trend_penalty,
    orderbook_imbalance_penalty,
    orderbook_spread_penalty,
)


class PaperBrokerTest(unittest.TestCase):
    def test_buy_and_sell_updates_cash_and_position(self):
        broker = PaperBroker("KRW-BTC", starting_cash=1_000_000, fee_rate=0.0005, slippage_bps=0)

        buy = broker.buy(price=50_000_000, cash_to_use=200_000, reason="test buy")
        self.assertIsNotNone(buy)
        self.assertGreater(broker.position.qty, 0)
        self.assertLess(broker.cash, 1_000_000)

        sell = broker.sell_all(price=51_000_000, reason="test sell")
        self.assertIsNotNone(sell)
        self.assertEqual(broker.position.qty, 0)
        self.assertGreater(sell.realized_pnl, 0)

    def test_partial_sell_keeps_remaining_position(self):
        broker = PaperBroker("KRW-BTC", starting_cash=1_000_000, fee_rate=0.0005, slippage_bps=0)
        broker.buy(price=50_000_000, cash_to_use=200_000, reason="test buy")

        fill = broker.sell_fraction(price=50_500_000, fraction=0.5, reason="partial")

        self.assertIsNotNone(fill)
        self.assertGreater(broker.position.qty, 0)
        self.assertTrue(broker.position.partial_exit_taken)


class RiskManagerTest(unittest.TestCase):
    def test_daily_loss_halts_trading(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=3,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(
            Signal(Side.BUY, "test", price=100.0),
            current_equity=989_000,
            position_fraction=0.2,
        )
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

    def test_sell_is_allowed_after_entry_limit(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=1.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=1,
                max_position_fraction=0.25,
                max_consecutive_losses=2,
            ),
            starting_equity=1_000_000,
        )
        risk.state.entries_today = 1

        approved, reason = risk.approve(
            Signal(Side.SELL, "exit", price=100.0),
            current_equity=1_000_000,
            position_fraction=0.2,
        )
        self.assertTrue(approved)
        self.assertIn("risk-reducing exit", reason)

    def test_new_24_hour_period_resets_target_base(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=5.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
            ),
            starting_equity=1_000_000,
        )
        risk.ensure_trading_day(datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc), 1_000_000)
        risk.state.entries_today = 3
        risk.state.halted = True
        risk.state.halt_reason = "daily profit target reached: 3.00%"

        risk.ensure_trading_day(datetime(2026, 4, 20, 0, 1, tzinfo=timezone.utc), 1_030_000)

        self.assertEqual(risk.state.starting_equity, 1_030_000)
        self.assertEqual(risk.state.entries_today, 0)
        self.assertFalse(risk.state.halted)

    def test_halt_cooldown_releases_trading(self):
        risk = RiskManager(
            RiskConfig(
                daily_profit_target_pct=3.0,
                daily_loss_limit_pct=1.0,
                max_entries_per_day=12,
                max_position_fraction=0.35,
                max_consecutive_losses=4,
                halt_cooldown_ticks=2,
            ),
            starting_equity=1_000_000,
        )

        approved, reason = risk.approve(Signal(Side.BUY, "test", price=100.0), 989_000, 0.2, tick=10)
        self.assertFalse(approved)
        self.assertIn("daily loss limit", reason)

        approved, _ = risk.approve(Signal(Side.BUY, "test", price=100.0), 1_000_000, 0.2, tick=12)
        self.assertTrue(approved)


class ReportMetricsTest(unittest.TestCase):
    def test_max_drawdown_uses_cumulative_realized_pnl(self):
        self.assertEqual(calculate_max_drawdown([1000, -300, -500, 200]), -800)

    def test_max_consecutive_losses(self):
        self.assertEqual(calculate_max_consecutive_losses([1000, -1, -2, 3, -4, -5, -6]), 3)


class StrategyFilterTest(unittest.TestCase):
    def test_entry_blocks_overextended_move(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_recent_momentum_pct=1.0,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 101, 102, 103, 104, 112], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("overextended", signal.reason)

    def test_btc_regime_blocks_weak_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            btc_short_window=3,
            btc_long_window=5,
            min_btc_momentum_pct=-0.5,
        )
        candles = make_candles([100, 99, 98, 97, 96, 94], volume=10.0)

        allowed, reason, _ = btc_regime_allows_entries(candles, config)

        self.assertFalse(allowed)
        self.assertIn("btc regime blocked", reason)

    def test_entry_blocks_high_rsi(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            max_entry_rsi=70.0,
            min_volume_ratio=0.5,
            max_recent_momentum_pct=10.0,
            long_trend_ema_window=0,
        )
        candles = make_candles([100, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("overextended: RSI", signal.reason)

    def test_volatility_reduces_position_fraction(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            target_recent_volatility_pct=1.0,
            min_volatility_position_fraction=0.4,
        )
        candles = make_candles([100, 105, 97, 108, 96, 110, 95, 112, 94, 115, 93, 118, 92, 120, 91, 122, 90, 124, 89, 126, 88], volume=10.0)

        fraction = volatility_adjusted_position_fraction(candles, config)

        self.assertLess(fraction, config.position_fraction)
        self.assertGreaterEqual(fraction, config.position_fraction * config.min_volatility_position_fraction)

    def test_long_ema_blocks_entries_below_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=8,
        )
        candles = make_candles([150, 140, 130, 100, 101, 102, 103, 104], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("long trend filter blocked", signal.reason)

    def test_generate_surfaces_ma_alignment_failure_reason(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([105, 104, 103, 102, 101, 100], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("ma alignment failed", signal.reason)

    def test_generate_surfaces_price_below_long_ma_reason(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_volume_ratio=0.5,
            long_trend_ema_window=0,
        )
        candles = make_candles([95, 96, 97, 110, 100, 98], volume=10.0)

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)
        self.assertIn("price below long MA", signal.reason)

    def test_range_rebound_can_buy_in_weak_trend(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_recent_momentum_pct=0.05,
            max_recent_momentum_pct=4.0,
            min_volume_ratio=1.2,
            long_trend_ema_window=8,
            rsi_period=5,
            enable_range_rebound=True,
            range_rebound_lookback=8,
            range_rebound_max_distance_from_low_pct=2.5,
            range_rebound_max_ema_gap_pct=3.0,
            range_rebound_min_bounce_pct=0.1,
            range_rebound_min_volume_ratio=0.9,
            range_rebound_min_expected_upside_pct=1.0,
            range_rebound_min_rsi=30.0,
            range_rebound_max_entry_rsi=60.0,
        )
        candles = make_variable_candles(
            [103, 102, 101, 100.5, 100, 99.5, 99, 100.2],
            [10, 10, 10, 10, 10, 10, 10, 16],
        )

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.BUY)
        self.assertIn("range rebound setup", signal.reason)

    def test_range_rebound_rejects_when_far_from_recent_low(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_recent_momentum_pct=0.05,
            max_recent_momentum_pct=4.0,
            min_volume_ratio=1.2,
            long_trend_ema_window=8,
            enable_range_rebound=True,
            range_rebound_lookback=8,
            range_rebound_max_distance_from_low_pct=1.0,
            range_rebound_max_ema_gap_pct=3.0,
            range_rebound_min_bounce_pct=0.1,
            range_rebound_min_volume_ratio=0.9,
            range_rebound_min_expected_upside_pct=1.0,
            range_rebound_min_rsi=30.0,
            range_rebound_max_entry_rsi=60.0,
        )
        candles = make_variable_candles(
            [120, 116, 112, 108, 104, 100, 98, 101],
            [10, 10, 10, 10, 10, 10, 10, 14],
        )

        signal = MovingAverageStrategy(config).generate(candles, Position())

        self.assertEqual(signal.side, Side.HOLD)

    def test_calculate_ema_returns_value_when_enough_closes(self):
        self.assertIsNotNone(calculate_ema([1, 2, 3, 4, 5], 5))
        self.assertIsNone(calculate_ema([1, 2, 3], 5))

    def test_estimated_upside_caps_at_target(self):
        candles = make_candles([100, 102, 101, 104, 103, 108], volume=10.0)

        upside = estimate_expected_upside_pct(candles, target_upside_pct=3.0)

        self.assertLessEqual(upside, 3.0)
        self.assertGreater(upside, 0.0)

    def test_estimated_downside_respects_stop_floor(self):
        candles = make_candles([100, 99, 101, 98, 102, 100], volume=10.0)

        downside = estimate_expected_downside_pct(candles, stop_loss_pct=1.0, volatility_multiplier=1.1)

        self.assertGreaterEqual(downside, 1.0)

    def test_market_breadth_ratio_counts_passing_markets(self):
        bullish = make_candles([100, 101, 102, 103, 104, 105], volume=10.0)
        bearish = make_candles([105, 104, 103, 102, 101, 100], volume=10.0)

        ratio = market_breadth_ratio({"A": bullish, "B": bearish}, short_window=3, long_window=5, ema_window=0)

        self.assertEqual(ratio, 0.5)

    def test_local_ai_decision_blocks_low_confidence(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_expected_upside_pct=0.5,
            target_upside_pct=3.0,
        )
        context = DecisionContext(True, "test context", 1.0, 0.2, 1.0)
        decision = review_entry_candidate(
            Signal(Side.BUY, "test", price=100.0, confidence=0.2),
            make_candles([100, 101, 102, 103, 104, 105], volume=10.0),
            context,
            config,
            AiDecisionConfig(enabled=True, min_confidence=0.55),
        )

        self.assertEqual(decision.action, "hold")

    def test_local_ai_decision_blocks_unfavorable_risk_reward(self):
        config = StrategyConfig(
            short_window=3,
            long_window=5,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            min_expected_upside_pct=0.5,
            target_upside_pct=3.0,
        )
        context = DecisionContext(True, "test context", 1.0, 0.2, 1.0)
        decision = review_entry_candidate(
            Signal(Side.BUY, "test", price=100.0, confidence=0.9),
            make_candles([100.0, 100.2, 100.1, 100.3, 100.2, 100.25], volume=10.0),
            context,
            config,
            AiDecisionConfig(enabled=True, min_confidence=0.55),
        )

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.grade, "C")
        self.assertTrue(any("downside" in note.lower() for note in decision.risk_notes))

    def test_orderbook_snapshot_metrics(self):
        snapshot = OrderbookSnapshot(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
            best_bid_price=100.0,
            best_bid_size=5.0,
            best_ask_price=100.1,
            best_ask_size=4.0,
            total_bid_size=12.0,
            total_ask_size=10.0,
        )

        self.assertGreater(snapshot.spread_bps, 0)
        self.assertGreater(snapshot.imbalance_ratio, 1.0)

    def test_five_minute_momentum_penalty_grows_with_shortfall(self):
        mild = five_minute_momentum_penalty(-0.05, 0.0)
        worse = five_minute_momentum_penalty(-0.25, 0.0)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)
        self.assertLessEqual(worse, 0.18)

    def test_five_minute_trend_penalty_grows_with_shortfall(self):
        mild = five_minute_trend_penalty(0.002)
        worse = five_minute_trend_penalty(0.008)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)

    def test_orderbook_spread_penalty_grows_with_excess_spread(self):
        mild = orderbook_spread_penalty(20.0, 18.0)
        worse = orderbook_spread_penalty(30.0, 18.0)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)

    def test_orderbook_imbalance_penalty_grows_with_shortfall(self):
        mild = orderbook_imbalance_penalty(0.82, 0.85)
        worse = orderbook_imbalance_penalty(0.70, 0.85)

        self.assertGreater(mild, 0.0)
        self.assertGreater(worse, mild)
        self.assertLessEqual(worse, 0.2)

    def test_openai_response_json_extraction(self):
        payload = {"output": [{"content": [{"text": "{\"action\":\"hold\"}"}]}]}

        parsed = extract_openai_json(payload)

        self.assertEqual(parsed["action"], "hold")

    def test_market_context_float_parser_is_soft(self):
        self.assertEqual(maybe_float("1.25"), 1.25)
        self.assertIsNone(maybe_float("not-a-number"))


class RangeReboundExitGraceTest(unittest.TestCase):
    def test_range_rebound_grace_suppresses_early_trend_break(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "range_rebound"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)

        suppressed = app._apply_range_rebound_exit_grace(11, 99.0, Signal(Side.SELL, "trend break", 99.0, 0.6))

        self.assertEqual(suppressed.side, Side.HOLD)
        self.assertIn("range rebound grace active", suppressed.reason)

    def test_range_rebound_grace_expires_after_configured_ticks(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "range_rebound"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)
        original = Signal(Side.SELL, "trend break", 99.0, 0.6)

        allowed = app._apply_range_rebound_exit_grace(13, 99.0, original)

        self.assertEqual(allowed, original)

    def test_trend_entries_do_not_get_rebound_grace(self):
        app = make_test_app(range_rebound_trend_break_grace_ticks=2)
        app.position_entry_strategy = "trend"
        app.position_entry_tick = 10
        app.broker.position = Position(qty=1.0, avg_price=100.0, peak_price=100.0)
        original = Signal(Side.SELL, "trend break", 99.0, 0.6)

        allowed = app._apply_range_rebound_exit_grace(11, 99.0, original)

        self.assertEqual(allowed, original)


def make_candles(closes: list[float], volume: float) -> list[Candle]:
    return [
        Candle(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, index, tzinfo=timezone.utc),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume,
        )
        for index, close in enumerate(closes)
    ]


def make_variable_candles(closes: list[float], volumes: list[float]) -> list[Candle]:
    return [
        Candle(
            market="KRW-BTC",
            timestamp=datetime(2026, 4, 20, index, tzinfo=timezone.utc),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volumes[index],
        )
        for index, close in enumerate(closes)
    ]


class DummyDataSource:
    def get_recent_candles(self, *args, **kwargs):
        raise NotImplementedError

    def get_orderbook_snapshot(self, *args, **kwargs):
        raise NotImplementedError

    def get_top_krw_markets(self, *args, **kwargs):
        return []


def make_test_app(range_rebound_trend_break_grace_ticks: int = 2) -> MultiMarketTradingApp:
    config = AppConfig(
        mode="paper",
        market="KRW-BTC",
        poll_seconds=0,
        starting_cash=1_000_000.0,
        fee_rate=0.0005,
        slippage_bps=5.0,
        strategy=StrategyConfig(
            short_window=5,
            long_window=20,
            take_profit_pct=1.0,
            stop_loss_pct=1.0,
            position_fraction=0.2,
            range_rebound_trend_break_grace_ticks=range_rebound_trend_break_grace_ticks,
        ),
        risk=RiskConfig(
            daily_profit_target_pct=3.0,
            daily_loss_limit_pct=5.0,
            max_entries_per_day=12,
            max_position_fraction=0.35,
            max_consecutive_losses=4,
        ),
        ai_decision=AiDecisionConfig(enabled=False),
        paths=PathConfig(
            trade_journal=Path("data/test_trades.csv"),
            event_log=Path("logs/test_events.jsonl"),
            state_file=Path("data/test_state.json"),
        ),
    )
    return MultiMarketTradingApp(config, DummyDataSource(), ["KRW-BTC"], request_delay=0.0)


if __name__ == "__main__":
    unittest.main()
