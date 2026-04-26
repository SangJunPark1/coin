from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .data import UpbitPublicDataSource
from .models import Position
from .report import render_report, read_events, read_trades
from .risk import RiskState
from .watch_multi import MultiMarketTradingApp

KST = timezone(timedelta(hours=9))
DEFAULT_START_KST = "2026-04-20T21:10:00+09:00"
DEFAULT_END_KST = "2026-04-25T18:00:00+09:00"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cloud paper-trading ticks and persist state.")
    parser.add_argument("--config", default="config.cloud.json")
    parser.add_argument("--start-kst", default=DEFAULT_START_KST)
    parser.add_argument("--end-kst", default=DEFAULT_END_KST)
    parser.add_argument("--top-markets", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=0.18)
    parser.add_argument("--cadence-minutes", type=int, default=5)
    parser.add_argument("--max-catch-up-ticks", type=int, default=12)
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    start_at = parse_kst(args.start_kst)
    end_at = parse_kst(args.end_kst)
    now = datetime.now(KST)
    outputs = [Path(value) for value in args.output] or [Path("docs/index.html")]

    if args.reset:
        reset_cloud_files(config, outputs)

    if now < start_at:
        if not all(output.exists() for output in outputs):
            write_status_report(config, outputs, f"대기 중: {start_at.strftime('%Y-%m-%d %H:%M:%S KST')}부터 시작합니다.")
        print("Cloud simulation is waiting for the configured start time.")
        return

    data_source = UpbitPublicDataSource()
    markets = data_source.get_top_krw_markets(args.top_markets)
    app = MultiMarketTradingApp(config, data_source, markets, request_delay=args.request_delay)
    state = load_state(config.paths.state_file)
    if state:
        apply_state(app, state)
    else:
        app.journal.event(
            "cloud_started",
            {
                "start_kst": start_at.isoformat(),
                "end_kst": end_at.isoformat(),
                "starting_cash": config.starting_cash,
                "markets": markets,
            },
        )

    previous_tick = int(state.get("tick", 0)) if state else 0
    next_tick = previous_tick + 1

    if now >= end_at:
        finish_simulation(app, next_tick, end_at)
        save_state(config.paths.state_file, app, next_tick, started_at=start_at, ended=True)
        refresh_outputs(config, outputs)
        print("Cloud simulation finished.")
        return

    target_tick = calculate_target_tick(start_at, now, args.cadence_minutes)
    target_tick = max(target_tick, next_tick)
    target_tick = min(target_tick, previous_tick + max(1, args.max_catch_up_ticks))

    completed_tick = previous_tick
    for tick in range(next_tick, target_tick + 1):
        app.run_tick(tick)
        completed_tick = tick

    save_state(config.paths.state_file, app, completed_tick, started_at=start_at, ended=False)
    refresh_outputs(config, outputs)
    print(f"Cloud simulation ticks completed: {next_tick}-{completed_tick}")


def parse_kst(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def calculate_target_tick(start_at: datetime, now: datetime, cadence_minutes: int) -> int:
    cadence_seconds = max(60, cadence_minutes * 60)
    elapsed_seconds = max(0.0, (now - start_at).total_seconds())
    return int(elapsed_seconds // cadence_seconds) + 1


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_state(app: MultiMarketTradingApp, state: dict[str, Any]) -> None:
    broker = state.get("broker", {})
    position = broker.get("position", {})
    app.broker.market = str(broker.get("market") or app.broker.market)
    app.broker.cash = float(broker.get("cash", app.broker.cash))
    app.broker.realized_pnl = float(broker.get("realized_pnl", app.broker.realized_pnl))
    app.broker.position = Position(
        qty=float(position.get("qty", 0.0)),
        avg_price=float(position.get("avg_price", 0.0)),
        peak_price=float(position.get("peak_price", 0.0)),
        partial_exit_taken=bool(position.get("partial_exit_taken", False)),
    )

    risk = state.get("risk", {})
    app.risk.state = RiskState(
        starting_equity=float(risk.get("starting_equity", app.risk.state.starting_equity)),
        day_key=str(risk.get("day_key", "")),
        entries_today=int(risk.get("entries_today", 0)),
        exits_today=int(risk.get("exits_today", 0)),
        consecutive_losses=int(risk.get("consecutive_losses", 0)),
        halted=bool(risk.get("halted", False)),
        halt_reason=str(risk.get("halt_reason", "")),
        period_started_at=str(risk.get("period_started_at", "")),
        halt_started_tick=int(risk["halt_started_tick"]) if risk.get("halt_started_tick") is not None else None,
        halt_until_tick=int(risk["halt_until_tick"]) if risk.get("halt_until_tick") is not None else None,
    )
    app.current_market = state.get("current_market")
    entry_tick = state.get("position_entry_tick")
    app.position_entry_tick = int(entry_tick) if entry_tick is not None else None
    reentry_until = state.get("market_reentry_until_tick", {})
    if isinstance(reentry_until, dict):
        app.market_reentry_until_tick = {str(key): int(value) for key, value in reentry_until.items()}
    stopout_ticks = state.get("market_stopout_ticks", {})
    if isinstance(stopout_ticks, dict):
        app.market_stopout_ticks = {
            str(key): [int(item) for item in value if item is not None]
            for key, value in stopout_ticks.items()
            if isinstance(value, list)
        }


def save_state(path: Path, app: MultiMarketTradingApp, tick: int, started_at: datetime, ended: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at.isoformat(),
        "last_run_at": datetime.now(KST).isoformat(timespec="seconds"),
        "ended": ended,
        "tick": tick,
        "current_market": app.current_market,
        "position_entry_tick": app.position_entry_tick,
        "market_reentry_until_tick": app.market_reentry_until_tick,
        "market_stopout_ticks": app.market_stopout_ticks,
        "broker": {
            "market": app.broker.market,
            "cash": app.broker.cash,
            "realized_pnl": app.broker.realized_pnl,
            "position": asdict(app.broker.position),
        },
        "risk": asdict(app.risk.state),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finish_simulation(app: MultiMarketTradingApp, tick: int, end_at: datetime) -> None:
    if app.broker.position.is_open:
        market = app.current_market or app.broker.market
        candles = app.data_source.get_recent_candles(market, app.config.strategy.long_window + 5)
        latest_price = candles[-1].close
        fill = app.broker.sell_all(latest_price, "simulation end forced exit")
        if fill is not None:
            app.risk.record_fill(fill)
            app.journal.trade(fill)
            app.journal.event("forced_exit", {"tick": tick, "fill": fill, "risk": app.risk.state})
    app.current_market = None
    app.position_entry_tick = None
    app.journal.event(
        "cloud_finished",
        {
            "tick": tick,
            "end_kst": end_at.isoformat(),
            "cash": app.broker.cash,
            "position": asdict(app.broker.position),
            "risk": asdict(app.risk.state),
        },
    )


def refresh_outputs(config: AppConfig, outputs: list[Path]) -> None:
    html = render_report(read_trades(config.paths.trade_journal), read_events(config.paths.event_log))
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")


def write_status_report(config: AppConfig, outputs: list[Path], message: str) -> None:
    config.paths.trade_journal.parent.mkdir(parents=True, exist_ok=True)
    config.paths.event_log.parent.mkdir(parents=True, exist_ok=True)
    if not config.paths.trade_journal.exists():
        config.paths.trade_journal.write_text(
            "timestamp,market,side,price,qty,fee,cash_after,position_qty_after,realized_pnl,reason\n",
            encoding="utf-8",
        )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "cloud_waiting",
        "payload": {"message": message},
    }
    with config.paths.event_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    refresh_outputs(config, outputs)


def reset_cloud_files(config: AppConfig, outputs: list[Path]) -> None:
    paths = [config.paths.trade_journal, config.paths.event_log, config.paths.state_file, *outputs]
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()


if __name__ == "__main__":
    main()
