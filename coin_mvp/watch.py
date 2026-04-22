from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv

from .app import TradingApp
from .config import load_config
from .data import SampleMarketDataSource, UpbitPublicDataSource, sleep_between_ticks
from .report import calculate_metrics, read_events, read_trades, render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long paper observation and refresh the report.")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--source", choices=["sample", "upbit"], default="upbit")
    parser.add_argument("--ticks", type=int, default=960, help="Total ticks. With 15s polling, 960 ticks is about 4 hours.")
    parser.add_argument("--report-every", type=int, default=20, help="Refresh report every N ticks.")
    parser.add_argument("--output", default="reports/latest_report.html")
    parser.add_argument("--continue-after-halt", action="store_true", help="Keep observing and refreshing reports after risk halt.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_source = SampleMarketDataSource() if args.source == "sample" else UpbitPublicDataSource()
    app = TradingApp(config=config, data_source=data_source, source_name=args.source)
    output = Path(args.output)

    app.journal.event(
        "watch_started",
        {
            "mode": config.mode,
            "market": config.market,
            "source": args.source,
            "ticks": args.ticks,
            "report_every": args.report_every,
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)

    for tick in range(1, args.ticks + 1):
        try:
            app._run_tick(tick)
            if tick % args.report_every == 0:
                refresh_report(config.paths.trade_journal, config.paths.event_log, output)
                print(f"Report refreshed at tick {tick}: {output}")
            if app.risk.state.halted and not args.continue_after_halt:
                print(f"Watch halted by risk manager: {app.risk.state.halt_reason}")
                break
        except Exception as exc:
            app.journal.event("watch_error", {"tick": tick, "error": repr(exc)})
            refresh_report(config.paths.trade_journal, config.paths.event_log, output)
            raise
        sleep_between_ticks(config.poll_seconds, args.source)

    app.journal.event(
        "watch_finished",
        {
            "cash": app.broker.cash,
            "position": asdict(app.broker.position),
            "risk": asdict(app.risk.state),
        },
    )
    refresh_report(config.paths.trade_journal, config.paths.event_log, output)
    print(f"Watch finished. Report: {output}")


def refresh_report(trade_path: Path, event_path: Path, output: Path) -> None:
    trades = read_trades(trade_path)
    events = read_events(event_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(trades, events), encoding="utf-8")
    append_metrics_snapshot(Path("data/metrics_snapshots.csv"), trades)


def append_metrics_snapshot(path: Path, trades: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(trades)
    is_new = not path.exists() or path.stat().st_size == 0
    fields = [
        "timestamp_utc",
        "exit_count",
        "win_count",
        "loss_count",
        "win_rate",
        "expectancy",
        "payoff_ratio",
        "profit_factor",
        "max_drawdown",
        "max_consecutive_losses",
        "total_fee",
        "total_realized",
    ]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{field: metrics.get(field, "") for field in fields if field != "timestamp_utc"},
    }
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
