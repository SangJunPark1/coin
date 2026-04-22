from __future__ import annotations

import argparse

from .app import TradingApp
from .config import load_config
from .data import SampleMarketDataSource, UpbitPublicDataSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the coin spot MVP in paper mode.")
    parser.add_argument("--config", default="config.example.json", help="Path to the JSON config file.")
    parser.add_argument(
        "--source",
        choices=["sample", "upbit"],
        default="sample",
        help="Market data source. sample is for offline tests.",
    )
    parser.add_argument("--ticks", type=int, default=80, help="Number of loop ticks to run.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_source = SampleMarketDataSource() if args.source == "sample" else UpbitPublicDataSource()

    app = TradingApp(config=config, data_source=data_source, source_name=args.source)
    app.run(ticks=args.ticks)
    print(f"Paper run finished. Trade journal: {config.paths.trade_journal}")
    print(f"Event log: {config.paths.event_log}")


if __name__ == "__main__":
    main()
