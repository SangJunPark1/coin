from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Fill


class Journal:
    def __init__(self, trade_path: Path, event_path: Path) -> None:
        self.trade_path = trade_path
        self.event_path = event_path
        self.trade_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_trade_header()

    def trade(self, fill: Fill) -> None:
        with self.trade_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_trade_fields())
            row = {
                "timestamp": fill.timestamp.isoformat(),
                "market": fill.market,
                "side": fill.side.value,
                "price": f"{fill.price:.8f}",
                "qty": f"{fill.qty:.12f}",
                "fee": f"{fill.fee:.8f}",
                "cash_after": f"{fill.cash_after:.8f}",
                "position_qty_after": f"{fill.position_qty_after:.12f}",
                "realized_pnl": f"{fill.realized_pnl:.8f}",
                "reason": fill.reason,
            }
            writer.writerow(row)

    def event(self, name: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": name,
            "payload": _json_safe(payload),
        }
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _ensure_trade_header(self) -> None:
        if self.trade_path.exists() and self.trade_path.stat().st_size > 0:
            return
        with self.trade_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_trade_fields())
            writer.writeheader()


def _trade_fields() -> list[str]:
    return [
        "timestamp",
        "market",
        "side",
        "price",
        "qty",
        "fee",
        "cash_after",
        "position_qty_after",
        "realized_pnl",
        "reason",
    ]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value
