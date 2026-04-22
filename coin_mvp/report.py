from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TradeRow:
    timestamp: str
    market: str
    side: str
    price: float
    qty: float
    fee: float
    cash_after: float
    position_qty_after: float
    realized_pnl: float
    reason: str


@dataclass(frozen=True)
class RoundTrip:
    entry: TradeRow
    exit: TradeRow

    @property
    def pnl(self) -> float:
        return self.exit.realized_pnl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the coin MVP HTML report.")
    parser.add_argument("--trades", default="data/trades.csv")
    parser.add_argument("--events", default="logs/events.jsonl")
    parser.add_argument("--output", default="reports/latest_report.html")
    args = parser.parse_args()

    trades = read_trades(Path(args.trades))
    events = read_events(Path(args.events))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(trades, events), encoding="utf-8")
    print(f"리포트 생성 완료: {output}")


def read_trades(path: Path) -> list[TradeRow]:
    if not path.exists():
        return []
    rows: list[TradeRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                TradeRow(
                    timestamp=row["timestamp"],
                    market=row["market"],
                    side=row["side"],
                    price=float(row["price"]),
                    qty=float(row["qty"]),
                    fee=float(row["fee"]),
                    cash_after=float(row["cash_after"]),
                    position_qty_after=float(row["position_qty_after"]),
                    realized_pnl=float(row["realized_pnl"]),
                    reason=row["reason"],
                )
            )
    return rows


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def render_report(trades: list[TradeRow], events: list[dict[str, Any]]) -> str:
    metrics = calculate_metrics(trades)
    pairs = pair_round_trips(trades)
    last_event = events[-1] if events else {}
    last_payload = last_event.get("payload", {}) if isinstance(last_event, dict) else {}
    risk = last_payload.get("risk", {}) if isinstance(last_payload, dict) else {}
    position = last_payload.get("position", {}) if isinstance(last_payload, dict) else {}
    cash = last_payload.get("cash") if isinstance(last_payload, dict) else None
    portfolio = portfolio_summary(trades, events)
    status = status_message(risk, position, float(metrics["total_realized"]))

    cards = [
        ("현재 평가금액", krw(portfolio["current_equity"])),
        ("시작 대비", krw(portfolio["change_amount"])),
        ("현재 수익률", pct(float(portfolio["return_pct"]) / 100.0)),
        ("전체 거래", str(len(trades))),
        ("완료 거래", str(metrics["exit_count"])),
        ("승률", pct(float(metrics["win_rate"]))),
        ("기대값", krw(float(metrics["expectancy"]))),
        ("손익비", ratio(float(metrics["payoff_ratio"]))),
        ("최대 낙폭", krw(float(metrics["max_drawdown"]))),
        ("실현 손익", krw(float(metrics["total_realized"]))),
        ("현재 현금", krw(cash) if isinstance(cash, (int, float)) else "-"),
    ]

    refreshed_at = display_time(str(last_event.get("timestamp", "아직 없음"))) if last_event else "아직 없음"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>코인 페이퍼 트레이딩 리포트</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f6fa;
      --panel: #ffffff;
      --ink: #182230;
      --muted: #667085;
      --line: #d6dde8;
      --soft-line: #e8edf4;
      --green: #087f5b;
      --red: #c23b3b;
      --blue: #2457c5;
      --amber: #a86108;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", "Malgun Gothic", Arial, sans-serif; font-size: 15px; }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 26px 18px 52px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 18px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    h2 {{ font-size: 17px; margin: 0; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .subtitle, .muted {{ color: var(--muted); }}
    .notice {{ background: #eef6ff; border: 1px solid #c9dffc; border-radius: 8px; padding: 13px 16px; margin-bottom: 16px; line-height: 1.55; }}
    .notice strong {{ color: var(--blue); }}
    .portfolio-hero {{ display: grid; grid-template-columns: minmax(260px, 1.4fr) repeat(3, minmax(160px, 1fr)); gap: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
    .portfolio-main, .portfolio-sub {{ padding: 16px 18px; border-right: 1px solid var(--line); }}
    .portfolio-sub:last-child {{ border-right: 0; }}
    .portfolio-title {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .portfolio-amount {{ font-size: 34px; line-height: 1.12; font-weight: 800; letter-spacing: 0; overflow-wrap: anywhere; }}
    .portfolio-value {{ font-size: 23px; line-height: 1.2; font-weight: 800; overflow-wrap: anywhere; }}
    .portfolio-note {{ margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .card {{ padding: 14px; min-height: 72px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 8px; overflow-wrap: anywhere; }}
    section {{ margin-top: 18px; overflow: hidden; }}
    .table-shell {{ overflow: hidden; }}
    .table-wrap {{ width: 100%; overflow: auto; -webkit-overflow-scrolling: touch; max-height: 62vh; border-top: 1px solid var(--soft-line); }}
    table {{ width: 100%; min-width: 980px; border-collapse: separate; border-spacing: 0; table-layout: auto; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--soft-line); text-align: left; vertical-align: top; background: #fff; }}
    th {{ position: sticky; top: 0; z-index: 3; color: var(--muted); font-size: 12px; background: #f8fafc; white-space: nowrap; }}
    th:first-child, td:first-child {{ position: sticky; left: 0; z-index: 2; box-shadow: 1px 0 0 var(--soft-line); }}
    th:first-child {{ z-index: 4; }}
    td {{ white-space: nowrap; }}
    td.text {{ white-space: normal; min-width: 260px; line-height: 1.45; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: 0; }}
    .buy, .blue {{ color: var(--blue); font-weight: 700; }}
    .sell, .neg {{ color: var(--red); font-weight: 700; }}
    .pos, .ok {{ color: var(--green); font-weight: 700; }}
    .warn {{ color: var(--amber); font-weight: 700; }}
    .state-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0; }}
    .state-item {{ padding: 14px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .state-item:nth-child(4n) {{ border-right: 0; }}
    .state-name {{ color: var(--muted); font-size: 12px; margin-bottom: 6px; }}
    .state-value {{ font-weight: 700; overflow-wrap: anywhere; }}
    .diagnosis {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0; }}
    .diagnosis div {{ padding: 14px 16px; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); line-height: 1.55; }}
    .diagnosis div:nth-child(2n) {{ border-right: 0; }}
    @media (max-width: 900px) {{
      main {{ padding: 20px 12px 40px; }}
      header {{ display: block; }}
      .portfolio-hero {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .portfolio-main {{ grid-column: 1 / -1; }}
      .grid, .state-grid, .diagnosis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ min-width: 900px; }}
    }}
    @media (max-width: 560px) {{
      h1 {{ font-size: 24px; }}
      .portfolio-hero {{ grid-template-columns: 1fr; }}
      .portfolio-main, .portfolio-sub {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .portfolio-amount {{ font-size: 29px; }}
      .grid, .state-grid, .diagnosis {{ grid-template-columns: 1fr; }}
      .table-wrap {{ max-height: 70vh; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>코인 페이퍼 트레이딩 리포트</h1>
      <div class="subtitle">수익률보다 먼저 기대값, 손익비, 표본 수, 낙폭을 함께 보는 화면입니다.</div>
    </div>
    <div class="subtitle">마지막 갱신: {html.escape(refreshed_at)}</div>
  </header>
  {render_portfolio_hero(portfolio)}
  <div class="notice"><strong>현재 모드: 모의거래</strong><br>이 화면은 수익을 보장하지 않습니다. 특히 완료 거래가 적을 때는 승률과 기대값이 크게 흔들리므로, 대수의 법칙 관점에서 표본 수와 신뢰구간을 먼저 보세요.</div>
  <div class="grid">{render_cards(cards)}</div>
  <section><h2>현재 상태</h2>{render_state_panel(risk, position, status)}</section>
  <section><h2>수익률 개선 진단</h2>{render_diagnosis(metrics)}</section>
  <section><h2>대수의 법칙 기반 표본 진단</h2>{render_sample_law(metrics)}</section>
  <section><h2>리서치 반영 체크리스트</h2>{render_research_checklist()}</section>
  <section><h2>BTC 분석 프레임워크 적용 상태</h2>{render_btc_framework()}</section>
  <section><h2>자동 수집 시장 데이터</h2>{render_market_context_table(events)}</section>
  <section><h2>AI 의사결정자 로그</h2>{render_ai_decision_table(events)}</section>
  <section><h2>통합 성과 지표</h2>{render_metric_table(metrics)}</section>
  <section><h2>매수 이유별 성과</h2>{render_group_table(group_by_entry_reason(pairs), "매수 이유")}</section>
  <section><h2>청산 이유별 성과</h2>{render_group_table(group_by_exit_reason(pairs), "청산 이유")}</section>
  <section><h2>필터 차단 분석</h2>{render_filter_block_table(events)}</section>
  <section><h2>시간대별 성과</h2>{render_group_table(group_by_exit_hour(pairs), "시간대")}</section>
  <section><h2>최근 거래</h2>{render_trade_table(trades[-60:])}</section>
  <section><h2>최근 판단 로그</h2>{render_event_table(events[-40:])}</section>
</main>
</body>
</html>
"""


def calculate_metrics(trades: list[TradeRow]) -> dict[str, float | int]:
    exits = [trade for trade in trades if trade.side == "sell"]
    pnls = [trade.realized_pnl for trade in exits]
    wins = [trade.realized_pnl for trade in exits if trade.realized_pnl > 0]
    losses = [trade.realized_pnl for trade in exits if trade.realized_pnl < 0]
    total_realized = sum(pnls)
    exit_count = len(exits)
    win_rate = len(wins) / exit_count if exit_count else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss) if exit_count else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else 0.0
    expectancy_se = standard_error(pnls)
    ci_low, ci_high = wilson_interval(win_rate, exit_count)
    return {
        "exit_count": exit_count,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": calculate_max_drawdown(pnls),
        "max_consecutive_losses": calculate_max_consecutive_losses(pnls),
        "expectancy_se": expectancy_se,
        "expectancy_ci_low": expectancy - 1.96 * expectancy_se if exit_count >= 2 else 0.0,
        "expectancy_ci_high": expectancy + 1.96 * expectancy_se if exit_count >= 2 else 0.0,
        "win_rate_ci_low": ci_low,
        "win_rate_ci_high": ci_high,
        "total_fee": sum(trade.fee for trade in trades),
        "total_realized": total_realized,
    }


def pair_round_trips(trades: list[TradeRow]) -> list[RoundTrip]:
    pairs: list[RoundTrip] = []
    open_entry: TradeRow | None = None
    for trade in trades:
        if trade.side == "buy":
            open_entry = trade
        elif trade.side == "sell" and open_entry is not None:
            pairs.append(RoundTrip(open_entry, trade))
            open_entry = None
    return pairs


def calculate_max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def calculate_max_consecutive_losses(pnls: list[float]) -> int:
    current = 0
    worst = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            worst = max(worst, current)
        elif pnl > 0:
            current = 0
    return worst


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def wilson_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    denominator = 1.0 + (z * z / n)
    centre = (rate + (z * z / (2 * n))) / denominator
    margin = (z * math.sqrt((rate * (1 - rate) / n) + (z * z / (4 * n * n)))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def render_cards(cards: list[tuple[str, str]]) -> str:
    return "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )


def portfolio_summary(trades: list[TradeRow], events: list[dict[str, Any]]) -> dict[str, float | str]:
    starting_cash = find_starting_cash(events) or 1_000_000.0
    latest_equity = find_latest_equity(events)
    latest_price = find_latest_price(events)
    cash = find_latest_cash(trades, events)
    open_position_qty = 0.0
    open_position_market = ""

    for trade in trades:
        if trade.side == "buy":
            open_position_qty = trade.position_qty_after
            open_position_market = trade.market
        elif trade.side == "sell":
            open_position_qty = 0.0
            open_position_market = ""

    if latest_equity is None:
        if cash is not None and open_position_qty > 0 and latest_price is not None:
            latest_equity = cash + (open_position_qty * latest_price)
        elif cash is not None:
            latest_equity = cash
        else:
            latest_equity = starting_cash

    change_amount = latest_equity - starting_cash
    return_pct = (change_amount / starting_cash) * 100.0 if starting_cash else 0.0
    position_status = f"{open_position_market} 보유 중" if open_position_qty > 0 else "미보유"
    return {
        "starting_cash": starting_cash,
        "current_equity": latest_equity,
        "change_amount": change_amount,
        "return_pct": return_pct,
        "cash": cash if cash is not None else latest_equity,
        "position_qty": open_position_qty,
        "position_status": position_status,
    }


def render_portfolio_hero(portfolio: dict[str, float | str]) -> str:
    change = float(portfolio["change_amount"])
    change_class = "pos" if change > 0 else "neg" if change < 0 else ""
    position_status = str(portfolio["position_status"])
    return f"""
  <div class="portfolio-hero">
    <div class="portfolio-main">
      <div class="portfolio-title">현재 평가금액</div>
      <div class="portfolio-amount">{html.escape(krw(float(portfolio["current_equity"])))}</div>
      <div class="portfolio-note">시작금액 {html.escape(krw(float(portfolio["starting_cash"])))} 기준입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">시작 대비 손익</div>
      <div class="portfolio-value {change_class}">{html.escape(krw(change))}</div>
      <div class="portfolio-note">실현/미실현 평가를 합친 기준입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">현재 수익률</div>
      <div class="portfolio-value {change_class}">{html.escape(pct(float(portfolio["return_pct"]) / 100.0))}</div>
      <div class="portfolio-note">100만원 대비 변화율입니다.</div>
    </div>
    <div class="portfolio-sub">
      <div class="portfolio-title">보유 상태</div>
      <div class="portfolio-value">{html.escape(position_status)}</div>
      <div class="portfolio-note">현금 {html.escape(krw(float(portfolio["cash"])))}</div>
    </div>
  </div>
"""


def find_starting_cash(events: list[dict[str, Any]]) -> float | None:
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("starting_cash"))
            if value is not None:
                return value
    return None


def find_latest_equity(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("equity"))
            if value is not None:
                return value
    return None


def find_latest_price(events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("price"))
            if value is not None:
                return value
    return None


def find_latest_cash(trades: list[TradeRow], events: list[dict[str, Any]]) -> float | None:
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            value = to_float(payload.get("cash"))
            if value is not None:
                return value
            fill = payload.get("fill")
            if isinstance(fill, dict):
                value = to_float(fill.get("cash_after"))
                if value is not None:
                    return value
    if trades:
        return trades[-1].cash_after
    return None


def render_state_panel(risk: dict[str, Any], position: dict[str, Any], status: str) -> str:
    halted = bool(risk.get("halted"))
    position_qty = to_float(position.get("qty")) or 0.0
    items = [
        ("운영 상태", "중지" if halted else "진행 가능", "warn" if halted else "ok"),
        ("상태 요약", status, ""),
        ("오늘 신규 진입", str(risk.get("entries_today", 0)), ""),
        ("오늘 청산", str(risk.get("exits_today", 0)), ""),
        ("연속 손실", str(risk.get("consecutive_losses", 0)), ""),
        ("중지 사유", korean_reason(str(risk.get("halt_reason") or "없음")), "warn" if halted else ""),
        ("보유 수량", f"{position_qty:.8f}", ""),
        ("평균 단가", krw(to_float(position.get("avg_price"))), ""),
    ]
    return '<div class="state-grid">' + "".join(render_state_item(*item) for item in items) + "</div>"


def render_state_item(label: str, value: str, klass: str) -> str:
    value_class = f' class="state-value {klass}"' if klass else ' class="state-value"'
    return f'<div class="state-item"><div class="state-name">{html.escape(label)}</div><div{value_class}>{html.escape(value)}</div></div>'


def render_diagnosis(metrics: dict[str, float | int]) -> str:
    items = []
    if int(metrics["exit_count"]) < 30:
        items.append(("표본 부족", "완료 거래가 30건 미만입니다. 지금의 승률은 우연의 영향이 커서 전략 우열을 판단하기 어렵습니다.", "warn"))
    if float(metrics["expectancy"]) <= 0:
        items.append(("기대값 음수", "거래 1회당 평균 기대 손익이 0 이하입니다. 진입 조건을 더 엄격하게 하고 과열 구간 진입을 줄여야 합니다.", "neg"))
    else:
        items.append(("기대값 양수", "현재 기록만 보면 거래 1회당 기대 손익은 양수입니다. 다만 표본이 충분한지 함께 확인해야 합니다.", "ok"))
    if float(metrics["payoff_ratio"]) < 1:
        items.append(("손익비 약함", "평균 이익이 평균 손실보다 작습니다. 승률이 아주 높지 않다면 장기적으로 불리해질 수 있습니다.", "warn"))
    if int(metrics["max_consecutive_losses"]) >= 3:
        items.append(("연속 손실 주의", "연속 손실이 3회 이상 발생했습니다. 손실 후 재진입 대기 시간을 늘리는 편이 좋습니다.", "warn"))
    if float(metrics["profit_factor"]) and float(metrics["profit_factor"]) < 1.2:
        items.append(("수익 팩터 낮음", "총이익 대비 총손실 비율이 낮습니다. 후보 선별과 BTC 장세 필터가 더 중요합니다.", "warn"))
    return '<div class="diagnosis">' + "".join(
        f'<div><strong class="{klass}">{html.escape(title)}</strong><br>{html.escape(body)}</div>'
        for title, body, klass in items
    ) + "</div>"


def render_metric_table(metrics: dict[str, float | int]) -> str:
    rows = [
        ("완료 거래 수", str(metrics["exit_count"]), "승률과 기대값 계산에 사용한 청산 완료 거래 수입니다."),
        ("승리 / 패배", f'{metrics["win_count"]} / {metrics["loss_count"]}', "실현 손익이 양수/음수인 청산 거래 수입니다."),
        ("평균 이익", krw(float(metrics["avg_win"])), "이익 거래의 평균 실현 손익입니다."),
        ("평균 손실", krw(float(metrics["avg_loss"])), "손실 거래의 평균 실현 손익입니다."),
        ("손익비", ratio(float(metrics["payoff_ratio"])), "평균 이익 / 평균 손실 절댓값입니다."),
        ("기대값", krw(float(metrics["expectancy"])), "거래 1회당 평균적으로 기대하는 손익입니다."),
        ("기대값 95% 근사 범위", f'{krw(float(metrics["expectancy_ci_low"]))} ~ {krw(float(metrics["expectancy_ci_high"]))}', "표본 기반 근사 구간입니다. 표본이 적으면 넓게 흔들립니다."),
        ("승률 95% 신뢰구간", f'{pct(float(metrics["win_rate_ci_low"]))} ~ {pct(float(metrics["win_rate_ci_high"]))}', "대수의 법칙 관점에서 승률 추정의 불확실성을 보여줍니다."),
        ("수익 팩터", ratio(float(metrics["profit_factor"])), "총이익 / 총손실 절댓값입니다. 1보다 커야 합니다."),
        ("최대 낙폭", krw(float(metrics["max_drawdown"])), "실현 손익 누적 기준 최고점 대비 최악 하락폭입니다."),
        ("최대 연속 손실", str(metrics["max_consecutive_losses"]), "손실 청산이 연속으로 발생한 최댓값입니다."),
        ("총 수수료", krw(float(metrics["total_fee"])), "모든 모의 체결에 반영된 수수료입니다."),
    ]
    return render_simple_table(["지표", "값", "해석"], rows, text_cols={2}, num_cols={1})


def render_sample_law(metrics: dict[str, float | int]) -> str:
    n = int(metrics["exit_count"])
    if n < 30:
        level = "낮음"
        note = "아직 표본이 매우 적습니다. 방향만 참고하고 전략을 자주 바꾸지 않는 편이 좋습니다."
    elif n < 100:
        level = "중간"
        note = "초기 판단은 가능하지만 오차가 큽니다. 최소 100건 이상 완료 거래를 쌓으면 안정성이 좋아집니다."
    else:
        level = "높음"
        note = "표본이 어느 정도 쌓였습니다. 그래도 시장 국면이 바뀌면 분포도 달라질 수 있습니다."
    rows = [
        ("완료 표본 수", str(n), "대수의 법칙은 완료 거래가 많아질수록 평균과 승률 추정이 안정된다는 아이디어입니다."),
        ("신뢰 수준", level, note),
        ("승률 추정 범위", f'{pct(float(metrics["win_rate_ci_low"]))} ~ {pct(float(metrics["win_rate_ci_high"]))}', "표본이 늘수록 구간이 좁아지는지 확인하세요."),
        ("기대값 추정 범위", f'{krw(float(metrics["expectancy_ci_low"]))} ~ {krw(float(metrics["expectancy_ci_high"]))}', "구간 전체가 0보다 높아질 때 전략 신뢰도가 올라갑니다."),
        ("운영 원칙", "표본 확보 전 급격한 수정 금지", "30건 전에는 관찰, 30~100건은 완만한 조정, 100건 이후부터 본격 비교가 적절합니다."),
    ]
    return render_simple_table(["항목", "값", "해석"], rows, text_cols={2}, num_cols={1})


def render_research_checklist() -> str:
    rows = [
        ("단일 신호 의존 축소", "적용", "이동평균만 보지 않고 200EMA, 모멘텀, 거래량, RSI 과열, BTC 장세를 함께 확인합니다."),
        ("실행비용 반영", "적용", "수수료와 슬리피지를 모의 체결에 반영합니다. 실제 호가 깊이 기반 비용은 다음 단계입니다."),
        ("변동성 기반 포지션 조절", "적용", "최근 변동성이 목표보다 크면 같은 신호라도 진입 금액을 자동 축소합니다."),
        ("PIT/상폐 편향 통제", "부분 적용", "현재 업비트 활성 상위 마켓 스캔이라 완전한 PIT 유니버스는 아닙니다. 리포트에서 한계로 추적합니다."),
        ("표본 기반 검증", "적용", "완료 거래 수, 승률 신뢰구간, 기대값 신뢰구간으로 대수의 법칙 관점의 신뢰도를 봅니다."),
        ("손절 + 시간손절", "적용", "가격 손절과 보유시간 기준 시간손절을 함께 적용합니다."),
        ("Walk-forward 검증", "미적용", "현재는 실시간 paper 관찰 단계입니다. 충분한 로그가 쌓인 뒤 구간별 검증을 추가해야 합니다."),
    ]
    return render_simple_table(["항목", "상태", "현재 MVP 반영"], rows, text_cols={2})


def render_btc_framework() -> str:
    rows = [
        ("거시 경제", "수동 확인", "금리, CPI/PPI, 달러 인덱스 같은 외부 변수는 아직 자동 수집하지 않습니다."),
        ("온체인 지표", "수동 확인", "MVRV, SOPR, Puell Multiple 등은 MVP 범위 밖입니다. 추후 별도 데이터 API가 필요합니다."),
        ("전략 비교", "부분 적용", "현재는 이동평균 추세 전략에 거래량, 과열, BTC 장세 필터를 추가하는 방향으로 개선했습니다."),
        ("총매수 분석", "수동 확인", "ETF 자금 유입, 기관 리포트, 규제 뉴스는 자동 매매 근거가 아니라 관찰 필터로 다루는 편이 안전합니다."),
        ("리스크 분석", "적용", "일 손실 한도, 일 진입 횟수, 연속 손실 한도, 변동성 기반 진입금액 축소, 표본 기반 진단을 함께 봅니다."),
        ("3일 관찰 목적", "적용", "목표는 단기 수익이 아니라 후보 선별, 기대값, 손익비, 낙폭, 표본 수가 개선되는지 확인하는 것입니다."),
    ]
    return render_simple_table(["분석 영역", "현재 상태", "적용 내용"], rows, text_cols={2})


def render_group_table(groups: list[dict[str, Any]], first_header: str) -> str:
    if not groups:
        return empty_block("아직 분석할 완료 거래가 없습니다.")
    rows = []
    for group in groups:
        cls = "pos" if group["total_pnl"] > 0 else "neg" if group["total_pnl"] < 0 else ""
        rows.append(
            (
                korean_reason(str(group["name"])),
                str(group["count"]),
                pct(group["win_rate"]),
                krw(group["avg_pnl"]),
                f'<span class="{cls}">{html.escape(krw(group["total_pnl"]))}</span>',
            )
        )
    return render_simple_table([first_header, "거래 수", "승률", "평균 손익", "총 손익"], rows, raw_cols={4}, text_cols={0}, num_cols={1, 2, 3, 4})


def render_filter_block_table(events: list[dict[str, Any]]) -> str:
    groups = analyze_filter_blocks(events)
    if not groups:
        return empty_block("아직 필터 차단 로그가 충분하지 않습니다. 다음 관찰부터 스캔 차단 사유가 누적됩니다.")
    rows = []
    for group in groups:
        avg_change = group["avg_next_change_pct"]
        avg_text = pct(avg_change / 100.0) if isinstance(avg_change, (int, float)) else "표본 부족"
        rows.append(
            (
                korean_reason(str(group["reason"])),
                str(group["count"]),
                str(group["priced_samples"]),
                avg_text,
                group["note"],
            )
        )
    return render_simple_table(["차단 이유", "차단 횟수", "가격 추적 표본", "이후 평균 변화", "해석"], rows, text_cols={0, 4}, num_cols={1, 2, 3})


def render_ai_decision_table(events: list[dict[str, Any]]) -> str:
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        decision = payload.get("ai_decision")
        if not isinstance(decision, dict):
            continue
        context = payload.get("decision_context", {})
        context_reason = context.get("reason", "") if isinstance(context, dict) else ""
        notes = decision.get("risk_notes", [])
        if isinstance(notes, list):
            risk_notes = "; ".join(str(note) for note in notes) or "-"
        else:
            risk_notes = str(notes or "-")
        rows.append(
            (
                display_time(str(event.get("timestamp", ""))),
                str(payload.get("market", "-")),
                str(decision.get("action", "-")),
                str(decision.get("source", "-")),
                pct((to_float(decision.get("confidence")) or 0.0)),
                pct((to_float(decision.get("expected_upside_pct")) or 0.0) / 100.0),
                pct((to_float(decision.get("expected_downside_pct")) or 0.0) / 100.0),
                str(decision.get("thesis", "-")),
                str(decision.get("invalidation", "-")),
                risk_notes,
                str(context_reason),
            )
        )
        if len(rows) >= 30:
            break
    if not rows:
        return empty_block("AI decision logs are not available yet. New candidate reviews will appear here.")
    return render_simple_table(
        ["시간", "마켓", "판단", "출처", "신뢰도", "예상 상승", "예상 하락", "판단 근거", "무효 조건", "위험 메모", "시장 컨텍스트"],
        rows,
        text_cols={7, 8, 9, 10},
        num_cols={4, 5, 6},
    )


def render_market_context_table(events: list[dict[str, Any]]) -> str:
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue
        context = payload.get("decision_context")
        if not isinstance(context, dict):
            continue
        rows.append(
            (
                display_time(str(event.get("timestamp", ""))),
                "허용" if context.get("allows_entries") else "차단",
                pct((to_float(context.get("score_multiplier")) or 0.0) - 1.0),
                pct((to_float(context.get("btc_momentum_pct")) or 0.0) / 100.0),
                pct((to_float(context.get("btc_volatility_pct")) or 0.0) / 100.0),
                str(context.get("fear_greed_value") if context.get("fear_greed_value") is not None else "-"),
                pct((to_float(context.get("global_market_cap_change_pct")) or 0.0) / 100.0)
                if context.get("global_market_cap_change_pct") is not None
                else "-",
                pct((to_float(context.get("btc_dominance_pct")) or 0.0) / 100.0)
                if context.get("btc_dominance_pct") is not None
                else "-",
                pct((to_float(context.get("binance_btcusdt_change_pct")) or 0.0) / 100.0)
                if context.get("binance_btcusdt_change_pct") is not None
                else "-",
                str(context.get("reason", "-")),
            )
        )
        if len(rows) >= 30:
            break
    if not rows:
        return empty_block("자동 수집 시장 데이터가 아직 없습니다. 다음 후보 스캔부터 기록됩니다.")
    return render_simple_table(
        ["시간", "진입", "점수 조정", "BTC 모멘텀", "BTC 변동성", "공포탐욕", "글로벌 시총 24h", "BTC 점유율", "Binance BTC 24h", "요약"],
        rows,
        text_cols={9},
        num_cols={2, 3, 4, 6, 7, 8},
    )


def analyze_filter_blocks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    observations: dict[str, list[tuple[int, float]]] = {}

    for index, event in enumerate(events):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if not isinstance(payload, dict):
            continue

        blocked_reasons = payload.get("blocked_reasons", {})
        if isinstance(blocked_reasons, dict):
            for reason, count in blocked_reasons.items():
                counts[str(reason)] = counts.get(str(reason), 0) + int(count)

        blocked_samples = payload.get("blocked_samples", [])
        if isinstance(blocked_samples, list):
            for sample in blocked_samples:
                if not isinstance(sample, dict):
                    continue
                market = str(sample.get("market", ""))
                reason = str(sample.get("reason", ""))
                price = to_float(sample.get("price"))
                if market and reason and price:
                    samples.append({"index": index, "market": market, "reason": reason, "price": price})
                    observations.setdefault(market, []).append((index, price))

        market = payload.get("market")
        price = to_float(payload.get("price"))
        if market and price:
            observations.setdefault(str(market), []).append((index, price))

        fill = payload.get("fill", {})
        if isinstance(fill, dict):
            fill_market = fill.get("market")
            fill_price = to_float(fill.get("price"))
            if fill_market and fill_price:
                observations.setdefault(str(fill_market), []).append((index, fill_price))

    changes_by_reason: dict[str, list[float]] = {}
    for sample in samples:
        later_price = next_later_price(observations.get(sample["market"], []), int(sample["index"]))
        if later_price is None:
            continue
        change_pct = (later_price / float(sample["price"]) - 1.0) * 100.0
        changes_by_reason.setdefault(str(sample["reason"]), []).append(change_pct)

    for reason in changes_by_reason:
        counts.setdefault(reason, 0)

    groups = []
    for reason, count in counts.items():
        changes = changes_by_reason.get(reason, [])
        if changes:
            avg_change = sum(changes) / len(changes)
            note = "차단 후 관측 가격 기준입니다. 양수면 놓친 상승, 음수면 회피한 하락 가능성을 뜻합니다."
        else:
            avg_change = None
            note = "이후 가격 표본이 아직 부족합니다."
        groups.append(
            {
                "reason": reason,
                "count": count,
                "priced_samples": len(changes),
                "avg_next_change_pct": avg_change,
                "note": note,
            }
        )
    return sorted(groups, key=lambda group: group["count"], reverse=True)


def next_later_price(observations: list[tuple[int, float]], current_index: int) -> float | None:
    for index, price in observations:
        if index > current_index:
            return price
    return None


def render_trade_table(trades: list[TradeRow]) -> str:
    if not trades:
        return empty_block("아직 거래 기록이 없습니다.")
    rows = []
    for trade in reversed(trades):
        pnl_class = "pos" if trade.realized_pnl > 0 else "neg" if trade.realized_pnl < 0 else ""
        side_class = "buy" if trade.side == "buy" else "sell"
        rows.append(
            (
                display_time(trade.timestamp),
                trade.market,
                f'<span class="{side_class}">{html.escape(korean_side(trade.side))}</span>',
                f"{trade.price:,.4f}",
                f"{trade.qty:.8f}",
                krw(trade.fee),
                f'<span class="{pnl_class}">{html.escape(krw(trade.realized_pnl))}</span>',
                korean_reason(trade.reason),
            )
        )
    return render_simple_table(["시간", "마켓", "구분", "가격", "수량", "수수료", "손익", "이유"], rows, raw_cols={2, 6}, text_cols={7}, num_cols={3, 4, 5, 6})


def render_event_table(events: list[dict[str, Any]]) -> str:
    if not events:
        return empty_block("아직 판단 로그가 없습니다.")
    rows = []
    for event in reversed(events):
        payload = event.get("payload", {})
        rows.append((display_time(str(event.get("timestamp", ""))), korean_event(str(event.get("event", ""))), summarize_event(str(event.get("event", "")), payload)))
    return render_simple_table(["시간", "이벤트", "요약"], rows, text_cols={2})


def render_simple_table(
    headers: list[str],
    rows: list[tuple[Any, ...]],
    raw_cols: set[int] | None = None,
    text_cols: set[int] | None = None,
    num_cols: set[int] | None = None,
) -> str:
    raw_cols = raw_cols or set()
    text_cols = text_cols or set()
    num_cols = num_cols or set()
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            classes = []
            if index in text_cols:
                classes.append("text")
            if index in num_cols:
                classes.append("num")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            if index in raw_cols:
                cells.append(f"<td{class_attr}>{value}</td>")
            else:
                cells.append(f"<td{class_attr}>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-shell"><div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div></div>'


def empty_block(message: str) -> str:
    return f'<div style="padding:16px;color:#667085;">{html.escape(message)}</div>'


def group_by_entry_reason(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: pair.entry.reason)


def group_by_exit_reason(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: pair.exit.reason.split(":")[0])


def group_by_exit_hour(pairs: list[RoundTrip]) -> list[dict[str, Any]]:
    return group_pairs(pairs, lambda pair: display_time(pair.exit.timestamp)[11:13] + "시")


def group_pairs(pairs: list[RoundTrip], key_func: Callable[[RoundTrip], str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    for pair in pairs:
        buckets.setdefault(str(key_func(pair)), []).append(pair.pnl)
    groups = []
    for name, pnls in buckets.items():
        wins = [pnl for pnl in pnls if pnl > 0]
        groups.append(
            {
                "name": name,
                "count": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                "total_pnl": sum(pnls),
            }
        )
    return sorted(groups, key=lambda group: group["total_pnl"], reverse=True)


def summarize_event(name: str, payload: dict[str, Any]) -> str:
    if name == "tick":
        signal = payload.get("signal", {})
        approved = "승인" if payload.get("approved") else "대기/차단"
        market = payload.get("market")
        market_part = f"{market}, " if market else ""
        candidate_part = f", 후보 {payload.get('candidate_count')}개" if payload.get("candidate_count") is not None else ""
        return (
            f'{market_part}가격 {num(payload.get("price"))}, 평가금액 {num(payload.get("equity"))}, '
            f'신호 {korean_side(str(signal.get("side")))} ({korean_reason(str(signal.get("reason")))}) , '
            f'판정 {approved}, 리스크 사유: {korean_reason(str(payload.get("risk_reason")))}{candidate_part}'
        )
    if name == "market_scan":
        reason = korean_reason(str(payload.get("reason", "")))
        return f'스캔 마켓 {payload.get("markets_scanned", 0)}개, 후보 {payload.get("candidates", 0)}개, 사유: {reason}'
    if name in {"fill", "forced_exit"}:
        fill = payload.get("fill", {})
        return (
            f'{korean_side(str(fill.get("side")))} 체결, '
            f'수량 {fill.get("qty")}, 손익 {krw(to_float(fill.get("realized_pnl")))}, '
            f'이유: {korean_reason(str(fill.get("reason")))}'
        )
    if name == "bot_finished":
        return f'종료 현금 {krw(to_float(payload.get("cash")))}, 상태 {korean_risk(payload.get("risk", {}))}'
    return json.dumps(payload, ensure_ascii=False)[:260]


def status_message(risk: dict[str, Any], position: dict[str, Any], total_realized: float) -> str:
    if risk.get("halted"):
        return f'거래 중지: {korean_reason(str(risk.get("halt_reason") or "사유 없음"))}'
    if (to_float(position.get("qty")) or 0.0) > 0:
        return "포지션 보유 중입니다. 청산 조건을 계속 감시하고 있습니다."
    if total_realized > 0:
        return "실현 손익이 플러스입니다. 표본이 더 쌓여도 유지되는지 확인하세요."
    if total_realized < 0:
        return "실현 손익이 마이너스입니다. 손실 제한과 진입 조건을 더 보수적으로 봐야 합니다."
    return "아직 충분한 실현 손익이 없습니다. 신호를 관찰 중입니다."


def korean_side(side: str) -> str:
    return {"buy": "매수", "sell": "매도", "hold": "대기", "none": "없음"}.get(side.lower(), side)


def korean_event(name: str) -> str:
    return {
        "bot_started": "봇 시작",
        "watch_started": "관찰 시작",
        "market_scan": "마켓 스캔",
        "market_scan_error": "스캔 오류",
        "tick": "판단",
        "fill": "체결",
        "forced_exit": "강제 청산",
        "fill_skipped": "체결 생략",
        "bot_error": "오류",
        "watch_error": "관찰 오류",
        "bot_finished": "봇 종료",
        "watch_finished": "관찰 종료",
    }.get(name, name)


def korean_reason(reason: str) -> str:
    if not reason or reason == "None":
        return "없음"
    replacements = [
        ("not enough candles", "캔들 데이터 부족"),
        ("position open, no exit condition", "보유 중이나 청산 조건 없음"),
        ("uptrend filter passed", "상승 추세 조건 충족"),
        ("btc regime blocked", "BTC 장세 필터 차단"),
        ("long trend filter blocked", "장기 EMA 필터 차단"),
        ("price below EMA", "가격이 장기 EMA 아래"),
        ("overextended", "단기 과열"),
        ("weak recent momentum", "최근 모멘텀 약함"),
        ("thin volume", "거래량 부족"),
        ("trend break", "추세 이탈"),
        ("no entry condition", "진입 조건 없음"),
        ("hold signal", "대기 신호"),
        ("approved risk-reducing exit", "리스크 축소 청산 승인"),
        ("approved", "승인"),
        ("max daily entries reached", "하루 신규 진입 한도 도달"),
        ("max consecutive losses reached", "연속 손실 한도 도달"),
        ("position fraction exceeds risk limit", "포지션 비중 한도 초과"),
        ("daily profit target reached", "하루 수익 목표 도달"),
        ("daily loss limit reached", "하루 손실 한도 도달"),
        ("stop loss reached", "손절 조건 도달"),
        ("take profit reached", "익절 조건 도달"),
        ("time stop reached", "시간손절 조건 도달"),
        ("forced exit", "강제 청산"),
        ("selected from top-volume scan", "상위 거래대금 스캔에서 선택"),
    ]
    translated = reason
    for source, target in replacements:
        translated = translated.replace(source, target)
    return translated


def korean_risk(risk: dict[str, Any]) -> str:
    if not isinstance(risk, dict):
        return str(risk)
    halted = "중지" if risk.get("halted") else "진행 가능"
    return (
        f'{halted}, 신규 진입 {risk.get("entries_today", 0)}회, '
        f'청산 {risk.get("exits_today", 0)}회, '
        f'연속 손실 {risk.get("consecutive_losses", 0)}회'
    )


def krw(value: float | int | None) -> str:
    if value is None:
        return "-"
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):,.0f} KRW"


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.1f}%"


def ratio(value: float | int | None) -> str:
    if value is None or float(value) == 0.0:
        return "-"
    return f"{float(value):.2f}x"


def num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def short_time(value: str) -> str:
    if "T" in value:
        date, rest = value.split("T", 1)
        return f"{date} {rest[:8]}"
    return value[:19]


def display_time(value: str) -> str:
    parsed = parse_utc_time(value)
    if parsed is None:
        return short_time(value)
    kst = parsed.astimezone(timezone(timedelta(hours=9)))
    return kst.strftime("%Y-%m-%d %H:%M:%S KST")


def parse_utc_time(value: str) -> datetime | None:
    if not value or "T" not in value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
