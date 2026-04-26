from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .config import AiDecisionConfig, StrategyConfig
from .market_context import DecisionContext
from .models import Candle, Signal
from .strategy import estimate_expected_downside_pct, estimate_expected_upside_pct, recent_volatility_pct


@dataclass(frozen=True)
class DecisionReview:
    action: str
    grade: str
    confidence: float
    thesis: str
    invalidation: str
    expected_upside_pct: float
    expected_downside_pct: float
    risk_notes: list[str]
    source: str = "local"
    input_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def review_entry_candidate(
    signal: Signal,
    candles: list[Candle],
    context: DecisionContext,
    strategy: StrategyConfig,
    ai_config: AiDecisionConfig,
) -> DecisionReview:
    """Review a deterministic entry signal before the risk layer can approve it.

    The current implementation is a local structured reviewer. It creates the
    same JSON-shaped output that an external AI provider should later return.
    """

    decision_input = build_decision_input(signal, candles, context, strategy)
    fallback_source = "local" if ai_config.provider.lower() == "local" else f"local-fallback:{ai_config.provider}"
    fallback_reason: str | None = None
    if ai_config.enabled and ai_config.provider.lower() == "openai":
        external, fallback_reason = try_openai_review(decision_input, ai_config)
        if external is not None:
            return external

    expected_upside = float(decision_input["candidate"]["expected_upside_pct"])
    expected_downside = estimate_expected_downside_pct(candles, strategy.stop_loss_pct, strategy.stop_volatility_multiplier)
    risk_notes: list[str] = []

    if not ai_config.enabled:
        return DecisionReview(
            action="buy",
            grade="A",
            confidence=signal.confidence,
            thesis="AI decision layer disabled; deterministic strategy signal accepted.",
            invalidation=f"Exit if stop loss reaches {strategy.stop_loss_pct:.2f}% or trend breaks.",
            expected_upside_pct=expected_upside,
            expected_downside_pct=expected_downside,
            risk_notes=[],
            source="disabled",
            input_snapshot=decision_input,
        )

    confidence = min(0.95, max(0.0, signal.confidence) * context.score_multiplier)
    if expected_upside < strategy.min_expected_upside_pct:
        risk_notes.append("Expected upside is below the configured minimum.")
    if expected_downside >= expected_upside:
        risk_notes.append("Expected downside is greater than or equal to expected upside.")
    if context.fear_greed_value is not None and context.fear_greed_value >= 85:
        risk_notes.append("Sentiment is overheated.")
    if context.fear_greed_value is not None and context.fear_greed_value <= 15:
        risk_notes.append("Market stress is high.")
    if not context.allows_entries:
        risk_notes.append("Market context does not allow new entries.")
    if fallback_reason:
        risk_notes.append(f"OpenAI fallback reason: {fallback_reason}")

    grade = decision_grade(confidence, expected_upside, expected_downside, context.allows_entries)

    if not context.allows_entries:
        action = "pause"
        thesis = "Market context is unfavorable, so the candidate should not be opened."
    elif expected_downside >= expected_upside:
        action = "hold"
        thesis = "Candidate risk/reward is unfavorable because downside is at least as large as upside."
    elif risk_notes and confidence < ai_config.min_confidence + 0.15:
        action = "hold"
        thesis = "Candidate has a deterministic signal, but the risk/reward evidence is not strong enough."
    elif confidence < ai_config.min_confidence or grade != "A":
        action = "hold"
        thesis = "Candidate grade is below the entry threshold."
    else:
        action = "buy"
        thesis = "Candidate has trend, volume, upside, and market-context evidence above the entry threshold."

    return DecisionReview(
        action=action,
        grade=grade,
        confidence=confidence,
        thesis=thesis,
        invalidation=f"Exit if stop loss reaches {strategy.stop_loss_pct:.2f}%, time stop triggers, or BTC context deteriorates.",
        expected_upside_pct=expected_upside,
        expected_downside_pct=expected_downside,
        risk_notes=risk_notes,
        source=fallback_source,
        input_snapshot=decision_input,
    )


def build_decision_input(
    signal: Signal,
    candles: list[Candle],
    context: DecisionContext,
    strategy: StrategyConfig,
) -> dict[str, Any]:
    closes = [candle.close for candle in candles]
    latest = closes[-1] if closes else 0.0
    recent_high = max((candle.high for candle in candles[-30:]), default=latest)
    recent_low = min((candle.low for candle in candles[-30:]), default=latest)
    return {
        "schema_version": "decision-input-v1",
        "candidate": {
            "market": candles[-1].market if candles else "",
            "latest_price": latest,
            "signal_side": signal.side.value if hasattr(signal.side, "value") else str(signal.side),
            "signal_reason": signal.reason,
            "signal_confidence": signal.confidence,
            "expected_upside_pct": estimate_expected_upside_pct(candles, strategy.target_upside_pct),
            "expected_downside_pct": estimate_expected_downside_pct(candles, strategy.stop_loss_pct, strategy.stop_volatility_multiplier),
            "recent_high": recent_high,
            "recent_low": recent_low,
            "recent_volatility_pct": recent_volatility_pct(candles),
        },
        "market_context": context.to_dict(),
        "strategy_limits": {
            "target_upside_pct": strategy.target_upside_pct,
            "min_expected_upside_pct": strategy.min_expected_upside_pct,
            "take_profit_pct": strategy.take_profit_pct,
            "stop_loss_pct": strategy.stop_loss_pct,
            "max_entry_rsi": strategy.max_entry_rsi,
        },
    }


def decision_grade(confidence: float, expected_upside: float, expected_downside: float, allows_entries: bool) -> str:
    if not allows_entries or expected_downside >= expected_upside:
        return "C"
    if confidence >= 0.75 and expected_upside >= 2.0:
        return "A"
    if confidence >= 0.60 and expected_upside >= 1.2:
        return "B"
    return "C"


def try_openai_review(decision_input: dict[str, Any], ai_config: AiDecisionConfig) -> tuple[DecisionReview | None, str | None]:
    api_key = os.environ.get(ai_config.api_key_env)
    if not api_key:
        return None, f"missing environment variable {ai_config.api_key_env}"
    try:
        payload = {
            "model": ai_config.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative crypto paper-trading decision reviewer. "
                        "Return only the structured decision. Never ignore risk limits."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(decision_input, ensure_ascii=False),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "paper_trade_decision_review",
                    "strict": True,
                    "schema": decision_review_schema(),
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        parsed = extract_openai_json(response_payload)
        return decision_from_mapping(parsed, source="openai", input_snapshot=decision_input), None
    except Exception as exc:
        reason = summarize_openai_error(exc)
        print(f"OpenAI decision review failed: {reason}")
        return None, reason


def summarize_openai_error(exc: Exception) -> str:
    status = getattr(exc, "code", None)
    body = ""
    try:
        body_bytes = exc.read()  # type: ignore[attr-defined]
        body = body_bytes.decode("utf-8", errors="replace")[:500]
    except Exception:
        body = str(exc)[:500]
    if status:
        return f"HTTP {status}: {body}"
    return body


def decision_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "grade",
            "confidence",
            "thesis",
            "invalidation",
            "expected_upside_pct",
            "expected_downside_pct",
            "risk_notes",
        ],
        "properties": {
            "action": {"type": "string", "enum": ["buy", "hold", "sell", "pause", "resume"]},
            "grade": {"type": "string", "enum": ["A", "B", "C"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "thesis": {"type": "string"},
            "invalidation": {"type": "string"},
            "expected_upside_pct": {"type": "number"},
            "expected_downside_pct": {"type": "number"},
            "risk_notes": {"type": "array", "items": {"type": "string"}},
        },
    }


def extract_openai_json(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("output_parsed"), dict):
        return payload["output_parsed"]
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("parsed"), dict):
                return content["parsed"]
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return json.loads(text)
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return json.loads(text)
    raise ValueError("OpenAI response did not contain structured JSON.")


def decision_from_mapping(payload: dict[str, Any], source: str, input_snapshot: dict[str, Any]) -> DecisionReview:
    return DecisionReview(
        action=str(payload["action"]),
        grade=str(payload.get("grade", "B")),
        confidence=float(payload["confidence"]),
        thesis=str(payload["thesis"]),
        invalidation=str(payload["invalidation"]),
        expected_upside_pct=float(payload["expected_upside_pct"]),
        expected_downside_pct=float(payload["expected_downside_pct"]),
        risk_notes=[str(note) for note in payload.get("risk_notes", [])],
        source=source,
        input_snapshot=input_snapshot,
    )
