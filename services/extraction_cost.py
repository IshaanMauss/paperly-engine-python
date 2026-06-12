from __future__ import annotations

import contextvars
import os
from typing import Any, Dict, List, Optional


_LEDGER: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "paperly_extraction_cost_ledger",
    default=None,
)


def start_cost_ledger() -> contextvars.Token:
    """Start a per-request in-memory Gemini usage ledger."""
    return _LEDGER.set([])


def reset_cost_ledger(token: contextvars.Token) -> None:
    _LEDGER.reset(token)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _model_rates(model: str) -> Dict[str, float]:
    model_l = (model or "").lower()
    if "flash-lite" in model_l:
        return {
            "input_usd_per_m": _float_env("GEMINI_FLASH_LITE_INPUT_USD_PER_M", 0.10),
            "output_usd_per_m": _float_env("GEMINI_FLASH_LITE_OUTPUT_USD_PER_M", 0.40),
        }
    if "flash" in model_l:
        return {
            "input_usd_per_m": _float_env("GEMINI_FLASH_INPUT_USD_PER_M", 0.30),
            "output_usd_per_m": _float_env("GEMINI_FLASH_OUTPUT_USD_PER_M", 2.50),
        }
    return {
        "input_usd_per_m": _float_env("GEMINI_DEFAULT_INPUT_USD_PER_M", 0.30),
        "output_usd_per_m": _float_env("GEMINI_DEFAULT_OUTPUT_USD_PER_M", 2.50),
    }


def _usage_metric(usage: Any, name: str) -> int:
    value = getattr(usage, name, None)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def estimate_usage_cost_inr(model: str, usage: Any) -> Dict[str, Any]:
    prompt_tokens = _usage_metric(usage, "prompt_token_count")
    candidates_tokens = _usage_metric(usage, "candidates_token_count")
    thoughts_tokens = _usage_metric(usage, "thoughts_token_count")
    total_tokens = _usage_metric(usage, "total_token_count")

    output_tokens = candidates_tokens + thoughts_tokens
    if output_tokens <= 0 and total_tokens > prompt_tokens:
        output_tokens = total_tokens - prompt_tokens

    rates = _model_rates(model)
    usd_to_inr = _float_env("USD_TO_INR", 84.0)
    estimated_usd = (
        (prompt_tokens / 1_000_000.0) * rates["input_usd_per_m"]
        + (output_tokens / 1_000_000.0) * rates["output_usd_per_m"]
    )

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": round(estimated_usd, 6),
        "estimated_inr": round(estimated_usd * usd_to_inr, 4),
        "rates": rates,
    }


def record_gemini_usage(
    *,
    model: str,
    document_type: str,
    page_num: Optional[int],
    attempt: int,
    component: str,
    usage: Any,
) -> None:
    ledger = _LEDGER.get()
    if ledger is None:
        return
    cost = estimate_usage_cost_inr(model, usage)
    ledger.append(
        {
            "kind": "usage",
            "component": component,
            "model": model,
            "document_type": document_type,
            "page_num": page_num,
            "attempt": attempt,
            **cost,
        }
    )


def record_gemini_failure(
    *,
    model: str,
    document_type: str,
    page_num: Optional[int],
    attempt: int,
    component: str,
    error: str,
) -> None:
    ledger = _LEDGER.get()
    if ledger is None:
        return
    ledger.append(
        {
            "kind": "failure",
            "component": component,
            "model": model,
            "document_type": document_type,
            "page_num": page_num,
            "attempt": attempt,
            "error": str(error)[:220],
            "estimated_usd": 0.0,
            "estimated_inr": 0.0,
        }
    )


def summarize_cost_ledger() -> Dict[str, Any]:
    ledger = _LEDGER.get() or []
    usage_rows = [row for row in ledger if row.get("kind") == "usage"]
    failure_rows = [row for row in ledger if row.get("kind") == "failure"]

    by_model: Dict[str, Dict[str, Any]] = {}
    for row in usage_rows:
        model = str(row.get("model") or "unknown")
        bucket = by_model.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "thoughts_tokens": 0,
                "total_tokens": 0,
                "estimated_inr": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        bucket["output_tokens"] += int(row.get("output_tokens") or 0)
        bucket["thoughts_tokens"] += int(row.get("thoughts_tokens") or 0)
        bucket["total_tokens"] += int(row.get("total_tokens") or 0)
        bucket["estimated_inr"] += float(row.get("estimated_inr") or 0)

    for bucket in by_model.values():
        bucket["estimated_inr"] = round(bucket["estimated_inr"], 4)

    return {
        "gemini_calls": len(usage_rows),
        "gemini_failures": len(failure_rows),
        "estimated_inr": round(sum(float(row.get("estimated_inr") or 0) for row in usage_rows), 4),
        "by_model": by_model,
        "failures": failure_rows[:8],
    }
