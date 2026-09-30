"""Live PKR -> USD conversion helpers for the presentation layer.

The trained model and stored historical dataset remain in their original PKR units.
Only user-facing monetary values are converted to USD. This avoids changing the
feature distribution the model was trained on.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from urllib.request import Request, urlopen

PRIMARY_URL = "https://api.frankfurter.dev/v2/providers/sbp/rate/pkr/usd"
SECONDARY_URL = "https://api.frankfurter.dev/v2/rate/pkr/usd"
FALLBACK_RATE = float(os.getenv("PKR_USD_FALLBACK_RATE", "0.00361"))
FALLBACK_DATE = os.getenv("PKR_USD_FALLBACK_DATE", "2026-09-30")


def _fetch_json(url: str, timeout: float) -> Dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": "cancellation-risk-school-project/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_pkr_to_usd_rate(timeout: float = 4.0) -> Dict[str, Any]:
    """Return the latest available PKR->USD rate with a safe offline fallback.

    Frankfurter is queried first using the State Bank of Pakistan provider. If that
    provider is unavailable, the blended Frankfurter rate is tried. The packaged
    fallback exists only so the Streamlit app remains usable during a network outage.
    """
    attempts = [
        (PRIMARY_URL, "State Bank of Pakistan via Frankfurter"),
        (SECONDARY_URL, "Frankfurter blended official rates"),
    ]
    errors = []
    for url, source in attempts:
        try:
            payload = _fetch_json(url, timeout)
            rate = float(payload["rate"])
            if rate <= 0:
                raise ValueError("exchange rate must be positive")
            return {
                "rate": rate,
                "date": str(payload.get("date") or "latest"),
                "source": source,
                "is_live": True,
                "base": "PKR",
                "quote": "USD",
            }
        except Exception as exc:  # network/provider errors should not break inference
            errors.append(f"{source}: {exc}")

    return {
        "rate": FALLBACK_RATE,
        "date": FALLBACK_DATE,
        "source": "Packaged fallback rate",
        "is_live": False,
        "base": "PKR",
        "quote": "USD",
        "warning": "; ".join(errors),
    }


def pkr_to_usd(amount, rate: float):
    return amount * float(rate)


def usd_to_pkr(amount, rate: float):
    rate = float(rate)
    if rate <= 0:
        raise ValueError("PKR->USD exchange rate must be positive")
    return amount / rate


def format_usd(amount_pkr: float, rate: float, decimals: int = 2) -> str:
    amount_usd = float(pkr_to_usd(float(amount_pkr), rate))
    sign = "-" if amount_usd < 0 else ""
    return f"{sign}${abs(amount_usd):,.{decimals}f}"


def rate_summary(context: Dict[str, Any]) -> str:
    rate = float(context["rate"])
    inverse = 1.0 / rate
    status = "Live" if context.get("is_live") else "Fallback"
    return (
        f"{status} FX: 1 PKR = ${rate:.6f} USD (about $1 = PKR {inverse:,.2f}) • "
        f"{context.get('source', 'exchange-rate source')} • rate date {context.get('date', 'latest')}"
    )
