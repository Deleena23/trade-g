"""Pre-trade risk + surveillance filter by Deleena K (Nasdaq Anti-Financial-Crime angle).
Pure-python so it is unit-testable. No ML deps.
"""
from __future__ import annotations
from dataclasses import dataclass

from config import RISK


@dataclass
class CheckResult:
    allowed: bool
    reason: str


def check_order(
    ticker: str,
    side: str,
    qty: float,
    price: float,
    portfolio_value: float,
    position_value: float,
    day_return: float = 0.0,
    orders_today: int = 0,
    is_wash: bool = False,
) -> CheckResult:
    """Apply circuit -> wash -> position -> turnover -> rate limits in order."""
    notional = abs(qty) * price
    if abs(day_return) > RISK["circuit_pct"]:
        return CheckResult(False, f"circuit-breaker: day move {day_return:.2%}")
    if is_wash:
        return CheckResult(False, "wash-trade-block: same beneficial owner both sides")
    if portfolio_value <= 0:
        return CheckResult(False, "invalid-portfolio")
    if (position_value + notional) / portfolio_value > RISK["max_position_pct"]:
        return CheckResult(False, "position-limit: >20% single name")
    if notional / portfolio_value > RISK["max_turnover_pct"]:
        return CheckResult(False, "turnover-limit: >50% rebalance")
    if orders_today >= RISK["max_orders_per_day"]:
        return CheckResult(False, "rate-limit: too many orders today")
    if side not in ("buy", "sell"):
        return CheckResult(False, "invalid-side")
    return CheckResult(True, "pass")


def detect_anomaly_row(row: dict) -> tuple[bool, str]:
    """Rule-based spoofing/wash proxy for demo + ML fallback.
    Returns (is_alert, rule_name).
    """
    spread = float(row.get("spread_proxy", 0))
    imb = abs(float(row.get("order_imbalance", 0)))
    cancel = float(row.get("cancel_rate", 0))
    turb = float(row.get("turbulence", 0))
    # High cancel + wide spread + one-sided imbalance = spoof-like
    if cancel > 0.02 and imb > 2.0:
        return True, "spoof-like: high cancel + one-sided imbalance"
    if spread > 0.05 and turb > 3.0:
        return True, "pump-like: wide spread + turbulence spike"
    if imb > 5.0:
        return True, "wash-like: extreme imbalance"
    return False, "normal"


def explain_alert(ticker: str, rule: str, price: float, date: str = "") -> str:
    """LLM-style template explainer (no API call; swap with FinGPT/OpenAI later)."""
    return (
        f"[{date or 'latest'}] {ticker} @ {price:.2f} flagged: {rule}. "
        "Likely compliance review: check order-to-trade ratio, cancel clustering, "
        "and beneficial-owner overlap per SEBI/SEC surveillance norms."
    )
