"""Risk filter tests - Nasdaq surveillance logic."""
from risk import check_order, detect_anomaly_row, explain_alert


def test_position_limit_blocks():
    r = check_order("AAPL", "buy", qty=1000, price=200, portfolio_value=100_000,
                    position_value=0)
    assert not r.allowed and "position-limit" in r.reason


def test_circuit_breaker_blocks():
    r = check_order("TCS.NS", "buy", qty=1, price=100, portfolio_value=100_000,
                    position_value=0, day_return=0.15)
    assert not r.allowed and "circuit" in r.reason


def test_wash_block():
    r = check_order("INFY.NS", "buy", qty=1, price=100, portfolio_value=100_000,
                    position_value=0, is_wash=True)
    assert not r.allowed and "wash" in r.reason


def test_valid_passes():
    r = check_order("QQQ", "buy", qty=10, price=100, portfolio_value=100_000,
                    position_value=0)
    assert r.allowed


def test_spoof_detection():
    row = {"spread_proxy": 0.01, "order_imbalance": 3.0, "cancel_rate": 0.05,
           "turbulence": 1.0}
    is_alert, rule = detect_anomaly_row(row)
    assert is_alert and "spoof" in rule


def test_normal_row():
    row = {"spread_proxy": 0.001, "order_imbalance": 0.2, "cancel_rate": 0.001,
           "turbulence": 0.5}
    is_alert, _ = detect_anomaly_row(row)
    assert not is_alert


def test_explainer_mentions_ticker():
    s = explain_alert("AAPL", "spoof-like", 220.5, "2025-01-02")
    assert "AAPL" in s and "220.5" in s
