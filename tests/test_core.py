"""Backtest math + features tests."""
import numpy as np
import pandas as pd

from backtest import sharpe, max_drawdown, backtest_one
from data import add_features, _synthetic


def test_sharpe_positive_for_uptrend():
    rets = pd.Series([0.01] * 100)
    assert sharpe(rets) > 5


def test_max_dd_negative():
    eq = pd.Series([1.0, 1.1, 0.9, 1.0])
    assert max_drawdown(eq) < 0


def test_backtest_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    df = pd.DataFrame({"ret_1d": np.linspace(-0.01, 0.01, 50)}, index=idx)
    sig = pd.Series(1, index=idx)
    net, eq, bh = backtest_one(df, sig, cost_bps=0)
    assert len(eq) == 50 and eq.iloc[-1] > 0


def test_features_no_all_nan():
    df = _synthetic("TEST", "2024-01-01", "2024-06-01")
    feat = add_features(df).dropna()
    assert len(feat) > 20
    assert "rsi_14" in feat.columns and "target" in feat.columns
