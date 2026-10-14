"""Backtest LGBM signals vs buy-hold with costs. Pure pandas (vectorbt optional).
Run: python backtest.py
"""
from __future__ import annotations
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from config import DATA_OUT, TRAINED_DIR, TXN_COST_BPS
from risk import detect_anomaly_row


def sharpe(rets: pd.Series) -> float:
    s = rets.std()
    return float(rets.mean() / (s + 1e-12) * np.sqrt(252)) if s > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def backtest_one(df: pd.DataFrame, signals: pd.Series, cost_bps: int = TXN_COST_BPS):
    pos = signals.shift(1).fillna(0)  # avoid lookahead
    strat_rets = pos * df["ret_1d"].fillna(0)
    turnover = pos.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10000)
    net = strat_rets - costs
    equity = (1 + net.fillna(0)).cumprod()
    bh_equity = (1 + df["ret_1d"].fillna(0)).cumprod()
    return net, equity, bh_equity


def main() -> dict:
    test_p = DATA_OUT / "trade_data.csv"
    if not test_p.exists():
        raise FileNotFoundError(f"missing {test_p}, run python data.py first")
    df_all = pd.read_csv(test_p)
    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all = df_all.set_index("date")
    else:
        try:
            df_all.index = pd.to_datetime(df_all.index)
        except Exception:
            pass
    FEATURES = ["rsi_14", "macd", "macd_signal", "sma_20", "sma_60",
                "boll_ub", "boll_lb", "vol_20", "turbulence",
                "spread_proxy", "order_imbalance", "cancel_rate", "ret_1d"]
    # load model if present else SMA rule
    model = None
    mp = TRAINED_DIR / "lgbm_model.pkl"
    if mp.exists():
        try:
            model = joblib.load(mp)
        except Exception as e:
            print(f"[backtest] model load failed: {e}")
    results = {}
    equities = {}
    for ticker, df in df_all.groupby("ticker"):
        df = df.sort_index()
        if model is not None:
            try:
                sig = pd.Series(model.predict(df[FEATURES]), index=df.index)
                sig = sig.map({0: -1, 1: 1})  # long/short
            except Exception:
                sig = pd.Series(np.where(df["Close"] > df["sma_20"], 1, -1), index=df.index)
        else:
            sig = pd.Series(np.where(df["Close"] > df["sma_20"], 1, -1), index=df.index)
        net, eq, bh = backtest_one(df, sig)
        results[ticker] = {
            "sharpe_strat": sharpe(net),
            "sharpe_bh": sharpe(df["ret_1d"].fillna(0)),
            "max_dd_strat": max_drawdown(eq),
            "ret_strat": float(eq.iloc[-1] - 1),
            "ret_bh": float(bh.iloc[-1] - 1),
        }
        equities[ticker] = (eq, bh)
        # surveillance count
        alerts = sum(detect_anomaly_row(r)[0] for _, r in df.iterrows())
        results[ticker]["alerts"] = int(alerts)
    (TRAINED_DIR / "backtest_metrics.json").write_text(json.dumps(results, indent=2))
    # plot first ticker
    t0 = next(iter(equities))
    eq, bh = equities[t0]
    plt.figure(figsize=(9, 4))
    plt.plot(eq.index, eq.values, label=f"strategy {t0}")
    plt.plot(bh.index, bh.values, label="buy-hold", alpha=0.7)
    plt.legend(); plt.tight_layout()
    plt.savefig(DATA_OUT / "backtest_result.png")
    print(f"[backtest] {json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    main()
