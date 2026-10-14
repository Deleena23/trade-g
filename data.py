"""Data pipeline by Deleena K: download (yfinance) -> features -> sqlite + csv split.
Pattern inspired by FinRL StockTrading_2026_1_data.py, trimmed to medium scope.
Run: python data.py
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_OUT, DB_PATH, TICKERS, START, END, TRAIN_END, TEST_START


def _synthetic(ticker: str, start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31) + seed)
    idx = pd.date_range(start=start, end=end, freq="B")
    n = len(idx)
    rets = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    vol = rng.integers(1_000_00, 5_000_000, n)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def download_one(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError("empty")
        # flatten multiindex from newer yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep]
    except Exception as e:
        print(f"[data] {ticker}: yfinance failed ({e}), using synthetic")
        return _synthetic(ticker, START, END)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["ret_1d"] = c.pct_change()
    df["sma_20"] = c.rolling(20).mean()
    df["sma_60"] = c.rolling(60).mean()
    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    # Bollinger
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["boll_ub"] = mid + 2 * std
    df["boll_lb"] = mid - 2 * std
    # VIX proxy + turbulence proxy
    df["vol_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    df["turbulence"] = (df["ret_1d"] / (df["ret_1d"].rolling(60).std() + 1e-9)).abs()
    # Microstructure proxies for surveillance demo
    df["spread_proxy"] = (df["High"] - df["Low"]) / c
    df["order_imbalance"] = (c - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    df["cancel_rate"] = df["spread_proxy"].rolling(5).mean()
    # Target: next-day direction
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def main() -> None:
    DATA_OUT.mkdir(exist_ok=True)
    all_frames = []
    for t in TICKERS:
        raw = download_one(t)
        feat = add_features(raw)
        feat["ticker"] = t
        feat.to_csv(DATA_OUT / f"{t.replace('^','').replace('.','_')}.csv")
        all_frames.append(feat)
        print(f"[data] {t}: {len(feat)} rows")
    full = pd.concat(all_frames)
    # sqlite store
    with sqlite3.connect(DB_PATH) as con:
        full.reset_index().rename(columns={"index": "date"}).to_sql(
            "market", con, if_exists="replace", index=False
        )
    # train/test split (use date column mask — index has duplicate dates across tickers)
    full = full.reset_index().rename(columns={"index": "date"})
    full["date"] = pd.to_datetime(full["date"])
    train = full[full["date"] <= TRAIN_END].dropna()
    test = full[full["date"] >= TEST_START].dropna()
    train.to_csv(DATA_OUT / "train_data.csv", index=False)
    test.to_csv(DATA_OUT / "trade_data.csv", index=False)
    print(f"[data] train {len(train)} test {len(test)} -> {DATA_OUT}")


if __name__ == "__main__":
    main()
