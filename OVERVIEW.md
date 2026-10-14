# trade-g — What it is, what it does, how we made it

*By Deleena K & Aadhitya S R (BTech, Christ University) — built for Nasdaq India AI/ML internship prep.*

## What it is

An equity algo-trading bot on NIFTY-50 + NASDAQ-100 stocks with a **trade-surveillance filter** bolted on. Think of it as two systems in one repo:

1. A **money bot** — reads market data, predicts next-day direction, sizes positions, backtests with costs.
2. A **watchdog** — screens every trade and every data row against compliance rules and anomaly checks, and explains each flag in plain English.

We picked this because Nasdaq is an exchange company: they don't just trade, they *run markets and police them* (Market Platforms + Anti-Financial Crime + Capital Access). A bot with a watchdog shows we think like an exchange, not just a trader.

Live repo: https://github.com/Deleena23/trade-g

## What it does

- **Downloads market data** for 7 tickers (`^NSEI`, RELIANCE.NS, TCS.NS, INFY.NS, QQQ, AAPL, MSFT) from Yahoo Finance. If the network fails it generates synthetic data so the pipeline never breaks.
- **Engineers 13 features** per ticker: returns, SMA-20/60, RSI-14, MACD + signal, Bollinger bands, volatility (VIX proxy), turbulence, spread, order-imbalance and cancel-rate proxies.
- **Predicts direction** with LightGBM (falls back to sklearn's HistGradientBoosting on CPU-only machines) — accuracy / precision / recall on a time-split validation set.
- **Sizes positions** with a PPO reinforcement-learning agent over a custom trend environment (last-5 returns → flat/long/short); falls back to a momentum rule without GPU/torch.
- **Backtests honestly**: signals shifted one day (no lookahead), 10 bps transaction costs, Sharpe ratio, max drawdown and total return vs buy-and-hold, per ticker, with an equity-curve chart.
- **Blocks bad trades** before they happen: circuit-breaker (>10% daily move), wash-trade block, 20% single-name position limit, 50% turnover limit, 20-orders/day rate limit.
- **Flags manipulation patterns** in the data: spoof-like (high cancels + one-sided book), pump-like (wide spread + turbulence spike), wash-like (extreme imbalance) — each with a one-line compliance explanation.
- **Serves it**: FastAPI (`/signal`, `/backtest`) + Streamlit dashboard (price chart, alert table, backtest metrics).
- **Tests it**: 11 pytest cases covering risk rules, backtest math and features.

## How we made it

1. **Data first (`data.py`, `config.py`)** — Deleena. Yahoo download with a synthetic GBM fallback, feature functions with pandas rolling/ewm, SQLite store + train (≤2024) / test (≥2025) CSV split. First bug we hit: slicing a concatenated multi-ticker index raised `KeyError: non-monotonic DatetimeIndex`, so we filter on a `date` column instead.
2. **Signal model (`train_lightgbm.py`)** — Deleena. 13-feature direction classifier, chronological 80/20 split (never shuffle time series), metrics + per-ticker latest signals for the API.
3. **DRL sizer (`train_drl.py`)** — Aadhitya. Custom 5-observation gym environment, PPO via stable-baselines3 when available, momentum-policy fallback otherwise. Keeps the repo runnable on any laptop.
4. **Backtest (`backtest.py`)** — joint. Pure-pandas engine, cost-aware, baseline-compared, chart-exported. On synthetic noise the strategy underperforms buy-hold — which is exactly what a correct backtest should show.
5. **Watchdog (`risk.py`)** — Aadhitya. Pure Python, zero ML deps, so it's fully unit-testable. Rule order matters: circuit → wash → position → turnover → rate.
6. **Serving + tests (`api.py`, `app.py`, `tests/`)** — joint. FastAPI + Streamlit, 7 risk tests + 4 math/feature tests, GitHub-ready with Dockerfile and a Windows `run.bat`.
7. **Docs** — `ARCHITECTURE.md` (full system design), `NOTES.md` (bugs we hit and fixed), this file.

## How to run it

```powershell
pip install -r requirements.txt
python data.py
python train_lightgbm.py
python train_drl.py
python backtest.py
pytest -q
uvicorn api:app --port 8000
streamlit run app.py
```

## What we'd do next

Real-time Kafka feed into `risk.py`, FinBERT news sentiment as a 14th feature, AWS deploy of the Docker image, and CI running pytest on every push.

*Educational project only — not financial advice.*
