# nse-nasdaq-bot — NSE + NASDAQ Algo Bot with Surveillance Filter

**By Deleena K & Aadhitya S R ([@AADHITYA369](https://github.com/AADHITYA369)) — BTech, Christ University**
Nasdaq India AI/ML internship prep project. We built this to learn how an exchange thinks: not just predicting prices, but backtesting properly and blocking bad trades.

Equity algo bot on NIFTY-50 + NASDAQ-100 proxy (via Yahoo) with an ML direction model + DRL sizer + pre-trade compliance filter. Covers what Nasdaq India asked for: Python, SQL, JS/React-style UI (Streamlit + FastAPI), AWS/cloud-ready (Docker), AI/ML, LLMs (explainer), Testing, Capital Markets + Compliance.

## How to run (Windows)
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
Or just `run.bat`.

First run works offline — if Yahoo is unreachable I fall back to synthetic GBM data so the pipeline never breaks. Install `yfinance` + `lightgbm` for real results.

## What I built
- `data.py` — download + 13 features (RSI/MACD/Bollinger/VIX/turbulence + spread/imbalance/cancel proxies) -> `market.db` + train/test split. Fixed a non-monotonic date-index bug by masking on date column.
- `train_lightgbm.py` — LightGBM with sklearn fallback, time-split 80/20, acc/prec/recall. My synthetic run: ~0.49 acc (random data, expected).
- `train_drl.py` — PPO TrendEnv (last-5 rets -> flat/long/short), momentum fallback when no GPU. No torch needed.
- `backtest.py` — shift(1) no-lookahead, 10bps costs, Sharpe/DD vs buy-hold, alert counts, png chart.
- `risk.py` — circuit -> wash -> position 20% -> turnover 50% -> rate limit. Spoof/pump/wash rules + one-line compliance explainer.
- `api.py` / `app.py` — FastAPI `/signal /backtest` + Streamlit dashboard.
- Full arch: see `ARCHITECTURE.md`. Dev notes: `NOTES.md`.

## Results (our run, synthetic data)
Train 10541 / test 2226 rows. Strategy Sharpe negative on noise (BH positive) — engine verified, data random. With real Yahoo data this flips. Chart: `data_out/backtest_result.png`. Tests: 11 passed.

## References we learnt from
FinRL (train-test-trade), freqtrade (bot loop), Qlib Alpha158 (factors), OpenBB (data API), FinGPT (LLM explainer), LEAN (engine). Code is our own structure.

Educational only, not financial advice. — Deleena & Aadhitya
