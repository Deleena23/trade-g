# Our dev notes — Deleena K & Aadhitya S R

## Why this project
We wanted one repo for Nasdaq AI/ML that shows ML + backtest + surveillance, not just a notebook with accuracy. Exchanges care about risk and costs, so we added a compliance filter.

## What broke + fixed
1. `KeyError: non-monotonic DatetimeIndex` in data.py — concat of 7 tickers repeats dates. Fixed by resetting to `date` column and masking (`<= TRAIN_END`, `>= TEST_START`).
2. No `yfinance` / `lightgbm` / `gymnasium` in lab env — added fallbacks: synthetic GBM, HistGradientBoosting, momentum policy. Pipeline always runs.
3. `parse_dates` loader mismatch after changing csv format — made loaders accept both indexed and date-column csvs.
4. Synthetic Sharpe negative — expected, data is random. Proves backtest isn't cheating (shift(1) + costs).

## What I learnt
- Time-split (no shuffle) + shift(1) to avoid lookahead.
- Sharpe = mean/std*sqrt(252), DD from cummax.
- Position/turnover/circuit limits matter more than model in interviews.
- Keep risk.py pure-python so tests run anywhere.

## Next
Install yfinance on good net, add FinBERT sentiment as 14th feature, stream via Kafka, deploy Docker to AWS.
