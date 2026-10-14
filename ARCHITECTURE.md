# nse-nasdaq-bot — Architecture (by Deleena K)

> BTech, Christ University — Nasdaq India AI/ML internship prep.
> Medium-scope equity algo bot: NIFTY-50 + NASDAQ-100 proxy with ML signal + DRL sizer + pre-trade surveillance filter.

## 1. System overview

```
yfinance (^NSEI, RELIANCE.NS, TCS.NS, INFY.NS, QQQ, AAPL, MSFT)
   |  fallback: synthetic GBM if offline
   v
data.py -> features (13) -> market.db (sqlite) + train_data.csv (<=2024-12-31) + trade_data.csv (>=2025-01-01)
   |
   +---> train_lightgbm.py -> LGBM / HistGradientBoosting -> lgbm_model.pkl + latest_signals.csv
   |         X=13 features, y=next-day up/down, time-split 80/20, metrics: acc/prec/recall
   |
   +---> train_drl.py -> TrendEnv (obs=last-5 rets, act=flat/long/short) -> PPO (SB3) or momentum-fallback
   |
   v
backtest.py (signals.shift(1), 10bps costs) -> Sharpe / DD / return vs buy-hold + alert counts -> backtest_result.png
   |
   +---> risk.py gate (circuit -> wash -> position 20% -> turnover 50% -> rate 20/day)
   +---> api.py (FastAPI /signal /backtest) + app.py (Streamlit price + alerts)
```

## 2. Data layer — `data.py`, `config.py`

- Universe (`config.py:TICKERS`): `^NSEI, RELIANCE.NS, TCS.NS, INFY.NS, QQQ, AAPL, MSFT`, `START=2019-01-01, END=2026-03-20`.
- `download_one()`: `yfinance.download(auto_adjust=True)`, flatten MultiIndex, keep OHLCV. On any exception -> `_synthetic()` GBM: `close=100*exp(cumsum(N(0.0004,0.012)))`, vol/spread synthesized. This is why pipeline runs offline.
- Store: per-ticker csv + `market.db:market` table + `train_data.csv` (train 10541 rows in my run) + `trade_data.csv` (2226 rows).
- Split is date-mask on `date` column (not index slice) because concatenated index has duplicate dates across tickers — fixed after first run hit `KeyError: non-monotonic DatetimeIndex`.

## 3. Feature layer — `data.py:add_features()` (13 features)

| # | name | formula | why |
|---|---|---|---|
| 1 | ret_1d | close.pct_change() | base return |
| 2-3 | sma_20/60 | close.rolling(N).mean() | trend |
| 4 | rsi_14 | 100-100/(1+gain14/loss14) | momentum |
| 5-6 | macd, macd_signal | ema12-ema26, signal ewm9 | momentum |
| 7-8 | boll_ub/lb | sma20 ± 2*std20 | volatility |
| 9 | vol_20 | std(ret,20)*sqrt(252) | VIX proxy |
| 10 | turbulence | abs(ret / std(ret,60)) | regime stress |
| 11 | spread_proxy | (H-L)/C | liquidity |
| 12 | order_imbalance | (C-O)/(H-L) | buying pressure |
| 13 | cancel_rate | mean(spread,5) | spoof proxy |
| y | target | (close.shift(-1) > close) | next-day direction |

All NaNs dropped before train. `Close` kept for backtest, `ticker` kept for groupby.

## 4. Model layer A — direction classifier `train_lightgbm.py`

- `FEATURES` (13 above), `load_train()` handles both `date`-column and indexed csv.
- `build_model()`: try `lightgbm.LGBMClassifier(random_state=42)` else `sklearn HistGradientBoostingClassifier`. My env used HGB (no lightgbm installed).
- Train: chronological 80/20 split (no shuffle — no lookahead), `model.fit(Xtr,ytr)`, metrics on valid: accuracy/precision/recall via sklearn. My synthetic run: acc 0.49 / prec 0.49 / rec 0.59 — random, expected on GBM noise. Real Yahoo data lifts this.
- Artifacts: `trained_models/lgbm_model.pkl`, `lgbm_metrics.json`, `data_out/latest_signals.csv` (per-ticker last row + `signal` 0/1 + `prob`).

## 5. Model layer B — DRL sizer `train_drl.py`

- `load_prices()`: first ticker's Close -> `ret` series.
- `TrendEnv(gym.Env)`: obs `Box(5,)` = last-5 rets, action `Discrete(3)` = {flat,long,short}, reward = `pos * ret[i]`, done at end. `PPO(MlpPolicy).learn(5000)` -> `ppo_policy.zip`.
- Fallback (my env, no gymnasium/torch): `momentum-5` rule saved to `ppo_policy.json` + `drl_metrics.json:{algo}`.
- Design choice: DRL sizes/times, LGBM gives direction — ensemble = direction × position. Keeps medium scope without full FinRL-Meta env zoo.

## 6. Surveillance + risk layer — `risk.py` (pure python, 7 tests)

Order in `check_order()`: circuit (abs(day_ret)>10%) -> wash (`is_wash`) -> position ((pos+notional)/pf>20%) -> turnover (notional/pf>50%) -> rate (>=20/day) -> side check -> pass.
- `detect_anomaly_row()`: spoof-like if `cancel>0.02 and |imb|>2`, pump-like if `spread>0.05 and turb>3`, wash-like if `|imb|>5`. Tuned for demo recall over precision.
- `explain_alert()`: template `"[date] TICKER @ price flagged: rule. review order-to-trade, cancel clustering, owner overlap per SEBI/SEC"` — swap with FinGPT/OpenAI later. No API call so tests stay offline.

## 7. Backtest engine — `backtest.py`

- `backtest_one()`: `pos=signals.shift(1)` (1-day delay), `strat=pos*ret`, `turnover=|Δpos|`, `cost=turnover*bps/1e4`, `net=strat-cost`, `equity=cumprod(1+net)`. Buy-hold = `cumprod(1+ret)`.
- Per ticker: `sharpe=mean/std*sqrt(252)`, `max_drawdown=min((eq-cummax)/cummax)`, `ret=eq[-1]-1`, `alerts=count(detect)`. My synthetic run: strat Sharpe negative (noise), BH positive — proves engine works, data is random.
- Signal map: LGBM `{0,1}->{ -1,+1}` long/short; no model -> SMA20 rule. Plot first ticker to `backtest_result.png`.

## 8. Serving — `api.py`, `app.py`

- FastAPI: `GET /health->{ok}`, `/signal` reads `latest_signals.csv`, `/backtest` reads `backtest_metrics.json`, 404 with hint if artifacts missing.
- Streamlit: selectbox over per-ticker csvs, `line_chart(Close.tail(250))`, loop `detect_anomaly_row` on tail, table + `explain_alert` for last alert, expandable backtest JSON. Per-ticker csvs still indexed format — `pd.read_csv(index_col=0, parse_dates=True)` works there.

## 9. Repo map + run order

```
config.py -> data.py -> train_lightgbm.py + train_drl.py -> backtest.py -> api.py / app.py
tests/test_risk.py (7) + tests/test_core.py (4: sharpe/DD/no-lookahead/features)
run: pip install -r requirements.txt; python data.py; python train_lightgbm.py; python train_drl.py; python backtest.py; pytest -q
```

## 10. What I'd do next (Nasdaq mapping)

1. `pip install yfinance lightgbm` + rerun — real data > synthetic.
2. Add FinBERT sentiment as 14th feature (FinGPT idea) for Capital Access story.
3. Kafka/Redpanda stream for `risk.py` to hit Nasdaq Kafka/Golang/cloud bullet.
4. Deploy `Dockerfile` (py3.11-slim + uvicorn:8000) to AWS EC2, add GitHub Actions CI.

Bases referred: FinRL (train-test-trade), freqtrade (bot loop), Qlib Alpha158 (factors+report), OpenBB (data API), FinGPT (LLM), LEAN (engine). Built from scratch in my own structure for internship submission.
