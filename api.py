"""FastAPI service: /health /signal /backtest.
Run: uvicorn api:app --reload --port 8000
"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

from config import DATA_OUT, TRAINED_DIR

app = FastAPI(title="trade-g")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/signal")
def signal():
    p = DATA_OUT / "latest_signals.csv"
    if not p.exists():
        return JSONResponse({"error": "run train_lightgbm.py first"}, status_code=404)
    df = pd.read_csv(p)
    return {"signals": df.to_dict(orient="records")}


@app.get("/backtest")
def backtest():
    p = TRAINED_DIR / "backtest_metrics.json"
    if not p.exists():
        return JSONResponse({"error": "run backtest.py first"}, status_code=404)
    return json.loads(Path(p).read_text())
