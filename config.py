"""Shared config for nse-nasdaq-bot — Deleena K (Christ University)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_OUT = BASE_DIR / "data_out"
TRAINED_DIR = BASE_DIR / "trained_models"
DB_PATH = DATA_OUT / "market.db"

# NSE proxy (works via Yahoo) + NASDAQ proxy. Keep small for medium scope.
TICKERS = [
    "^NSEI",          # Nifty 50 index
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "QQQ",            # Nasdaq-100 ETF
    "AAPL",
    "MSFT",
]

START = "2019-01-01"
END = "2026-03-20"
TRAIN_END = "2024-12-31"  # >= train, > test
TEST_START = "2025-01-01"

TXN_COST_BPS = 10  # 10 bps per side
RANDOM_STATE = 42

# Risk limits (Nasdaq surveillance angle)
RISK = {
    "max_position_pct": 0.20,   # max 20% in single name
    "max_turnover_pct": 0.50,    # max 50% turnover per rebalance
    "circuit_pct": 0.10,         # block if single-day move >10%
    "max_orders_per_day": 20,
}
