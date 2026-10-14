"""Train direction model: LightGBM if available else sklearn fallback.
Run: python train_lightgbm.py  (requires data_out/train_data.csv)
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd

from config import DATA_OUT, TRAINED_DIR, RANDOM_STATE

FEATURES = ["rsi_14", "macd", "macd_signal", "sma_20", "sma_60",
            "boll_ub", "boll_lb", "vol_20", "turbulence",
            "spread_proxy", "order_imbalance", "cancel_rate", "ret_1d"]


def load_train() -> pd.DataFrame:
    p = DATA_OUT / "train_data.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}, run python data.py first")
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df.dropna(subset=FEATURES + ["target"])


def build_model():
    try:
        from lightgbm import LGBMClassifier
        print("[train_lgbm] using LightGBM")
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print(f"[train_lgbm] LightGBM unavailable ({e}), using HistGradientBoosting")
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)


def main() -> dict:
    TRAINED_DIR.mkdir(exist_ok=True)
    df = load_train()
    X, y = df[FEATURES], df["target"]
    # time split: last 20% as valid
    n = len(df)
    split = int(n * 0.8)
    Xtr, Xva, ytr, yva = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    model = build_model()
    model.fit(Xtr, ytr)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    pred = model.predict(Xva)
    metrics = {
        "train_rows": len(Xtr),
        "valid_rows": len(Xva),
        "accuracy": float(accuracy_score(yva, pred)),
        "precision": float(precision_score(yva, pred, zero_division=0)),
        "recall": float(recall_score(yva, pred, zero_division=0)),
        "features": FEATURES,
    }
    joblib.dump(model, TRAINED_DIR / "lgbm_model.pkl")
    (TRAINED_DIR / "lgbm_metrics.json").write_text(json.dumps(metrics, indent=2))
    # per-ticker latest signal for API
    latest = df.groupby("ticker").tail(1)[FEATURES + ["ticker", "Close"]].copy()
    try:
        latest["signal"] = model.predict(latest[FEATURES])
        # predict_proba may not exist on all fallbacks
        if hasattr(model, "predict_proba"):
            latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    except Exception:
        latest["signal"] = 0
    latest[["ticker", "Close", "signal"]].to_csv(DATA_OUT / "latest_signals.csv", index=False)
    print(f"[train_lgbm] {metrics}")
    return metrics


if __name__ == "__main__":
    main()
