"""Streamlit dashboard: price + signals + surveillance alerts.
Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st

from config import DATA_OUT
from risk import detect_anomaly_row, explain_alert

st.set_page_config(page_title="trade-g", layout="wide")
st.title("trade-g - surveillance + backtest")

files = sorted(DATA_OUT.glob("*.csv"))
names = [f.name for f in files if f.name not in ("train_data.csv", "trade_data.csv")]
choice = st.selectbox("Ticker file", names or ["— run data.py first —"])
if choice.startswith("—"):
    st.stop()

df = pd.read_csv(DATA_OUT / choice, index_col=0, parse_dates=True)
st.line_chart(df["Close"].tail(250))

alerts = []
for idx, row in df.tail(250).iterrows():
    is_alert, rule = detect_anomaly_row(row.to_dict())
    if is_alert:
        alerts.append({"date": str(idx), "ticker": choice,
                       "price": float(row["Close"]),
                       "rule": rule,
                       "explain": explain_alert(choice, rule, float(row["Close"]), str(idx))})
st.subheader(f"Alerts: {len(alerts)}")
if alerts:
    st.dataframe(pd.DataFrame(alerts).tail(20))
    st.write(alerts[-1]["explain"])
else:
    st.write("No alerts in window.")

try:
    import json
    from config import TRAINED_DIR
    m = (TRAINED_DIR / "backtest_metrics.json").read_text()
    st.subheader("Backtest metrics")
    st.json(m[:4000])
except Exception:
    st.info("Run backtest.py to see Sharpe/drawdown metrics.")
