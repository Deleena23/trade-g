@echo off
REM Deleena K - one-click run
pip install -r requirements.txt
python data.py
python train_lightgbm.py
python train_drl.py
python backtest.py
python -m pytest tests -q
echo Done. API: uvicorn api:app --port 8000  UI: streamlit run app.py
