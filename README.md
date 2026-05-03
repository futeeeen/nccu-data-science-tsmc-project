# TSMC Stock Prediction and Backtesting System

This project predicts next-day TSMC stock direction and backtests a trading strategy based on model signals.

## Quick Start
Use this section if this is your first time running the project.

1. Clone the repository and enter the folder
```bash
git clone https://github.com/futeeeen/nccu-data-science-tsmc-project.git
cd nccu-data-science-tsmc-project
```

2. Install dependencies
```bash
pip install pandas numpy yfinance scikit-learn streamlit
```

3. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```

4. Open the browser page shown by Streamlit and run
- Keep `Ticker` as `2330.TW` (or change it)
- Set `Start date` and `End date`
- Set `Validation ratio`, `Test ratio`, and `Threshold`
- Click `Run`

You will immediately see model comparison, backtest results, and next-day prediction.

## What the App Does
- Downloads OHLCV data from Yahoo Finance (`yfinance`)
- Builds technical features (MA, RSI, MACD)
- Splits data into `train / validation / test` by time order
- Trains models only on `train`
- Selects the best model by validation strategy return
- Reports final performance on `test`
- Generates next trading day prediction (`P(up)` and signal)

## Main Files
- `tsmc_stock_system.py`: core training, evaluation, and backtesting pipeline
- `streamlit_app.py`: interactive dashboard for running the full flow
- `backtest_result.csv`: script backtest output
- `streamlit_backtest_result.csv`: Streamlit test backtest output

## Optional: Run Script Directly
```bash
python tsmc_stock_system.py
```

## Streamlit Output
The dashboard includes:
- Classification metrics (accuracy, precision, recall, F1)
- Strategy metrics (return, max drawdown, Sharpe)
- Strategy vs buy-and-hold equity curve
- Signal table and downloadable CSV
- Next trading day prediction from the trained best model

## CSV Columns (`streamlit_backtest_result.csv`)
- `Close`: close price
- `pred_prob_up`: predicted probability of up move
- `pred_up`: binary signal from threshold
- `position`: executed position (signal shifted by one day)
- `asset_ret`: daily asset return
- `strategy_ret`: daily strategy return
- `buy_hold_ret`: daily buy-and-hold return
- `strategy_cum`: cumulative strategy curve
- `buy_hold_cum`: cumulative buy-and-hold curve

## Optional Dependency
Install XGBoost model support:
```bash
pip install xgboost
```
