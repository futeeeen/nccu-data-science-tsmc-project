"""
TSMC stock price direction prediction and trading strategy backtesting system.

Core flow based on the project specification:
1) Data collection with yfinance
2) Feature engineering (MA / RSI / MACD)
3) Classification models (Logistic Regression, Random Forest, optional XGBoost)
4) Time-series split evaluation
5) Strategy backtesting and investment performance metrics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    name: str
    model: object
    y_true: pd.Series
    y_pred: np.ndarray
    proba_up: np.ndarray
    test_df: pd.DataFrame


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Cannot download data for ticker={ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_ratio"] = out["ma_5"] / out["ma_20"]
    out["vol_chg"] = out["Volume"].pct_change()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()
    return out


def time_series_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1500, random_state=42)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass

    return models


def evaluate_model(name: str, model: object, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame) -> ModelResult:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba_up = model.predict_proba(X_test)[:, 1]
    else:
        proba_up = y_pred.astype(float)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return ModelResult(
        name=name,
        model=model,
        y_true=y_test,
        y_pred=y_pred,
        proba_up=proba_up,
        test_df=test_df.copy(),
    )


def backtest_strategy(result: ModelResult, threshold: float = 0.5) -> pd.DataFrame:
    bt = result.test_df.copy()
    bt["pred_up"] = (result.proba_up >= threshold).astype(int)
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)

    # Signal at t uses information known at close of t, execute at t+1 open/close approximation.
    bt["position"] = bt["pred_up"].shift(1).fillna(0).astype(int)
    bt["strategy_ret"] = bt["position"] * bt["asset_ret"]
    bt["buy_hold_ret"] = bt["asset_ret"]

    bt["strategy_cum"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_cum"] = (1 + bt["buy_hold_ret"]).cumprod()
    return bt


def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1
    return float(dd.min())


def sharpe_ratio(daily_returns: pd.Series, annual_factor: int = 252) -> float:
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(annual_factor))


def performance_report(bt: pd.DataFrame) -> Dict[str, float]:
    strat_total_return = float(bt["strategy_cum"].iloc[-1] - 1)
    hold_total_return = float(bt["buy_hold_cum"].iloc[-1] - 1)
    strat_mdd = max_drawdown(bt["strategy_cum"])
    hold_mdd = max_drawdown(bt["buy_hold_cum"])
    strat_sharpe = sharpe_ratio(bt["strategy_ret"])
    hold_sharpe = sharpe_ratio(bt["buy_hold_ret"])

    return {
        "strategy_total_return": strat_total_return,
        "buy_hold_total_return": hold_total_return,
        "strategy_max_drawdown": strat_mdd,
        "buy_hold_max_drawdown": hold_mdd,
        "strategy_sharpe": strat_sharpe,
        "buy_hold_sharpe": hold_sharpe,
    }


def pick_best_model(results: List[ModelResult]) -> ModelResult:
    # Prioritize profitability over plain accuracy.
    best = None
    best_score = -np.inf
    for r in results:
        bt = backtest_strategy(r)
        perf = performance_report(bt)
        score = perf["strategy_total_return"]
        if score > best_score:
            best_score = score
            best = r
    if best is None:
        raise RuntimeError("No model results available.")
    return best


def main() -> None:
    ticker = "2330.TW"  # TSMC listed in Taiwan
    start = "2015-01-01"
    end = "2026-01-01"

    print("Downloading stock data...")
    raw = download_data(ticker=ticker, start=start, end=end)
    df = add_indicators(raw)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "return_1d",
        "ma_5",
        "ma_20",
        "ma_ratio",
        "vol_chg",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]

    train_df, test_df = time_series_split(df, test_size=0.2)
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    models = get_models()
    results: List[ModelResult] = []
    for name, model in models.items():
        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df,
        )
        results.append(result)

    best = pick_best_model(results)
    bt = backtest_strategy(best)
    perf = performance_report(bt)

    print("\n===== Best Model By Backtest Return =====")
    print(best.name)
    print("\n===== Backtest Performance =====")
    for k, v in perf.items():
        print(f"{k}: {v:.4f}")

    output = pd.DataFrame(
        {
            "close": bt["Close"],
            "pred_up": bt["pred_up"],
            "position": bt["position"],
            "strategy_ret": bt["strategy_ret"],
            "buy_hold_ret": bt["buy_hold_ret"],
            "strategy_cum": bt["strategy_cum"],
            "buy_hold_cum": bt["buy_hold_cum"],
        }
    )
    output.to_csv("backtest_result.csv", index=True)
    print("\nSaved: backtest_result.csv")


if __name__ == "__main__":
    main()
