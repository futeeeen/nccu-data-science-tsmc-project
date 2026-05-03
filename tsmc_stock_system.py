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
from sklearn.decomposition import PCA

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
    
    #均線 (僅作為計算基準，不放入最終特徵)
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    
    #1. 相對均線(MA Ratio)與 乖離率 (Bias: Distance to MA)
    out["ma_ratio"] = out["ma_5"] / out["ma_20"]
    out["bias_5"] = (out["Close"] / out["ma_5"]) - 1
    out["bias_20"] = (out["Close"] / out["ma_20"]) - 1
    
    #2. 成交量變化率(標準化體積)
    out["vol_chg"] = out["Volume"].pct_change()

    #3. RSI 本身已是平穩化指標(0~100)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    #4. 百分比 MACD (Percentage MACD) : 確保不受絕對價格上漲的影響
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd_pct"] = (ema12 - ema26) / out["Close"]
    out["macd_signal_pct"] = out["macd_pct"].ewm(span=9,adjust=False).mean()
    out["macd_hist_pct"] = out["macd_pct"] - out["macd_signal_pct"]
    
    #Target
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()
    return out


def time_series_split_three(
    df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("Require val_size > 0, test_size > 0, and val_size + test_size < 1.")

    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def get_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        # Ridge 正規化 (L2 懲罰) - 收縮權重去噪
        "LogisticRegression_Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty='l2', C=1.0, max_iter=1500, random_state=42)),
            ]
        ),
        # Lasso 正規化 (L1 懲罰) - 自動特徵篩選 (會把不重要的特徵權重歸零)
        "LogisticRegression_Lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1500, random_state=42)),
            ]
        ),
        # PCA 主成份分析 + 原本的羅吉斯迴歸
        "LogisticRegression_PCA": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.90)), # 擷取解釋 90% 變異數的主成份
                ("clf", LogisticRegression(penalty='l2', max_iter=1500, random_state=42)),
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


def evaluate_model(
    name: str,
    model: object,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_df: pd.DataFrame,
    split_name: str,
) -> ModelResult:
    y_pred = model.predict(X_eval)

    if hasattr(model, "predict_proba"):
        proba_up = model.predict_proba(X_eval)[:, 1]
    else:
        proba_up = y_pred.astype(float)

    print(f"\n===== {name} | {split_name} =====")
    print(f"Accuracy : {accuracy_score(y_eval, y_pred):.4f}")
    print(f"Precision: {precision_score(y_eval, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_eval, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_eval, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_eval, y_pred))
    print("Classification Report:")
    print(classification_report(y_eval, y_pred, digits=4, zero_division=0))

    return ModelResult(
        name=name,
        model=model,
        y_true=y_eval,
        y_pred=y_pred,
        proba_up=proba_up,
        test_df=eval_df.copy(),
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

    train_df, val_df, test_df = time_series_split_three(df, val_size=0.2, test_size=0.2)
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    models = get_models()
    val_results: List[ModelResult] = []
    test_results: List[ModelResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        val_result = evaluate_model(
            name=name,
            model=model,
            X_eval=X_val,
            y_eval=y_val,
            eval_df=val_df,
            split_name="Validation",
        )
        val_results.append(val_result)

        test_result = evaluate_model(
            name=name,
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            eval_df=test_df,
            split_name="Test",
        )
        test_results.append(test_result)

    best = pick_best_model(val_results)
    best_test = next(r for r in test_results if r.name == best.name)
    bt_val = backtest_strategy(best)
    perf_val = performance_report(bt_val)
    bt_test = backtest_strategy(best_test)
    perf_test = performance_report(bt_test)

    print("\n===== Best Model By Validation Backtest Return =====")
    print(best.name)
    print("\n===== Validation Backtest Performance =====")
    for k, v in perf_val.items():
        print(f"{k}: {v:.4f}")
    print("\n===== Test Backtest Performance =====")
    for k, v in perf_test.items():
        print(f"{k}: {v:.4f}")

    output = pd.DataFrame(
        {
            "close": bt_test["Close"],
            "pred_up": bt_test["pred_up"],
            "position": bt_test["position"],
            "strategy_ret": bt_test["strategy_ret"],
            "buy_hold_ret": bt_test["buy_hold_ret"],
            "strategy_cum": bt_test["strategy_cum"],
            "buy_hold_cum": bt_test["buy_hold_cum"],
        }
    )
    output.to_csv("backtest_result.csv", index=True)
    print("\nSaved: backtest_result.csv")


if __name__ == "__main__":
    main()
