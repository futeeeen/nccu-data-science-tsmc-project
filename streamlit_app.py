from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tsmc_stock_system import (
    add_indicators,
    download_data,
    get_models,
    performance_report,
    time_series_split_three,
)


FEATURE_COLS = [
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


def run_backtest(eval_df: pd.DataFrame, proba_up: np.ndarray, threshold: float) -> pd.DataFrame:
    bt = eval_df.copy()
    bt["pred_prob_up"] = proba_up
    bt["pred_up"] = (proba_up >= threshold).astype(int)
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)
    bt["position"] = bt["pred_up"].shift(1).fillna(0).astype(int)
    bt["strategy_ret"] = bt["position"] * bt["asset_ret"]
    bt["buy_hold_ret"] = bt["asset_ret"]
    bt["strategy_cum"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_cum"] = (1 + bt["buy_hold_ret"]).cumprod()
    return bt


def evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict], str]:
    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    rows: list[dict] = []
    detail: dict[str, dict] = {}
    best_model_name = ""
    best_val_return = -np.inf

    for name, model in get_models().items():
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else val_pred.astype(float)
        val_bt = run_backtest(val_df, val_proba, threshold)
        val_perf = performance_report(val_bt)

        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else test_pred.astype(float)
        test_bt = run_backtest(test_df, test_proba, threshold)
        test_perf = performance_report(test_bt)

        rows.append(
            {
                "model": name,
                "val_accuracy": accuracy_score(y_val, val_pred),
                "val_precision": precision_score(y_val, val_pred, zero_division=0),
                "val_recall": recall_score(y_val, val_pred, zero_division=0),
                "val_f1": f1_score(y_val, val_pred, zero_division=0),
                "val_strategy_total_return": val_perf["strategy_total_return"],
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_precision": precision_score(y_test, test_pred, zero_division=0),
                "test_recall": recall_score(y_test, test_pred, zero_division=0),
                "test_f1": f1_score(y_test, test_pred, zero_division=0),
                "test_strategy_total_return": test_perf["strategy_total_return"],
                "test_buy_hold_total_return": test_perf["buy_hold_total_return"],
                "test_strategy_max_drawdown": test_perf["strategy_max_drawdown"],
                "test_strategy_sharpe": test_perf["strategy_sharpe"],
            }
        )

        detail[name] = {
            "model": model,
            "val_bt": val_bt,
            "val_perf": val_perf,
            "test_bt": test_bt,
            "test_perf": test_perf,
        }

        if val_perf["strategy_total_return"] > best_val_return:
            best_val_return = val_perf["strategy_total_return"]
            best_model_name = name

    table = pd.DataFrame(rows).sort_values(by="val_strategy_total_return", ascending=False)
    return table, detail, best_model_name


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    st.set_page_config(page_title="TSMC Stock Backtest Dashboard", layout="wide")
    st.title("TSMC Stock Direction Prediction and Backtest")
    st.caption("Train only on train split, select best model by validation return, and report final test performance.")

    with st.sidebar:
        st.header("Parameters")
        ticker = st.text_input("Ticker", value="2330.TW")
        start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
        end = st.date_input("End date", value=pd.Timestamp("2026-01-01"))
        val_size = st.slider("Validation ratio", 0.1, 0.3, 0.2, 0.05)
        test_size = st.slider("Test ratio", 0.1, 0.3, 0.2, 0.05)
        threshold = st.slider("Long signal threshold (P(up))", 0.3, 0.8, 0.5, 0.01)
        run = st.button("Run", type="primary")

    if not run:
        st.info("Set parameters in the sidebar and click Run.")
        return

    if start >= end:
        st.error("Start date must be earlier than end date.")
        return

    if val_size + test_size >= 1:
        st.error("Validation ratio + test ratio must be less than 1.")
        return

    with st.spinner("Downloading data and training models..."):
        raw = download_data(ticker=ticker, start=str(start), end=str(end))
        df = add_indicators(raw)
        train_df, val_df, test_df = time_series_split_three(df, val_size=val_size, test_size=test_size)
        table, detail, best_name = evaluate_models(train_df, val_df, test_df, threshold)

    st.subheader("Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Train rows", f"{len(train_df):,}")
    c3.metric("Validation rows", f"{len(val_df):,}")
    c4.metric("Test rows", f"{len(test_df):,}")

    st.subheader("Model Comparison")
    show_cols = [
        "model",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_strategy_total_return",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_strategy_total_return",
        "test_buy_hold_total_return",
        "test_strategy_max_drawdown",
        "test_strategy_sharpe",
    ]
    st.dataframe(
        table[show_cols].style.format(
            {
                "val_accuracy": "{:.4f}",
                "val_precision": "{:.4f}",
                "val_recall": "{:.4f}",
                "val_f1": "{:.4f}",
                "val_strategy_total_return": "{:.4f}",
                "test_accuracy": "{:.4f}",
                "test_precision": "{:.4f}",
                "test_recall": "{:.4f}",
                "test_f1": "{:.4f}",
                "test_strategy_total_return": "{:.4f}",
                "test_buy_hold_total_return": "{:.4f}",
                "test_strategy_max_drawdown": "{:.4f}",
                "test_strategy_sharpe": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    best = detail[best_name]
    bt_test = best["test_bt"]
    perf_test = best["test_perf"]

    st.subheader(f"Best Model (by Validation Return): {best_name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test strategy return", fmt_pct(perf_test["strategy_total_return"]))
    m2.metric("Test max drawdown", fmt_pct(perf_test["strategy_max_drawdown"]))
    m3.metric("Test Sharpe ratio", f"{perf_test['strategy_sharpe']:.3f}")

    n1, n2, n3 = st.columns(3)
    n1.metric("Test Buy & Hold return", fmt_pct(perf_test["buy_hold_total_return"]))
    n2.metric("Test Buy & Hold max drawdown", fmt_pct(perf_test["buy_hold_max_drawdown"]))
    n3.metric("Test Buy & Hold Sharpe", f"{perf_test['buy_hold_sharpe']:.3f}")

    st.subheader("Test Equity Curve")
    curve_df = bt_test[["strategy_cum", "buy_hold_cum"]].copy()
    curve_df.columns = ["Strategy", "Buy & Hold"]
    st.line_chart(curve_df, use_container_width=True)

    st.subheader("Latest Signal Snapshot (test split)")
    signal_df = bt_test[["Close", "pred_prob_up", "pred_up", "position"]].copy()
    st.dataframe(signal_df.tail(100), use_container_width=True)

    latest_features = df[FEATURE_COLS].tail(1)
    latest_date = df.index[-1]
    best_model = best["model"]
    next_prob_up = (
        float(best_model.predict_proba(latest_features)[:, 1][0])
        if hasattr(best_model, "predict_proba")
        else float(best_model.predict(latest_features)[0])
    )
    next_pred_up = int(next_prob_up >= threshold)
    st.subheader("Next Trading Day Prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric("Feature date", str(pd.to_datetime(latest_date).date()))
    p2.metric("P(up)", f"{next_prob_up:.4f}")
    p3.metric("Signal", "UP (long)" if next_pred_up == 1 else "DOWN/NO LONG")

    csv_df = bt_test[
        [
            "Close",
            "pred_prob_up",
            "pred_up",
            "position",
            "asset_ret",
            "strategy_ret",
            "buy_hold_ret",
            "strategy_cum",
            "buy_hold_cum",
        ]
    ].copy()
    csv_df.index.name = "date"
    csv_bytes = csv_df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "Download streamlit_backtest_result.csv",
        data=csv_bytes,
        file_name="streamlit_backtest_result.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
