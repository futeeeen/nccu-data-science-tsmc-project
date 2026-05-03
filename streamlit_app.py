from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tsmc_stock_system import (
    backtest_strategy,
    download_data,
    get_models,
    performance_report,
    time_series_split,
    add_indicators,
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


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict], str]:
    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    model_table_rows = []
    detail = {}
    best_model_name = ""
    best_return = -np.inf

    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba_up = model.predict_proba(X_test)[:, 1]
        else:
            proba_up = y_pred.astype(float)

        sim_df = test_df.copy()
        sim_df["pred_prob_up"] = proba_up
        sim_df["pred_up"] = (proba_up >= threshold).astype(int)
        sim_df["asset_ret"] = sim_df["Close"].pct_change().fillna(0.0)
        sim_df["position"] = sim_df["pred_up"].shift(1).fillna(0).astype(int)
        sim_df["strategy_ret"] = sim_df["position"] * sim_df["asset_ret"]
        sim_df["buy_hold_ret"] = sim_df["asset_ret"]
        sim_df["strategy_cum"] = (1 + sim_df["strategy_ret"]).cumprod()
        sim_df["buy_hold_cum"] = (1 + sim_df["buy_hold_ret"]).cumprod()

        perf = performance_report(sim_df)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "strategy_total_return": perf["strategy_total_return"],
            "buy_hold_total_return": perf["buy_hold_total_return"],
            "strategy_max_drawdown": perf["strategy_max_drawdown"],
            "strategy_sharpe": perf["strategy_sharpe"],
        }
        model_table_rows.append(row)
        detail[name] = {"model": model, "y_pred": y_pred, "proba_up": proba_up, "bt": sim_df, "perf": perf}

        if perf["strategy_total_return"] > best_return:
            best_return = perf["strategy_total_return"]
            best_model_name = name

    table = pd.DataFrame(model_table_rows).sort_values(by="strategy_total_return", ascending=False)
    return table, detail, best_model_name


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    st.set_page_config(page_title="TSMC 預測與交易策略儀表板", layout="wide")
    st.title("台積電股票價格預測與交易策略分析系統")
    st.caption("以策略報酬作為模型挑選主目標，並同步檢視分類指標。")

    with st.sidebar:
        st.header("參數設定")
        ticker = st.text_input("股票代號", value="2330.TW")
        start = st.date_input("開始日期", value=pd.Timestamp("2015-01-01"))
        end = st.date_input("結束日期", value=pd.Timestamp("2026-01-01"))
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
        threshold = st.slider("買進訊號門檻 (P(up))", 0.3, 0.8, 0.5, 0.01)
        run = st.button("開始分析", type="primary")

    if not run:
        st.info("請先在左側設定參數，然後按「開始分析」。")
        return

    if start >= end:
        st.error("開始日期必須早於結束日期。")
        return

    with st.spinner("下載資料與訓練模型中..."):
        raw = download_data(ticker=ticker, start=str(start), end=str(end))
        df = add_indicators(raw)
        train_df, test_df = time_series_split(df, test_size=test_size)
        table, detail, best_name = evaluate_models(train_df, test_df, threshold)

    st.subheader("資料概況")
    c1, c2, c3 = st.columns(3)
    c1.metric("總樣本數", f"{len(df):,}")
    c2.metric("訓練集筆數", f"{len(train_df):,}")
    c3.metric("測試集筆數", f"{len(test_df):,}")

    st.subheader("模型比較（依策略報酬排序）")
    show_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "strategy_total_return",
        "buy_hold_total_return",
        "strategy_max_drawdown",
        "strategy_sharpe",
    ]
    st.dataframe(
        table[show_cols].style.format(
            {
                "accuracy": "{:.4f}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1": "{:.4f}",
                "strategy_total_return": "{:.4f}",
                "buy_hold_total_return": "{:.4f}",
                "strategy_max_drawdown": "{:.4f}",
                "strategy_sharpe": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    best = detail[best_name]
    bt = best["bt"]
    perf = best["perf"]

    st.subheader(f"最佳模型：{best_name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("策略總報酬", fmt_pct(perf["strategy_total_return"]))
    m2.metric("最大回撤", fmt_pct(perf["strategy_max_drawdown"]))
    m3.metric("Sharpe Ratio", f"{perf['strategy_sharpe']:.3f}")

    n1, n2, n3 = st.columns(3)
    n1.metric("Buy & Hold 總報酬", fmt_pct(perf["buy_hold_total_return"]))
    n2.metric("Buy & Hold 最大回撤", fmt_pct(perf["buy_hold_max_drawdown"]))
    n3.metric("Buy & Hold Sharpe", f"{perf['buy_hold_sharpe']:.3f}")

    st.subheader("策略淨值曲線")
    curve_df = bt[["strategy_cum", "buy_hold_cum"]].copy()
    curve_df.columns = ["Strategy", "Buy & Hold"]
    st.line_chart(curve_df, use_container_width=True)

    st.subheader("交易訊號與收盤價")
    signal_df = bt[["Close", "pred_up", "position"]].copy()
    st.dataframe(signal_df.tail(100), use_container_width=True)

    csv_df = bt.copy()
    csv_bytes = csv_df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "下載回測結果 CSV",
        data=csv_bytes,
        file_name="streamlit_backtest_result.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
