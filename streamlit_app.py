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
    time_series_split_three
)


FEATURE_COLS = [
    "return_1d",
    "ma_ratio",
    "bias_5",
    "bias_20",
    "vol_chg",
    "rsi_14",
    "macd",        
    "macd_signal",  
    "macd_hist",    
]

def execute_dynamic_leverage(p: float, th: float) -> float:
    # 自訂動態槓桿邏輯
    if p >= 0.60:
        return 2.0  # 很有把握：開 2 倍槓桿
    elif p >= th:
        return 1.0  # 過門檻：正常做多
    elif p <= 0.40:
        return -1.0  # 很有把握下跌(上漲機率極低) : 做空
    else:
        return 0.0  # 完全沒信心：空手觀望 > 機率在0.40 ~ threshold 之間
        
def run_backtest(
    eval_df: pd.DataFrame, proba_up: np.ndarray, threshold: float, cost_per_trade: float
) -> pd.DataFrame:
    bt = eval_df.copy()
    bt["pred_prob_up"] = proba_up
    
    # 決定部位大小 (動態槓桿)
    v_func = np.vectorize(execute_dynamic_leverage)
    bt["target_position"] = v_func(proba_up, threshold)
    
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)
    
    # 使用前一天的決定執行
    bt["position"] = bt["target_position"].shift(1).fillna(0)
    bt["position_chg"] = bt["position"].diff().abs().fillna(bt["position"])
    
    # 成本根據變動的槓桿倍數計算
    bt["cost"] = bt["position_chg"] * cost_per_trade
    bt["strategy_ret_gross"] = bt["position"] * bt["asset_ret"]
    bt["strategy_ret"] = bt["strategy_ret_gross"] - bt["cost"]
    bt["buy_hold_ret"] = bt["asset_ret"]
    
    bt["strategy_cum"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_cum"] = (1 + bt["buy_hold_ret"]).cumprod()
    return bt


def pick_threshold(
    val_df: pd.DataFrame,
    proba_up: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    step: float,
    cost_per_trade: float,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(min_threshold, max_threshold + 1e-12, step)
    best_threshold = min_threshold
    best_score = -np.inf
    rows: list[dict] = []

    for th in thresholds:
        bt = run_backtest(val_df, proba_up, float(th), cost_per_trade)
        perf = performance_report(bt)
        rows.append(
            {
                "threshold": float(th),
                "val_strategy_total_return": perf["strategy_total_return"],
                "val_strategy_sharpe": perf["strategy_sharpe"],
                "val_strategy_max_drawdown": perf["strategy_max_drawdown"],
            }
        )
        if perf["strategy_total_return"] > best_score:
            best_score = perf["strategy_total_return"]
            best_threshold = float(th)

    return best_threshold, pd.DataFrame(rows)


def strategy_diagnostics(bt: pd.DataFrame) -> dict[str, float]:
    in_position = bt["position"] > 0
    entry_count = int(((bt["position"] > 0) & (bt["position"].shift(1).fillna(0) == 0)).sum())
    holding_ratio = float(bt["position"].mean())
    active_days = int(in_position.sum())
    active_ret = bt.loc[in_position, "strategy_ret"]
    win_rate = float((active_ret > 0).mean()) if len(active_ret) > 0 else 0.0
    avg_win = float(active_ret[active_ret > 0].mean()) if (active_ret > 0).any() else 0.0
    avg_loss = float(active_ret[active_ret < 0].mean()) if (active_ret < 0).any() else 0.0
    expectancy = float(active_ret.mean()) if len(active_ret) > 0 else 0.0
    trade_days = int((bt["position_chg"] > 0).sum())
    total_cost = float((bt["position_chg"] * bt["cost"]).sum())
    return {
        "entry_count": entry_count,
        "holding_ratio": holding_ratio,
        "active_days": active_days,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "trade_days": trade_days,
        "total_cost": total_cost,
    }


def evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_threshold: float,
    auto_threshold: bool,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    cost_per_trade: float,
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
        selected_threshold = base_threshold
        threshold_grid = pd.DataFrame()
        if auto_threshold:
            selected_threshold, threshold_grid = pick_threshold(
                val_df=val_df,
                proba_up=val_proba,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
                step=threshold_step,
                cost_per_trade=cost_per_trade,
            )
        val_bt = run_backtest(val_df, val_proba, selected_threshold, cost_per_trade)
        val_perf = performance_report(val_bt)

        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else test_pred.astype(float)
        test_bt = run_backtest(test_df, test_proba, selected_threshold, cost_per_trade)
        test_perf = performance_report(test_bt)
        test_diag = strategy_diagnostics(test_bt)

        rows.append(
            {
                "model": name,
                "threshold_used": selected_threshold,
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
                "test_holding_ratio": test_diag["holding_ratio"],
                "test_entry_count": test_diag["entry_count"],
            }
        )

        detail[name] = {
            "model": model,
            "threshold": selected_threshold,
            "threshold_grid": threshold_grid,
            "val_bt": val_bt,
            "val_perf": val_perf,
            "test_bt": test_bt,
            "test_perf": test_perf,
            "test_diag": test_diag,
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
        auto_threshold = st.checkbox("Auto-search threshold on validation", value=True)
        threshold = st.slider("Base threshold (used when auto-search is off)", 0.3, 0.8, 0.5, 0.01)
        threshold_min = st.slider("Threshold search min", 0.3, 0.8, 0.4, 0.01)
        threshold_max = st.slider("Threshold search max", 0.3, 0.8, 0.7, 0.01)
        threshold_step = st.select_slider("Threshold search step", options=[0.01, 0.02, 0.05], value=0.01)
        cost_bps = st.number_input("Round-trip cost (bps per position change)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
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
    if threshold_min > threshold_max:
        st.error("Threshold search min must be <= max.")
        return

    cost_per_trade = cost_bps / 10000.0

    with st.spinner("Downloading data and training models..."):
        raw = download_data(ticker=ticker, start=str(start), end=str(end))
        df = add_indicators(raw)
        train_df, val_df, test_df = time_series_split_three(df, val_size=val_size, test_size=test_size)
        table, detail, best_name = evaluate_models(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            base_threshold=threshold,
            auto_threshold=auto_threshold,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            cost_per_trade=cost_per_trade,
        )

    st.subheader("Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Train rows", f"{len(train_df):,}")
    c3.metric("Validation rows", f"{len(val_df):,}")
    c4.metric("Test rows", f"{len(test_df):,}")

    st.subheader("Model Comparison")
    show_cols = [
        "model",
        "threshold_used",
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
        "test_holding_ratio",
        "test_entry_count",
    ]
    st.dataframe(
        table[show_cols].style.format(
            {
                "val_accuracy": "{:.4f}",
                "threshold_used": "{:.2f}",
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
                "test_holding_ratio": "{:.2%}",
                "test_entry_count": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    best = detail[best_name]
    bt_test = best["test_bt"]
    perf_test = best["test_perf"]
    diag_test = best["test_diag"]
    selected_threshold = best["threshold"]

    st.subheader("Current Strategy Definition")
    strategy_note = (
        f"- Signal rule: Dynamic leverage (`target_position = f(P(up), threshold={selected_threshold:.2f})`)\n"
        "- Execution rule: use previous-day target position (`position = target_position.shift(1)`)\n"
        f"- Trading cost: {cost_bps:.1f} bps per position change (Δ leverage)\n"
        f"- Threshold mode: {'Auto-search on validation' if auto_threshold else 'Fixed threshold'}\n"
        "- Model selection rule: choose model with highest validation strategy total return"
    )
    st.markdown(strategy_note)

    st.subheader(f"Best Model (by Validation Return): {best_name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test strategy return", fmt_pct(perf_test["strategy_total_return"]))
    m2.metric("Test max drawdown", fmt_pct(perf_test["strategy_max_drawdown"]))
    m3.metric("Test Sharpe ratio", f"{perf_test['strategy_sharpe']:.3f}")

    n1, n2, n3 = st.columns(3)
    n1.metric("Test Buy & Hold return", fmt_pct(perf_test["buy_hold_total_return"]))
    n2.metric("Test Buy & Hold max drawdown", fmt_pct(perf_test["buy_hold_max_drawdown"]))
    n3.metric("Test Buy & Hold Sharpe", f"{perf_test['buy_hold_sharpe']:.3f}")

    st.subheader("Strategy Diagnostics (test split)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Entry count", f"{diag_test['entry_count']}")
    d2.metric("Holding ratio", fmt_pct(diag_test["holding_ratio"]))
    d3.metric("Win rate (holding days)", fmt_pct(diag_test["win_rate"]))
    d4.metric("Total trading cost", fmt_pct(diag_test["total_cost"]))

    e1, e2, e3 = st.columns(3)
    e1.metric("Avg win (daily)", fmt_pct(diag_test["avg_win"]))
    e2.metric("Avg loss (daily)", fmt_pct(diag_test["avg_loss"]))
    e3.metric("Expectancy (daily)", fmt_pct(diag_test["expectancy"]))

    if auto_threshold and not best["threshold_grid"].empty:
        st.subheader("Threshold Search Result (best model, validation split)")
        st.dataframe(
            best["threshold_grid"].style.format(
                {
                    "threshold": "{:.2f}",
                    "val_strategy_total_return": "{:.4f}",
                    "val_strategy_sharpe": "{:.4f}",
                    "val_strategy_max_drawdown": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    st.subheader("Test Equity Curve")
    curve_df = bt_test[["strategy_cum", "buy_hold_cum"]].copy()
    curve_df.columns = ["Strategy", "Buy & Hold"]
    st.line_chart(curve_df, use_container_width=True)

    st.subheader("Latest Signal Snapshot (test split)")
    signal_df = bt_test[["Close", "pred_prob_up", "target_position", "position"]].copy()
    st.dataframe(signal_df.tail(100), use_container_width=True)

    latest_features = df[FEATURE_COLS].tail(1)
    latest_date = df.index[-1]
    best_model = best["model"]
    next_prob_up = (
        float(best_model.predict_proba(latest_features)[:, 1][0])
        if hasattr(best_model, "predict_proba")
        else float(best_model.predict(latest_features)[0])
    )
    
    # 棄用原本的 0/1 邏輯，改用你的動態槓桿函數來計算明天的目標部位
    next_target_position = execute_dynamic_leverage(next_prob_up, selected_threshold)
    
    st.subheader("Next Trading Day Prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric("Feature date", str(pd.to_datetime(latest_date).date()))
    p2.metric("P(up)", f"{next_prob_up:.4f}")
    
    # 顯示目標部位 (動態槓桿倍數) 而不是單純的 UP/DOWN
    p3.metric("Target Position (Leverage)", f"{next_target_position:.2f}x")

    csv_df = bt_test[
        [
            "Close",
            "pred_prob_up",
            "target_position",
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
