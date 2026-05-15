from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from project_glossary import render_sticky_title_glossary
from technical_system import (
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


TEXT = {
    "en": {
        "app_title": "TSMC Stock Direction Prediction and Backtest",
        "app_caption": "Run once to generate both the backtest dashboard and the project lifecycle view.",
        "parameters": "Parameters",
        "language": "Language / 語言",
        "run": "Run",
        "initial_info": "Set parameters in the sidebar and click Run. Both dashboards will appear below after one run.",
        "tab_backtest": "Backtest Dashboard",
        "tab_training_data": "Training Data",
        "tab_lifecycle": "Project Lifecycle",
        "tab_architecture": "Architecture",
        "lifecycle_title": "Data Science Project Lifecycle",
        "lifecycle_caption": "This view maps the final project lifecycle to the concrete outputs produced by this app.",
        "problem": "Problem",
        "data_rows": "Data rows",
        "best_model": "Best model",
        "test_return": "Test return",
        "stage_detail": "Stage Detail",
        "guiding_question": "Guiding question",
        "method_detail": "Method detail",
        "project_evidence": "Project Evidence",
        "actual_model_comparison": "Actual Model Comparison",
        "lifecycle_scorecard": "Lifecycle Scorecard",
        "data_summary": "Data Summary",
        "train_rows": "Train rows",
        "validation_rows": "Validation rows",
        "test_rows": "Test rows",
        "total_rows": "Total rows",
        "model_comparison": "Model Comparison",
        "strategy_definition": "Current Strategy Definition",
        "strategy_diagnostics": "Strategy Diagnostics (test split)",
        "threshold_search": "Threshold Search Result (best model, validation split)",
        "equity_curve": "Test Equity Curve",
        "signal_snapshot": "Latest Signal Snapshot (test split)",
        "next_prediction": "Next Trading Day Prediction",
        "feature_date": "Feature date",
        "target_position": "Target Position (Leverage)",
        "download_csv": "Download streamlit_backtest_result.csv",
        "choose_stage4": "Choose Stage 4 to see the full model comparison table used for selection.",
        "architecture_title": "Interactive Project Architecture",
        "architecture_caption": "Switch granularity, inspect each module, and connect the graph to what the dashboard produces.",
        "granularity": "Granularity",
        "overview_level": "Overview",
        "detail_level": "Detailed",
        "inspect_module": "Inspect module",
        "module_role": "Role",
        "module_inputs": "Inputs",
        "module_outputs": "Outputs",
        "module_evidence": "Evidence in this app",
    },
    "zh": {
        "app_title": "台積電股價方向預測與回測系統",
        "app_caption": "按一次 Run 後，同時產生回測 Dashboard 與資料科學專案流程頁。",
        "parameters": "參數設定",
        "language": "Language / 語言",
        "run": "執行 Run",
        "initial_info": "請先在左側設定參數並按 Run，執行一次後下方會同時出現兩個 Dashboard。",
        "tab_backtest": "回測 Dashboard",
        "tab_training_data": "訓練資料",
        "tab_lifecycle": "專案流程",
        "tab_architecture": "架構探索",
        "lifecycle_title": "資料科學專案生命週期",
        "lifecycle_caption": "此頁把資料科學專案各階段，對應到本系統實際完成的內容與結果。",
        "problem": "問題",
        "data_rows": "資料筆數",
        "best_model": "最佳模型",
        "test_return": "測試集報酬",
        "stage_detail": "階段說明",
        "guiding_question": "引導問題",
        "method_detail": "方法細節",
        "project_evidence": "專案證據",
        "actual_model_comparison": "實際模型比較結果",
        "lifecycle_scorecard": "流程重點摘要",
        "data_summary": "資料摘要",
        "train_rows": "訓練資料筆數",
        "validation_rows": "驗證資料筆數",
        "test_rows": "測試資料筆數",
        "total_rows": "總資料筆數",
        "model_comparison": "模型比較",
        "strategy_definition": "目前策略定義",
        "strategy_diagnostics": "策略診斷（測試集）",
        "threshold_search": "Threshold 搜尋結果（最佳模型，驗證集）",
        "equity_curve": "測試集累積報酬曲線",
        "signal_snapshot": "最新訊號快照（測試集）",
        "next_prediction": "下一交易日預測",
        "feature_date": "特徵日期",
        "target_position": "目標部位（槓桿）",
        "download_csv": "下載 streamlit_backtest_result.csv",
        "choose_stage4": "選擇 Stage 4 可查看用於選模的完整模型比較表。",
        "architecture_title": "互動式專案架構探索",
        "architecture_caption": "切換架構層級、點選模組，並把資料流連回 dashboard 產出的結果。",
        "granularity": "架構層級",
        "overview_level": "總覽",
        "detail_level": "細節",
        "inspect_module": "查看模組",
        "module_role": "角色",
        "module_inputs": "輸入",
        "module_outputs": "輸出",
        "module_evidence": "本工具中的證據",
    },
}


def tr(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))


def execute_dynamic_leverage(p: float, th: float) -> float:
    # ?芾???瑽▼?摩
    if p >= 0.60:
        return 2.0  # 敺??嚗? 2 ??獢?
    elif p >= th:
        return 1.0  # ??瑼鳴?甇?虜??
    elif p <= 0.40:
        return -1.0  # 敺??銝?(銝撞璈?璆萎?) : ?征
    else:
        return 0.0  # 摰瘝縑敹?蝛箸?閫??> 璈???.40 ~ threshold 銋?
        
def run_backtest(
    eval_df: pd.DataFrame, proba_up: np.ndarray, threshold: float, cost_per_trade: float
) -> pd.DataFrame:
    bt = eval_df.copy()
    bt["pred_prob_up"] = proba_up
    
    # 瘙箏??其?憭批? (??瑽▼)
    v_func = np.vectorize(execute_dynamic_leverage)
    bt["target_position"] = v_func(proba_up, threshold)
    
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)
    
    # 雿輻??憭拍?瘙箏??瑁?
    bt["position"] = bt["target_position"].shift(1).fillna(0)
    bt["position_chg"] = bt["position"].diff().abs().fillna(bt["position"])
    
    # ??寞?霈???獢踹閮?
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
    total_cost = float(bt["cost"].sum())
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


def render_project_lifecycle_dashboard(
    lang: str,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    table: pd.DataFrame,
    best_name: str,
    best: dict,
    selected_threshold: float,
    cost_bps: float,
    next_prob_up: float,
    next_target_position: float,
) -> None:
    perf_test = best["test_perf"]
    diag_test = best["test_diag"]

    st.subheader(tr(lang, "lifecycle_title"))
    st.caption(tr(lang, "lifecycle_caption"))

    summary_cols = st.columns(4)
    summary_cols[0].metric(tr(lang, "problem"), "Next-day direction" if lang == "en" else "預測隔日漲跌")
    summary_cols[1].metric(tr(lang, "data_rows"), f"{len(df):,}")
    summary_cols[2].metric(tr(lang, "best_model"), best_name)
    summary_cols[3].metric(tr(lang, "test_return"), fmt_pct(perf_test["strategy_total_return"]))

    graph = """
    digraph {
        graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.65"];
        node [shape=box, style="rounded,filled", color="#7b8794", fillcolor="#eef6fb", fontname="Arial", fontsize=12, margin="0.15,0.10"];
        edge [color="#8c96a3", arrowsize=0.8, penwidth=1.4];

        goal [label="1. Define goal\\nPredict next-day direction"];
        data [label="2. Collect data\\nyfinance OHLCV"];
        build [label="3. Build model\\nTechnical indicators + ML"];
        eval [label="4. Evaluate model\\nValidation + test backtest"];
        present [label="5. Present results\\nDashboard + CSV"];
        deploy [label="6. Deploy model\\nNext-day signal"];

        goal -> data -> build -> eval -> present -> deploy;
        eval -> build [label=" critique", fontsize=10, color="#b07d62"];
        deploy -> goal [label=" improve", fontsize=10, color="#b07d62"];
    }
    """
    st.graphviz_chart(graph, use_container_width=True)

    if lang == "zh":
        lifecycle_rows = [
            {
                "Lifecycle step": "Define the goal",
                "Question": "我要解決什麼問題？",
                "Project result": "預測台積電下一個交易日是否上漲，並測試這個訊號是否能形成可用的交易策略。",
                "Method detail": "建立二元分類目標：若明日收盤價高於今日收盤價，target=1；否則 target=0。",
                "Evidence in app": "Target = next-day Close > current Close。",
            },
            {
                "Lifecycle step": "Collect and manage data",
                "Question": "我需要哪些資料？",
                "Project result": f"下載 {ticker} 從 {start} 到 {end} 的 OHLCV 股價資料。特徵工程與清理後，保留 {len(df):,} 筆資料。",
                "Method detail": "Feature cleaning 包含計算 daily return、MA ratio、5/20 日 bias、成交量變化、RSI、MACD、MACD signal 與 MACD histogram；將 inf 轉成缺值；並移除 rolling window、pct_change 與 target shift 造成的缺值列。",
                "Evidence in app": f"清理後資料依時間順序切成 Train/Validation/Test = {len(train_df):,}/{len(val_df):,}/{len(test_df):,}。",
            },
            {
                "Lifecycle step": "Build the model",
                "Question": "資料中有哪些模式可以用來解決問題？",
                "Project result": f"建立 {len(table):,} 個候選模型，使用 return、MA ratio、bias、volume change、RSI 與 MACD 類技術指標訓練。",
                "Method detail": "模型只使用 training split 訓練。線性模型先用 StandardScaler 標準化再進 Logistic Regression；PCA_LogReg 會先降維再分類；樹模型則從相同特徵中學習非線性規則。Validation/Test 只用於評估，不參與訓練。",
                "Evidence in app": "候選模型包含 LogisticRegression_Ridge、PCA_LogReg、RandomForest，以及環境有安裝時的 XGBoost。",
            },
            {
                "Lifecycle step": "Evaluate and critique model",
                "Question": "模型是否真的解決問題？",
                "Project result": f"實際比較 {len(table):,} 個模型。依 validation strategy total return 選出最佳模型 {best_name}；其測試集 Sharpe = {perf_test['strategy_sharpe']:.3f}，最大回撤 = {fmt_pct(perf_test['strategy_max_drawdown'])}。",
                "Method detail": "比較同時使用分類指標（accuracy、precision、recall、F1）與投資績效指標（strategy return、buy-and-hold return、max drawdown、Sharpe、entry count、holding ratio）。選模依 validation strategy total return，test split 則保留做最終樣本外報告。",
                "Evidence in app": f"Selected threshold = {selected_threshold:.2f}；test entries = {diag_test['entry_count']}；holding ratio = {fmt_pct(diag_test['holding_ratio'])}。",
            },
            {
                "Lifecycle step": "Present results and document",
                "Question": "我能否清楚呈現結果與方法？",
                "Project result": "Dashboard 顯示模型比較、策略定義、診斷指標、累積報酬曲線、最新訊號與可下載 CSV。",
                "Method detail": "Backtest Dashboard 提供數字表格與圖表；Project Lifecycle 則用報告友善的方式整理每個階段。",
                "Evidence in app": "可在 Backtest Dashboard 查看完整模型表格、策略圖與下載結果。",
            },
            {
                "Lifecycle step": "Deploy model",
                "Question": "模型如何在真實情境使用？",
                "Project result": f"目前訓練出的最佳模型會輸出下一交易日上漲機率 P(up) = {next_prob_up:.4f}。",
                "Method detail": "訓練後將最新一筆特徵輸入最佳模型，得到 P(up)，再由交易規則轉換成下一交易日的 target position。",
                "Evidence in app": f"建議目標部位 = {next_target_position:.2f}x；交易成本假設 = {cost_bps:.1f} bps。",
            },
        ]
    else:
        lifecycle_rows = [
        {
            "Lifecycle step": "Define the goal",
            "Question": "What problem am I solving?",
            "Project result": "Predict whether TSMC will rise on the next trading day and test whether the signal can support a trading strategy.",
            "Method detail": "Create a binary classification target: 1 means tomorrow's close is higher than today's close; 0 means otherwise.",
            "Evidence in app": "Target = next-day Close > current Close.",
        },
        {
            "Lifecycle step": "Collect and manage data",
            "Question": "What information do I need?",
            "Project result": f"Downloaded {ticker} OHLCV data from {start} to {end}. After feature engineering and cleaning, {len(df):,} rows remain.",
            "Method detail": "Feature cleaning includes calculating daily return, MA ratio, 5/20-day bias, volume change, RSI, MACD, MACD signal, and MACD histogram; replacing inf values with missing values; and dropping rows with missing values caused by rolling windows, percentage change, or future target shifting.",
            "Evidence in app": f"Cleaned feature rows are split by time order into Train/Validation/Test = {len(train_df):,}/{len(val_df):,}/{len(test_df):,}.",
        },
        {
            "Lifecycle step": "Build the model",
            "Question": "What patterns can lead to solutions?",
            "Project result": f"Built {len(table):,} candidate models using technical indicators: return, MA ratio, bias, volume change, RSI, and MACD-based features.",
            "Method detail": "Training uses only the training split. Linear models use StandardScaler before Logistic Regression; PCA_LogReg additionally reduces dimensions before classification. Tree-based models learn nonlinear rules from the same engineered features. Each model is fit once on X_train/y_train, then evaluated on validation and test without retraining on those splits.",
            "Evidence in app": "Candidate models include LogisticRegression_Ridge, PCA_LogReg, RandomForest, and XGBoost when the package is installed.",
        },
        {
            "Lifecycle step": "Evaluate and critique model",
            "Question": "Does the model solve my problem?",
            "Project result": f"Compared {len(table):,} models. Best model by validation strategy return is {best_name}; test Sharpe = {perf_test['strategy_sharpe']:.3f}, max drawdown = {fmt_pct(perf_test['strategy_max_drawdown'])}.",
            "Method detail": "The comparison uses classification metrics (accuracy, precision, recall, F1) plus investment metrics (strategy return, buy-and-hold return, max drawdown, Sharpe ratio, entry count, and holding ratio). The selected model is chosen by validation strategy total return, while the test split is kept for final out-of-sample reporting.",
            "Evidence in app": f"Selected threshold = {selected_threshold:.2f}; test entries = {diag_test['entry_count']}; holding ratio = {fmt_pct(diag_test['holding_ratio'])}.",
        },
        {
            "Lifecycle step": "Present results and document",
            "Question": "Can I explain the result and how?",
            "Project result": "The dashboard shows model comparison, strategy definition, diagnostics, equity curve, latest signal, and downloadable CSV.",
            "Method detail": "The Backtest Dashboard provides numeric tables and charts; this Project Lifecycle view summarizes the work in report-friendly stages.",
            "Evidence in app": "Use the Backtest Dashboard view for tables and charts.",
        },
        {
            "Lifecycle step": "Deploy model",
            "Question": "How can the model be used in the real world?",
            "Project result": f"The current trained best model produces a next-trading-day probability P(up) = {next_prob_up:.4f}.",
            "Method detail": "After training, the latest feature row is passed into the selected model to estimate P(up). The trading rule converts that probability into a target position for the next trading day.",
            "Evidence in app": f"Suggested target position = {next_target_position:.2f}x; trading cost assumption = {cost_bps:.1f} bps.",
        },
        ]

    st.subheader(tr(lang, "stage_detail"))
    stage_labels = [
        f"Stage {idx}: {row['Lifecycle step']}" for idx, row in enumerate(lifecycle_rows, start=1)
    ]
    selected_stage = st.radio(
        "Choose a lifecycle stage",
        stage_labels,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_idx = stage_labels.index(selected_stage)
    selected_row = lifecycle_rows[selected_idx]

    left, right = st.columns([1.1, 1.4])
    with left:
        st.markdown(f"### {selected_stage}")
        st.markdown(f"**{tr(lang, 'guiding_question')}:** {selected_row['Question']}")
        st.info(selected_row["Project result"])
        st.markdown(f"**{tr(lang, 'method_detail')}:** {selected_row['Method detail']}")
    with right:
        st.markdown(f"### {tr(lang, 'project_evidence')}")
        st.success(selected_row["Evidence in app"])
        if selected_idx == 3:
            st.markdown(f"### {tr(lang, 'actual_model_comparison')}")
            comparison_cols = [
                "model",
                "threshold_used",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_strategy_total_return",
                "test_accuracy",
                "test_f1",
                "test_strategy_total_return",
                "test_buy_hold_total_return",
                "test_strategy_sharpe",
                "test_entry_count",
            ]
            comparison_df = table[comparison_cols].copy()
            comparison_df.insert(1, "selected_best_model", comparison_df["model"] == best_name)
            winner_cols = [
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_strategy_total_return",
                "test_accuracy",
                "test_f1",
                "test_strategy_total_return",
                "test_strategy_sharpe",
            ]
            selection_return = comparison_df.loc[
                comparison_df["model"] == best_name, "val_strategy_total_return"
            ].iloc[0]
            st.markdown(
                (
                    f"**Key selection reason:** `{best_name}` is selected because it has the highest "
                    f"`val_strategy_total_return` ({selection_return:.4f}) on the validation split."
                    if lang == "en"
                    else f"**關鍵選模原因：** `{best_name}` 在驗證集的 `val_strategy_total_return` 最高（{selection_return:.4f}），因此被選為最佳模型。"
                )
            )

            def highlight_model_comparison(row: pd.Series) -> list[str]:
                styles = []
                is_selected = row["model"] == best_name
                for col, value in row.items():
                    style_parts = []
                    if is_selected:
                        style_parts.append("background-color: #fff4e6")
                    if col in winner_cols and value == comparison_df[col].max():
                        style_parts.append("color: #c62828")
                        style_parts.append("font-weight: 700")
                    if col == "selected_best_model" and value:
                        style_parts.append("color: #c62828")
                        style_parts.append("font-weight: 700")
                    styles.append("; ".join(style_parts))
                return styles

            st.dataframe(
                comparison_df.style.apply(highlight_model_comparison, axis=1).format(
                    {
                        "threshold_used": "{:.2f}",
                        "val_accuracy": "{:.4f}",
                        "val_precision": "{:.4f}",
                        "val_recall": "{:.4f}",
                        "val_f1": "{:.4f}",
                        "val_strategy_total_return": "{:.4f}",
                        "test_accuracy": "{:.4f}",
                        "test_f1": "{:.4f}",
                        "test_strategy_total_return": "{:.4f}",
                        "test_buy_hold_total_return": "{:.4f}",
                        "test_strategy_sharpe": "{:.4f}",
                        "test_entry_count": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                (
                    "Red bold values mark the winner for each metric where higher is better. "
                    "The orange row is the selected model. The final selection uses validation strategy total return, "
                    "while test metrics are reserved for out-of-sample reporting."
                    if lang == "en"
                    else "紅色粗體代表該指標表現最佳（越高越好）；淡橘色列是最後選出的模型。最終選模使用 validation strategy total return，test 指標保留作為樣本外報告。"
                )
            )
        else:
            st.caption(tr(lang, "choose_stage4"))

    with st.expander("Lifecycle overview table"):
        st.dataframe(
            pd.DataFrame(lifecycle_rows)[["Lifecycle step", "Project result", "Evidence in app"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(tr(lang, "lifecycle_scorecard"))
    scorecard = pd.DataFrame(
        [
            {"Metric": "Best validation-selected model", "Value": best_name},
            {"Metric": "Selected threshold", "Value": f"{selected_threshold:.2f}"},
            {"Metric": "Test strategy return", "Value": fmt_pct(perf_test["strategy_total_return"])},
            {"Metric": "Test buy-and-hold return", "Value": fmt_pct(perf_test["buy_hold_total_return"])},
            {"Metric": "Test strategy Sharpe", "Value": f"{perf_test['strategy_sharpe']:.3f}"},
            {"Metric": "Entry count", "Value": f"{diag_test['entry_count']}"},
            {"Metric": "Win rate on holding days", "Value": fmt_pct(diag_test["win_rate"])},
            {"Metric": "Total trading cost", "Value": fmt_pct(diag_test["total_cost"])},
        ]
    )
    st.dataframe(scorecard, use_container_width=True, hide_index=True)


def render_backtest_dashboard(
    lang: str,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    table: pd.DataFrame,
    detail: dict[str, dict],
    best_name: str,
    selected_threshold: float,
    cost_bps: float,
    auto_threshold: bool,
    next_prob_up: float,
    next_target_position: float,
) -> None:
    st.subheader(tr(lang, "data_summary"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr(lang, "total_rows"), f"{len(df):,}")
    c2.metric(tr(lang, "train_rows"), f"{len(train_df):,}")
    c3.metric(tr(lang, "validation_rows"), f"{len(val_df):,}")
    c4.metric(tr(lang, "test_rows"), f"{len(test_df):,}")

    st.subheader(tr(lang, "model_comparison"))
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
                "threshold_used": "{:.2f}",
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

    st.subheader(tr(lang, "strategy_definition"))
    if lang == "zh":
        strategy_note = (
            f"- 訊號規則：動態槓桿 (`target_position = f(P(up), threshold={selected_threshold:.2f})`)\n"
            "- 執行規則：使用前一日目標部位 (`position = target_position.shift(1)`)\n"
            f"- 交易成本：每次部位變化扣 {cost_bps:.1f} bps\n"
            f"- Threshold 模式：{'在 validation 自動搜尋' if auto_threshold else '固定 threshold'}\n"
            "- 模型選擇規則：選擇 validation strategy total return 最高的模型"
        )
    else:
        strategy_note = (
            f"- Signal rule: Dynamic leverage (`target_position = f(P(up), threshold={selected_threshold:.2f})`)\n"
            "- Execution rule: use previous-day target position (`position = target_position.shift(1)`)\n"
            f"- Trading cost: {cost_bps:.1f} bps per position change\n"
            f"- Threshold mode: {'Auto-search on validation' if auto_threshold else 'Fixed threshold'}\n"
            "- Model selection rule: choose model with highest validation strategy total return"
        )
    st.markdown(strategy_note)

    st.subheader(
        f"Best Model (by Validation Return): {best_name}"
        if lang == "en"
        else f"最佳模型（依驗證集報酬選出）：{best_name}"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Test strategy return" if lang == "en" else "測試集策略報酬", fmt_pct(perf_test["strategy_total_return"]))
    m2.metric("Test max drawdown" if lang == "en" else "測試集最大回撤", fmt_pct(perf_test["strategy_max_drawdown"]))
    m3.metric("Test Sharpe ratio" if lang == "en" else "測試集 Sharpe", f"{perf_test['strategy_sharpe']:.3f}")

    n1, n2, n3 = st.columns(3)
    n1.metric("Test Buy & Hold return", fmt_pct(perf_test["buy_hold_total_return"]))
    n2.metric("Test Buy & Hold max drawdown", fmt_pct(perf_test["buy_hold_max_drawdown"]))
    n3.metric("Test Buy & Hold Sharpe", f"{perf_test['buy_hold_sharpe']:.3f}")

    st.subheader(tr(lang, "strategy_diagnostics"))
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
        st.subheader(tr(lang, "threshold_search"))
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

    st.subheader(tr(lang, "equity_curve"))
    curve_df = bt_test[["strategy_cum", "buy_hold_cum"]].copy()
    curve_df.columns = ["Strategy", "Buy & Hold"]
    st.line_chart(curve_df, use_container_width=True)

    st.subheader(tr(lang, "signal_snapshot"))
    signal_df = bt_test[["Close", "pred_prob_up", "target_position", "position"]].copy()
    st.dataframe(signal_df.tail(100), use_container_width=True)

    st.subheader(tr(lang, "next_prediction"))
    p1, p2, p3 = st.columns(3)
    p1.metric(tr(lang, "feature_date"), str(pd.to_datetime(df.index[-1]).date()))
    p2.metric("P(up)", f"{next_prob_up:.4f}")
    p3.metric(tr(lang, "target_position"), f"{next_target_position:.2f}x")

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
        tr(lang, "download_csv"),
        data=csv_bytes,
        file_name="streamlit_backtest_result.csv",
        mime="text/csv",
    )


def architecture_modules(lang: str, result: dict | None = None) -> dict[str, dict[str, str]]:
    ticker = result["ticker"] if result else "2330.TW"
    best_name = result["best_name"] if result else "selected best model"
    if lang == "zh":
        return {
            "goal": {
                "label": "1. 研究目標",
                "role": "定義問題：預測下一交易日是否上漲，並檢查訊號是否能支援交易策略。",
                "inputs": "股票代號、日期區間、validation/test 比例、threshold 與交易成本。",
                "outputs": "target：隔日收盤價是否高於今日收盤價。",
                "evidence": "Backtest Dashboard 的 Strategy Definition 顯示目前訊號規則與交易成本。",
            },
            "data": {
                "label": "2. 資料取得",
                "role": "從 Yahoo Finance 下載每日 OHLCV 股價資料。",
                "inputs": f"ticker={ticker}、start date、end date。",
                "outputs": "Open、High、Low、Close、Volume。",
                "evidence": "Data Summary 顯示清理後總資料、train、validation、test 筆數。",
            },
            "features": {
                "label": "3. 特徵工程",
                "role": "將價格量資料轉成技術指標，讓模型可以學習短期趨勢與動能。",
                "inputs": "OHLCV 原始資料。",
                "outputs": "return_1d、ma_ratio、bias_5、bias_20、vol_chg、rsi_14、macd、macd_signal、macd_hist。",
                "evidence": "Project Lifecycle Stage 2 說明 feature cleaning 與缺值處理。",
            },
            "split": {
                "label": "4. 時間序切分",
                "role": "依時間順序切成 train / validation / test，避免未來資料洩漏。",
                "inputs": "完整特徵資料表。",
                "outputs": "訓練集、驗證集、測試集。",
                "evidence": "模型只用 train 訓練；validation 用於選模型與 threshold；test 用於最終樣本外報告。",
            },
            "models": {
                "label": "5. 候選模型",
                "role": "訓練多個模型並比較分類與交易績效。",
                "inputs": "技術指標特徵與 target。",
                "outputs": "各模型的 P(up)、分類指標與回測結果。",
                "evidence": f"Model Comparison 會列出所有候選模型；目前最佳模型為 {best_name}。",
            },
            "selection": {
                "label": "6. 選模與門檻",
                "role": "用 validation strategy total return 選出最佳模型與 threshold。",
                "inputs": "validation 預測機率與 threshold 搜尋範圍。",
                "outputs": "best model、selected threshold。",
                "evidence": "Threshold Search Result 與 Project Lifecycle Stage 4 顯示實際選模依據。",
            },
            "backtest": {
                "label": "7. 回測策略",
                "role": "將 P(up) 轉成 target_position，並計算交易成本與策略報酬。",
                "inputs": "P(up)、threshold、交易成本。",
                "outputs": "position、strategy_ret、strategy_cum、buy_hold_cum。",
                "evidence": "Backtest Dashboard 顯示績效、診斷指標與權益曲線。",
            },
            "present": {
                "label": "8. 視覺化與部署",
                "role": "將結果整理成 dashboard、CSV 與下一交易日預測。",
                "inputs": "最佳模型、測試集回測、最新特徵。",
                "outputs": "模型比較、生命週期說明、CSV、P(up)、target position。",
                "evidence": "Next Trading Day Prediction 與 Download CSV 是最後輸出。",
            },
        }
    return {
        "goal": {
            "label": "1. Goal",
            "role": "Define the problem: predict whether the next trading day will rise and test whether the signal supports a trading strategy.",
            "inputs": "Ticker, date range, validation/test ratios, threshold, and trading cost.",
            "outputs": "Target: whether tomorrow's close is above today's close.",
            "evidence": "Strategy Definition shows the signal rule and trading cost assumption.",
        },
        "data": {
            "label": "2. Data Collection",
            "role": "Download daily OHLCV price data from Yahoo Finance.",
            "inputs": f"ticker={ticker}, start date, and end date.",
            "outputs": "Open, High, Low, Close, and Volume.",
            "evidence": "Data Summary shows cleaned total, train, validation, and test rows.",
        },
        "features": {
            "label": "3. Feature Engineering",
            "role": "Convert OHLCV data into technical indicators for short-term trend and momentum learning.",
            "inputs": "Raw OHLCV data.",
            "outputs": "return_1d, ma_ratio, bias_5, bias_20, vol_chg, rsi_14, macd, macd_signal, macd_hist.",
            "evidence": "Project Lifecycle Stage 2 explains feature cleaning and missing-value handling.",
        },
        "split": {
            "label": "4. Time Split",
            "role": "Split chronologically into train / validation / test to avoid leakage.",
            "inputs": "Full feature table.",
            "outputs": "Train, validation, and test splits.",
            "evidence": "Train fits models; validation selects model and threshold; test reports final out-of-sample results.",
        },
        "models": {
            "label": "5. Candidate Models",
            "role": "Train multiple models and compare classification and trading performance.",
            "inputs": "Technical indicator features and target.",
            "outputs": "P(up), classification metrics, and backtest results for each model.",
            "evidence": f"Model Comparison lists all candidate models; current best model is {best_name}.",
        },
        "selection": {
            "label": "6. Model & Threshold Selection",
            "role": "Select the best model and threshold using validation strategy total return.",
            "inputs": "Validation probabilities and threshold search range.",
            "outputs": "Best model and selected threshold.",
            "evidence": "Threshold Search Result and Project Lifecycle Stage 4 show the selection evidence.",
        },
        "backtest": {
            "label": "7. Backtest Strategy",
            "role": "Convert P(up) into target_position and calculate trading cost and strategy return.",
            "inputs": "P(up), threshold, and trading cost.",
            "outputs": "position, strategy_ret, strategy_cum, and buy_hold_cum.",
            "evidence": "Backtest Dashboard shows performance metrics, diagnostics, and equity curve.",
        },
        "present": {
            "label": "8. Visualization & Deployment",
            "role": "Package results into dashboards, CSV, and next-day prediction.",
            "inputs": "Best model, test backtest, and latest feature row.",
            "outputs": "Model comparison, lifecycle explanation, CSV, P(up), and target position.",
            "evidence": "Next Trading Day Prediction and Download CSV are final outputs.",
        },
    }


def architecture_graph(granularity: str, lang: str, result: dict | None = None) -> str:
    modules = architecture_modules(lang, result)
    if granularity == "detail":
        return f"""
digraph G {{
  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.55"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, color="#64748b", fillcolor="#eff6ff"];
  edge [color="#64748b", arrowsize=0.8, fontname="Arial", fontsize=10];
  goal [label="{modules["goal"]["label"]}\\nNext-day direction", fillcolor="#e0f2fe"];
  yfin [label="Yahoo Finance\\nOHLCV", fillcolor="#f8fafc"];
  features [label="{modules["features"]["label"]}\\nRSI / MACD / MA", fillcolor="#ecfeff"];
  split [label="{modules["split"]["label"]}\\nTrain / Validation / Test", fillcolor="#fefce8"];
  logreg [label="Logistic / PCA", fillcolor="#ede9fe"];
  forest [label="RandomForest", fillcolor="#ede9fe"];
  xgb [label="XGBoost optional", fillcolor="#ede9fe"];
  selection [label="{modules["selection"]["label"]}\\nValidation return", fillcolor="#fdf2f8"];
  backtest [label="{modules["backtest"]["label"]}\\nDynamic leverage", fillcolor="#fee2e2"];
  present [label="{modules["present"]["label"]}\\nCharts + CSV + prediction", fillcolor="#dcfce7"];
  goal -> yfin -> features -> split;
  split -> logreg -> selection;
  split -> forest -> selection;
  split -> xgb -> selection;
  selection -> backtest -> present;
}}
"""
    return f"""
digraph G {{
  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.5", ranksep="0.7"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, color="#64748b", fillcolor="#eff6ff"];
  edge [color="#64748b", arrowsize=0.8];
  goal [label="{modules["goal"]["label"]}", fillcolor="#e0f2fe"];
  data [label="{modules["data"]["label"]}", fillcolor="#f8fafc"];
  features [label="{modules["features"]["label"]}", fillcolor="#ecfeff"];
  split [label="{modules["split"]["label"]}", fillcolor="#fefce8"];
  models [label="{modules["models"]["label"]}", fillcolor="#ede9fe"];
  selection [label="{modules["selection"]["label"]}", fillcolor="#fdf2f8"];
  backtest [label="{modules["backtest"]["label"]}", fillcolor="#fee2e2"];
  present [label="{modules["present"]["label"]}", fillcolor="#dcfce7"];
  goal -> data -> features -> split -> models -> selection -> backtest -> present;
}}
"""


def render_architecture_explorer(lang: str, result: dict | None = None) -> None:
    st.subheader(tr(lang, "architecture_title"))
    st.caption(tr(lang, "architecture_caption"))
    level_options = {
        tr(lang, "overview_level"): "overview",
        tr(lang, "detail_level"): "detail",
    }
    selected_level = st.radio(tr(lang, "granularity"), list(level_options.keys()), horizontal=True)
    granularity = level_options[selected_level]
    st.graphviz_chart(architecture_graph(granularity, lang, result), use_container_width=True)

    modules = architecture_modules(lang, result)
    labels = {info["label"]: key for key, info in modules.items()}
    selected_label = st.selectbox(tr(lang, "inspect_module"), list(labels.keys()))
    selected = modules[labels[selected_label]]

    left, right = st.columns(2)
    with left:
        st.markdown(f"### {selected['label']}")
        st.markdown(f"**{tr(lang, 'module_role')}**")
        st.write(selected["role"])
        st.markdown(f"**{tr(lang, 'module_inputs')}**")
        st.write(selected["inputs"])
    with right:
        st.markdown("### " + ("Outputs" if lang == "en" else "輸出與證據"))
        st.markdown(f"**{tr(lang, 'module_outputs')}**")
        st.write(selected["outputs"])
        st.markdown(f"**{tr(lang, 'module_evidence')}**")
        st.write(selected["evidence"])

    if result is not None:
        st.markdown("### " + ("Live run snapshot" if lang == "en" else "目前執行結果快照"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr(lang, "train_rows"), f"{len(result['train_df']):,}")
        c2.metric(tr(lang, "validation_rows"), f"{len(result['val_df']):,}")
        c3.metric(tr(lang, "test_rows"), f"{len(result['test_df']):,}")
        c4.metric(tr(lang, "best_model"), result["best_name"])


def render_training_data_tab(lang: str, result: dict) -> None:
    st.subheader("Training Data" if lang == "en" else "訓練資料")
    if lang == "zh":
        st.markdown(
            """
            這個頁籤說明技術面模型實際使用哪些資料來訓練，以及資料從原始價格轉成模型特徵的流程。

            - **資料來源**：Yahoo Finance OHLCV 價格與成交量資料。
            - **處理前資料**：每日 `Open / High / Low / Close / Volume`。
            - **處理後資料**：加入報酬率、均線相對位置、RSI、MACD、成交量變化率等技術指標。
            - **訓練目標**：`target = 1` 代表下一個交易日收盤價高於當日收盤價，否則為 `0`。
            - **切分方式**：依時間順序切成 train / validation / test，不隨機打散。
            """
        )
    else:
        st.markdown(
            """
            This tab explains what data the technical models train on and how raw prices become model features.

            - **Data source**: Yahoo Finance OHLCV price and volume data.
            - **Before processing**: daily `Open / High / Low / Close / Volume`.
            - **After processing**: returns, relative moving-average position, RSI, MACD, and volume-change features.
            - **Training target**: `target = 1` when the next trading day's close is above today's close; otherwise `0`.
            - **Split rule**: chronological train / validation / test split without random shuffling.
            """
        )

    train_df = result["train_df"]
    val_df = result["val_df"]
    test_df = result["test_df"]
    df = result["df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr(lang, "train_rows"), f"{len(train_df):,}")
    c2.metric(tr(lang, "validation_rows"), f"{len(val_df):,}")
    c3.metric(tr(lang, "test_rows"), f"{len(test_df):,}")
    c4.metric(tr(lang, "total_rows"), f"{len(df):,}")

    st.markdown("### " + ("Feature Columns" if lang == "en" else "特徵欄位"))
    st.dataframe(pd.DataFrame({"feature": FEATURE_COLS}), use_container_width=True, hide_index=True)

    st.markdown("### " + ("Model Training Summary" if lang == "en" else "模型訓練摘要"))
    metric_cols = ["model", "val_accuracy", "val_precision", "val_recall", "val_f1", "test_accuracy", "test_f1"]
    st.dataframe(result["table"][[col for col in metric_cols if col in result["table"].columns]], use_container_width=True, hide_index=True)

    st.markdown("### " + ("Processed Data Preview" if lang == "en" else "處理後資料預覽"))
    preview_cols = ["Open", "High", "Low", "Close", "Volume", *FEATURE_COLS, "target"]
    preview_cols = [col for col in preview_cols if col in df.columns]
    st.dataframe(df[preview_cols].tail(100), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="TSMC Stock Backtest Dashboard", layout="wide")

    with st.sidebar:
        language_label = st.selectbox("Language / 語言", ["English", "繁體中文"], index=0)
        lang = "zh" if language_label == "繁體中文" else "en"
        st.header(tr(lang, "parameters"))
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
        cost_bps = st.number_input(
            "Round-trip cost (bps per position change)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )
        run = st.button(tr(lang, "run"), type="primary")

    render_sticky_title_glossary(
        tr(lang, "app_title"),
        tr(lang, "app_caption"),
        lang,
        key_prefix="technical_glossary",
    )

    if run:
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

            best = detail[best_name]
            selected_threshold = best["threshold"]
            latest_features = df[FEATURE_COLS].tail(1)
            best_model = best["model"]
            next_prob_up = (
                float(best_model.predict_proba(latest_features)[:, 1][0])
                if hasattr(best_model, "predict_proba")
                else float(best_model.predict(latest_features)[0])
            )
            next_target_position = execute_dynamic_leverage(next_prob_up, selected_threshold)

        st.session_state["dashboard_result"] = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "df": df,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "table": table,
            "detail": detail,
            "best_name": best_name,
            "selected_threshold": selected_threshold,
            "cost_bps": cost_bps,
            "auto_threshold": auto_threshold,
            "lang": lang,
            "next_prob_up": next_prob_up,
            "next_target_position": next_target_position,
        }

    result = st.session_state.get("dashboard_result")
    if result is None:
        st.info(tr(lang, "initial_info"))
        render_architecture_explorer(lang)
        return

    tab_backtest, tab_training_data, tab_lifecycle, tab_architecture = st.tabs(
        [tr(lang, "tab_backtest"), tr(lang, "tab_training_data"), tr(lang, "tab_lifecycle"), tr(lang, "tab_architecture")]
    )
    with tab_backtest:
        render_backtest_dashboard(
            lang=lang,
            df=result["df"],
            train_df=result["train_df"],
            val_df=result["val_df"],
            test_df=result["test_df"],
            table=result["table"],
            detail=result["detail"],
            best_name=result["best_name"],
            selected_threshold=result["selected_threshold"],
            cost_bps=result["cost_bps"],
            auto_threshold=result["auto_threshold"],
            next_prob_up=result["next_prob_up"],
            next_target_position=result["next_target_position"],
        )

    with tab_training_data:
        render_training_data_tab(lang, result)

    with tab_lifecycle:
        render_project_lifecycle_dashboard(
            lang=lang,
            ticker=result["ticker"],
            start=result["start"],
            end=result["end"],
            df=result["df"],
            train_df=result["train_df"],
            val_df=result["val_df"],
            test_df=result["test_df"],
            table=result["table"],
            best_name=result["best_name"],
            best=result["detail"][result["best_name"]],
            selected_threshold=result["selected_threshold"],
            cost_bps=result["cost_bps"],
            next_prob_up=result["next_prob_up"],
            next_target_position=result["next_target_position"],
        )

    with tab_architecture:
        render_architecture_explorer(lang, result)


if __name__ == "__main__":
    main()
