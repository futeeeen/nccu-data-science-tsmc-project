from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from project_glossary import render_sticky_title_glossary
from triple_barrier_system import (
    CHIP_COLS,
    FUNDAMENTAL_COLS,
    TECHNICAL_COLS,
    run_multi_factor_pipeline as run_triple_barrier_pipeline,
)


TEXT = {
    "en": {
        "app_title": "TSMC Triple Barrier Strategy Analysis",
        "app_caption": "Technical, fundamental, and chip factors are trained against Triple Barrier labels, calibrated to 0-100, then converted into trading signals.",
        "data": "Data",
        "ticker": "Ticker",
        "start_date": "Start date",
        "end_date": "End date",
        "fetch_finmind": "Fetch fundamental/chip data from FinMind",
        "finmind_token": "FinMind token (optional)",
        "fundamental_csv": "Fundamental CSV (date, eps)",
        "chip_csv": "Chip CSV (date, foreign_net_buy, investment_trust_net_buy, dealer_net_buy)",
        "weights": "Weights",
        "technical_weight": "Technical weight",
        "fundamental_weight": "Fundamental weight",
        "chip_weight": "Chip weight",
        "auto_threshold": "Auto-search final threshold on validation",
        "strategy_settings": "Trading Strategy",
        "strategy_mode": "Trading rule",
        "single_threshold": "Single threshold",
        "hysteresis": "Dual threshold buffer",
        "fixed_threshold": "Fixed final score threshold",
        "fixed_sell_threshold": "Fixed sell score threshold",
        "threshold_min": "Threshold search min",
        "threshold_max": "Threshold search max",
        "threshold_step": "Threshold search step",
        "cost_bps": "Trading cost (bps per position change)",
        "barrier_settings": "Triple Barrier Labeling",
        "take_profit_pct": "Take-profit barrier (%)",
        "stop_loss_pct": "Stop-loss barrier (%)",
        "max_holding_days": "Vertical barrier (trading days)",
        "run": "Run triple-barrier analysis",
        "score_design": "Triple Barrier Score Design",
        "data_source": "Data Source",
        "expected_csv": "Expected CSV Format",
        "running": "Running triple-barrier pipeline...",
        "initial_info": "Upload factor data if available, set weights and barrier parameters, then click Run triple-barrier analysis.",
        "cached": "Showing cached result from the last run",
        "data_notes": "Data Notes",
        "tab_overview": "Overview",
        "tab_compare": "Strategy Comparison",
        "tab_explain": "Factor Explanation",
        "tab_training_data": "Training Data",
        "tab_architecture": "Architecture",
        "tab_signals": "Signals & Download",
        "tab_references": "References",
        "factor_status": "Factor Model Status",
        "performance": "Performance",
        "strategy_return": "Strategy return",
        "buy_hold_return": "Buy & Hold return",
        "strategy_sharpe": "Strategy Sharpe",
        "max_drawdown": "Max drawdown",
        "entry_count": "Entry count",
        "holding_ratio": "Holding ratio",
        "total_cost": "Total trading cost",
        "selected_threshold": "Selected final score threshold",
        "selected_sell_threshold": "Selected sell score threshold",
        "current_strategy": "Current Trading Strategy",
        "validation_threshold": "Validation Threshold Search",
        "score_contributions": "Score Contributions",
        "equity_curve": "Equity Curve",
        "factor_explanation": "Factor Contribution Explanation",
        "factor_explanation_caption": "Select a test date to inspect why the Triple Barrier trading score and factor contributions are high or low.",
        "strategy_to_explain": "Strategy to explain",
        "manual_score": "Manual weighted score",
        "meta_score": "Meta model score",
        "select_date": "Select test date",
        "final_score": "Final score",
        "buy_threshold": "Buy threshold",
        "sell_threshold": "Sell threshold",
        "decision": "Decision",
        "next_position": "Next target position",
        "technical_score": "Technical score",
        "fundamental_score": "Fundamental score",
        "chip_score": "Chip score",
        "contribution": "contribution",
        "breakdown": "Factor contribution breakdown",
        "threshold_check": "Threshold check",
        "buy_msg": "The selected date reaches the buy threshold, so the strategy sets target_position = 1.",
        "cash_msg": "The selected date is below the buy threshold, so the strategy stays out of the market.",
        "single_rule": "Single threshold rule: target_position = 1 when final_score >= buy_threshold; otherwise target_position = 0.",
        "hysteresis_rule": "Dual threshold buffer rule: enter only when cash and final_score >= buy_threshold; exit only when holding and final_score <= sell_threshold.",
        "why": "Why this date looks this way",
        "technical_drivers": "Technical drivers",
        "fundamental_drivers": "Fundamental drivers",
        "chip_drivers": "Chip drivers",
        "no_technical": "No technical driver data is available for this date.",
        "no_fundamental": "No fundamental driver data is available for this date.",
        "no_chip": "No chip driver data is available for this date.",
        "contribution_over_time": "Contribution Over Time",
        "strategy_compare_title": "Manual Weighted Score vs Meta Model Score",
        "strategy_compare_caption": "Manual weighted score uses your chosen weights. Meta model learns how the three factor scores relate to Triple Barrier outcomes.",
        "meta_summary": "Meta Model Training Summary",
        "meta_threshold": "Meta Threshold Search",
        "meta_skipped": "Meta threshold search was skipped because validation data was insufficient.",
        "equity_compare": "Equity Curve Comparison",
        "latest_signals": "Latest Test Signals",
        "signal_strategy": "Signal table strategy",
        "download": "Download triple_barrier_backtest_result.csv",
        "architecture_title": "Triple Barrier Architecture",
        "architecture_caption": "Switch granularity, inspect each module, and connect the graph to what the dashboard produces.",
        "granularity": "Granularity",
        "overview_level": "Overview",
        "detail_level": "Detailed",
        "inspect_module": "Inspect module",
        "module_role": "Role",
        "module_inputs": "Inputs",
        "module_outputs": "Outputs",
        "module_evidence": "Evidence in this app",
        "architecture_note": "This page explains the Triple Barrier strategy architecture.",
        "references_title": "References",
        "references_caption": "Triple Barrier Labeling is a financial machine learning labeling method commonly attributed to Marcos López de Prado.",
        "references_project_note": "This app adapts the idea for a long-only TSMC strategy: the label asks whether price reaches the take-profit barrier before the stop-loss barrier or the vertical time barrier.",
        "buy_decision": "BUY / HOLD",
        "cash_decision": "NO BUY / CASH",
        "high": "high",
        "moderately_high": "moderately high",
        "neutral": "neutral",
        "moderately_low": "moderately low",
        "low": "low",
    },
    "zh": {
        "app_title": "台積電 Triple Barrier 策略分析工具",
        "app_caption": "技術面、基本面、籌碼面會以 Triple Barrier 標籤訓練成分數，校正為 0-100 後再轉成交易訊號。",
        "data": "資料設定",
        "ticker": "股票代號",
        "start_date": "開始日期",
        "end_date": "結束日期",
        "fetch_finmind": "從 FinMind 抓取基本面 / 籌碼面資料",
        "finmind_token": "FinMind token（選填）",
        "fundamental_csv": "基本面 CSV（date, eps）",
        "chip_csv": "籌碼面 CSV（date, foreign_net_buy, investment_trust_net_buy, dealer_net_buy）",
        "weights": "權重設定",
        "technical_weight": "技術面權重",
        "fundamental_weight": "基本面權重",
        "chip_weight": "籌碼面權重",
        "auto_threshold": "用驗證資料自動搜尋買入門檻",
        "strategy_settings": "交易策略",
        "strategy_mode": "買賣規則",
        "single_threshold": "單一門檻",
        "hysteresis": "雙門檻緩衝策略",
        "fixed_threshold": "固定最終分數門檻",
        "fixed_sell_threshold": "固定賣出分數門檻",
        "threshold_min": "門檻搜尋最小值",
        "threshold_max": "門檻搜尋最大值",
        "threshold_step": "門檻搜尋間距",
        "cost_bps": "交易成本（每次部位變動 bps）",
        "barrier_settings": "Triple Barrier 標籤設定",
        "take_profit_pct": "停利障礙（%）",
        "stop_loss_pct": "停損障礙（%）",
        "max_holding_days": "垂直時間障礙（交易日）",
        "run": "執行 Triple Barrier 分析",
        "score_design": "Triple Barrier 分數設計",
        "data_source": "資料來源",
        "expected_csv": "CSV 格式範例",
        "running": "正在執行 Triple Barrier 分析...",
        "initial_info": "若有因子資料可先上傳，設定權重與障礙參數後按下執行 Triple Barrier 分析。",
        "cached": "目前顯示上一次執行結果",
        "data_notes": "資料備註",
        "tab_overview": "總覽",
        "tab_compare": "策略比較",
        "tab_explain": "因子解釋",
        "tab_training_data": "訓練資料",
        "tab_architecture": "架構",
        "tab_signals": "訊號與下載",
        "tab_references": "References",
        "factor_status": "因子模型狀態",
        "performance": "績效表現",
        "strategy_return": "策略報酬",
        "buy_hold_return": "買進持有報酬",
        "strategy_sharpe": "策略 Sharpe",
        "max_drawdown": "最大回撤",
        "entry_count": "進場次數",
        "holding_ratio": "持有比例",
        "total_cost": "總交易成本",
        "selected_threshold": "選出的最終分數門檻",
        "selected_sell_threshold": "選出的賣出分數門檻",
        "current_strategy": "目前買賣策略",
        "validation_threshold": "驗證集門檻搜尋",
        "score_contributions": "分數貢獻",
        "equity_curve": "權益曲線",
        "factor_explanation": "因子貢獻解釋",
        "factor_explanation_caption": "選擇測試日期，查看 Triple Barrier 交易分數與各因子貢獻為何偏高或偏低。",
        "strategy_to_explain": "要解釋的策略",
        "manual_score": "人工加權分數",
        "meta_score": "Meta model 分數",
        "select_date": "選擇測試日期",
        "final_score": "最終分數",
        "buy_threshold": "買入門檻",
        "sell_threshold": "賣出門檻",
        "decision": "決策結果",
        "next_position": "下一期目標部位",
        "technical_score": "技術面分數",
        "fundamental_score": "基本面分數",
        "chip_score": "籌碼面分數",
        "contribution": "貢獻分數",
        "breakdown": "因子貢獻拆解",
        "threshold_check": "買入門檻檢查",
        "buy_msg": "此日期的最終分數達到買入門檻，因此策略將 target_position 設為 1。",
        "cash_msg": "此日期的最終分數低於買入門檻，因此策略選擇不進場。",
        "single_rule": "單一門檻規則：final_score >= buy_threshold 時 target_position = 1，否則 target_position = 0。",
        "hysteresis_rule": "雙門檻緩衝規則：空手時 final_score >= buy_threshold 才進場；持有時 final_score <= sell_threshold 才退場。",
        "why": "為什麼這一天會得到這個結果",
        "technical_drivers": "技術面指標解釋",
        "fundamental_drivers": "基本面指標解釋",
        "chip_drivers": "籌碼面指標解釋",
        "no_technical": "此日期沒有可用的技術面指標解釋資料。",
        "no_fundamental": "此日期沒有可用的基本面指標解釋資料。",
        "no_chip": "此日期沒有可用的籌碼面指標解釋資料。",
        "contribution_over_time": "因子貢獻時間序列",
        "strategy_compare_title": "人工加權分數 vs Meta Model 分數",
        "strategy_compare_caption": "人工加權分數使用你設定的權重；Meta model 則學習三個因子分數與 Triple Barrier 結果之間的關係。",
        "meta_summary": "Meta Model 訓練摘要",
        "meta_threshold": "Meta Model 門檻搜尋",
        "meta_skipped": "因驗證資料不足，略過 Meta Model 門檻搜尋。",
        "equity_compare": "權益曲線比較",
        "latest_signals": "近期測試訊號",
        "signal_strategy": "訊號表使用的策略",
        "download": "下載 triple_barrier_backtest_result.csv",
        "architecture_title": "Triple Barrier 架構",
        "architecture_caption": "切換架構層級、點選模組，並把資料流連回 Triple Barrier dashboard 產出的結果。",
        "granularity": "架構層級",
        "overview_level": "總覽",
        "detail_level": "詳細",
        "inspect_module": "檢視模組",
        "module_role": "角色",
        "module_inputs": "輸入",
        "module_outputs": "輸出",
        "module_evidence": "此 app 中的證據",
        "architecture_note": "此頁說明 Triple Barrier 策略架構。",
        "references_title": "References",
        "references_caption": "Triple Barrier Labeling 是金融機器學習常見的標籤方法，通常引用 Marcos López de Prado 的方法。",
        "references_project_note": "本 app 將此概念調整成台積電多因子 long-only 策略：標籤判斷價格是否先碰到停利障礙，而不是先停損或時間到。",
        "buy_decision": "買進 / 持有",
        "cash_decision": "不買 / 空手",
        "high": "偏高",
        "moderately_high": "中度偏高",
        "neutral": "中性",
        "moderately_low": "中度偏低",
        "low": "偏低",
    },
}


def tr(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def read_optional_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def resolve_finmind_token(sidebar_token: str | None) -> tuple[str | None, str]:
    token = (sidebar_token or "").strip()
    if token:
        return token, "sidebar input"

    try:
        token = str(st.secrets.get("FINMIND_TOKEN", "")).strip()
    except Exception:
        token = ""
    if token:
        return token, "Streamlit secrets"

    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        return token, "environment variable"

    return None, "not configured"


def normalize_weights(technical: float, fundamental: float, chip: float) -> tuple[float, float, float]:
    total = technical + fundamental + chip
    if total == 0:
        return 0.2, 0.4, 0.4
    return technical / total, fundamental / total, chip / total


def render_factor_status(result, lang: str) -> None:
    st.subheader("Factor Training Overview" if lang == "en" else "因子訓練總覽")
    status_map = result.factor_data_summary.set_index("factor").to_dict("index")
    cols = st.columns(3)
    for idx, factor in enumerate(["technical", "fundamental", "chip"]):
        info = status_map.get(factor, {})
        status = str(info.get("status", "unknown"))
        label = status.replace("_", " ").title()
        if lang == "zh":
            label = {
                "model": "已訓練模型",
                "neutral": "中性分數",
                "neutral_insufficient_data": "中性分數：資料不足",
            }.get(status, label)
        cols[idx].metric(
            f"{factor.title()} factor" if lang == "en" else {
                "technical": "技術面因子",
                "fundamental": "基本面因子",
                "chip": "籌碼面因子",
            }.get(factor, factor),
            label,
            f"{int(info.get('feature_count', 0))} features" if lang == "en" else f"{int(info.get('feature_count', 0))} 個特徵",
        )

    st.dataframe(
        result.factor_data_summary.style.format(
            {
                "overall_coverage": "{:.2%}",
                "feature_count": "{:.0f}",
                "train_usable_rows": "{:.0f}",
                "validation_usable_rows": "{:.0f}",
                "test_usable_rows": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def describe_score_level(score: float, lang: str = "en") -> str:
    if score >= 75:
        return tr(lang, "high")
    if score >= 55:
        return tr(lang, "moderately_high")
    if score >= 45:
        return tr(lang, "neutral")
    if score >= 25:
        return tr(lang, "moderately_low")
    return tr(lang, "low")


FEATURE_DIRECTIONS = {
    "return_1d": "positive_with_overheat",
    "ma_ratio": "trend_above_one",
    "bias_5": "moderate_positive_bias",
    "bias_20": "moderate_positive_bias",
    "vol_chg": "moderate_positive_volume",
    "rsi_14": "rsi_range",
    "macd": "positive",
    "macd_signal": "positive",
    "macd_hist": "positive",
    "eps": "positive",
    "eps_growth_yoy": "positive",
    "eps_growth_qoq": "positive",
    "foreign_net_buy_5d": "positive",
    "investment_trust_net_buy_5d": "positive",
    "dealer_net_buy_5d": "positive",
    "total_institutional_net_buy_5d": "positive",
    "total_institutional_net_buy_20d": "positive",
    "foreign_net_buy_5d_ratio": "positive",
    "investment_trust_net_buy_5d_ratio": "positive",
    "dealer_net_buy_5d_ratio": "positive",
    "total_institutional_net_buy_5d_ratio": "positive",
    "foreign_consecutive_buy_days": "positive",
    "investment_trust_consecutive_buy_days": "positive",
    "margin_balance_change_5d": "lower_or_mild",
    "short_balance_change_5d": "negative",
}

DIRECTION_TEXT = {
    "en": {
        "positive": "higher is better",
        "negative": "lower is better",
        "positive_with_overheat": "positive is good; extreme jump is overheated",
        "trend_above_one": "above 1 is uptrend; too extended is caution",
        "moderate_positive_bias": "mild positive bias is good; extreme bias is overheated",
        "moderate_positive_volume": "mild volume expansion is good; extreme spike is caution",
        "rsi_range": "healthy range is best",
        "lower_or_mild": "lower or mild leverage is safer",
    },
    "zh": {
        "positive": "越高越好",
        "negative": "越低越好",
        "positive_with_overheat": "正報酬偏好；暴衝過熱需留意",
        "trend_above_one": "高於 1 偏多；過度延伸需留意",
        "moderate_positive_bias": "溫和正乖離較好；過大代表過熱",
        "moderate_positive_volume": "溫和放量較好；爆量需留意",
        "rsi_range": "落在健康區間最好",
        "lower_or_mild": "融資低或溫和較安全",
    },
}


def direction_text(feature: str, lang: str) -> str:
    direction = FEATURE_DIRECTIONS.get(feature, "positive")
    return DIRECTION_TEXT.get(lang, DIRECTION_TEXT["en"]).get(direction, direction)


def bounded_quality(value: float, good_low: float, good_high: float, okay_low: float, okay_high: float) -> float:
    if good_low <= value <= good_high:
        return 85.0
    if okay_low <= value <= okay_high:
        return 60.0
    return 20.0


def feature_quality_score(feature: str, value: float, percentile: float) -> float:
    direction = FEATURE_DIRECTIONS.get(feature, "positive")
    if pd.isna(value) or pd.isna(percentile):
        return 50.0

    if direction == "positive":
        return float(percentile)
    if direction == "negative":
        return float(100.0 - percentile)
    if direction == "positive_with_overheat":
        if value < 0:
            return max(5.0, 50.0 + value * 1000.0)
        if value <= 0.05:
            return min(90.0, 55.0 + value * 700.0)
        return 45.0 if value <= 0.08 else 25.0
    if direction == "trend_above_one":
        if 1.0 <= value <= 1.08:
            return 85.0
        if 0.98 <= value < 1.0 or 1.08 < value <= 1.15:
            return 55.0
        return 25.0
    if direction == "moderate_positive_bias":
        if 0.0 <= value <= 0.06:
            return 85.0
        if -0.03 <= value < 0.0 or 0.06 < value <= 0.12:
            return 55.0
        return 25.0
    if direction == "moderate_positive_volume":
        if 0.0 <= value <= 0.5:
            return 80.0
        if -0.3 <= value < 0.0 or 0.5 < value <= 1.5:
            return 55.0
        return 30.0
    if direction == "rsi_range":
        return bounded_quality(value, good_low=45.0, good_high=70.0, okay_low=35.0, okay_high=80.0)
    if direction == "lower_or_mild":
        if value <= 0:
            return 80.0
        if percentile <= 55:
            return 60.0
        return max(10.0, 100.0 - percentile)
    return float(percentile)


def describe_signal_quality(score: float, lang: str = "en") -> str:
    if score >= 75:
        return "Good" if lang == "en" else "良好"
    if score >= 55:
        return "Positive" if lang == "en" else "偏好"
    if score >= 45:
        return "Neutral" if lang == "en" else "普通"
    if score >= 25:
        return "Weak" if lang == "en" else "偏弱"
    return "Poor" if lang == "en" else "不佳"


FEATURE_DEFINITIONS = {
    "return_1d": {
        "zh": "一日報酬率：今日收盤價相對前一交易日的漲跌幅，用來捕捉最近價格動能。",
        "en": "One-day return: today's close versus the previous trading day's close. It captures recent price momentum.",
    },
    "ma_ratio": {
        "zh": "短長均線比：5 日均線除以 20 日均線。大於 1 代表短期價格相對強於中期趨勢。",
        "en": "Moving-average ratio: 5-day MA divided by 20-day MA. Above 1 means short-term price is stronger than the medium-term trend.",
    },
    "bias_5": {
        "zh": "5 日乖離率：收盤價相對 5 日均線的偏離程度。正值代表價格高於短期均線。",
        "en": "5-day bias: close price deviation from the 5-day moving average. Positive values mean price is above the short-term average.",
    },
    "bias_20": {
        "zh": "20 日乖離率：收盤價相對 20 日均線的偏離程度，用來觀察中期過熱或轉弱。",
        "en": "20-day bias: close price deviation from the 20-day moving average. It helps identify medium-term overheating or weakness.",
    },
    "vol_chg": {
        "zh": "成交量變化率：今日成交量相對前一交易日的變化。量增常代表市場關注度提升。",
        "en": "Volume change: today's volume versus the previous trading day's volume. Rising volume often means stronger market attention.",
    },
    "rsi_14": {
        "zh": "14 日 RSI：衡量近期上漲與下跌力道的相對強弱指標。過高可能過熱，過低可能超賣。",
        "en": "14-day RSI: relative strength of recent gains versus losses. Very high values may be overheated; very low values may be oversold.",
    },
    "macd": {
        "zh": "MACD：短期與長期 EMA 的差距，本系統會除以收盤價做標準化。正值通常代表偏多動能。",
        "en": "MACD: difference between short-term and long-term EMA, normalized by close price here. Positive values usually indicate bullish momentum.",
    },
    "macd_signal": {
        "zh": "MACD 訊號線：MACD 的平滑均線，用來判斷 MACD 動能是否延續或轉弱。",
        "en": "MACD signal line: smoothed average of MACD, used to judge whether MACD momentum is continuing or weakening.",
    },
    "macd_hist": {
        "zh": "MACD 柱狀體：MACD 減訊號線。正值且擴大通常代表上漲動能增強。",
        "en": "MACD histogram: MACD minus signal line. Positive and rising values usually mean strengthening upward momentum.",
    },
    "eps": {
        "zh": "EPS：每股盈餘，代表公司每一股能分配到的獲利，是基本面獲利能力指標。",
        "en": "EPS: earnings per share. It measures profit attributable to each share and represents fundamental profitability.",
    },
    "eps_growth_yoy": {
        "zh": "EPS 年增率：本期 EPS 相對去年同期的成長率，用來觀察獲利是否長期改善。",
        "en": "EPS YoY growth: current EPS compared with the same period last year. It shows whether profitability is improving over the longer term.",
    },
    "eps_growth_qoq": {
        "zh": "EPS 季增率：本期 EPS 相對上一季的成長率，用來觀察近期獲利動能。",
        "en": "EPS QoQ growth: current EPS compared with the previous quarter. It captures near-term earnings momentum.",
    },
    "foreign_net_buy_5d": {
        "zh": "外資 5 日買賣超：外資近 5 個交易日買進減賣出的合計。正值代表外資偏買。",
        "en": "Foreign investors 5-day net buy: buy minus sell over the last 5 trading days. Positive means foreign investors are net buyers.",
    },
    "investment_trust_net_buy_5d": {
        "zh": "投信 5 日買賣超：投信近 5 個交易日買進減賣出的合計。常用來觀察本土法人態度。",
        "en": "Investment trust 5-day net buy: buy minus sell over the last 5 trading days. It reflects local institutional positioning.",
    },
    "dealer_net_buy_5d": {
        "zh": "自營商 5 日買賣超：自營商近 5 個交易日買進減賣出的合計。",
        "en": "Dealer 5-day net buy: dealer buy minus sell over the last 5 trading days.",
    },
    "total_institutional_net_buy_5d": {
        "zh": "三大法人 5 日買賣超：外資、投信、自營商近 5 日買賣超合計。",
        "en": "All institutions 5-day net buy: combined net buy of foreign investors, investment trusts, and dealers over 5 days.",
    },
    "total_institutional_net_buy_20d": {
        "zh": "三大法人 20 日買賣超：外資、投信、自營商近 20 日買賣超合計，用來看較長期籌碼方向。",
        "en": "All institutions 20-day net buy: combined institutional net buy over 20 days, useful for longer-term chip direction.",
    },
    "foreign_net_buy_5d_ratio": {
        "zh": "外資 5 日買賣超占量比：外資 5 日買賣超除以近 5 日成交量，避免只看絕對張數。",
        "en": "Foreign 5-day net-buy ratio: foreign net buy divided by 5-day volume, scaling institutional buying by trading activity.",
    },
    "investment_trust_net_buy_5d_ratio": {
        "zh": "投信 5 日買賣超占量比：投信 5 日買賣超除以近 5 日成交量。",
        "en": "Investment trust 5-day net-buy ratio: investment trust net buy divided by 5-day volume.",
    },
    "dealer_net_buy_5d_ratio": {
        "zh": "自營商 5 日買賣超占量比：自營商 5 日買賣超除以近 5 日成交量。",
        "en": "Dealer 5-day net-buy ratio: dealer net buy divided by 5-day volume.",
    },
    "total_institutional_net_buy_5d_ratio": {
        "zh": "三大法人 5 日買賣超占量比：三大法人 5 日買賣超除以近 5 日成交量。",
        "en": "All institutions 5-day net-buy ratio: total institutional net buy divided by 5-day volume.",
    },
    "foreign_consecutive_buy_days": {
        "zh": "外資連續買超天數：外資連續呈現淨買超的交易日數。",
        "en": "Foreign consecutive buy days: number of consecutive trading days where foreign investors are net buyers.",
    },
    "investment_trust_consecutive_buy_days": {
        "zh": "投信連續買超天數：投信連續呈現淨買超的交易日數。",
        "en": "Investment trust consecutive buy days: number of consecutive trading days where investment trusts are net buyers.",
    },
    "margin_balance_change_5d": {
        "zh": "融資餘額 5 日變化：近 5 日融資餘額變化。融資增加常代表散戶槓桿買盤增加。",
        "en": "5-day margin balance change: change in margin financing balance over 5 days. Rising margin can imply more leveraged retail buying.",
    },
    "short_balance_change_5d": {
        "zh": "融券餘額 5 日變化：近 5 日融券餘額變化。融券增加常代表放空力道上升。",
        "en": "5-day short balance change: change in short-selling balance over 5 days. Rising short balance often means stronger bearish pressure.",
    },
}


def feature_definition(feature: str, lang: str = "en") -> str:
    item = FEATURE_DEFINITIONS.get(feature, {})
    return item.get(lang) or item.get("en") or "Definition not available."


def feature_short_definition(feature: str, lang: str = "en") -> str:
    definition = feature_definition(feature, lang)
    return definition.split("。")[0] if lang == "zh" else definition.split(".")[0] + "."


def style_factor_signal(row: pd.Series) -> list[str]:
    score = row.get("quality_score", row.get("validation_percentile", 50))
    if pd.isna(score):
        score = 50
    if score >= 75:
        style = "background-color: #dcfce7; color: #14532d; font-weight: 700;"
    elif score >= 55:
        style = "background-color: #ecfdf5; color: #166534;"
    elif score >= 45:
        style = "background-color: #f8fafc; color: #334155;"
    elif score >= 25:
        style = "background-color: #fef9c3; color: #854d0e;"
    else:
        style = "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
    return [style] * len(row)


def styled_factor_table(table: pd.DataFrame):
    return (
        table.style.format({"value": "{:.4f}", "validation_percentile": "{:.1f}", "quality_score": "{:.1f}"})
        .apply(style_factor_signal, axis=1)
        .hide(axis="columns", subset=["quality_score"])
    )


def factor_explanation_table(row: pd.Series, feature_cols: list[str], lang: str = "en") -> pd.DataFrame:
    columns = [
        "feature",
        "definition",
        "value",
        "validation_percentile",
        "direction",
        "signal",
        "interpretation",
        "quality_score",
    ]
    records = []
    for col in feature_cols:
        pct_col = f"{col}_pct_rank"
        if col in row.index and pct_col in row.index and pd.notna(row[col]) and pd.notna(row[pct_col]):
            percentile = float(row[pct_col])
            value = float(row[col])
            quality = feature_quality_score(col, value, percentile)
            records.append(
                {
                    "feature": col,
                    "definition": feature_short_definition(col, lang),
                    "value": value,
                    "validation_percentile": percentile,
                    "direction": direction_text(col, lang),
                    "signal": describe_signal_quality(quality, lang),
                    "interpretation": describe_score_level(percentile, lang),
                    "quality_score": quality,
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values("quality_score", ascending=False)


def render_feature_glossary(features: list[str], lang: str) -> None:
    title = "指標定義" if lang == "zh" else "Feature glossary"
    with st.expander(title, expanded=False):
        for feature in features:
            st.markdown(f"**`{feature}`**  \n{feature_definition(feature, lang)}")


def contribution_value(row: pd.Series, col: str) -> float:
    return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0


def strategy_mode_label(lang: str, strategy_mode: str) -> str:
    return tr(lang, "hysteresis") if strategy_mode == "hysteresis" else tr(lang, "single_threshold")


def render_current_strategy(result, lang: str, is_meta_strategy: bool = False) -> None:
    buy_threshold = result.meta_selected_threshold if is_meta_strategy else result.selected_threshold
    sell_threshold = result.meta_selected_sell_threshold if is_meta_strategy else result.selected_sell_threshold
    st.subheader(tr(lang, "current_strategy"))
    c1, c2, c3 = st.columns(3)
    c1.metric(tr(lang, "strategy_mode"), strategy_mode_label(lang, result.strategy_mode))
    c2.metric(tr(lang, "buy_threshold"), f"{buy_threshold:.1f}")
    c3.metric(tr(lang, "sell_threshold"), f"{sell_threshold:.1f}")
    rule_key = "hysteresis_rule" if result.strategy_mode == "hysteresis" else "single_rule"
    st.info(tr(lang, rule_key))


def render_factor_explanation(result, lang: str) -> None:
    st.subheader(tr(lang, "factor_explanation"))
    st.caption(tr(lang, "factor_explanation_caption"))

    strategy_options = [tr(lang, "manual_score"), tr(lang, "meta_score")]
    strategy = st.radio(
        tr(lang, "strategy_to_explain"),
        strategy_options,
        horizontal=True,
    )
    is_meta_strategy = strategy == tr(lang, "meta_score")
    bt = result.meta_backtest_df.copy() if is_meta_strategy else result.backtest_df.copy()
    selected_threshold = result.meta_selected_threshold if is_meta_strategy else result.selected_threshold
    selected_sell_threshold = result.meta_selected_sell_threshold if is_meta_strategy else result.selected_sell_threshold
    render_current_strategy(result, lang, is_meta_strategy=is_meta_strategy)
    dates = list(bt.index)
    selected_date = st.selectbox(
        tr(lang, "select_date"),
        dates,
        index=len(dates) - 1,
        format_func=lambda x: str(pd.to_datetime(x).date()),
    )
    row = bt.loc[selected_date]
    manual_row = result.backtest_df.loc[selected_date] if selected_date in result.backtest_df.index else row

    technical_contribution = contribution_value(manual_row, "technical_contribution")
    fundamental_contribution = contribution_value(manual_row, "fundamental_contribution")
    chip_contribution = contribution_value(manual_row, "chip_contribution")
    final_score = float(row["final_score"])
    decision = tr(lang, "buy_decision") if float(row["target_position"]) >= 1.0 else tr(lang, "cash_decision")
    threshold_gap = final_score - selected_threshold

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr(lang, "final_score"), f"{final_score:.1f}", f"{threshold_gap:+.1f} vs threshold")
    c2.metric(tr(lang, "buy_threshold"), f"{selected_threshold:.1f}")
    c3.metric(tr(lang, "decision"), decision)
    c4.metric(tr(lang, "next_position"), f"{int(row['target_position'])}")

    m1, m2, m3 = st.columns(3)
    m1.metric(tr(lang, "technical_score"), f"{row['technical_score']:.1f}", f"{technical_contribution:.1f} {tr(lang, 'contribution')}")
    m2.metric(tr(lang, "fundamental_score"), f"{row['fundamental_score']:.1f}", f"{fundamental_contribution:.1f} {tr(lang, 'contribution')}")
    m3.metric(tr(lang, "chip_score"), f"{row['chip_score']:.1f}", f"{chip_contribution:.1f} {tr(lang, 'contribution')}")

    chart_col, threshold_col = st.columns([2, 1])
    contribution_df = pd.DataFrame(
        {
            "factor": ["Technical", "Fundamental", "Chip"] if lang == "en" else ["技術面", "基本面", "籌碼面"],
            tr(lang, "contribution"): [technical_contribution, fundamental_contribution, chip_contribution],
        }
    ).set_index("factor")
    with chart_col:
        st.markdown(f"### {tr(lang, 'breakdown')}")
        st.bar_chart(contribution_df, use_container_width=True)
    with threshold_col:
        st.markdown(f"### {tr(lang, 'threshold_check')}")
        st.progress(min(max(final_score / 100.0, 0.0), 1.0), text=f"{tr(lang, 'final_score')} {final_score:.1f} / 100")
        st.progress(min(max(selected_threshold / 100.0, 0.0), 1.0), text=f"{tr(lang, 'buy_threshold')} {selected_threshold:.1f} / 100")
        if result.strategy_mode == "hysteresis":
            st.progress(min(max(selected_sell_threshold / 100.0, 0.0), 1.0), text=f"{tr(lang, 'sell_threshold')} {selected_sell_threshold:.1f} / 100")
        if float(row["target_position"]) >= 1.0:
            st.success(tr(lang, "buy_msg"))
        else:
            st.warning(tr(lang, "cash_msg"))

    if lang == "zh":
        st.markdown(
            f"""
            **{tr(lang, 'why')}**

            - 技術面貢獻為 `{technical_contribution:.1f}`，因為校正後技術面分數是 `{row['technical_score']:.1f}`，再乘上人工設定的技術面權重。
            - 基本面貢獻為 `{fundamental_contribution:.1f}`，因為校正後基本面分數是 `{row['fundamental_score']:.1f}`，再乘上人工設定的基本面權重。
            - 籌碼面貢獻為 `{chip_contribution:.1f}`，因為校正後籌碼面分數是 `{row['chip_score']:.1f}`，再乘上人工設定的籌碼面權重。
            - 最終分數是 `{final_score:.1f}`，買入門檻是 `{selected_threshold:.1f}`，賣出門檻是 `{selected_sell_threshold:.1f}`，因此這一天的決策是 `{decision}`。
            - 如果選擇 `Meta model 分數`，最終交易分數會由模型學習三個因子分數之間的關係；貢獻欄位仍保留人工加權拆解作為參考。
            - 下方 percentile 會把該日特徵值與驗證期間的分布做比較，用來判斷該指標偏高或偏低。
            """
        )
    else:
        st.markdown(
            f"""
            **{tr(lang, 'why')}**

            - Technical contribution is `{technical_contribution:.1f}` because the calibrated technical score is
              `{row['technical_score']:.1f}` and the manual technical weight is applied after calibration.
            - Fundamental contribution is `{fundamental_contribution:.1f}` because the calibrated
              fundamental score is `{row['fundamental_score']:.1f}` and the manual fundamental weight is applied after calibration.
            - Chip contribution is `{chip_contribution:.1f}` because the calibrated chip score is
              `{row['chip_score']:.1f}` and the manual chip weight is applied after calibration.
            - The final score is `{final_score:.1f}`, the selected buy threshold is `{selected_threshold:.1f}`,
              and the selected sell threshold is `{selected_sell_threshold:.1f}`,
              so this date is classified as `{decision}`.
            - If `Meta model score` is selected, the final trading score is learned from the three calibrated factor scores,
              while the contribution columns still show the manual weighted decomposition for reference.
            - Percentiles below compare the selected date's feature value against the validation-period distribution.
            """
        )

    technical_features = TECHNICAL_COLS
    fundamental_features = ["eps", "eps_growth_yoy", "eps_growth_qoq"]
    chip_features = [
        "foreign_net_buy_5d",
        "investment_trust_net_buy_5d",
        "dealer_net_buy_5d",
        "total_institutional_net_buy_5d",
        "total_institutional_net_buy_20d",
        "foreign_net_buy_5d_ratio",
        "investment_trust_net_buy_5d_ratio",
        "dealer_net_buy_5d_ratio",
        "total_institutional_net_buy_5d_ratio",
        "foreign_consecutive_buy_days",
        "investment_trust_consecutive_buy_days",
        "margin_balance_change_5d",
        "short_balance_change_5d",
    ]

    t_col, f_col, c_col = st.columns(3)
    with t_col:
        st.markdown(f"### {tr(lang, 'technical_drivers')}")
        t_table = factor_explanation_table(row, technical_features, lang)
        if t_table.empty:
            st.warning(tr(lang, "no_technical"))
        else:
            st.dataframe(
                styled_factor_table(t_table),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("feature", help="Feature name / 指標名稱"),
                    "definition": st.column_config.TextColumn("definition", help="Short definition / 指標短定義", width="medium"),
                },
            )
            render_feature_glossary(technical_features, lang)
    with f_col:
        st.markdown(f"### {tr(lang, 'fundamental_drivers')}")
        f_table = factor_explanation_table(row, fundamental_features, lang)
        if f_table.empty:
            st.warning(tr(lang, "no_fundamental"))
        else:
            st.dataframe(
                styled_factor_table(f_table),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("feature", help="Feature name / 指標名稱"),
                    "definition": st.column_config.TextColumn("definition", help="Short definition / 指標短定義", width="medium"),
                },
            )
            render_feature_glossary(fundamental_features, lang)
    with c_col:
        st.markdown(f"### {tr(lang, 'chip_drivers')}")
        c_table = factor_explanation_table(row, chip_features, lang)
        if c_table.empty:
            st.warning(tr(lang, "no_chip"))
        else:
            st.dataframe(
                styled_factor_table(c_table),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("feature", help="Feature name / 指標名稱"),
                    "definition": st.column_config.TextColumn("definition", help="Short definition / 指標短定義", width="medium"),
                },
            )
            render_feature_glossary(chip_features, lang)

    st.subheader(tr(lang, "contribution_over_time"))
    chart_cols = ["fundamental_contribution", "chip_contribution", "final_score"]
    if "technical_contribution" in bt.columns:
        chart_cols.insert(0, "technical_contribution")
    if "meta_score" in bt.columns:
        chart_cols.append("meta_score")
    timeline_source = bt if all(col in bt.columns for col in chart_cols) else result.backtest_df
    st.line_chart(timeline_source[[col for col in chart_cols if col in timeline_source.columns]], use_container_width=True)


def render_strategy_comparison(result, lang: str) -> None:
    st.subheader(tr(lang, "strategy_compare_title"))
    st.caption(tr(lang, "strategy_compare_caption"))
    st.dataframe(
        result.strategy_comparison.style.format(
            {
                "selected_buy_threshold": "{:.1f}",
                "selected_sell_threshold": "{:.1f}",
                "strategy_total_return": "{:.4f}",
                "buy_hold_total_return": "{:.4f}",
                "strategy_max_drawdown": "{:.4f}",
                "strategy_sharpe": "{:.4f}",
                "entry_count": "{:.0f}",
                "holding_ratio": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {tr(lang, 'meta_summary')}")
        st.dataframe(result.meta_model_summary, use_container_width=True, hide_index=True)
    with c2:
        st.markdown(f"### {tr(lang, 'meta_threshold')}")
        if result.meta_threshold_table.empty:
            st.warning(tr(lang, "meta_skipped"))
        else:
            st.dataframe(
                result.meta_threshold_table.style.format(
                    {
                        "buy_threshold": "{:.1f}",
                        "sell_threshold": "{:.1f}",
                        "threshold": "{:.1f}",
                        "validation_strategy_return": "{:.4f}",
                        "validation_sharpe": "{:.4f}",
                        "validation_max_drawdown": "{:.4f}",
                        "validation_entry_count": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader(tr(lang, "equity_compare"))
    comparison_curve = pd.DataFrame(
        {
            "Manual weighted strategy" if lang == "en" else "人工加權策略": result.backtest_df["strategy_cum"],
            "Meta model strategy" if lang == "en" else "Meta model 策略": result.meta_backtest_df["strategy_cum"],
            "Buy & Hold" if lang == "en" else "買進持有": result.backtest_df["buy_hold_cum"],
        }
    )
    st.line_chart(comparison_curve, use_container_width=True)


def architecture_modules(lang: str) -> dict[str, dict[str, str]]:
    if lang == "zh":
        return {
            "goal": {
                "label": "1. 研究目標",
                "role": "定義問題：用停利、停損與最長持有期建立 Triple Barrier 交易事件，評估訊號是否能抓到波段機會。",
                "inputs": "股票代號、日期區間、交易成本、策略模式、停利障礙、停損障礙、垂直時間障礙。",
                "outputs": "target：交易事件是否先碰到停利障礙，而不是先停損或時間到。",
                "evidence": "Triple Barrier Score Design 與 Current Trading Strategy 顯示目前標籤目標與交易規則。",
            },
            "data": {
                "label": "2. 資料層",
                "role": "取得價格、基本面與籌碼面資料。",
                "inputs": "Yahoo Finance OHLCV、FinMind EPS、三大法人、融資融券，或手動 CSV。",
                "outputs": "可合併到每日交易日的原始資料表。",
                "evidence": "Data Notes 會說明哪些資料成功取得、哪些因子因缺資料維持中性 50 分。",
            },
            "features": {
                "label": "3. 特徵工程",
                "role": "把原始資料轉成模型可學習的技術面、基本面、籌碼面特徵。",
                "inputs": "價格量資料、EPS、法人買賣超、融資融券餘額。",
                "outputs": "RSI、MACD、MA ratio、EPS growth、法人 rolling net buy 等特徵。",
                "evidence": "Factor Explanation 的 driver tables 顯示各特徵目前狀態與紅黃綠燈。",
            },
            "split": {
                "label": "4. 時間序切分",
                "role": "避免資料外洩，依時間順序切成 train / validation / test。",
                "inputs": "完整特徵資料表。",
                "outputs": "訓練集、驗證集、測試集。",
                "evidence": "模型只用 train 學習；threshold 只用 validation 搜尋；test 只做最終報告。",
            },
            "submodels": {
                "label": "5. 三個子模型",
                "role": "分別訓練 technical、fundamental、chip 模型，學習哪些因子狀態較容易先達成停利。",
                "inputs": "各因子自己的特徵欄位與 Triple Barrier target。",
                "outputs": "technical_score、fundamental_score、chip_score。",
                "evidence": "Overview 的 Factor Model Status 顯示每個因子是 model、neutral 或資料不足。",
            },
            "score": {
                "label": "6. 分數合成",
                "role": "比較人工權重與 Meta model 兩種合成方式。",
                "inputs": "三個校正後的 0-100 因子分數。",
                "outputs": "Manual weighted final_score 與 Meta model final_score。",
                "evidence": "Strategy Comparison 比較 Manual Weighted Score 和 Meta Model Score。",
            },
            "strategy": {
                "label": "7. 交易策略",
                "role": "把 final_score 轉成 target_position，並由 Triple Barrier 風控規則管理出場。",
                "inputs": "final_score、buy_threshold、sell_threshold、策略模式、停利 / 停損 / 最長持有期。",
                "outputs": "target_position、position、交易成本、策略報酬、Triple Barrier 出場效果。",
                "evidence": "Current Trading Strategy 顯示門檻進場邏輯，Triple Barrier settings 定義停利、停損與時間出場。",
            },
            "dashboard": {
                "label": "8. 視覺化與下載",
                "role": "把模型結果轉成可解釋的 dashboard 與 CSV。",
                "inputs": "回測結果、因子分數、交易部位、績效指標。",
                "outputs": "績效圖、因子解釋、紅黃綠燈、下載檔。",
                "evidence": "Overview、Factor Explanation、Signals & Download 都是最終展示層。",
            },
        }
    return {
        "goal": {
            "label": "1. Goal",
            "role": "Define the problem: build Triple Barrier trade events with take-profit, stop-loss, and max holding period rules.",
            "inputs": "Ticker, date range, trading cost, strategy mode, take-profit barrier, stop-loss barrier, and vertical barrier.",
            "outputs": "Target: whether a trade reaches the upside barrier before the downside barrier or timeout.",
            "evidence": "Triple Barrier Score Design and Current Trading Strategy show the label target and trading rule.",
        },
        "data": {
            "label": "2. Data Layer",
            "role": "Fetch price, fundamental, and chip data.",
            "inputs": "Yahoo Finance OHLCV, FinMind EPS, institutional flows, margin/short data, or uploaded CSV.",
            "outputs": "Raw daily-aligned data ready for feature engineering.",
            "evidence": "Data Notes explain which sources loaded and which missing factors stay neutral at 50.",
        },
        "features": {
            "label": "3. Feature Engineering",
            "role": "Transform raw data into technical, fundamental, and chip model features.",
            "inputs": "Price/volume, EPS, institutional net buy, margin/short balances.",
            "outputs": "RSI, MACD, MA ratio, EPS growth, rolling institutional net buy, and related features.",
            "evidence": "Factor Explanation driver tables show feature values and red/yellow/green health colors.",
        },
        "split": {
            "label": "4. Time Split",
            "role": "Avoid leakage by splitting data chronologically into train / validation / test.",
            "inputs": "Full feature table.",
            "outputs": "Train, validation, and test splits.",
            "evidence": "Train fits models; validation searches thresholds; test reports final out-of-sample results.",
        },
        "submodels": {
            "label": "5. Factor Submodels",
            "role": "Train technical, fundamental, and chip models separately to estimate which factor states tend to reach take-profit first.",
            "inputs": "Each factor's feature columns and the Triple Barrier target.",
            "outputs": "technical_score, fundamental_score, and chip_score.",
            "evidence": "Overview Factor Model Status shows whether each factor is model, neutral, or insufficient.",
        },
        "score": {
            "label": "6. Score Fusion",
            "role": "Compare manual weighting against a learned meta model.",
            "inputs": "Three calibrated 0-100 factor scores.",
            "outputs": "Manual weighted final_score and Meta model final_score.",
            "evidence": "Strategy Comparison compares Manual Weighted Score and Meta Model Score.",
        },
        "strategy": {
            "label": "7. Trading Strategy",
            "role": "Convert final_score into target_position while Triple Barrier risk controls manage exits.",
            "inputs": "final_score, buy_threshold, sell_threshold, strategy mode, take-profit, stop-loss, and max holding days.",
            "outputs": "target_position, executed position, trading cost, strategy return, and Triple Barrier exit behavior.",
            "evidence": "Current Trading Strategy shows entry logic, while Triple Barrier settings define take-profit, stop-loss, and timeout exits.",
        },
        "dashboard": {
            "label": "8. Visualization & Download",
            "role": "Turn model outputs into an explainable dashboard and CSV.",
            "inputs": "Backtest results, factor scores, positions, and performance metrics.",
            "outputs": "Performance charts, factor explanations, health colors, and downloadable files.",
            "evidence": "Overview, Factor Explanation, and Signals & Download are the final presentation layer.",
        },
    }


def architecture_graph(granularity: str, lang: str) -> str:
    modules = architecture_modules(lang)
    detailed = granularity == "detail"
    if detailed:
        return f"""
digraph G {{
  graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.55"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, color="#64748b", fillcolor="#eff6ff"];
  edge [color="#64748b", arrowsize=0.8, fontname="Arial", fontsize=10];
  goal [label="{modules["goal"]["label"]}\\nTriple Barrier target", fillcolor="#e0f2fe"];
  price [label="Yahoo Finance\\nOHLCV", fillcolor="#f8fafc"];
  finmind [label="FinMind / CSV\\nEPS + Chip", fillcolor="#f8fafc"];
  tech [label="Technical features\\nRSI / MACD / MA", fillcolor="#ecfeff"];
  fund [label="Fundamental features\\nEPS growth", fillcolor="#f0fdf4"];
  chip [label="Chip features\\nInstitutional flow", fillcolor="#fff7ed"];
  split [label="{modules["split"]["label"]}\\nTrain / Validation / Test", fillcolor="#fefce8"];
  submodels [label="{modules["submodels"]["label"]}\\n3 calibrated scores", fillcolor="#ede9fe"];
  manual [label="Manual barrier score\\nUser weights", fillcolor="#fdf2f8"];
  meta [label="Meta barrier score\\nLearned outcome relationship", fillcolor="#fdf2f8"];
  strategy [label="{modules["strategy"]["label"]}\\nThreshold + barrier exits", fillcolor="#fee2e2"];
  dashboard [label="{modules["dashboard"]["label"]}\\nCharts + explanations + CSV", fillcolor="#dcfce7"];
  goal -> price;
  goal -> finmind;
  price -> tech;
  finmind -> fund;
  finmind -> chip;
  tech -> split;
  fund -> split;
  chip -> split;
  split -> submodels;
  submodels -> manual;
  submodels -> meta;
  manual -> strategy;
  meta -> strategy;
  strategy -> dashboard;
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
  submodels [label="{modules["submodels"]["label"]}", fillcolor="#ede9fe"];
  score [label="{modules["score"]["label"]}", fillcolor="#fdf2f8"];
  strategy [label="{modules["strategy"]["label"]}", fillcolor="#fee2e2"];
  dashboard [label="{modules["dashboard"]["label"]}", fillcolor="#dcfce7"];
  goal -> data -> features -> submodels -> score -> strategy -> dashboard;
}}
"""


def render_architecture_explorer(result, lang: str) -> None:
    title = tr(lang, "architecture_title")
    caption = tr(lang, "architecture_caption")
    st.subheader(title)
    st.caption(caption)
    levels = {
        tr(lang, "overview_level") if lang == "en" else "總覽": "overview",
        tr(lang, "detail_level") if lang == "en" else "細節": "detail",
    }
    selected_level = st.radio(
        tr(lang, "granularity") if lang == "en" else "架構層級",
        list(levels.keys()),
        horizontal=True,
    )
    granularity = levels[selected_level]
    st.graphviz_chart(architecture_graph(granularity, lang), use_container_width=True)

    modules = architecture_modules(lang)
    labels = {info["label"]: key for key, info in modules.items()}
    selected_label = st.selectbox(
        tr(lang, "inspect_module") if lang == "en" else "查看模組",
        list(labels.keys()),
    )
    selected = modules[labels[selected_label]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {selected['label']}")
        st.markdown(f"**{tr(lang, 'module_role') if lang == 'en' else '角色'}**")
        st.write(selected["role"])
        st.markdown(f"**{tr(lang, 'module_inputs') if lang == 'en' else '輸入'}**")
        st.write(selected["inputs"])
    with c2:
        st.markdown("### " + ("Outputs" if lang == "en" else "輸出與證據"))
        st.markdown(f"**{tr(lang, 'module_outputs') if lang == 'en' else '輸出'}**")
        st.write(selected["outputs"])
        st.markdown(f"**{tr(lang, 'module_evidence') if lang == 'en' else '本工具中的證據'}**")
        st.write(selected["evidence"])

    if result is not None:
        st.markdown("### " + ("Live run snapshot" if lang == "en" else "目前執行結果快照"))
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Train rows" if lang == "en" else "訓練筆數", f"{len(result.train_df):,}")
        s2.metric("Validation rows" if lang == "en" else "驗證筆數", f"{len(result.val_df):,}")
        s3.metric("Test rows" if lang == "en" else "測試筆數", f"{len(result.test_df):,}")
        s4.metric("Strategy mode" if lang == "en" else "策略模式", strategy_mode_label(lang, result.strategy_mode))


def render_references(lang: str) -> None:
    st.subheader(tr(lang, "references_title"))
    st.caption(tr(lang, "references_caption"))

    if lang == "zh":
        st.markdown(
            """
            #### 方法來源

            Triple Barrier Method 通常引用 Marcos López de Prado 的 *Advances in Financial Machine Learning*
            第 3 章 Labeling。該章包含固定時間標籤、動態門檻、Triple Barrier Method、meta-labeling 等金融機器學習標籤設計。

            #### 本專案如何使用

            本 app 使用三個障礙定義一個交易事件：

            - **上方水平障礙**：停利門檻，價格先碰到代表較正面的 long trade 結果。
            - **下方水平障礙**：停損門檻，價格先碰到代表交易失敗或風險事件。
            - **垂直時間障礙**：最長持有期，若停利與停損都沒先發生，就在時間到時結束事件。

            本專案把這個事件標籤用於監督式學習，讓模型不只學隔日漲跌，而是學「是否較可能先達成停利」。

            #### 參考資料

            - Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
              O'Reilly 書籍頁面列出 Chapter 3 Labeling，其中包含 **The Triple-Barrier Method**。
              <https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml>
            - MLFinPy documentation, **Data Labelling**.
              文件說明 triple-barrier method 與 meta-labeling 的實作脈絡。
              <https://mlfinpy.readthedocs.io/en/stable/Labelling.html>
            """
        )
    else:
        st.markdown(
            """
            #### Method Origin

            The Triple Barrier Method is commonly cited from Marcos López de Prado's
            *Advances in Financial Machine Learning*, Chapter 3, Labeling. That chapter covers fixed-time
            horizon labels, dynamic thresholds, the Triple Barrier Method, and meta-labeling.

            #### How This Project Uses It

            This app defines a trade event with three barriers:

            - **Upper horizontal barrier**: take-profit threshold. Reaching this first is treated as a favorable long-trade outcome.
            - **Lower horizontal barrier**: stop-loss threshold. Reaching this first is treated as a failed or risk event.
            - **Vertical time barrier**: maximum holding period. If neither price barrier is reached first, the event expires at this time limit.

            The project uses this event label for supervised learning, so the model learns whether a setup is more likely
            to reach take-profit first instead of only predicting next-day direction.

            #### References

            - Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
              The O'Reilly book page lists Chapter 3, Labeling, including **The Triple-Barrier Method**.
              <https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml>
            - MLFinPy documentation, **Data Labelling**.
              This documentation describes the triple-barrier method implementation context with meta-labeling.
              <https://mlfinpy.readthedocs.io/en/stable/Labelling.html>
            """
        )

    st.info(tr(lang, "references_project_note"))


def render_training_data_tab(result, lang: str) -> None:
    st.subheader("Training Data" if lang == "en" else "訓練資料")
    if lang == "zh":
        st.markdown(
            """
            這個頁籤說明 Triple Barrier 模型實際使用的資料、處理流程與監督式學習標籤。

            - **資料來源**：Yahoo Finance / FinMind 價格資料、FinMind 或 CSV 的 EPS、三大法人與融資融券資料。
            - **處理前資料**：價格量、EPS、法人買賣超、融資融券餘額。
            - **處理後資料**：技術面、基本面、籌碼面特徵，並加入 Triple Barrier 事件欄位。
            - **訓練目標**：`target = 1` 代表先碰到停利障礙；`target = -1` 代表先碰到停損；`target = 0` 代表時間到。
            - **訓練方式**：technical / fundamental / chip 各自學習 Triple Barrier target，再校正為 0-100 分數；Meta model 使用三個因子分數作為輸入。
            """
        )
    else:
        st.markdown(
            """
            This tab explains the actual data used by the Triple Barrier models, the processing flow, and the supervised label.

            - **Data sources**: Yahoo Finance / FinMind price data, plus FinMind or CSV EPS, institutional flow, and margin/short data.
            - **Before processing**: price/volume, EPS, institutional net buy, and margin/short balances.
            - **After processing**: technical, fundamental, and chip features plus Triple Barrier event columns.
            - **Training target**: `target = 1` when take-profit is reached first; `target = -1` when stop-loss is reached first; `target = 0` when the event times out.
            - **Training design**: technical / fundamental / chip submodels learn the Triple Barrier target and are calibrated to 0-100 scores; the Meta model uses the three factor scores as inputs.
            """
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows" if lang == "en" else "訓練筆數", f"{len(result.train_df):,}")
    c2.metric("Validation rows" if lang == "en" else "驗證筆數", f"{len(result.val_df):,}")
    c3.metric("Test rows" if lang == "en" else "測試筆數", f"{len(result.test_df):,}")
    c4.metric("Total rows" if lang == "en" else "總筆數", f"{len(result.df):,}")

    b1, b2, b3 = st.columns(3)
    b1.metric("Take-profit" if lang == "en" else "停利障礙", fmt_pct(result.take_profit_pct))
    b2.metric("Stop-loss" if lang == "en" else "停損障礙", fmt_pct(result.stop_loss_pct))
    b3.metric("Max holding days" if lang == "en" else "最長持有日", f"{result.max_holding_days}")

    st.markdown("### " + ("Triple Barrier Label Distribution" if lang == "en" else "Triple Barrier 標籤分布"))
    label_counts = result.df["target"].value_counts(dropna=False).sort_index().rename("rows").reset_index()
    label_counts.columns = ["target", "rows"]
    st.dataframe(label_counts, use_container_width=True, hide_index=True)

    st.markdown("### " + ("Data Source Notes" if lang == "en" else "資料來源備註"))
    for note in result.data_notes:
        st.write(f"- {note}")

    st.markdown("### " + ("Feature Groups" if lang == "en" else "特徵群組"))
    feature_table = pd.DataFrame(
        [{"factor": "technical", "feature": feature} for feature in TECHNICAL_COLS]
        + [{"factor": "fundamental", "feature": feature} for feature in FUNDAMENTAL_COLS]
        + [{"factor": "chip", "feature": feature} for feature in CHIP_COLS]
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("### " + ("Usable Rows by Factor" if lang == "en" else "各因子可用資料筆數"))
    st.dataframe(result.factor_data_summary, use_container_width=True, hide_index=True)

    st.markdown("### " + ("Processed Training Data Preview" if lang == "en" else "處理後訓練資料預覽"))
    preview_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "target",
        "tb_label",
        "tb_event",
        "tb_event_date",
        "tb_event_return",
        "tb_holding_days",
        *TECHNICAL_COLS,
        *FUNDAMENTAL_COLS,
        *CHIP_COLS,
    ]
    preview_cols = [col for col in preview_cols if col in result.df.columns]
    st.dataframe(result.df[preview_cols].tail(100), use_container_width=True)

    st.markdown("### " + ("Score Data Generated for Modeling" if lang == "en" else "模型產生的分數資料"))
    st.dataframe(result.score_df.tail(100), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="TSMC Triple Barrier Strategy", layout="wide")

    with st.sidebar:
        language_label = st.selectbox("Language / 語言", ["English", "繁體中文"], index=0)
        lang = "zh" if language_label == "繁體中文" else "en"

        st.header(tr(lang, "data"))
        ticker = st.text_input(tr(lang, "ticker"), value="2330.TW")
        start = st.date_input(tr(lang, "start_date"), value=pd.Timestamp("2015-01-01"))
        end = st.date_input(tr(lang, "end_date"), value=pd.Timestamp("2026-01-01"))
        use_finmind = st.checkbox(tr(lang, "fetch_finmind"), value=True)
        finmind_token = st.text_input(tr(lang, "finmind_token"), value="", type="password")
        fundamental_file = st.file_uploader(tr(lang, "fundamental_csv"), type=["csv"])
        chip_file = st.file_uploader(
            tr(lang, "chip_csv"),
            type=["csv"],
        )

        st.header(tr(lang, "weights"))
        technical_w = st.slider(tr(lang, "technical_weight"), 0.0, 1.0, 0.2, 0.05)
        fundamental_w = st.slider(tr(lang, "fundamental_weight"), 0.0, 1.0, 0.4, 0.05)
        chip_w = st.slider(tr(lang, "chip_weight"), 0.0, 1.0, 0.4, 0.05)

        st.header(tr(lang, "strategy_settings"))
        strategy_options = {
            tr(lang, "single_threshold"): "single_threshold",
            tr(lang, "hysteresis"): "hysteresis",
        }
        strategy_label = st.selectbox(tr(lang, "strategy_mode"), list(strategy_options.keys()), index=1)
        strategy_mode = strategy_options[strategy_label]
        auto_threshold = st.checkbox(tr(lang, "auto_threshold"), value=True)
        if auto_threshold:
            st.caption(
                "Auto-search is enabled. Fixed thresholds are ignored."
                if lang == "en"
                else "目前啟用自動搜尋，固定買入 / 賣出門檻不會被使用。"
            )
        else:
            st.caption(
                "Auto-search is disabled. Search range settings are ignored."
                if lang == "en"
                else "目前關閉自動搜尋，門檻搜尋範圍不會被使用。"
            )
        final_threshold = st.slider(
            tr(lang, "fixed_threshold"),
            0.0,
            100.0,
            60.0,
            1.0,
            disabled=auto_threshold,
        )
        sell_threshold = st.slider(
            tr(lang, "fixed_sell_threshold"),
            0.0,
            100.0,
            45.0,
            1.0,
            disabled=auto_threshold,
        )
        threshold_min = st.slider(
            tr(lang, "threshold_min"),
            0.0,
            100.0,
            40.0,
            1.0,
            disabled=not auto_threshold,
        )
        threshold_max = st.slider(
            tr(lang, "threshold_max"),
            0.0,
            100.0,
            80.0,
            1.0,
            disabled=not auto_threshold,
        )
        threshold_step = st.select_slider(
            tr(lang, "threshold_step"),
            options=[1.0, 2.0, 5.0],
            value=2.0,
            disabled=not auto_threshold,
        )
        cost_bps = st.number_input(tr(lang, "cost_bps"), 0.0, 100.0, 10.0, 1.0)

        st.header(tr(lang, "barrier_settings"))
        take_profit_pct = st.number_input(tr(lang, "take_profit_pct"), 1.0, 50.0, 8.0, 0.5)
        stop_loss_pct = st.number_input(tr(lang, "stop_loss_pct"), 1.0, 50.0, 5.0, 0.5)
        max_holding_days = st.number_input(tr(lang, "max_holding_days"), 3, 120, 20, 1)
        run = st.button(tr(lang, "run"), type="primary")

    render_sticky_title_glossary(
        tr(lang, "app_title"),
        tr(lang, "app_caption"),
        lang,
        key_prefix="triple_barrier_glossary",
    )

    st.subheader(tr(lang, "score_design"))
    if lang == "zh":
        st.markdown(
            """
        - Triple Barrier 會用停利障礙、停損障礙與最長持有期，將每個交易日標記成較貼近波段交易的事件結果。
        - 每個因子模型會先產生「較可能先碰到停利障礙」的原始分數。
        - 原始分數會用驗證期間的百分位排名校正成 0-100 分。
        - 最終分數 = 技術面分數 * 技術面權重 + 基本面分數 * 基本面權重 + 籌碼面分數 * 籌碼面權重。
        - 即使某個模型的原始機率比較保守，百分位校正仍可讓它保有設定好的決策權重。
        - EPS 會依估計財報公告日對齊，降低偷看未來資料的風險。
        - 另外也會訓練第二層 Meta model，讓你比較「人工設定權重」與「模型學習三因子和 Triple Barrier 結果關係」的差異。
        """
        )
    else:
        st.markdown(
            """
        - Triple Barrier labeling turns each trading day into an event outcome using take-profit, stop-loss, and max holding period rules.
        - Each factor model produces a raw score for the probability of reaching the upside barrier first.
        - Raw scores are calibrated with validation percentile ranking into a 0-100 factor score.
        - Final score = technical score * technical weight + fundamental score * fundamental weight + chip score * chip weight.
        - This keeps the technical factor at the intended weight even if its raw model probabilities are conservative.
        - EPS is aligned by estimated report availability date to reduce look-ahead bias.
        - A second-stage meta model is also trained on the three calibrated factor scores, so you can compare manual weights against learned Triple Barrier outcome relationships.
        """
        )

    st.subheader(tr(lang, "data_source"))
    if lang == "zh":
        st.markdown(
            """
        此工具可以從 FinMind 抓取 EPS 與三大法人等籌碼資料。如果 FinMind 無法使用，
        也可以手動上傳 CSV。若有上傳 CSV，會優先使用上傳資料。
        """
        )
    else:
        st.markdown(
            """
        The app can fetch EPS and institutional investor data from FinMind. If FinMind is unavailable,
        you can upload CSV files manually. Uploaded CSV files take priority over FinMind data.
        """
        )

    st.subheader(tr(lang, "expected_csv"))
    c1, c2 = st.columns(2)
    with c1:
        st.code(
            "date,eps\n2021-05-15,5.39\n2021-08-14,5.18\n2021-11-14,6.03\n\n"
            "# Or use period_end and the app will estimate report availability date:\n"
            "period_end,eps\n2021-03-31,5.39\n",
            language="csv",
        )
    with c2:
        st.code(
            "date,foreign_net_buy,investment_trust_net_buy,dealer_net_buy,margin_balance,short_balance\n"
            "2024-01-02,1200000,300000,-50000,15000000,200000\n",
            language="csv",
        )

    if run:
        if threshold_min > threshold_max:
            st.error("Threshold search min must be less than or equal to threshold search max." if lang == "en" else "門檻搜尋最小值必須小於或等於最大值。")
            return
        if strategy_mode == "hysteresis" and not auto_threshold and sell_threshold > final_threshold:
            st.error("Sell threshold must be less than or equal to buy threshold." if lang == "en" else "賣出門檻必須小於或等於買入門檻。")
            return

        tech_weight, fund_weight, chip_weight = normalize_weights(technical_w, fundamental_w, chip_w)
        fundamental_df = read_optional_csv(fundamental_file)
        chip_df = read_optional_csv(chip_file)
        resolved_finmind_token, finmind_token_source = resolve_finmind_token(finmind_token)

        with st.spinner(tr(lang, "running")):
            result = run_triple_barrier_pipeline(
                ticker=ticker,
                start=str(start),
                end=str(end),
                fundamental_df=fundamental_df,
                chip_df=chip_df,
                use_finmind=use_finmind,
                finmind_token=resolved_finmind_token,
                final_threshold=final_threshold,
                sell_threshold=sell_threshold,
                auto_threshold=auto_threshold,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                cost_bps=cost_bps,
                technical_weight=tech_weight,
                fundamental_weight=fund_weight,
                chip_weight=chip_weight,
                strategy_mode=strategy_mode,
                take_profit_pct=take_profit_pct / 100.0,
                stop_loss_pct=stop_loss_pct / 100.0,
                max_holding_days=int(max_holding_days),
            )
        if use_finmind:
            result.data_notes.insert(0, f"FinMind token source: {finmind_token_source}.")
        st.session_state["triple_barrier_result"] = result
        st.session_state["triple_barrier_params"] = {
            "ticker": ticker,
            "start": str(start),
            "end": str(end),
            "technical_weight": tech_weight,
            "fundamental_weight": fund_weight,
            "chip_weight": chip_weight,
            "auto_threshold": auto_threshold,
            "cost_bps": cost_bps,
            "strategy_mode": strategy_mode,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
            "max_holding_days": int(max_holding_days),
        }

    result = st.session_state.get("triple_barrier_result")
    if result is None:
        st.info(tr(lang, "initial_info"))
        initial_architecture, initial_references = st.tabs(
            [tr(lang, "tab_architecture"), tr(lang, "tab_references")]
        )
        with initial_architecture:
            render_architecture_explorer(None, lang)
        with initial_references:
            render_references(lang)
        return

    params = st.session_state.get("triple_barrier_params", {})
    st.caption(
        tr(lang, "cached")
        + (f": {params.get('ticker')} ({params.get('start')} to {params.get('end')})." if params else ".")
    )

    with st.expander(tr(lang, "data_notes"), expanded=False):
        for note in result.data_notes:
            display_note = note
            if lang == "zh":
                display_note = (
                    note.replace("Fundamental CSV not provided. Fundamental factor will use a neutral 50 score.", "未提供基本面 CSV，基本面因子會使用中性 50 分。")
                    .replace("Chip CSV not provided. Chip factor will use a neutral 50 score.", "未提供籌碼面 CSV，籌碼面因子會使用中性 50 分。")
                    .replace("FinMind fundamental EPS data fetched. EPS is shifted to estimated report availability dates.", "已從 FinMind 抓取基本面 EPS，並將 EPS 對齊到估計財報可取得日期。")
                    .replace("FinMind chip data fetched. Institutional net buy is transformed into rolling features.", "已從 FinMind 抓取籌碼面資料，並轉換為三大法人買賣超的滾動特徵。")
                )
            if "not provided" in note or "failed" in note:
                st.warning(display_note)
            else:
                st.success(display_note)

    tab_overview, tab_compare, tab_explain, tab_training_data, tab_architecture, tab_signals, tab_references = st.tabs(
        [
            tr(lang, "tab_overview"),
            tr(lang, "tab_compare"),
            tr(lang, "tab_explain"),
            tr(lang, "tab_training_data"),
            tr(lang, "tab_architecture"),
            tr(lang, "tab_signals"),
            tr(lang, "tab_references"),
        ]
    )

    with tab_overview:
        render_current_strategy(result, lang)
        st.subheader(tr(lang, "factor_status"))
        st.dataframe(result.factor_table, use_container_width=True, hide_index=True)
        render_factor_status(result, lang)

        st.subheader(tr(lang, "performance"))
        d = result.diagnostics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr(lang, "strategy_return"), fmt_pct(d["strategy_total_return"]))
        m2.metric(tr(lang, "buy_hold_return"), fmt_pct(d["buy_hold_total_return"]))
        m3.metric(tr(lang, "strategy_sharpe"), f"{d['strategy_sharpe']:.3f}")
        m4.metric(tr(lang, "max_drawdown"), fmt_pct(d["strategy_max_drawdown"]))

        e1, e2, e3 = st.columns(3)
        e1.metric(tr(lang, "entry_count"), f"{d['entry_count']}")
        e2.metric(tr(lang, "holding_ratio"), fmt_pct(d["holding_ratio"]))
        e3.metric(tr(lang, "total_cost"), fmt_pct(d["total_cost"]))
        t1, t2 = st.columns(2)
        t1.metric(tr(lang, "selected_threshold"), f"{result.selected_threshold:.1f}")
        t2.metric(tr(lang, "selected_sell_threshold"), f"{result.selected_sell_threshold:.1f}")

        if auto_threshold and not result.threshold_table.empty:
            st.subheader(tr(lang, "validation_threshold"))
            st.dataframe(
                result.threshold_table.style.format(
                    {
                        "buy_threshold": "{:.1f}",
                        "sell_threshold": "{:.1f}",
                        "threshold": "{:.1f}",
                        "validation_strategy_return": "{:.4f}",
                        "validation_sharpe": "{:.4f}",
                        "validation_max_drawdown": "{:.4f}",
                        "validation_entry_count": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(tr(lang, "score_contributions"))
        latest = result.score_df.tail(1).T
        latest.columns = ["latest_value"]
        st.dataframe(latest, use_container_width=True)

        contribution_cols = [
            "technical_contribution",
            "fundamental_contribution",
            "chip_contribution",
            "final_score",
        ]
        st.line_chart(result.score_df[contribution_cols], use_container_width=True)

        st.subheader(tr(lang, "equity_curve"))
        curve = result.backtest_df[["strategy_cum", "buy_hold_cum"]].copy()
        curve.columns = ["Triple Barrier strategy", "Buy & Hold"] if lang == "en" else ["Triple Barrier 策略", "買進持有"]
        st.line_chart(curve, use_container_width=True)

    with tab_compare:
        render_strategy_comparison(result, lang)

    with tab_explain:
        render_factor_explanation(result, lang)

    with tab_training_data:
        render_training_data_tab(result, lang)

    with tab_architecture:
        render_architecture_explorer(result, lang)

    with tab_signals:
        st.subheader(tr(lang, "latest_signals"))
        strategy_options = [tr(lang, "manual_score"), tr(lang, "meta_score")]
        strategy_for_table = st.radio(
            tr(lang, "signal_strategy"),
            strategy_options,
            horizontal=True,
        )
        signal_df = result.backtest_df if strategy_for_table == tr(lang, "manual_score") else result.meta_backtest_df
        show_cols = [
            "Close",
            "technical_score",
            "fundamental_score",
            "chip_score",
            "final_score",
            "fundamental_contribution",
            "chip_contribution",
            "target_position",
            "position",
            "strategy_cum",
            "buy_hold_cum",
        ]
        extra_cols = ["meta_probability", "meta_score"]
        visible_cols = [col for col in show_cols + extra_cols if col in signal_df.columns]
        st.dataframe(signal_df[visible_cols].tail(100), use_container_width=True)

        csv_bytes = signal_df.to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            tr(lang, "download"),
            data=csv_bytes,
            file_name="triple_barrier_backtest_result.csv",
            mime="text/csv",
        )

    with tab_references:
        render_references(lang)


if __name__ == "__main__":
    main()
