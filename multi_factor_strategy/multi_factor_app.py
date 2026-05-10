from __future__ import annotations

import pandas as pd
import streamlit as st

from multi_factor_system import run_multi_factor_pipeline


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def read_optional_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def normalize_weights(technical: float, fundamental: float, chip: float) -> tuple[float, float, float]:
    total = technical + fundamental + chip
    if total == 0:
        return 0.2, 0.4, 0.4
    return technical / total, fundamental / total, chip / total


def render_factor_status(result) -> None:
    st.subheader("Factor Training Overview")
    status_map = result.factor_data_summary.set_index("factor").to_dict("index")
    cols = st.columns(3)
    for idx, factor in enumerate(["technical", "fundamental", "chip"]):
        info = status_map.get(factor, {})
        status = str(info.get("status", "unknown"))
        label = status.replace("_", " ").title()
        cols[idx].metric(
            f"{factor.title()} factor",
            label,
            f"{int(info.get('feature_count', 0))} features",
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


def describe_score_level(score: float) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "moderately high"
    if score >= 45:
        return "neutral"
    if score >= 25:
        return "moderately low"
    return "low"


def factor_explanation_table(row: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    records = []
    for col in feature_cols:
        pct_col = f"{col}_pct_rank"
        if col in row.index and pct_col in row.index and pd.notna(row[col]):
            records.append(
                {
                    "feature": col,
                    "value": row[col],
                    "validation_percentile": row[pct_col],
                    "interpretation": describe_score_level(float(row[pct_col])),
                }
            )
    return pd.DataFrame(records).sort_values("validation_percentile", ascending=False)


def render_factor_explanation(result) -> None:
    st.subheader("Factor Contribution Explanation")
    st.caption("Select a test date to inspect why fundamental and chip contributions are high or low.")

    strategy = st.radio(
        "Strategy to explain",
        ["Manual weighted score", "Meta model score"],
        horizontal=True,
    )
    bt = result.backtest_df.copy() if strategy == "Manual weighted score" else result.meta_backtest_df.copy()
    dates = list(bt.index)
    selected_date = st.selectbox(
        "Select test date",
        dates,
        index=len(dates) - 1,
        format_func=lambda x: str(pd.to_datetime(x).date()),
    )
    row = bt.loc[selected_date]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fundamental score", f"{row['fundamental_score']:.1f}")
    c2.metric("Fundamental contribution", f"{row['fundamental_contribution']:.1f}")
    c3.metric("Chip score", f"{row['chip_score']:.1f}")
    c4.metric("Chip contribution", f"{row['chip_contribution']:.1f}")

    st.markdown(
        f"""
        **Why this date looks this way**

        - Fundamental contribution is `{row['fundamental_contribution']:.1f}` because the calibrated
          fundamental score is `{row['fundamental_score']:.1f}` and the manual fundamental weight is applied after calibration.
        - Chip contribution is `{row['chip_contribution']:.1f}` because the calibrated chip score is
          `{row['chip_score']:.1f}` and the manual chip weight is applied after calibration.
        - If `Meta model score` is selected, the final trading score is learned from the three calibrated factor scores,
          while the contribution columns still show the manual weighted decomposition for reference.
        - Percentiles below compare the selected date's feature value against the validation-period distribution.
        """
    )

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

    f_col, c_col = st.columns(2)
    with f_col:
        st.markdown("### Fundamental drivers")
        f_table = factor_explanation_table(row, fundamental_features)
        if f_table.empty:
            st.warning("No fundamental driver data is available for this date.")
        else:
            st.dataframe(
                f_table.style.format({"value": "{:.4f}", "validation_percentile": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
            )
    with c_col:
        st.markdown("### Chip drivers")
        c_table = factor_explanation_table(row, chip_features)
        if c_table.empty:
            st.warning("No chip driver data is available for this date.")
        else:
            st.dataframe(
                c_table.style.format({"value": "{:.4f}", "validation_percentile": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Contribution Over Time")
    chart_cols = ["fundamental_contribution", "chip_contribution", "final_score"]
    if "meta_score" in bt.columns:
        chart_cols.append("meta_score")
    st.line_chart(bt[chart_cols], use_container_width=True)


def render_strategy_comparison(result) -> None:
    st.subheader("Manual Weighted Score vs Meta Model Score")
    st.caption(
        "Manual weighted score uses your chosen weights. Meta model learns how the three factor scores interact."
    )
    st.dataframe(
        result.strategy_comparison.style.format(
            {
                "selected_threshold": "{:.1f}",
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
        st.markdown("### Meta Model Training Summary")
        st.dataframe(result.meta_model_summary, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Meta Threshold Search")
        if result.meta_threshold_table.empty:
            st.warning("Meta threshold search was skipped because validation data was insufficient.")
        else:
            st.dataframe(
                result.meta_threshold_table.style.format(
                    {
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

    st.subheader("Equity Curve Comparison")
    comparison_curve = pd.DataFrame(
        {
            "Manual weighted strategy": result.backtest_df["strategy_cum"],
            "Meta model strategy": result.meta_backtest_df["strategy_cum"],
            "Buy & Hold": result.backtest_df["buy_hold_cum"],
        }
    )
    st.line_chart(comparison_curve, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="TSMC Multi-Factor Strategy", layout="wide")
    st.title("TSMC Multi-Factor Investment Analysis")
    st.caption("Technical, fundamental, and chip factors are scored separately, calibrated to 0-100, then blended.")

    with st.sidebar:
        st.header("Data")
        ticker = st.text_input("Ticker", value="2330.TW")
        start = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
        end = st.date_input("End date", value=pd.Timestamp("2026-01-01"))
        use_finmind = st.checkbox("Fetch fundamental/chip data from FinMind", value=True)
        finmind_token = st.text_input("FinMind token (optional)", value="", type="password")
        fundamental_file = st.file_uploader("Fundamental CSV (date, eps)", type=["csv"])
        chip_file = st.file_uploader(
            "Chip CSV (date, foreign_net_buy, investment_trust_net_buy, dealer_net_buy)",
            type=["csv"],
        )

        st.header("Weights")
        technical_w = st.slider("Technical weight", 0.0, 1.0, 0.2, 0.05)
        fundamental_w = st.slider("Fundamental weight", 0.0, 1.0, 0.4, 0.05)
        chip_w = st.slider("Chip weight", 0.0, 1.0, 0.4, 0.05)
        auto_threshold = st.checkbox("Auto-search final threshold on validation", value=True)
        final_threshold = st.slider("Fixed final score threshold", 0.0, 100.0, 60.0, 1.0)
        threshold_min = st.slider("Threshold search min", 0.0, 100.0, 40.0, 1.0)
        threshold_max = st.slider("Threshold search max", 0.0, 100.0, 80.0, 1.0)
        threshold_step = st.select_slider("Threshold search step", options=[1.0, 2.0, 5.0], value=2.0)
        cost_bps = st.number_input("Trading cost (bps per position change)", 0.0, 100.0, 10.0, 1.0)
        run = st.button("Run multi-factor analysis", type="primary")

    st.subheader("Score Design")
    st.markdown(
        """
        - Each factor model produces a raw probability-like score.
        - Raw scores are calibrated with validation percentile ranking into a 0-100 factor score.
        - Final score = technical score * technical weight + fundamental score * fundamental weight + chip score * chip weight.
        - This keeps the technical factor at the intended weight even if its raw model probabilities are conservative.
        - EPS is aligned by estimated report availability date to reduce look-ahead bias.
        - A second-stage meta model is also trained on the three calibrated factor scores, so you can compare learned relationships against manual weights.
        """
    )

    st.subheader("Data Source")
    st.markdown(
        """
        The app can fetch EPS and institutional investor data from FinMind. If FinMind is unavailable,
        you can upload CSV files manually. Uploaded CSV files take priority over FinMind data.
        """
    )

    st.subheader("Expected CSV Format")
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
            st.error("Threshold search min must be less than or equal to threshold search max.")
            return

        tech_weight, fund_weight, chip_weight = normalize_weights(technical_w, fundamental_w, chip_w)
        fundamental_df = read_optional_csv(fundamental_file)
        chip_df = read_optional_csv(chip_file)

        with st.spinner("Running multi-factor pipeline..."):
            result = run_multi_factor_pipeline(
                ticker=ticker,
                start=str(start),
                end=str(end),
                fundamental_df=fundamental_df,
                chip_df=chip_df,
                use_finmind=use_finmind,
                finmind_token=finmind_token.strip() or None,
                final_threshold=final_threshold,
                auto_threshold=auto_threshold,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                cost_bps=cost_bps,
                technical_weight=tech_weight,
                fundamental_weight=fund_weight,
                chip_weight=chip_weight,
            )
        st.session_state["multi_factor_result"] = result
        st.session_state["multi_factor_params"] = {
            "ticker": ticker,
            "start": str(start),
            "end": str(end),
            "technical_weight": tech_weight,
            "fundamental_weight": fund_weight,
            "chip_weight": chip_weight,
            "auto_threshold": auto_threshold,
            "cost_bps": cost_bps,
        }

    result = st.session_state.get("multi_factor_result")
    if result is None:
        st.info("Upload factor data if available, set weights, and click Run multi-factor analysis.")
        return

    params = st.session_state.get("multi_factor_params", {})
    st.caption(
        "Showing cached result from the last run"
        + (f": {params.get('ticker')} ({params.get('start')} to {params.get('end')})." if params else ".")
    )

    with st.expander("Data Notes", expanded=False):
        for note in result.data_notes:
            if "not provided" in note or "failed" in note:
                st.warning(note)
            else:
                st.success(note)

    tab_overview, tab_compare, tab_explain, tab_signals = st.tabs(
        ["Overview", "Strategy Comparison", "Factor Explanation", "Signals & Download"]
    )

    with tab_overview:
        st.subheader("Factor Model Status")
        st.dataframe(result.factor_table, use_container_width=True, hide_index=True)
        render_factor_status(result)

        st.subheader("Performance")
        d = result.diagnostics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategy return", fmt_pct(d["strategy_total_return"]))
        m2.metric("Buy & Hold return", fmt_pct(d["buy_hold_total_return"]))
        m3.metric("Strategy Sharpe", f"{d['strategy_sharpe']:.3f}")
        m4.metric("Max drawdown", fmt_pct(d["strategy_max_drawdown"]))

        e1, e2, e3 = st.columns(3)
        e1.metric("Entry count", f"{d['entry_count']}")
        e2.metric("Holding ratio", fmt_pct(d["holding_ratio"]))
        e3.metric("Total trading cost", fmt_pct(d["total_cost"]))
        st.metric("Selected final score threshold", f"{result.selected_threshold:.1f}")

        if auto_threshold and not result.threshold_table.empty:
            st.subheader("Validation Threshold Search")
            st.dataframe(
                result.threshold_table.style.format(
                    {
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

        st.subheader("Score Contributions")
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

        st.subheader("Equity Curve")
        curve = result.backtest_df[["strategy_cum", "buy_hold_cum"]].copy()
        curve.columns = ["Multi-factor strategy", "Buy & Hold"]
        st.line_chart(curve, use_container_width=True)

    with tab_compare:
        render_strategy_comparison(result)

    with tab_explain:
        render_factor_explanation(result)

    with tab_signals:
        st.subheader("Latest Test Signals")
        strategy_for_table = st.radio(
            "Signal table strategy",
            ["Manual weighted score", "Meta model score"],
            horizontal=True,
        )
        signal_df = result.backtest_df if strategy_for_table == "Manual weighted score" else result.meta_backtest_df
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
            "Download multi_factor_backtest_result.csv",
            data=csv_bytes,
            file_name="multi_factor_backtest_result.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
