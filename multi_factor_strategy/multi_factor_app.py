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

    if not run:
        st.info("Upload factor data if available, set weights, and click Run multi-factor analysis.")
        return
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

    with st.expander("Data Notes", expanded=False):
        for note in result.data_notes:
            if "not provided" in note or "failed" in note:
                st.warning(note)
            else:
                st.success(note)

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

    st.subheader("Latest Test Signals")
    show_cols = [
        "Close",
        "technical_score",
        "fundamental_score",
        "chip_score",
        "final_score",
        "target_position",
        "position",
        "strategy_cum",
        "buy_hold_cum",
    ]
    st.dataframe(result.backtest_df[show_cols].tail(100), use_container_width=True)

    csv_bytes = result.backtest_df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "Download multi_factor_backtest_result.csv",
        data=csv_bytes,
        file_name="multi_factor_backtest_result.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
