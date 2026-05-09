from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TECHNICAL_COLS = [
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

FUNDAMENTAL_COLS = [
    "eps",
    "eps_growth_yoy",
    "eps_growth_qoq",
]

CHIP_COLS = [
    "foreign_net_buy_5d",
    "investment_trust_net_buy_5d",
    "dealer_net_buy_5d",
    "total_institutional_net_buy_5d",
    "total_institutional_net_buy_20d",
    "margin_balance_change_5d",
    "short_balance_change_5d",
]


@dataclass
class FactorRunResult:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    factor_table: pd.DataFrame
    score_df: pd.DataFrame
    backtest_df: pd.DataFrame
    diagnostics: dict[str, float]
    data_notes: list[str]


def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Cannot download data for ticker={ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_ratio"] = out["ma_5"] / out["ma_20"]
    out["bias_5"] = (out["Close"] / out["ma_5"]) - 1
    out["bias_20"] = (out["Close"] / out["ma_20"]) - 1
    out["vol_chg"] = out["Volume"].pct_change()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = (ema12 - ema26) / out["Close"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out


def _standardize_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        raise ValueError(f"CSV must include a `{date_col}` column.")
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).set_index(date_col)
    out.index.name = None
    return out


def merge_fundamental_data(price_df: pd.DataFrame, fundamental_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    out = price_df.copy()
    if fundamental_df is None or fundamental_df.empty:
        for col in FUNDAMENTAL_COLS:
            out[col] = np.nan
        return out, "Fundamental CSV not provided. Fundamental factor will use a neutral 50 score."

    fund = _standardize_date_index(fundamental_df)
    if "eps" not in fund.columns:
        raise ValueError("Fundamental CSV must include columns: date, eps.")
    fund["eps_growth_yoy"] = fund["eps"].pct_change(4)
    fund["eps_growth_qoq"] = fund["eps"].pct_change(1)

    fund = fund[["eps", "eps_growth_yoy", "eps_growth_qoq"]].replace([np.inf, -np.inf], np.nan)
    merged = pd.merge_asof(
        out.sort_index(),
        fund.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged, "Fundamental CSV loaded. EPS growth is aligned by report date and forward-filled through merge_asof."


def merge_chip_data(price_df: pd.DataFrame, chip_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    out = price_df.copy()
    if chip_df is None or chip_df.empty:
        for col in CHIP_COLS:
            out[col] = np.nan
        return out, "Chip CSV not provided. Chip factor will use a neutral 50 score."

    chip = _standardize_date_index(chip_df)
    required = ["foreign_net_buy", "investment_trust_net_buy", "dealer_net_buy"]
    missing = [col for col in required if col not in chip.columns]
    if missing:
        raise ValueError(f"Chip CSV missing required columns: {missing}.")

    chip["total_institutional_net_buy"] = (
        chip["foreign_net_buy"] + chip["investment_trust_net_buy"] + chip["dealer_net_buy"]
    )
    chip["foreign_net_buy_5d"] = chip["foreign_net_buy"].rolling(5).sum()
    chip["investment_trust_net_buy_5d"] = chip["investment_trust_net_buy"].rolling(5).sum()
    chip["dealer_net_buy_5d"] = chip["dealer_net_buy"].rolling(5).sum()
    chip["total_institutional_net_buy_5d"] = chip["total_institutional_net_buy"].rolling(5).sum()
    chip["total_institutional_net_buy_20d"] = chip["total_institutional_net_buy"].rolling(20).sum()

    if "margin_balance" in chip.columns:
        chip["margin_balance_change_5d"] = chip["margin_balance"].diff(5)
    else:
        chip["margin_balance_change_5d"] = 0.0
    if "short_balance" in chip.columns:
        chip["short_balance_change_5d"] = chip["short_balance"].diff(5)
    else:
        chip["short_balance_change_5d"] = 0.0

    chip = chip[CHIP_COLS].replace([np.inf, -np.inf], np.nan)
    merged = pd.merge_asof(
        out.sort_index(),
        chip.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged, "Chip CSV loaded. Institutional and margin/short features are aligned by date."


def time_series_split_three(
    df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("Require val_size > 0, test_size > 0, and val_size + test_size < 1.")
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def get_factor_model() -> object:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def get_tree_factor_model() -> object:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
    )


def _has_usable_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(col in df.columns for col in cols) and not df[cols].isna().all().all()


def fit_predict_factor(
    factor_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    neutral_when_missing: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    all_eval = pd.concat([val_df, test_df]).copy()
    if not _has_usable_columns(train_df, feature_cols) or not _has_usable_columns(all_eval, feature_cols):
        if not neutral_when_missing:
            raise ValueError(f"{factor_name} features are not usable.")
        score = pd.DataFrame(
            {
                f"{factor_name}_raw": 0.5,
                f"{factor_name}_score": 50.0,
            },
            index=all_eval.index,
        )
        return score, {"factor": factor_name, "status": "neutral", "validation_auc_proxy": np.nan}

    train_clean = train_df[feature_cols + ["target"]].replace([np.inf, -np.inf], np.nan).dropna()
    val_clean = val_df[feature_cols + ["target"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_clean) < 50 or len(val_clean) < 20 or train_clean["target"].nunique() < 2:
        score = pd.DataFrame(
            {
                f"{factor_name}_raw": 0.5,
                f"{factor_name}_score": 50.0,
            },
            index=all_eval.index,
        )
        return score, {"factor": factor_name, "status": "neutral_insufficient_data", "validation_auc_proxy": np.nan}

    model = get_tree_factor_model() if factor_name in {"fundamental", "chip"} else get_factor_model()
    model.fit(train_clean[feature_cols], train_clean["target"])

    val_raw = pd.Series(model.predict_proba(val_clean[feature_cols])[:, 1], index=val_clean.index)
    eval_features = all_eval[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    eval_raw = pd.Series(model.predict_proba(eval_features)[:, 1], index=all_eval.index)
    eval_score = percentile_score(eval_raw, val_raw)

    score = pd.DataFrame(
        {
            f"{factor_name}_raw": eval_raw,
            f"{factor_name}_score": eval_score,
        },
        index=all_eval.index,
    )
    return score, {"factor": factor_name, "status": "model", "validation_raw_mean": float(val_raw.mean())}


def percentile_score(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = reference.dropna().sort_values().to_numpy()
    if len(ref) == 0:
        return pd.Series(50.0, index=values.index)
    ranks = np.searchsorted(ref, values.to_numpy(), side="right") / len(ref)
    return pd.Series(ranks * 100.0, index=values.index).clip(0, 100)


def combine_scores(
    score_df: pd.DataFrame,
    technical_weight: float = 0.2,
    fundamental_weight: float = 0.4,
    chip_weight: float = 0.4,
) -> pd.DataFrame:
    out = score_df.copy()
    out["final_score"] = (
        technical_weight * out["technical_score"]
        + fundamental_weight * out["fundamental_score"]
        + chip_weight * out["chip_score"]
    )
    out["technical_contribution"] = technical_weight * out["technical_score"]
    out["fundamental_contribution"] = fundamental_weight * out["fundamental_score"]
    out["chip_contribution"] = chip_weight * out["chip_score"]
    return out


def run_score_backtest(df: pd.DataFrame, score_df: pd.DataFrame, threshold: float, cost_bps: float) -> pd.DataFrame:
    bt = df.loc[score_df.index].copy()
    bt = bt.join(score_df, how="left")
    bt["target_position"] = (bt["final_score"] >= threshold).astype(float)
    bt["position"] = bt["target_position"].shift(1).fillna(0.0)
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)
    bt["position_chg"] = bt["position"].diff().abs().fillna(bt["position"])
    bt["cost"] = bt["position_chg"] * (cost_bps / 10000.0)
    bt["strategy_ret"] = bt["position"] * bt["asset_ret"] - bt["cost"]
    bt["buy_hold_ret"] = bt["asset_ret"]
    bt["strategy_cum"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_cum"] = (1 + bt["buy_hold_ret"]).cumprod()
    return bt


def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    return float((cum / roll_max - 1).min())


def sharpe_ratio(daily_returns: pd.Series, annual_factor: int = 252) -> float:
    std = daily_returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(annual_factor))


def diagnostics(bt: pd.DataFrame) -> dict[str, float]:
    return {
        "strategy_total_return": float(bt["strategy_cum"].iloc[-1] - 1),
        "buy_hold_total_return": float(bt["buy_hold_cum"].iloc[-1] - 1),
        "strategy_max_drawdown": max_drawdown(bt["strategy_cum"]),
        "buy_hold_max_drawdown": max_drawdown(bt["buy_hold_cum"]),
        "strategy_sharpe": sharpe_ratio(bt["strategy_ret"]),
        "buy_hold_sharpe": sharpe_ratio(bt["buy_hold_ret"]),
        "entry_count": int(((bt["position"] == 1) & (bt["position"].shift(1).fillna(0) == 0)).sum()),
        "holding_ratio": float(bt["position"].mean()),
        "total_cost": float(bt["cost"].sum()),
    }


def run_multi_factor_pipeline(
    ticker: str,
    start: str,
    end: str,
    fundamental_df: Optional[pd.DataFrame] = None,
    chip_df: Optional[pd.DataFrame] = None,
    val_size: float = 0.2,
    test_size: float = 0.2,
    final_threshold: float = 60.0,
    cost_bps: float = 10.0,
    technical_weight: float = 0.2,
    fundamental_weight: float = 0.4,
    chip_weight: float = 0.4,
) -> FactorRunResult:
    notes: list[str] = []
    raw = download_price_data(ticker, start, end)
    df = add_technical_features(raw)
    df, fund_note = merge_fundamental_data(df, fundamental_df)
    notes.append(fund_note)
    df, chip_note = merge_chip_data(df, chip_df)
    notes.append(chip_note)
    df = add_target(df).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TECHNICAL_COLS + ["target"]).copy()

    train_df, val_df, test_df = time_series_split_three(df, val_size=val_size, test_size=test_size)
    score_parts = []
    factor_rows = []
    for factor_name, cols in [
        ("technical", TECHNICAL_COLS),
        ("fundamental", FUNDAMENTAL_COLS),
        ("chip", CHIP_COLS),
    ]:
        score, row = fit_predict_factor(factor_name, train_df, val_df, test_df, cols)
        score_parts.append(score)
        factor_rows.append(row)

    score_df = pd.concat(score_parts, axis=1)
    score_df = combine_scores(score_df, technical_weight, fundamental_weight, chip_weight)
    backtest_df = run_score_backtest(df, score_df.loc[test_df.index], final_threshold, cost_bps)
    return FactorRunResult(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        factor_table=pd.DataFrame(factor_rows),
        score_df=score_df,
        backtest_df=backtest_df,
        diagnostics=diagnostics(backtest_df),
        data_notes=notes,
    )
