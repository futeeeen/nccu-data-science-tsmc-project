from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    "foreign_net_buy_5d_ratio",
    "investment_trust_net_buy_5d_ratio",
    "dealer_net_buy_5d_ratio",
    "total_institutional_net_buy_5d_ratio",
    "foreign_consecutive_buy_days",
    "investment_trust_consecutive_buy_days",
    "margin_balance_change_5d",
    "short_balance_change_5d",
]

META_SCORE_COLS = ["technical_score", "fundamental_score", "chip_score"]


def taiwan_stock_id_from_ticker(ticker: str) -> str:
    return ticker.split(".")[0]


def is_taiwan_stock_ticker(ticker: str) -> bool:
    stock_id = taiwan_stock_id_from_ticker(ticker.strip())
    return ticker.upper().strip().endswith(".TW") or stock_id.isdigit()


@dataclass
class FactorRunResult:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    factor_table: pd.DataFrame
    factor_data_summary: pd.DataFrame
    score_df: pd.DataFrame
    backtest_df: pd.DataFrame
    diagnostics: dict[str, float]
    meta_backtest_df: pd.DataFrame
    meta_diagnostics: dict[str, float]
    meta_model_summary: pd.DataFrame
    strategy_comparison: pd.DataFrame
    data_notes: list[str]
    selected_threshold: float
    selected_sell_threshold: float
    threshold_table: pd.DataFrame
    meta_selected_threshold: float
    meta_selected_sell_threshold: float
    meta_threshold_table: pd.DataFrame
    strategy_mode: str
    take_profit_pct: float
    stop_loss_pct: float
    max_holding_days: int


@lru_cache(maxsize=64)
def fetch_finmind_price_data(stock_id: str, start_date: str, end_date: str, token: str | None = None) -> pd.DataFrame:
    try:
        from FinMind.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("FinMind is not installed. Run `pip install FinMind`.") from exc

    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    price = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
    if price.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = price.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").set_index("date")
    out = out.rename(
        columns={
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
            "Trading_Volume": "Volume",
        }
    )
    return out[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()


def download_price_data(ticker: str, start: str, end: str, finmind_token: str | None = None) -> pd.DataFrame:
    yfinance_error = ""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        df = pd.DataFrame()
        yfinance_error = str(exc)

    if df.empty and is_taiwan_stock_ticker(ticker):
        stock_id = taiwan_stock_id_from_ticker(ticker)
        fallback = fetch_finmind_price_data(stock_id, start, end, finmind_token)
        if not fallback.empty:
            fallback.attrs["price_source"] = "FinMind TaiwanStockPrice"
            if yfinance_error:
                fallback.attrs["price_note"] = f"Yahoo Finance price download failed first: {yfinance_error}"
            else:
                fallback.attrs["price_note"] = "Yahoo Finance returned 0 rows, so FinMind price data was used."
            return fallback

    if df.empty:
        detail = f" Yahoo Finance error: {yfinance_error}" if yfinance_error else ""
        raise ValueError(f"Cannot download data for ticker={ticker}.{detail}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.attrs["price_source"] = "Yahoo Finance"
    return out


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


def add_triple_barrier_target(
    df: pd.DataFrame,
    take_profit_pct: float = 0.08,
    stop_loss_pct: float = 0.05,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    out = df.copy()
    labels: list[float] = []
    events: list[str | None] = []
    event_dates: list[pd.Timestamp | pd.NaT] = []
    event_returns: list[float] = []
    holding_days: list[float] = []
    closes = out["Close"].to_numpy()
    highs = out["High"].to_numpy()
    lows = out["Low"].to_numpy()
    index = out.index

    for i in range(len(out)):
        if i + max_holding_days >= len(out):
            labels.append(np.nan)
            events.append(None)
            event_dates.append(pd.NaT)
            event_returns.append(np.nan)
            holding_days.append(np.nan)
            continue

        entry_price = closes[i]
        upper = entry_price * (1.0 + take_profit_pct)
        lower = entry_price * (1.0 - stop_loss_pct)
        label = 0
        event = "vertical_barrier"
        event_idx = i + max_holding_days

        for j in range(i + 1, i + max_holding_days + 1):
            hit_upper = highs[j] >= upper
            hit_lower = lows[j] <= lower
            if hit_lower:
                label = -1
                event = "stop_loss"
                event_idx = j
                break
            if hit_upper:
                label = 1
                event = "take_profit"
                event_idx = j
                break

        labels.append(label)
        events.append(event)
        event_dates.append(index[event_idx])
        event_returns.append(float(closes[event_idx] / entry_price - 1.0))
        holding_days.append(float(event_idx - i))

    out["tb_label"] = labels
    out["target"] = out["tb_label"]
    out["tb_event"] = events
    out["tb_event_date"] = event_dates
    out["tb_event_return"] = event_returns
    out["tb_holding_days"] = holding_days
    return out


def _standardize_date_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        raise ValueError(f"CSV must include a `{date_col}` column.")
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).set_index(date_col)
    out.index.name = None
    return out


def estimate_financial_available_date(period_end: pd.Timestamp) -> pd.Timestamp:
    period_end = pd.Timestamp(period_end)
    year = period_end.year
    month = period_end.month
    if month == 3:
        return pd.Timestamp(year=year, month=5, day=15)
    if month == 6:
        return pd.Timestamp(year=year, month=8, day=14)
    if month == 9:
        return pd.Timestamp(year=year, month=11, day=14)
    if month == 12:
        return pd.Timestamp(year=year + 1, month=3, day=31)
    return period_end + pd.Timedelta(days=45)


@lru_cache(maxsize=64)
def fetch_finmind_fundamental_eps(
    stock_id: str, start_date: str, end_date: str, token: str | None = None
) -> pd.DataFrame:
    try:
        from FinMind.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("FinMind is not installed. Run `pip install FinMind`.") from exc

    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    financial = dl.taiwan_stock_financial_statement(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )
    if financial.empty:
        return pd.DataFrame(columns=["date", "period_end", "eps"])

    eps = financial[financial["type"].eq("EPS")].copy()
    eps["period_end"] = pd.to_datetime(eps["date"])
    eps["date"] = eps["period_end"].map(estimate_financial_available_date)
    eps["eps"] = pd.to_numeric(eps["value"], errors="coerce")
    eps = eps.sort_values("period_end")
    return eps[["date", "period_end", "eps"]].dropna(subset=["eps"])


def _pivot_finmind_institutional(institutional: pd.DataFrame) -> pd.DataFrame:
    if institutional.empty:
        return pd.DataFrame(columns=["date", "foreign_net_buy", "investment_trust_net_buy", "dealer_net_buy"])

    df = institutional.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["net_buy"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")
    pivot = df.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum").fillna(0.0)

    out = pd.DataFrame(index=pivot.index)
    out["foreign_net_buy"] = pivot.get("Foreign_Investor", 0.0) + pivot.get("Foreign_Dealer_Self", 0.0)
    out["investment_trust_net_buy"] = pivot.get("Investment_Trust", 0.0)
    out["dealer_net_buy"] = pivot.get("Dealer_self", 0.0) + pivot.get("Dealer_Hedging", 0.0)
    out = out.reset_index()
    return out


def _extract_margin_short(margin_df: pd.DataFrame) -> pd.DataFrame:
    if margin_df.empty:
        return pd.DataFrame(columns=["date", "margin_balance", "short_balance"])

    df = margin_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    margin_candidates = [
        "MarginPurchaseTodayBalance",
        "MarginPurchaseYesterdayBalance",
        "MarginPurchaseBuy",
    ]
    short_candidates = [
        "ShortSaleTodayBalance",
        "ShortSaleYesterdayBalance",
        "ShortSaleSell",
    ]
    out = pd.DataFrame({"date": df["date"]})
    for target, candidates in [("margin_balance", margin_candidates), ("short_balance", short_candidates)]:
        match = next((col for col in candidates if col in df.columns), None)
        out[target] = pd.to_numeric(df[match], errors="coerce") if match else np.nan
    return out.drop_duplicates("date")


@lru_cache(maxsize=64)
def fetch_finmind_chip_data(
    stock_id: str, start_date: str, end_date: str, token: str | None = None
) -> pd.DataFrame:
    try:
        from FinMind.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("FinMind is not installed. Run `pip install FinMind`.") from exc

    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    institutional = dl.taiwan_stock_institutional_investors(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )
    chip = _pivot_finmind_institutional(institutional)

    try:
        margin = dl.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        margin = _extract_margin_short(margin)
        chip = chip.merge(margin, on="date", how="left")
    except Exception:
        chip["margin_balance"] = np.nan
        chip["short_balance"] = np.nan
    return chip


def merge_fundamental_data(price_df: pd.DataFrame, fundamental_df: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    out = price_df.copy()
    if fundamental_df is None or fundamental_df.empty:
        for col in FUNDAMENTAL_COLS:
            out[col] = np.nan
        return out, "Fundamental CSV not provided. Fundamental factor will use a neutral 50 score."

    fund_input = fundamental_df.copy()
    if "period_end" in fund_input.columns:
        fund_input["period_end"] = pd.to_datetime(fund_input["period_end"])
        fund_input["date"] = fund_input["period_end"].map(estimate_financial_available_date)
    fund = _standardize_date_index(fund_input)
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
    chip["foreign_consecutive_buy_days"] = consecutive_positive_days(chip["foreign_net_buy"])
    chip["investment_trust_consecutive_buy_days"] = consecutive_positive_days(chip["investment_trust_net_buy"])

    if "margin_balance" in chip.columns:
        chip["margin_balance_change_5d"] = chip["margin_balance"].diff(5)
    else:
        chip["margin_balance_change_5d"] = 0.0
    if "short_balance" in chip.columns:
        chip["short_balance_change_5d"] = chip["short_balance"].diff(5)
    else:
        chip["short_balance_change_5d"] = 0.0

    merge_cols = [
        "foreign_net_buy_5d",
        "investment_trust_net_buy_5d",
        "dealer_net_buy_5d",
        "total_institutional_net_buy_5d",
        "total_institutional_net_buy_20d",
        "foreign_consecutive_buy_days",
        "investment_trust_consecutive_buy_days",
        "margin_balance_change_5d",
        "short_balance_change_5d",
    ]
    chip = chip[merge_cols].replace([np.inf, -np.inf], np.nan)
    merged = pd.merge_asof(
        out.sort_index(),
        chip.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    vol_5d = merged["Volume"].rolling(5).sum().replace(0, np.nan)
    merged["foreign_net_buy_5d_ratio"] = merged["foreign_net_buy_5d"] / vol_5d
    merged["investment_trust_net_buy_5d_ratio"] = merged["investment_trust_net_buy_5d"] / vol_5d
    merged["dealer_net_buy_5d_ratio"] = merged["dealer_net_buy_5d"] / vol_5d
    merged["total_institutional_net_buy_5d_ratio"] = merged["total_institutional_net_buy_5d"] / vol_5d
    return merged, "Chip CSV loaded. Institutional and margin/short features are aligned by date."


def consecutive_positive_days(series: pd.Series) -> pd.Series:
    count = 0
    values = []
    for value in series.fillna(0):
        count = count + 1 if value > 0 else 0
        values.append(count)
    return pd.Series(values, index=series.index)


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


def positive_class_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


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

    val_raw = pd.Series(positive_class_proba(model, val_clean[feature_cols]), index=val_clean.index)
    eval_features = all_eval[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    eval_raw = pd.Series(positive_class_proba(model, eval_features), index=all_eval.index)
    eval_score = percentile_score(eval_raw, val_raw)

    score = pd.DataFrame(
        {
            f"{factor_name}_raw": eval_raw,
            f"{factor_name}_score": eval_score,
        },
        index=all_eval.index,
    )
    return score, {"factor": factor_name, "status": "model", "validation_raw_mean": float(val_raw.mean())}


def factor_data_summary(
    factor_name: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    status: str,
) -> dict[str, float | int | str]:
    def usable_rows(part: pd.DataFrame) -> int:
        if not all(col in part.columns for col in feature_cols):
            return 0
        return int(part[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all").shape[0])

    coverage_base = len(train_df) + len(val_df) + len(test_df)
    usable_total = usable_rows(train_df) + usable_rows(val_df) + usable_rows(test_df)
    return {
        "factor": factor_name,
        "status": status,
        "feature_count": len(feature_cols),
        "train_usable_rows": usable_rows(train_df),
        "validation_usable_rows": usable_rows(val_df),
        "test_usable_rows": usable_rows(test_df),
        "overall_coverage": usable_total / coverage_base if coverage_base else 0.0,
    }


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


def build_target_position(
    scores: pd.Series,
    buy_threshold: float,
    sell_threshold: float | None = None,
    strategy_mode: str = "single_threshold",
) -> pd.Series:
    if strategy_mode == "hysteresis":
        exit_threshold = buy_threshold if sell_threshold is None else sell_threshold
        position = 0.0
        positions = []
        for score in scores:
            if position == 0.0 and score >= buy_threshold:
                position = 1.0
            elif position == 1.0 and score <= exit_threshold:
                position = 0.0
            positions.append(position)
        return pd.Series(positions, index=scores.index, dtype=float)
    return (scores >= buy_threshold).astype(float)


def run_score_backtest(
    df: pd.DataFrame,
    score_df: pd.DataFrame,
    threshold: float,
    cost_bps: float,
    sell_threshold: float | None = None,
    strategy_mode: str = "single_threshold",
    take_profit_pct: float = 0.08,
    stop_loss_pct: float = 0.05,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    bt = df.loc[score_df.index].copy()
    bt = bt.join(score_df, how="left")
    bt["buy_threshold"] = float(threshold)
    bt["sell_threshold"] = float(threshold if sell_threshold is None else sell_threshold)
    bt["strategy_mode"] = strategy_mode
    bt["asset_ret"] = bt["Close"].pct_change().fillna(0.0)
    bt["target_position"] = 0.0
    bt["position"] = 0.0
    bt["tb_trade_event"] = ""
    bt["tb_days_held_live"] = 0

    position = 0.0
    entry_price = 0.0
    entry_pos = -1
    scores = bt["final_score"].fillna(0.0).to_numpy()
    highs = bt["High"].to_numpy()
    lows = bt["Low"].to_numpy()
    closes = bt["Close"].to_numpy()
    positions = []
    targets = []
    events = []
    held_days = []
    exit_threshold = threshold if sell_threshold is None else sell_threshold

    for i, score in enumerate(scores):
        event = ""
        if position == 1.0:
            days_held = i - entry_pos
            hit_stop = lows[i] <= entry_price * (1.0 - stop_loss_pct)
            hit_profit = highs[i] >= entry_price * (1.0 + take_profit_pct)
            hit_time = days_held >= max_holding_days
            hit_model_exit = strategy_mode == "hysteresis" and score <= exit_threshold
            if hit_stop:
                position = 0.0
                event = "stop_loss"
            elif hit_profit:
                position = 0.0
                event = "take_profit"
            elif hit_time:
                position = 0.0
                event = "vertical_barrier"
            elif hit_model_exit:
                position = 0.0
                event = "model_exit"

        if position == 0.0 and score >= threshold and event == "":
            position = 1.0
            entry_price = closes[i]
            entry_pos = i
            event = "entry"

        targets.append(position)
        positions.append(position)
        events.append(event)
        held_days.append(float(i - entry_pos) if position == 1.0 else 0.0)

    bt["target_position"] = targets
    bt["position"] = pd.Series(positions, index=bt.index).shift(1).fillna(0.0)
    bt["tb_trade_event"] = events
    bt["tb_days_held_live"] = held_days
    bt["take_profit_pct"] = take_profit_pct
    bt["stop_loss_pct"] = stop_loss_pct
    bt["max_holding_days"] = max_holding_days
    bt["position_chg"] = bt["position"].diff().abs().fillna(bt["position"])
    bt["cost"] = bt["position_chg"] * (cost_bps / 10000.0)
    bt["strategy_ret"] = bt["position"] * bt["asset_ret"] - bt["cost"]
    bt["buy_hold_ret"] = bt["asset_ret"]
    bt["strategy_cum"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buy_hold_cum"] = (1 + bt["buy_hold_ret"]).cumprod()
    return bt


def add_feature_percentiles(
    backtest_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    out = backtest_df.copy()
    for col in feature_cols:
        if col in out.columns and col in val_df.columns:
            out[f"{col}_pct_rank"] = percentile_score(out[col], val_df[col])
    return out


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


def choose_final_threshold(
    df: pd.DataFrame,
    val_score_df: pd.DataFrame,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    cost_bps: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_days: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best_threshold = threshold_min
    best_sell_threshold = threshold_min
    best_return = -np.inf
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    for threshold in thresholds:
        bt = run_score_backtest(
            df,
            val_score_df,
            float(threshold),
            cost_bps,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
        )
        d = diagnostics(bt)
        rows.append(
            {
                "threshold": float(threshold),
                "validation_strategy_return": d["strategy_total_return"],
                "validation_sharpe": d["strategy_sharpe"],
                "validation_max_drawdown": d["strategy_max_drawdown"],
                "validation_entry_count": d["entry_count"],
            }
        )
        if d["strategy_total_return"] > best_return:
            best_return = d["strategy_total_return"]
            best_threshold = float(threshold)
            best_sell_threshold = float(threshold)
    return best_threshold, best_sell_threshold, pd.DataFrame(rows)


def choose_hysteresis_thresholds(
    df: pd.DataFrame,
    val_score_df: pd.DataFrame,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    cost_bps: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_days: int,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best_buy_threshold = threshold_min
    best_sell_threshold = threshold_min
    best_return = -np.inf
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    for buy_threshold in thresholds:
        for sell_threshold in thresholds:
            if sell_threshold > buy_threshold:
                continue
            bt = run_score_backtest(
                df,
                val_score_df,
                float(buy_threshold),
                cost_bps,
                sell_threshold=float(sell_threshold),
                strategy_mode="hysteresis",
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                max_holding_days=max_holding_days,
            )
            d = diagnostics(bt)
            rows.append(
                {
                    "buy_threshold": float(buy_threshold),
                    "sell_threshold": float(sell_threshold),
                    "validation_strategy_return": d["strategy_total_return"],
                    "validation_sharpe": d["strategy_sharpe"],
                    "validation_max_drawdown": d["strategy_max_drawdown"],
                    "validation_entry_count": d["entry_count"],
                    "validation_holding_ratio": d["holding_ratio"],
                }
            )
            if d["strategy_total_return"] > best_return:
                best_return = d["strategy_total_return"]
                best_buy_threshold = float(buy_threshold)
                best_sell_threshold = float(sell_threshold)
    return best_buy_threshold, best_sell_threshold, pd.DataFrame(rows)


def split_validation_for_meta(val_df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    split_idx = max(1, len(val_df) // 2)
    return val_df.index[:split_idx], val_df.index[split_idx:]


def run_meta_model_strategy(
    df: pd.DataFrame,
    score_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    cost_bps: float,
    strategy_mode: str,
    fixed_sell_threshold: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_days: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, float, float, pd.DataFrame]:
    meta_train_idx, meta_val_idx = split_validation_for_meta(val_df)
    meta_train = score_df.loc[meta_train_idx, META_SCORE_COLS].join(df["target"]).dropna()
    meta_val = score_df.loc[meta_val_idx, META_SCORE_COLS].join(df["target"]).dropna()

    if len(meta_train) < 30 or len(meta_val) < 20 or meta_train["target"].nunique() < 2:
        test_score = score_df.loc[test_df.index, META_SCORE_COLS].copy()
        test_score["final_score"] = score_df.loc[test_df.index, "final_score"]
        bt = run_score_backtest(
            df,
            test_score,
            50.0,
            cost_bps,
            sell_threshold=fixed_sell_threshold if strategy_mode == "hysteresis" else None,
            strategy_mode=strategy_mode,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
        )
        summary = pd.DataFrame(
            [{"item": "meta_model_status", "value": "neutral_insufficient_validation_data"}]
        )
        return bt, diagnostics(bt), summary, 50.0, fixed_sell_threshold, pd.DataFrame()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    model.fit(meta_train[META_SCORE_COLS], meta_train["target"])

    meta_val_probability = pd.Series(
        positive_class_proba(model, meta_val[META_SCORE_COLS]),
        index=meta_val.index,
    )
    meta_val_score = score_df.loc[meta_val.index, META_SCORE_COLS].copy()
    meta_val_score["meta_probability"] = meta_val_probability
    meta_val_score["meta_score"] = percentile_score(meta_val_probability, meta_val_probability)
    meta_val_score["final_score"] = meta_val_score["meta_score"]
    if strategy_mode == "hysteresis":
        selected_threshold, selected_sell_threshold, threshold_table = choose_hysteresis_thresholds(
            df=df,
            val_score_df=meta_val_score,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            cost_bps=cost_bps,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
        )
    else:
        selected_threshold, selected_sell_threshold, threshold_table = choose_final_threshold(
            df=df,
            val_score_df=meta_val_score,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            cost_bps=cost_bps,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_holding_days=max_holding_days,
        )

    test_features = score_df.loc[test_df.index, META_SCORE_COLS].copy()
    test_features["meta_probability"] = pd.Series(
        positive_class_proba(model, test_features[META_SCORE_COLS]),
        index=test_features.index,
    )
    test_features["meta_score"] = percentile_score(test_features["meta_probability"], meta_val_probability)
    test_score = test_features[META_SCORE_COLS + ["meta_probability", "meta_score"]].copy()
    test_score["final_score"] = test_features["meta_score"]
    bt = run_score_backtest(
        df,
        test_score,
        selected_threshold,
        cost_bps,
        sell_threshold=selected_sell_threshold if strategy_mode == "hysteresis" else None,
        strategy_mode=strategy_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )

    clf = model.named_steps["clf"]
    clf_classes = list(clf.classes_)
    coef_idx = clf_classes.index(1) if 1 in clf_classes and clf.coef_.shape[0] > 1 else 0
    coef = clf.coef_[coef_idx]
    summary_rows = [
        {"item": "meta_model_status", "value": "model"},
        {"item": "meta_train_rows", "value": len(meta_train)},
        {"item": "meta_validation_rows", "value": len(meta_val)},
        {"item": "meta_selected_buy_threshold", "value": selected_threshold},
        {"item": "meta_selected_sell_threshold", "value": selected_sell_threshold},
        {"item": "strategy_mode", "value": strategy_mode},
        {"item": "meta_validation_probability_mean", "value": float(meta_val_probability.mean())},
        {"item": "meta_validation_probability_min", "value": float(meta_val_probability.min())},
        {"item": "meta_validation_probability_max", "value": float(meta_val_probability.max())},
        {"item": "meta_score_calibration", "value": "validation_percentile_rank"},
    ]
    for feature, value in zip(META_SCORE_COLS, coef):
        summary_rows.append({"item": f"coef_{feature}", "value": float(value)})
    return bt, diagnostics(bt), pd.DataFrame(summary_rows), selected_threshold, selected_sell_threshold, threshold_table


def build_strategy_comparison(
    weighted_diagnostics: dict[str, float],
    meta_diagnostics: dict[str, float],
    selected_threshold: float,
    selected_sell_threshold: float,
    meta_selected_threshold: float,
    meta_selected_sell_threshold: float,
    strategy_mode: str,
) -> pd.DataFrame:
    rows = []
    for name, d, buy_threshold, sell_threshold in [
        ("manual_weighted_score", weighted_diagnostics, selected_threshold, selected_sell_threshold),
        ("meta_model_score", meta_diagnostics, meta_selected_threshold, meta_selected_sell_threshold),
    ]:
        rows.append(
            {
                "strategy": name,
                "strategy_mode": strategy_mode,
                "selected_buy_threshold": buy_threshold,
                "selected_sell_threshold": sell_threshold,
                "strategy_total_return": d["strategy_total_return"],
                "buy_hold_total_return": d["buy_hold_total_return"],
                "strategy_max_drawdown": d["strategy_max_drawdown"],
                "strategy_sharpe": d["strategy_sharpe"],
                "entry_count": d["entry_count"],
                "holding_ratio": d["holding_ratio"],
            }
        )
    return pd.DataFrame(rows)


def run_multi_factor_pipeline(
    ticker: str,
    start: str,
    end: str,
    fundamental_df: Optional[pd.DataFrame] = None,
    chip_df: Optional[pd.DataFrame] = None,
    use_finmind: bool = False,
    finmind_token: str | None = None,
    val_size: float = 0.2,
    test_size: float = 0.2,
    final_threshold: float = 60.0,
    sell_threshold: float = 45.0,
    auto_threshold: bool = True,
    threshold_min: float = 40.0,
    threshold_max: float = 80.0,
    threshold_step: float = 2.0,
    cost_bps: float = 10.0,
    technical_weight: float = 0.2,
    fundamental_weight: float = 0.4,
    chip_weight: float = 0.4,
    strategy_mode: str = "single_threshold",
    take_profit_pct: float = 0.08,
    stop_loss_pct: float = 0.05,
    max_holding_days: int = 20,
) -> FactorRunResult:
    notes: list[str] = []
    raw = download_price_data(ticker, start, end, finmind_token)
    price_source = raw.attrs.get("price_source")
    price_note = raw.attrs.get("price_note")
    if price_source:
        notes.append(f"Price data source: {price_source}.")
    if price_note:
        notes.append(str(price_note))
    df = add_technical_features(raw)
    stock_id = taiwan_stock_id_from_ticker(ticker)
    can_use_finmind = is_taiwan_stock_ticker(ticker)
    if use_finmind and not can_use_finmind:
        notes.append(
            f"FinMind auto-fetch skipped for {ticker}. FinMind Taiwan stock data only supports Taiwan stock tickers, "
            "so uploaded CSV data will be used if provided; otherwise fundamental and chip factors stay neutral."
        )
    if use_finmind and can_use_finmind and fundamental_df is None:
        try:
            fundamental_df = fetch_finmind_fundamental_eps(stock_id, start, end, finmind_token)
            if fundamental_df.empty:
                notes.append(
                    "FinMind fundamental fetch returned 0 rows. Check token/quota, stock id, and date range; "
                    "fundamental factor will use a neutral 50 score."
                )
            else:
                notes.append(
                    f"FinMind fundamental EPS data fetched ({len(fundamental_df)} rows). "
                    "EPS is shifted to estimated report availability dates."
                )
        except Exception as exc:
            notes.append(f"FinMind fundamental fetch failed: {exc}")
    if use_finmind and can_use_finmind and chip_df is None:
        try:
            chip_df = fetch_finmind_chip_data(stock_id, start, end, finmind_token)
            if chip_df.empty:
                notes.append(
                    "FinMind chip fetch returned 0 rows. Check token/quota, stock id, and date range; "
                    "chip factor will use a neutral 50 score."
                )
            else:
                notes.append(
                    f"FinMind chip data fetched ({len(chip_df)} rows). "
                    "Institutional net buy is transformed into rolling features."
                )
        except Exception as exc:
            notes.append(f"FinMind chip fetch failed: {exc}")
    df, fund_note = merge_fundamental_data(df, fundamental_df)
    notes.append(fund_note)
    df, chip_note = merge_chip_data(df, chip_df)
    notes.append(chip_note)
    df = add_triple_barrier_target(
        df,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    ).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TECHNICAL_COLS + ["target"]).copy()
    label_counts = df["target"].value_counts(dropna=True).sort_index().to_dict()
    notes.append(
        "Triple Barrier labels applied: "
        f"take_profit={take_profit_pct:.2%}, stop_loss={stop_loss_pct:.2%}, "
        f"max_holding_days={max_holding_days}, label_counts={label_counts}."
    )

    train_df, val_df, test_df = time_series_split_three(df, val_size=val_size, test_size=test_size)
    score_parts = []
    factor_rows = []
    summary_rows = []
    for factor_name, cols in [
        ("technical", TECHNICAL_COLS),
        ("fundamental", FUNDAMENTAL_COLS),
        ("chip", CHIP_COLS),
    ]:
        score, row = fit_predict_factor(factor_name, train_df, val_df, test_df, cols)
        score_parts.append(score)
        factor_rows.append(row)
        summary_rows.append(factor_data_summary(factor_name, cols, train_df, val_df, test_df, str(row["status"])))

    score_df = pd.concat(score_parts, axis=1)
    score_df = combine_scores(score_df, technical_weight, fundamental_weight, chip_weight)
    threshold_table = pd.DataFrame()
    selected_threshold = final_threshold
    selected_sell_threshold = sell_threshold if strategy_mode == "hysteresis" else final_threshold
    if auto_threshold:
        if strategy_mode == "hysteresis":
            selected_threshold, selected_sell_threshold, threshold_table = choose_hysteresis_thresholds(
                df=df,
                val_score_df=score_df.loc[val_df.index],
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                cost_bps=cost_bps,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                max_holding_days=max_holding_days,
            )
        else:
            selected_threshold, selected_sell_threshold, threshold_table = choose_final_threshold(
                df=df,
                val_score_df=score_df.loc[val_df.index],
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                cost_bps=cost_bps,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                max_holding_days=max_holding_days,
            )
    backtest_df = run_score_backtest(
        df,
        score_df.loc[test_df.index],
        selected_threshold,
        cost_bps,
        sell_threshold=selected_sell_threshold if strategy_mode == "hysteresis" else None,
        strategy_mode=strategy_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )
    backtest_df = add_feature_percentiles(backtest_df, val_df, TECHNICAL_COLS + FUNDAMENTAL_COLS + CHIP_COLS)
    meta_backtest_df, meta_d, meta_summary, meta_threshold, meta_sell_threshold, meta_threshold_table = run_meta_model_strategy(
        df=df,
        score_df=score_df,
        val_df=val_df,
        test_df=test_df,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        cost_bps=cost_bps,
        strategy_mode=strategy_mode,
        fixed_sell_threshold=sell_threshold,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )
    meta_backtest_df = add_feature_percentiles(meta_backtest_df, val_df, TECHNICAL_COLS + FUNDAMENTAL_COLS + CHIP_COLS)
    weighted_d = diagnostics(backtest_df)
    return FactorRunResult(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        factor_table=pd.DataFrame(factor_rows),
        factor_data_summary=pd.DataFrame(summary_rows),
        score_df=score_df,
        backtest_df=backtest_df,
        diagnostics=weighted_d,
        meta_backtest_df=meta_backtest_df,
        meta_diagnostics=meta_d,
        meta_model_summary=meta_summary,
        strategy_comparison=build_strategy_comparison(
            weighted_d,
            meta_d,
            selected_threshold,
            selected_sell_threshold,
            meta_threshold,
            meta_sell_threshold,
            strategy_mode,
        ),
        data_notes=notes,
        selected_threshold=selected_threshold,
        selected_sell_threshold=selected_sell_threshold,
        threshold_table=threshold_table,
        meta_selected_threshold=meta_threshold,
        meta_selected_sell_threshold=meta_sell_threshold,
        meta_threshold_table=meta_threshold_table,
        strategy_mode=strategy_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_holding_days=max_holding_days,
    )
