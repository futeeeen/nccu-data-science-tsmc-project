# Multi-Factor Strategy Prototype

This folder is a standalone prototype for the next version of the investment analysis tool.

It separates the signal into three factor groups:

- Technical factor: 20% default weight
- Fundamental factor: 40% default weight
- Chip factor: 40% default weight

Each factor produces its own raw model score. The raw scores are calibrated with validation percentile ranking into a 0-100 score before blending. This prevents a conservative technical model from being permanently underweighted just because its raw probabilities are compressed.

The final score threshold can be fixed manually or searched on the validation split. When auto-search is enabled, the threshold with the highest validation strategy return is selected, and the test split is used only for final out-of-sample reporting.

The app also trains a second-stage meta model. This model takes the three calibrated scores (`technical_score`, `fundamental_score`, `chip_score`) as inputs and learns their relationship to next-day direction. This lets you compare:

- Manual weighted score: user-defined weights such as 20/40/40
- Meta model score: learned relationship among the three factor scores

For the meta model, the validation split is divided chronologically: the first half trains the meta model, and the second half selects the threshold. The test split remains reserved for final reporting.

## Quick Start

Install optional FinMind support:

```bash
pip install FinMind
```

Run from this folder:

```bash
streamlit run multi_factor_app.py
```

Or from the project root:

```bash
streamlit run multi_factor_strategy/multi_factor_app.py
```

## Fundamental CSV Format

The app can fetch EPS from FinMind automatically. It uses `taiwan_stock_financial_statement`, filters `type == "EPS"`, and estimates when the statement becomes available to avoid look-ahead bias:

- Q1 period end 03/31 -> available 05/15
- Q2 period end 06/30 -> available 08/14
- Q3 period end 09/30 -> available 11/14
- Q4 period end 12/31 -> available next year 03/31

If you upload CSV manually, use either already-available dates:

```csv
date,eps
2021-05-15,5.39
2021-08-14,5.18
2021-11-14,6.03
```

Or use period-end dates and let the app estimate report availability:

```csv
period_end,eps
2021-03-31,5.39
2021-06-30,5.18
2021-09-30,6.03
```

The system computes:

- `eps_growth_yoy`
- `eps_growth_qoq`

The data is aligned to daily stock data by report date using backward `merge_asof`.

## Chip CSV Format

The app can fetch chip data from FinMind automatically using `taiwan_stock_institutional_investors`. It converts `buy - sell` into:

- `foreign_net_buy`
- `investment_trust_net_buy`
- `dealer_net_buy`

It also attempts to fetch margin/short balance with `taiwan_stock_margin_purchase_short_sale`. If that endpoint is unavailable, margin/short features are treated as missing or neutral.

Feature engineering includes:

- 5-day rolling net buy for foreign investors, investment trusts, dealers, and all institutions
- 20-day rolling total institutional net buy
- 5-day net buy divided by 5-day total trading volume
- consecutive foreign/investment-trust buying days
- 5-day margin/short balance changes when available

If you upload CSV manually, use:

```csv
date,foreign_net_buy,investment_trust_net_buy,dealer_net_buy,margin_balance,short_balance
2024-01-02,1200000,300000,-50000,15000000,200000
2024-01-03,-800000,100000,20000,15100000,220000
```

Required columns:

- `date`
- `foreign_net_buy`
- `investment_trust_net_buy`
- `dealer_net_buy`

Optional columns:

- `margin_balance`
- `short_balance`

The system computes 5-day and 20-day chip features.

## Missing Data Behavior

If no fundamental CSV is uploaded, the fundamental factor becomes neutral at 50.

If no chip CSV is uploaded, the chip factor becomes neutral at 50.

This keeps the app runnable while making it clear which factor data is missing.
