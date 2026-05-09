# Multi-Factor Strategy Prototype

This folder is a standalone prototype for the next version of the investment analysis tool.

It separates the signal into three factor groups:

- Technical factor: 20% default weight
- Fundamental factor: 40% default weight
- Chip factor: 40% default weight

Each factor produces its own raw model score. The raw scores are calibrated with validation percentile ranking into a 0-100 score before blending. This prevents a conservative technical model from being permanently underweighted just because its raw probabilities are compressed.

## Quick Start

Run from this folder:

```bash
streamlit run multi_factor_app.py
```

Or from the project root:

```bash
streamlit run multi_factor_strategy/multi_factor_app.py
```

## Fundamental CSV Format

The fundamental CSV currently supports EPS growth:

```csv
date,eps
2021-03-31,5.39
2021-06-30,5.18
2021-09-30,6.03
2021-12-31,6.41
2022-03-31,7.82
```

The system computes:

- `eps_growth_yoy`
- `eps_growth_qoq`

The data is aligned to daily stock data by report date using backward `merge_asof`.

## Chip CSV Format

The chip CSV supports institutional and optional margin/short balance data:

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
