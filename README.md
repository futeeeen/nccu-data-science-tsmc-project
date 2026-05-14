# TSMC Stock Prediction and Backtesting System

This project contains three strategy versions for TSMC stock prediction and backtesting.

## Project Layout

| Version | Path | Purpose | Run command |
| --- | --- | --- | --- |
| Technical baseline | `technical_strategy/` | Technical indicators only, multi-model comparison, baseline backtest | `streamlit run technical_strategy/technical_app.py` |
| Multi-factor strategy | `multi_factor_strategy/` | Technical + fundamental + chip factors, weighted score and meta model | `streamlit run multi_factor_strategy/multi_factor_app.py` |
| Triple-barrier experiment | `triple_barrier_strategy/` | Experimental copy for Triple Barrier Labeling and wave-oriented targets | `streamlit run triple_barrier_strategy/triple_barrier_app.py` |

See `PROJECT_STRUCTURE.md` for the full folder map.

## Quick Start

1. Clone the repository and enter the folder.

```bash
git clone https://github.com/futeeeen/nccu-data-science-tsmc-project.git
cd nccu-data-science-tsmc-project
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run one strategy app.

```bash
streamlit run technical_strategy/technical_app.py
```

```bash
streamlit run multi_factor_strategy/multi_factor_app.py
```

```bash
streamlit run triple_barrier_strategy/triple_barrier_app.py
```

## Technical Strategy

Path:

```text
technical_strategy/
```

Purpose:

- Technical indicators only
- Multi-model comparison
- Train / validation / test split by time order
- Backtest model signals against Buy & Hold

Main files:

- `technical_strategy/technical_system.py`
- `technical_strategy/technical_app.py`

## Multi-Factor Strategy

Path:

```text
multi_factor_strategy/
```

Purpose:

- Technical factor
- Fundamental factor
- Chip factor
- Manual weighted score
- Second-stage meta model comparison
- Factor contribution explanation by date

Main files:

- `multi_factor_strategy/multi_factor_system.py`
- `multi_factor_strategy/multi_factor_app.py`

### FinMind Token

For Streamlit Cloud deployment, add this secret so FinMind data works reliably:

```toml
FINMIND_TOKEN = "your_finmind_token"
```

The app reads the token in this order:

- Sidebar token input
- Streamlit Cloud secret `FINMIND_TOKEN`
- Environment variable `FINMIND_TOKEN`

## Triple Barrier Strategy

Path:

```text
triple_barrier_strategy/
```

Purpose:

- Experimental version copied from the multi-factor strategy
- Reserved for Triple Barrier Labeling
- Intended to replace next-day direction labels with wave-oriented labels

Planned target:

```text
+1: take-profit barrier is hit first
 0: neither barrier is hit before max holding period
-1: stop-loss barrier is hit first
```

Main files:

- `triple_barrier_strategy/triple_barrier_system.py`
- `triple_barrier_strategy/triple_barrier_app.py`

## Local Secrets

Local secrets should stay in:

```text
.streamlit/secrets.toml
```

This file is ignored by Git and must not be pushed.

## Original Project PDF

The original project specification is kept at the root:

```text
資科_台積電股票價格預測與交易策略分析系統.pdf
```
