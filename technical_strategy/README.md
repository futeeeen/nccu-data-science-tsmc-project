# Technical Strategy

This folder contains the technical-indicator baseline strategy.

It is copied from the original root-level implementation so the project can keep three strategy families side by side:

- Technical-only model comparison
- Multi-factor strategy
- Triple-barrier strategy experiment

## Files

- `technical_system.py`: data download, technical feature engineering, model training, and backtesting logic
- `technical_app.py`: Streamlit dashboard for the technical-only strategy

## Run

From the project root:

```bash
streamlit run technical_strategy/technical_app.py
```

The original root-level files are kept for backward compatibility:

- `tsmc_stock_system.py`
- `streamlit_app.py`
