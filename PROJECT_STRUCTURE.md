# Project Structure

This project now keeps three strategy families side by side.

## 1. Technical Strategy

Path:

```text
technical_strategy/
```

Purpose:

- Technical indicators only
- Multi-model comparison
- Baseline supervised classification and backtesting

Run:

```bash
streamlit run technical_strategy/technical_app.py
```

The root-level technical files were removed. Use the files inside `technical_strategy/`.

## 2. Multi-Factor Strategy

Path:

```text
multi_factor_strategy/
```

Purpose:

- Technical factor
- Fundamental factor
- Chip factor
- Manual weighted score
- Meta model comparison
- Current deployed Streamlit Cloud version

Run:

```bash
streamlit run multi_factor_strategy/multi_factor_app.py
```

## 3. Triple Barrier Strategy

Path:

```text
triple_barrier_strategy/
```

Purpose:

- Experimental version copied from `multi_factor_strategy/`
- Intended to replace next-day direction labels with Triple Barrier Labeling
- Used for wave-oriented training and backtesting research

Run:

```bash
streamlit run triple_barrier_strategy/triple_barrier_app.py
```

## Notes

- `multi_factor_strategy/` should remain stable because it is already deployed.
- `triple_barrier_strategy/` can be changed freely for the new labeling method.
- Generated CSV files, local secrets, cache folders, and model artifacts are ignored by `.gitignore`.
