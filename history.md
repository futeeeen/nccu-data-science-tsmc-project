# History

## 2026-05-30
- Recorded and prepared the Triple Barrier strategy upgrade for GitHub backup.
- Included the 2026-05-26 Triple Barrier training plan document describing completed and future modeling directions.
- Confirmed the Triple Barrier app now includes statistical-edge threshold selection, trend/timing entry gates, risk-based position sizing, drawdown de-risking, purged/embargo validation, and walk-forward threshold folds.
- Validation: ran `python -m py_compile triple_barrier_strategy\triple_barrier_system.py triple_barrier_strategy\triple_barrier_app.py triple_barrier_strategy\project_glossary.py` and `git diff --check`.

## 2026-05-26
- Added a Triple Barrier training plan documenting statistical edge, anti-chasing filters, drawdown discipline, and future trend/entry model separation.
- Updated the Triple Barrier strategy validation objective to prefer expectancy, Sharpe reward, drawdown control, and lower turnover instead of selecting thresholds only by return.
- Added trade-level statistical edge metrics including win rate, average win/loss, profit factor, payoff ratio, and expectancy.
- Added anti-chasing entry filters that can block high-score entries when RSI, price extension, ATR percentile, or volume-change percentile are too hot.
- Exposed validation objective weights, max validation drawdown limit, and entry filter controls in `triple_barrier_app`.
- Added glossary coverage for expectancy, profit factor, validation objective, and entry discipline filter columns.
- Added trend/timing gates so Triple Barrier entries require both broader trend support and a patient entry setup.
- Added risk-based position sizing plus drawdown-based exposure reduction to avoid all-in/all-out trading.
- Added purged/embargo split handling and walk-forward validation folds for threshold search to reduce label overlap leakage and single-window overfitting.
- Expanded Triple Barrier glossary coverage for trend score, entry timing score, position sizing, drawdown de-risking, purged embargo validation, and walk-forward folds.

## 2026-05-23
- Added a GitHub Actions workflow to deploy `project_site/` automatically to GitHub Pages when changes are pushed to `main`.
- Added `project_site/.nojekyll` so the static HTML site is served directly without Jekyll processing.
- Updated `project_site/README.md` with GitHub Pages deployment instructions and the required Pages source setting.

## 2026-05-21
- Updated the Triple Barrier strategy to support ATR-based dynamic barriers, with default take-profit at `2 * ATR(14)` and stop-loss at `1 * ATR(14)`.
- Added `atr_14`, `atr_14_pct`, `tb_take_profit_pct`, and `tb_stop_loss_pct` so training data previews show the volatility-adjusted barrier design.
- Aligned Triple Barrier labeling and backtest exits so submodels and trading rules use the same dynamic barrier logic.
- Added `class_weight="balanced"` and event-based `sample_weight` to make factor models pay more attention to large gain/loss turning-point samples.
- Added validation ROC AUC, weighted ROC AUC, Average Precision, positive-label rate, and training weight diagnostics to factor model status.
- Expanded the Triple Barrier glossary with ATR, dynamic barrier, sample weight, class weight, ROC AUC, and Average Precision definitions.
- Made sticky Glossary search results scrollable in the technical, multi-factor, and Triple Barrier apps so multiple matches remain readable.

## 2026-05-16
- Made glossary search results language-aware so definitions, usage notes, interpretations, and UI labels follow the page English/Chinese selection.
- Added training-data glossary coverage for the technical, multi-factor, and triple-barrier apps.
- Included OHLCV fields, technical/fundamental/chip features, generated score columns, model metrics, and Triple Barrier event fields in glossary search.
