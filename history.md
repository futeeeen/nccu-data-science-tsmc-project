# History

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
