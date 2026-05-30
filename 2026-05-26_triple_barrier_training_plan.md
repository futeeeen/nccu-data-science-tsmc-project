# 2026-05-26 Triple Barrier Training Plan

## Goal

Adjust `triple_barrier_app` from a simple score-threshold trading model into a more disciplined trading research framework.

The core direction is:

- Optimize for statistical edge, not only prediction accuracy.
- Separate trend judgment from entry timing.
- Add strict drawdown and position-risk control.
- Avoid overtrading and late-stage crowded entries.

## Current System Baseline

The current Triple Barrier version already includes:

- Triple Barrier labels instead of next-day direction labels.
- ATR-based dynamic take-profit and stop-loss barriers.
- Factor models for technical, fundamental, and chip signals.
- Event-weighted training samples for large gain/loss events.
- ROC AUC, weighted ROC AUC, Average Precision, and positive-label rate in factor model status.
- Manual weighted score and Meta model score comparison.
- Validation threshold search and test-period backtest.

This means the next changes can focus on the trading decision layer and evaluation objective rather than replacing the whole pipeline.

## Changes To Do First

### 1. Add Statistical Edge Metrics

Status: implemented.

Add metrics that describe whether trades have positive expectancy.

New metrics:

- `trade_count`
- `win_count`
- `loss_count`
- `win_rate`
- `avg_win`
- `avg_loss`
- `payoff_ratio`
- `profit_factor`
- `expectancy`

Suggested formula:

```text
expectancy = win_rate * avg_win - loss_rate * abs(avg_loss)
```

Why:

Accuracy can be misleading. A model can be right often but lose money if average losses are larger than average wins.

Where to change:

- `triple_barrier_strategy/triple_barrier_system.py`
- `diagnostics()`
- strategy comparison tables in `triple_barrier_app.py`
- glossary definitions for the new metrics

### 2. Change Threshold Search Objective

Status: implemented.

Currently threshold search mostly selects the threshold with the best validation return.

Change it to select by a risk-adjusted statistical edge objective.

Suggested validation objective:

```text
objective =
  expectancy
  + sharpe_weight * strategy_sharpe
  - drawdown_penalty * abs(strategy_max_drawdown)
  - turnover_penalty * entry_count
```

Initial simple default:

```text
objective =
  expectancy
  + 0.10 * strategy_sharpe
  - 0.50 * abs(strategy_max_drawdown)
  - 0.001 * entry_count
```

Why:

This makes threshold selection prefer stable, repeatable edge instead of one lucky high-return validation window.

Where to change:

- `choose_final_threshold()`
- `choose_hysteresis_thresholds()`
- threshold search tables
- sidebar controls for objective weights later if needed

### 3. Add Anti-Chasing Entry Filters

Status: implemented.

Before entering a position, block trades that look too extended or crowded.

Initial filters:

- RSI too high, for example `rsi_14 > 75`
- Price too far above MA20, for example `bias_20 > 0.12`
- ATR percentile too high, for example `atr_14_pct_pct_rank > 85`
- Volume change too extreme, for example `vol_chg_pct_rank > 90`

Expected logic:

```text
can_enter =
  final_score >= buy_threshold
  and not overextended
  and not crowded
```

Why:

This helps the model avoid buying after an emotional price spike, reducing FOMO-style late entries.

Where to change:

- `run_score_backtest()`
- add filter columns to backtest output
- add UI controls for enabling/disabling filters
- add glossary terms for overextended and crowded filters

### 4. Add Basic Drawdown Penalty To Validation

Status: implemented.

Before implementing full position sizing, enforce drawdown discipline in threshold search.

Initial rule:

- If validation max drawdown exceeds a selected limit, penalize heavily or reject the threshold.

Example:

```text
if abs(strategy_max_drawdown) > max_allowed_drawdown:
    objective -= large_penalty
```

Why:

This prevents the optimizer from choosing a strategy that earns high return but requires unacceptable drawdown.

Where to change:

- `choose_final_threshold()`
- `choose_hysteresis_thresholds()`
- sidebar setting: `Max validation drawdown limit`

## Changes To Do Later

### 1. Split Trend Model And Entry Timing Model

Status: implemented as rule-based trend/timing gates.

Current system uses one final score to decide entry.

Future structure:

```text
trend_score >= trend_threshold
entry_score >= entry_threshold
not overextended
not crowded
```

Trend model features:

- MA60 / MA120 slope
- 20-day / 60-day return
- price above long-term moving average
- trend strength indicators

Entry model features:

- pullback to MA20 / MA60
- RSI not overheated
- price distance from ATR bands
- volatility contraction then breakout

Why later:

This requires a larger data and UI change because factor scores, explanations, and backtest logic need to show two model layers.

### 2. Position Sizing And Risk Per Trade

Status: implemented.

Move from all-in / all-out `0` or `1` position to dynamic position sizing.

Suggested formula:

```text
position_size = risk_per_trade / stop_loss_pct
position_size = min(position_size, max_position_size)
```

Example:

```text
risk_per_trade = 1%
stop_loss_pct = 4%
position_size = 25%
```

Additional controls:

- max position size
- max portfolio exposure
- reduce size after drawdown

Why later:

This changes the interpretation of `position`, `holding_ratio`, returns, costs, and signal tables.

### 3. Drawdown-Based De-Risking

Status: implemented.

Reduce position size when the strategy equity curve enters drawdown.

Example:

```text
if current_drawdown < -10%:
    position_size *= 0.5
```

Why later:

This belongs with position sizing and should be tested carefully.

### 4. Walk-Forward Validation

Status: implemented for threshold search by averaging validation objective across chronological folds.

Replace a single train / validation / test split with multiple rolling windows.

Why:

Financial regimes change. A strategy that works in one validation window may fail in another.

Possible implementation:

- rolling train window
- validation window
- test window
- aggregate objective across folds

### 5. Purged And Embargoed Validation

Status: implemented for train / validation / test split boundaries.

Triple Barrier labels can overlap in time, so nearby samples may leak information.

Future improvement:

- Purge overlapping events between train and validation.
- Add embargo days after validation windows.

Why later:

It is methodologically important, but more complex than the first trading-rule improvements.

### 6. Meta-Labeling

Status: partially implemented. The project already has a second-stage meta model over calibrated factor scores; true candidate-entry meta-labeling remains a future research extension.

Use one model to generate candidate entries, then train a second model to decide whether each candidate should actually be taken.

Why:

This matches the idea of separating direction / setup from trade quality.

Possible structure:

```text
primary model: identifies possible long setups
meta model: filters whether the setup has positive expected value
```

## Recommended Implementation Order

1. Add expectancy and trade-level statistics.
2. Use expectancy / Sharpe / drawdown / turnover objective in threshold search.
3. Add anti-chasing filters.
4. Add max drawdown validation constraint.
5. Add UI controls and glossary terms.
6. Later, split trend and entry models.
7. Later, add dynamic position sizing.
8. Later, add walk-forward and purged validation.

## Notes

The safest first version should keep the existing model training pipeline intact and improve the validation objective plus entry rules.

This avoids breaking the current dashboard while moving the strategy closer to:

- statistical edge
- patience
- risk discipline
- reduced emotional / FOMO-style entries
