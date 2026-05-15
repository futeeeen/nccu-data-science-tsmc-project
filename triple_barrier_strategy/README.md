# Triple Barrier Strategy

This folder is the experimental strategy branch for Triple Barrier Labeling.

It starts as a copy of `multi_factor_strategy/` so the current multi-factor model can remain stable while this version changes the training target from next-day direction into a more trading-oriented wave label.

## Goal

The current multi-factor strategy predicts next-day direction:

```text
target = next close > today close
```

This version uses Triple Barrier Labeling:

```text
target = which event happens first within a fixed horizon

+1: take-profit barrier is hit first
 0: neither barrier is hit before the time limit
-1: stop-loss barrier is hit first
```

This should better match the trading goal: catching meaningful upward waves and avoiding important drawdowns.

## Files

- `triple_barrier_system.py`: copied multi-factor pipeline, ready for Triple Barrier target changes
- `triple_barrier_app.py`: copied Streamlit dashboard, independent from the deployed multi-factor app

## Current Method

The training label and trading rule are aligned around the same three barriers:

- Take-profit barrier: default `+8%`
- Stop-loss barrier: default `-5%`
- Vertical time barrier: default `20` trading days

For each date, the system looks forward up to the vertical barrier:

```text
+1: the take-profit barrier is hit first
-1: the stop-loss barrier is hit first
 0: neither barrier is hit before the max holding period
```

The factor models are trained as supervised classifiers on this label. The strategy score uses:

```text
P(label = +1)
```

This means the model is no longer trained on simple next-day direction. It is trained to estimate the probability that a long trade reaches the upside barrier before hitting the downside barrier or timing out.

The backtest uses the same barrier settings:

```text
Enter when final_score >= selected threshold
Exit when take-profit, stop-loss, or max holding period is reached
```

If the dual-threshold strategy is selected, a model-based exit threshold can still close a position earlier, but take-profit, stop-loss, and max holding period remain the core risk controls.

## Run

From the project root:

```bash
streamlit run triple_barrier_strategy/triple_barrier_app.py
```

Or from this folder:

```bash
cd triple_barrier_strategy
streamlit run triple_barrier_app.py
```

If the browser shows a blank page with only the Streamlit toolbar, stop the old Streamlit process and restart it. This usually means the previous session on that port is still running or stuck.

```bash
streamlit run triple_barrier_app.py --server.port 8512
```

## Next Implementation Steps

1. Compare different barrier settings with walk-forward validation.
2. Add volatility-scaled barriers instead of fixed percentage barriers.
3. Add meta-labeling for candidate-entry filtering.
4. Add purged / embargoed validation to reduce label overlap leakage.
5. Compare results against the technical and multi-factor baselines.
