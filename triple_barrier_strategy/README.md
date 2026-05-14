# Triple Barrier Strategy

This folder is the experimental strategy branch for Triple Barrier Labeling.

It starts as a copy of `multi_factor_strategy/` so the current multi-factor model can remain stable while this version changes the training target from next-day direction into a more trading-oriented wave label.

## Goal

The current multi-factor strategy predicts next-day direction:

```text
target = next close > today close
```

This version will evolve toward Triple Barrier Labeling:

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

## Run

From the project root:

```bash
streamlit run triple_barrier_strategy/triple_barrier_app.py
```

## Next Implementation Steps

1. Add `add_triple_barrier_target()`.
2. Replace the current next-day `target`.
3. Update model training to support three labels or meta-labeling.
4. Update backtesting to exit on take-profit, stop-loss, or max holding period.
5. Compare results against the technical and multi-factor baselines.
