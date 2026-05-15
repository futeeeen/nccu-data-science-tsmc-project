# Troubleshooting

This file records recurring or important issues found while building the TSMC stock prediction and backtesting project.

Use it as a living project log. When a new issue appears, add a new entry under the most relevant category and keep the same structure:

```md
## Category

### Issue Title

Status:
Date:
Affected area:

#### Symptom
What the user sees.

#### Root Cause
Why it happens.

#### Fix
What changed.

#### Validation
How the fix was checked.

#### Files Changed
Related files.

#### Notes
Any remaining caveats.
```

## Index

| Category | Issue | Status | Affected area |
| --- | --- | --- | --- |
| Modeling / Score Calibration | [Meta Model Scores Are Too Conservative](#meta-model-scores-are-too-conservative) | Fixed | Multi-factor strategy, Triple-barrier strategy |

## Modeling / Score Calibration

### Meta Model Scores Are Too Conservative

Status: Fixed  
Date: 2026-05-15  
Affected area: `multi_factor_strategy`, `triple_barrier_strategy`

#### Symptom

In the multi-factor strategy comparison, the Meta Model strategy showed extremely low or inactive trading behavior.

Example observation:

| strategy | selected_buy_threshold | selected_sell_threshold | entry_count | holding_ratio |
| --- | ---: | ---: | ---: | ---: |
| manual_weighted_score | 80.0 | 62.0 | 6 | 4.18% |
| meta_model_score | 40.0 | 40.0 | 0 | 0.00% |

The Meta Model selected the lowest available threshold but still did not enter any trade on the test period.

#### Root Cause

The original implementation used the Logistic Regression output probability directly as the trading score:

```python
meta_score = predict_proba(...) * 100
```

For financial classification problems, model probabilities are often compressed into a narrow range because the signal-to-noise ratio is low. In one observed run, the validation probabilities were approximately:

```text
0.417 to 0.554
```

After multiplying by 100, the Meta Model score only became:

```text
41.7 to 55.4
```

This score range is technically valid as probability, but it is not well aligned with the app's threshold search range of `40` to `80`. As a result, even the lowest searched threshold could still be too high for many test-period scores, making the strategy look overly conservative or completely inactive.

#### Fix

Keep the raw model probability for diagnostics, but convert the trading score into a validation-percentile score.

The updated design is:

| Column | Meaning |
| --- | --- |
| `meta_probability` | Raw Logistic Regression probability. Useful for checking model confidence and probability compression. |
| `meta_score` | Percentile rank of `meta_probability` against validation-period probabilities, scaled to `0-100`. |
| `final_score` | The Meta Model trading score used by threshold search and backtesting. Same value as `meta_score` for the Meta Model strategy. |

Conceptually:

```python
meta_val_probability = model.predict_proba(meta_val_features)[:, 1]
meta_score = percentile_score(meta_probability, meta_val_probability)
final_score = meta_score
```

This is consistent with how the individual technical, fundamental, and chip factor scores are calibrated in the project.

#### Why This Works Better

The threshold search operates on a `0-100` score scale. A percentile-calibrated score tells the strategy how strong today's Meta Model signal is relative to the validation period.

For example:

| Raw probability | Percentile score interpretation |
| ---: | --- |
| `0.48` | Could be weak, neutral, or strong depending on the validation distribution. |
| `80` | Stronger than roughly 80% of validation-period Meta Model signals. |

This avoids treating a compressed probability range as if it naturally occupied the full `0-100` trading score range.

#### Validation

After applying validation-percentile calibration, a quick local run showed:

```text
meta_score min/median/max: 0.52 / 45.83 / 100.00
meta entries: 67
meta holding ratio: 52.47%
```

The Meta Model no longer stayed inactive simply because its raw probabilities were compressed.

#### Files Changed

- `multi_factor_strategy/multi_factor_system.py`
- `triple_barrier_strategy/triple_barrier_system.py`

#### Notes

This change does not guarantee that the Meta Model will outperform the manual weighted strategy. It only fixes the score-scale mismatch so that the Meta Model can be compared more fairly under the same threshold search framework.

When reviewing Meta Model results, check both:

- `meta_probability`: whether the classifier is inherently uncertain.
- `meta_score`: whether the signal is relatively strong compared with validation-period signals.
