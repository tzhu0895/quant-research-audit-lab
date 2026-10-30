# Quant Research Audit Lab

A compact Python toolkit for statistical diagnostics on time-ordered research data. It demonstrates a few reusable checks using synthetic observations; it does not include a trading strategy, real data or a claim of predictive performance.

## What is included?

| Tool | Purpose |
| --- | --- |
| Rank information coefficient | Spearman correlation between a cross-sectional signal and a forward return. |
| Moving-block bootstrap | Percentile confidence interval for a sample mean while retaining within-block order. |
| Newey–West standard error | Heteroskedasticity- and autocorrelation-consistent standard error for a mean. |
| Purged time split | Exclude training observations whose label windows reach into validation. |
| Synthetic plot | Generate a simple cumulative-observation figure without market data. |

## Setup and run

Python 3.10+ is recommended. In a virtual environment:

```bash
pip install -r requirements.txt
python -m unittest -v
python demo.py
```

The demo prints a synthetic bootstrap interval, HAC standard error and purged split sizes, and creates `synthetic_report.png`. Results depend on the specified random seed and are illustrative only.

## Example API

```python
import numpy as np
from audit import moving_block_mean_ci, newey_west_mean_se, purged_split, rank_ic

series = np.array([0.01, -0.01, 0.02, 0.00, 0.01, -0.02])
print(moving_block_mean_ci(series, block_length=2, iterations=1000, seed=42))
print(newey_west_mean_se(series, lags=1))
print(rank_ic([1, 2, 3], [3, 2, 1]))
train, valid = purged_split(np.arange(20), np.arange(20) + 2, validation_start=15)
```

`rank_ic` omits non-finite pairs and returns NaN if fewer than three remain or rankings are constant. The bootstrap uses non-circular blocks drawn from the observed sequence; it returns a percentile interval for the mean. The HAC function applies Bartlett weights. The split retains training rows only when their label end is strictly before the validation boundary.

## Interpretation and limitations

These are building blocks, not a complete research audit. A positive interval does not prove a strategy will work, and the tools do not establish absence of leakage or overfitting. Block length, lag selection, dependence, multiple testing, data vintages and validation design require separate judgment. The split does not automatically add an embargo after validation or perform nested model selection.

## Files

- `audit.py` — statistical functions.
- `demo.py` — synthetic example and chart.
- `test_audit.py` — numerical and input checks.
- `requirements.txt` — runtime dependencies.

This repository was assembled with AI-assisted development and human-directed review.
