# Marketing Mix Model

Bayesian Marketing Mix Model for measuring channel effectiveness and optimizing budget allocation. Built with PyMC for MCMC inference.

## Overview

Estimates the incremental contribution of each marketing channel (TV, digital, search, social, etc.) to revenue, accounting for:
- **Adstock**: Carryover/decay effects of advertising
- **Saturation**: Diminishing returns at high spend levels
- **Seasonality**: Weekly and monthly patterns
- **Control variables**: Pricing, promotions, holidays

The model outputs posterior distributions of channel ROI, enabling data-driven budget reallocation.

## Install

```bash
pip install -e .
```

## Usage

```python
from mmm import BayesianMMM
from mmm.data import load_sample_data

df = load_sample_data()
model = BayesianMMM(target='revenue', channels=['tv', 'digital', 'search', 'social'])
model.fit(df, samples=2000)
model.plot_contributions()
model.optimize_budget(total_budget=1000000)
```

## Project Structure

```
mmm/
├── model.py          # Bayesian MMM with PyMC
├── adstock.py        # Geometric and Weibull adstock transforms
├── saturation.py     # Hill and logistic saturation curves
├── decomposition.py  # Channel contribution decomposition
├── optimizer.py      # Budget allocation optimization
├── plots.py          # Response curves, waterfall, diagnostics
└── data.py           # Data loading and validation
```

See `examples/` for notebooks.
