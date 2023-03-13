"""Adstock transformation functions."""
import numpy as np

def geometric_adstock(x, decay_rate, max_lag=8):
    """Apply geometric decay adstock transformation.
    
    Each period's effect carries over to subsequent periods,
    decaying by decay_rate each step.
    """
    weights = decay_rate ** np.arange(max_lag)
    weights = weights / weights.sum()
    
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        for j in range(min(max_lag, i + 1)):
            result[i] += x[i - j] * weights[j]
    return result

def weibull_adstock(x, shape, scale, max_lag=12):
    """Weibull CDF-based adstock (more flexible than geometric)."""
    lags = np.arange(max_lag)
    weights = 1 - np.exp(-(lags / scale) ** shape)
    weights = np.diff(np.concatenate([[0], weights]))
    weights = weights / weights.sum()
    
    result = np.convolve(x, weights[::-1])[:len(x)]
    return result
