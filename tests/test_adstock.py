"""Tests for adstock transforms."""
import numpy as np
from mmm.adstock import geometric_adstock, weibull_adstock

def test_geometric_adstock_decay():
    x = np.zeros(20)
    x[0] = 100
    result = geometric_adstock(x, decay_rate=0.5, max_lag=8)
    assert result[0] > result[1] > result[2]
    assert result[0] > 50  # first period should retain most

def test_geometric_adstock_no_decay():
    x = np.array([100, 0, 0, 0])
    result = geometric_adstock(x, decay_rate=0.0, max_lag=4)
    assert result[0] == 100
    assert result[1] == 0

def test_weibull_shape():
    x = np.zeros(20)
    x[0] = 100
    result = weibull_adstock(x, shape=2, scale=3, max_lag=12)
    assert len(result) == 20
