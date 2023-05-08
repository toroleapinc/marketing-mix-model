"""Tests for saturation functions."""
import numpy as np
from mmm.saturation import hill_saturation, logistic_saturation

def test_hill_monotonic():
    x = np.linspace(0, 1000, 100)
    y = hill_saturation(x, K=500, S=2)
    assert all(np.diff(y) >= 0)

def test_hill_half_point():
    y = hill_saturation(np.array([500.0]), K=500, S=2)
    assert abs(y[0] - 0.5) < 0.01

def test_logistic_bounds():
    x = np.linspace(-10, 10, 100)
    y = logistic_saturation(x, L=1.0, k=1.0)
    assert all(y >= 0) and all(y <= 1)
