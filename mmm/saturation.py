"""Saturation (diminishing returns) functions."""
import numpy as np

def hill_saturation(x, K, S):
    """Hill function saturation.
    
    Args:
        x: Spend/impressions
        K: Half-saturation point
        S: Slope parameter
    """
    return 1 / (1 + (x / K) ** (-S))

def logistic_saturation(x, L=1.0, k=0.001, x0=0):
    """Logistic saturation curve."""
    return L / (1 + np.exp(-k * (x - x0)))

def michaelis_menten(x, Vmax, Km):
    """Michaelis-Menten saturation (similar to Hill with S=1)."""
    return Vmax * x / (Km + x)
