"""Budget allocation optimization."""
import numpy as np
from scipy.optimize import minimize
from .adstock import geometric_adstock
from .saturation import hill_saturation

def optimize_allocation(model, total_budget, constraints=None):
    """Find optimal budget allocation using scipy minimize."""
    n_channels = len(model.channels)
    trace = model.trace
    
    def objective(allocations):
        """Negative total effect (we minimize)."""
        total = 0
        for i, ch in enumerate(model.channels):
            beta = np.mean(trace.posterior[f'{ch}_beta'].values.flatten())
            K = np.mean(trace.posterior[f'{ch}_K'].values.flatten())
            S = np.mean(trace.posterior[f'{ch}_S'].values.flatten())
            effect = beta * hill_saturation(np.array([allocations[i]]), K, S)[0]
            total += effect
        return -total
    
    # constraints: allocations sum to total_budget, all >= 0
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}]
    bounds = [(0, total_budget)] * n_channels
    
    x0 = np.full(n_channels, total_budget / n_channels)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    
    allocation = {ch: val for ch, val in zip(model.channels, result.x)}
    return allocation
