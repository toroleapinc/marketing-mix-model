"""Channel contribution decomposition."""
import numpy as np
import pandas as pd

def decompose_contributions(model, df):
    """Decompose fitted model into per-channel contributions."""
    trace = model.trace
    contributions = {}
    
    for ch in model.channels:
        beta_samples = trace.posterior[f'{ch}_beta'].values.flatten()
        decay_samples = trace.posterior[f'{ch}_decay'].values.flatten()
        
        # use posterior mean
        beta = np.mean(beta_samples)
        decay = np.mean(decay_samples)
        
        from .adstock import geometric_adstock
        from .saturation import hill_saturation
        
        x = df[ch].values
        x_adstock = geometric_adstock(x, decay)
        K = np.mean(trace.posterior[f'{ch}_K'].values.flatten())
        S = np.mean(trace.posterior[f'{ch}_S'].values.flatten())
        x_sat = hill_saturation(x_adstock, K, S)
        
        contributions[ch] = beta * x_sat
    
    return pd.DataFrame(contributions, index=df.index)
