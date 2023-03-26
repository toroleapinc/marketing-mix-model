"""Bayesian Marketing Mix Model."""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from .adstock import geometric_adstock
from .saturation import hill_saturation

class BayesianMMM:
    """Bayesian MMM with adstock and saturation."""
    
    def __init__(self, target='revenue', channels=None, control_vars=None):
        self.target = target
        self.channels = channels or []
        self.control_vars = control_vars or []
        self.trace = None
        self.model = None
    
    def fit(self, df, samples=2000, tune=1000, chains=4):
        """Fit the model using MCMC."""
        y = df[self.target].values
        y_scaled = (y - y.mean()) / y.std()
        
        with pm.Model() as self.model:
            # intercept
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            
            # channel effects
            channel_effects = []
            for ch in self.channels:
                x = df[ch].values
                decay = pm.Beta(f'{ch}_decay', alpha=3, beta=3)
                half_sat = pm.HalfNormal(f'{ch}_K', sigma=np.max(x))
                slope = pm.HalfNormal(f'{ch}_S', sigma=2)
                beta = pm.HalfNormal(f'{ch}_beta', sigma=1)
                
                # apply adstock then saturation
                x_adstock = geometric_adstock(x, decay.eval() if hasattr(decay, 'eval') else 0.5)
                x_saturated = hill_saturation(x_adstock, half_sat, slope)
                channel_effects.append(beta * x_saturated)
            
            # control variables
            for ctrl in self.control_vars:
                beta_ctrl = pm.Normal(f'{ctrl}_beta', mu=0, sigma=1)
                channel_effects.append(beta_ctrl * df[ctrl].values)
            
            mu = intercept + sum(channel_effects) if channel_effects else intercept
            sigma = pm.HalfNormal('sigma', sigma=1)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y_scaled)
            
            self.trace = pm.sample(samples, tune=tune, chains=chains, return_inferencedata=True)
        
        return self
    
    def summary(self):
        """Return model summary."""
        if self.trace is None:
            raise RuntimeError("Model not fitted yet")
        return az.summary(self.trace)
    
    def plot_contributions(self, save=None):
        """Plot channel contribution decomposition."""
        from .plots import plot_waterfall
        contributions = self._decompose()
        plot_waterfall(contributions, save=save)
    
    def _decompose(self):
        """Decompose revenue into channel contributions."""
        if self.trace is None:
            raise RuntimeError("Model not fitted")
        contributions = {}
        for ch in self.channels:
            beta = self.trace.posterior[f'{ch}_beta'].mean().item()
            contributions[ch] = beta
        return contributions
    
    def optimize_budget(self, total_budget, constraints=None):
        """Optimize budget allocation across channels."""
        from .optimizer import optimize_allocation
        return optimize_allocation(self, total_budget, constraints)
