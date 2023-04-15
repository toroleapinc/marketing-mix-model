"""Visualization for MMM results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_waterfall(contributions, save=None):
    """Waterfall chart of channel contributions."""
    channels = list(contributions.keys())
    values = list(contributions.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    ax.barh(channels, values, color=colors)
    ax.set_xlabel('Contribution')
    ax.set_title('Channel Contributions')
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.close()

def plot_response_curves(model, channel, spend_range=None, save=None):
    """Plot response curve for a channel."""
    from .saturation import hill_saturation
    trace = model.trace
    K = np.mean(trace.posterior[f'{channel}_K'].values.flatten())
    S = np.mean(trace.posterior[f'{channel}_S'].values.flatten())
    beta = np.mean(trace.posterior[f'{channel}_beta'].values.flatten())
    
    if spend_range is None:
        spend_range = np.linspace(0, K * 3, 200)
    response = beta * hill_saturation(spend_range, K, S)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(spend_range, response)
    ax.set_xlabel(f'{channel} Spend')
    ax.set_ylabel('Estimated Effect')
    ax.set_title(f'{channel} Response Curve')
    ax.axvline(K, ls='--', color='gray', alpha=0.5, label=f'Half-saturation: {K:.0f}')
    ax.legend()
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150)
    plt.close()

def plot_diagnostics(trace, save=None):
    """Plot MCMC diagnostics."""
    import arviz as az
    az.plot_trace(trace)
    if save: plt.savefig(save, dpi=100)
    plt.close()
