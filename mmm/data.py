"""Data loading and validation."""
import pandas as pd
import numpy as np

def load_sample_data():
    """Generate sample marketing data for testing."""
    np.random.seed(42)
    n = 104  # 2 years of weekly data
    
    data = {
        'week': pd.date_range('2021-01-04', periods=n, freq='W'),
        'tv': np.random.lognormal(10, 0.5, n),
        'digital': np.random.lognormal(9, 0.6, n),
        'search': np.random.lognormal(8, 0.4, n),
        'social': np.random.lognormal(7, 0.7, n),
        'price': 50 + np.random.randn(n) * 5,
        'promo': np.random.binomial(1, 0.3, n),
    }
    
    # generate revenue as function of channels + noise
    revenue = (
        50000 +
        0.05 * data['tv'] +
        0.08 * data['digital'] +
        0.12 * data['search'] +
        0.03 * data['social'] -
        200 * data['price'] +
        5000 * data['promo'] +
        np.random.randn(n) * 3000
    )
    data['revenue'] = revenue
    return pd.DataFrame(data)

def validate_data(df, target, channels):
    """Basic data validation."""
    missing = [c for c in [target] + channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[channels].isnull().any().any():
        print("Warning: NaN values in channel data, filling with 0")
        df[channels] = df[channels].fillna(0)
    return df
