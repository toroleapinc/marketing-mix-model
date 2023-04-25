"""Basic MMM fitting example."""
from mmm import BayesianMMM
from mmm.data import load_sample_data

df = load_sample_data()
print(f"Data shape: {df.shape}")
print(df.head())

model = BayesianMMM(
    target='revenue',
    channels=['tv', 'digital', 'search', 'social'],
    control_vars=['price', 'promo']
)

# This would take a while to run
# model.fit(df, samples=1000, tune=500, chains=2)
# print(model.summary())
# model.plot_contributions(save='contributions.png')

print("Example ready - uncomment fit() to run")
