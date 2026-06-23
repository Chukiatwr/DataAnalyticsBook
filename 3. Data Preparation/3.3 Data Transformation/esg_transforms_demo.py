# esg_transforms_demo.py
#
# Data transformation and feature engineering demo for ESG data
# - Standardization (z-score)
# - Variable transforms (log, sqrt)
# - Feature engineering (averages, ratios)
# - Binning (quantiles)
# - One-hot encoding (categorical variables)

import numpy as np
import pandas as pd

# Set the display option
pd.set_option('display.float_format', '{:.3f}'.format)
# Set the option to display ALL columns
pd.set_option('display.max_columns', None)

np.random.seed(123)
n = 10
sectors = ['Energy', 'Financials', 'Industrials']

df = pd.DataFrame({
    "cid"   : [f"C{i:02d}" for i in range(1, n+1)],
    "sector": np.random.choice(sectors, size=n, p=[0.25,0.35,0.40]),
    "E_score": np.clip(np.random.normal(60, 12, n), 0, 100),
    "S_score": np.clip(np.random.normal(55, 12, n), 0, 100),
    "G_score": np.clip(np.random.normal(65, 10, n), 0, 100),
    "revenue": np.abs(np.random.lognormal(mean=4.2, sigma=0.8, size=n)),
    "carbon" : np.abs(np.random.lognormal(mean=3.6, sigma=0.5, size=n))
})
print("Original Data:\n", df[::2], "\n")
df.to_csv("esg_data.csv", index=False, float_format='%.4f')

# Standardization (z-score)
for col in ["E_score","S_score","G_score"]:
    mu, sd = df[col].mean(), df[col].std(ddof=0)
    df[f"{col}_z"] = (df[col] - mu) / (sd if sd != 0 else 1.0)

# Variable transforms
df["revenue_log"] = np.log1p(df["revenue"])
df["intensity_sqrt"] = np.sqrt(df["carbon"])

# Feature engineering
df["ESG_avg"] = df[["E_score","S_score","G_score"]].mean(axis=1)
df["ESG_weighted"] = 0.5*df["E_score"] + 0.3*df["S_score"] + 0.2*df["G_score"]
df["carbon_per_rev"] = df["carbon"] / (df["revenue"] + 1e-9)

# Binning
df["size_bucket"] = pd.qcut(df["revenue"], q=3, labels=["small","mid","large"])

# One-hot encode sector
df = pd.concat([df, pd.get_dummies(df["sector"], prefix="sector", dtype=int)], axis=1)

# Save transformed data and print preview
df.to_csv("esg_transformed.csv", index=False, float_format='%.4f')
print("Saved esg_transformed.csv with", len(df), "rows and", df.shape[1], "columns")
print("Preview:\n", df[::2])