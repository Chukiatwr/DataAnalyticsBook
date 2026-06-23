# fill.py

import pandas as pd

# Create demo dataset with missing values
data = {
    "Date": pd.date_range(start="2026-01-01", periods=7, freq="D"),
    "Temperature": [25, None, None, 28, None, 30, None]
}
df = pd.DataFrame(data)

print("Original Dataset\n", df)

df['Ffill'] = df['Temperature'].ffill()
df['Bfill'] = df['Temperature'].bfill()

print("\nAfter Forward and Backward Filling\n", df)