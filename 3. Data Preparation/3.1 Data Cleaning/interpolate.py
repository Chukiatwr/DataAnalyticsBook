# interpolate.py

import pandas as pd
import numpy as np

# Sample data
data = {
    "Date": pd.date_range("2026-01-01", periods=8, freq="D"),
    "Temp": [25.0, np.nan, 26.5, np.nan, np.nan, 28.0, np.nan, 30.0]
}
df = pd.DataFrame(data).set_index("Date")

print("Raw data:")
print(df)

# Perform interpolation with each method
df["Temp_linear"] = df["Temp"].interpolate(method="linear")
df["Temp_quadratic"] = df["Temp"].interpolate(method="quadratic")   # ต้องใช้ SciPy
df["Temp_cubic"] = df["Temp"].interpolate(method="cubic")           # ต้องใช้ SciPy

print("\nInterpolated results:")
print(df)