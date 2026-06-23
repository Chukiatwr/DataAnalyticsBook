# missing.py

import pandas as pd
import numpy as np

# สร้าง DataFrame จำลองเกี่ยวกับ ESG
data = {
    "Company": ["A Corp", "B Ltd", "C Inc", "D Group", "E Co"],
    "Carbon_Emission": [120, np.nan, 95, 140, np.nan],   # Environmental (E)
    "Employee_Satisfaction": [4.2, 3.8, 4.1, 4.5, 4.0], # Social (S)
    "Board_Diversity": [0.35, 0.40, np.nan, 0.30, 0.25], # Governance (G)
}

df = pd.DataFrame(data)
print("Raw ESG Data:")
print(df)

# Remove rows with any missing values
df_drop_rows = df.dropna()
print("\nAfter dropping rows with missing values:")
print(df_drop_rows)

# Remove columns with any missing values
df_drop_cols = df.dropna(axis=1)
print("\nAfter dropping columns with missing values:")
print(df_drop_cols)
