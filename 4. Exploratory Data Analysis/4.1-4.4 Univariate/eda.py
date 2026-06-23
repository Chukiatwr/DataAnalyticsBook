# eda.py - Exploratory Data Analysis (EDA) for ESG datasets

# # -----------------------------------------------------
# 📘 ESG EDA Demo: Central Tendency, Dispersion, Shape
# -----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# -------------------------------
# 1️⃣ Create Example ESG Dataset
# -------------------------------
np.random.seed(123)

N = 100
data = {
    'ESG_Score': np.random.normal(loc=70, scale=10, size=N),  # Overall ESG Score
    # Carbon Emission ~ log-normal (right skewed)
    # mean of underlying normal ≈ log(mean) - ½σ² to control scale
    'Carbon_Emission': np.random.lognormal(mean=np.log(120) - 0.5, sigma=0.6, size=N)
}
df = pd.DataFrame(data)
print("Head of Dataset:")
print(df.head(), "\n")

# ------------------------------------------
# 2️⃣ Measures of Central Tendency
# ------------------------------------------
print("=== Measures of Central Tendency ===")
central_tendency = df[['ESG_Score', 'Carbon_Emission']].agg(['mean', 'median'])
print(central_tendency)

# ------------------------------------------
# 3️⃣ Measures of Dispersion
# ------------------------------------------
print("\n=== Measures of Dispersion ===")
dispersion = df[['ESG_Score', 'Carbon_Emission']].agg(['min', 'max', 'std', 'var'])
dispersion.loc['range'] = dispersion.loc['max'] - dispersion.loc['min']
print(dispersion)

# ------------------------------------------
# 4️⃣ Shape of Distribution (Skewness & Kurtosis)
# ------------------------------------------
print("\n=== Shape of Distribution ===")
for col in ['ESG_Score', 'Carbon_Emission']:
    skew_val = skew(df[col])
    kurt_val = kurtosis(df[col], fisher=False)  # Fisher=False → Normal = 3
    print(f"{col}: Skewness = {skew_val:.3f}, Kurtosis = {kurt_val:.3f}")

# ------------------------------------------
# 5️⃣ Visualization
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cols = ['ESG_Score', 'Carbon_Emission']

for ax, col in zip(axes, cols):
    sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
    ax.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
    ax.axvline(df[col].median(), color='green', linestyle='--', label='Median')
    ax.set_title(f"{col}\n(Skew={skew(df[col]):.2f}, Kurt={kurtosis(df[col], fisher=False):.2f})")
    ax.legend()

plt.suptitle("ESG Data Distribution Analysis", fontsize=14)
plt.tight_layout()
plt.show()