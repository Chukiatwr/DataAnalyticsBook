# scatterplot.py

# ------------------------------------------------------------
# Demo: Positive, Negative, and Near-Zero Correlation Examples
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.random.seed(123)   # Set random seed for reproducibility
n = 100
# -------------------------------------------
# Create three datasets with different correlations
# 1. Positive correlation (Y increases with X)
x_pos = np.linspace(0, 10, n)
y_pos = 2 * x_pos + np.random.normal(0, 5, n)

# 2. Negative correlation (Y decreases with X)
x_neg = np.linspace(0, 10, n)
y_neg = -2 * x_neg + np.random.normal(0, 5, n)

# 3. Near-zero correlation (No linear relationship)
x_zero = np.linspace(0, 10, n)
y_zero = np.random.normal(0, 10, n)

# Combine into one DataFrame
df = pd.DataFrame({
    'X_Pos': x_pos, 'Y_Pos': y_pos,
    'X_Neg': x_neg, 'Y_Neg': y_neg,
    'X_Zero': x_zero, 'Y_Zero': y_zero
})

# -------------------------------------------
# Compute Pearson correlations
r_pos, _  = pearsonr(df['X_Pos'], df['Y_Pos'])
r_neg, _  = pearsonr(df['X_Neg'], df['Y_Neg'])
r_zero, _ = pearsonr(df['X_Zero'], df['Y_Zero'])

# -------------------------------------------
# Create scatter plots side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Positive correlation plot
sns.scatterplot(x='X_Pos', y='Y_Pos', data=df, ax=axes[0], color='seagreen')
sns.regplot(x='X_Pos', y='Y_Pos', data=df, ax=axes[0], scatter=False, color='red')
axes[0].set_title(f"Positive Correlation\nr = {r_pos:.2f}")

# Near-zero correlation plot
sns.scatterplot(x='X_Zero', y='Y_Zero', data=df, ax=axes[1], color='steelblue')
sns.regplot(x='X_Zero', y='Y_Zero', data=df, ax=axes[1], scatter=False, color='red')
axes[1].set_title(f"Near-Zero Correlation\nr = {r_zero:.2f}")

# Negative correlation plot
sns.scatterplot(x='X_Neg', y='Y_Neg', data=df, ax=axes[2], color='darkorange')
sns.regplot(x='X_Neg', y='Y_Neg', data=df, ax=axes[2], scatter=False, color='red')
axes[2].set_title(f"Negative Correlation\nr = {r_neg:.2f}")

for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.suptitle("Scatter Plots Showing Different Types of Correlation", fontsize=14)
plt.tight_layout()
plt.show()