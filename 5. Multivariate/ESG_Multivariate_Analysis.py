# multivariate_analysis.py
#
# This code demonstrates multivariate analysis techniques on a synthetic ESG dataset.
# Techniques include PCA, Factor Analysis, Clustering, and Regression.
# The dataset simulates various ESG metrics and their relationships with financial outcomes.
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, silhouette_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None) # Show all columns in DataFrame
pd.set_option('display.width', 100)        # Set display width for better readability
pd.set_option('display.precision', 4)      # Set decimal precision for floats

np.random.seed(42)  # For reproducibility

# ==== Synthesize ESG-like dataset ====
n = 600  # number of firms

# Environmental metrics
emissions_intensity = np.random.gamma(2.0, 50.0, n)                 # lower is better
energy_efficiency   = np.random.normal(65, 10, n).clip(20, 95)      # higher is better
renewables_share    = np.random.beta(2, 5, n) * 100                 # percentage 0-100

# Social metrics
injury_rate         = np.random.gamma(2.5, 1.0, n)                  # lower is better
diversity_index     = np.random.normal(55, 12, n).clip(0,100)       # percentage-like score
training_hours      = np.random.normal(20, 5, n).clip(0,60)         # hours/employee

# Governance metrics
board_independence  = np.random.normal(58, 15, n).clip(0,100)       # % independent
exec_comp_alignment = np.random.normal(60, 10, n).clip(0,100)       # higher ~ pay tied to LT value
controversies       = np.random.poisson(0.7, n)                     # count of negative events

# Latent ESG score (composed with noise; lower E harms, higher S/G helps)
esg_score = (
    -0.25 * (emissions_intensity / emissions_intensity.max()) * 100
    + 0.20 * energy_efficiency
    + 0.15 * renewables_share
    - 0.30 * injury_rate * 2
    + 0.20 * diversity_index
    + 0.10 * training_hours
    + 0.20 * board_independence
    + 0.20 * exec_comp_alignment
    - 5.0  * controversies
    + np.random.normal(0, 5, n)
)

# Financial outcome variables (toy relationships + noise)
revenue_growth   = 3 + 0.03 * esg_score + np.random.normal(0, 2.0, n)   # %
cost_of_capital  = 12 - 0.02 * esg_score + np.random.normal(0, 1.0, n)  # %
volatility       = 35 - 0.05 * esg_score + np.random.normal(0, 3.0, n)  # annualized %, lower is better

# Classification target for LDA (controversy risk tiers)
risk_tier = np.where(controversies >= 2, "High", np.where(controversies == 1, "Medium", "Low"))

df = pd.DataFrame({
    "emissions_intensity": emissions_intensity,
    "energy_efficiency": energy_efficiency,
    "renewables_share": renewables_share,
    "injury_rate": injury_rate,
    "diversity_index": diversity_index,
    "training_hours": training_hours,
    "board_independence": board_independence,
    "exec_comp_alignment": exec_comp_alignment,
    "controversies": controversies,
    "esg_score": esg_score,
    "revenue_growth": revenue_growth,
    "cost_of_capital": cost_of_capital,
    "volatility": volatility,
    "risk_tier": risk_tier
})

cols = ["emissions_intensity", "injury_rate", "training_hours", "exec_comp_alignment"]
print(df[cols].head())
print("=" * 60)
print(df[cols].describe())
print("=" * 60)
print(df["risk_tier"].value_counts())
print("=" * 60)

"""
Creates a 2x2 grid showing basic information about four important ESG and financial variables
"""
# Select the 4 most important variables for analysis
important_vars = ['esg_score', 'emissions_intensity', 'revenue_growth', 'volatility']

# Create the 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.suptitle('ESG Data Analysis - Key Variables Overview', fontsize=18, fontweight='bold', y=0.98)
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for idx, var in enumerate(important_vars):
    # Determine the position in the 2x2 grid
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Get basic statistics
    data = df[var].dropna()
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    skewness = stats.skew(data)
    
    # Create histogram with KDE
    ax.hist(data, bins=30, alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
    
    # Add KDE curve
    ax2 = ax.twinx()
    data.plot.kde(ax=ax2, color='red', linewidth=2, alpha=0.8)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    
    # Customize the plot
    ax.set_title(f'{var.replace("_", " ").title()}', fontsize=12, pad=15)
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Min: {min_val:.2f}\n'
    stats_text += f'Max: {max_val:.2f}\n'
    stats_text += f'Skew: {skewness:.2f}'
    
    # Position text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add mean and median lines
    ax.axvline(mean_val,   color='red',  linestyle='--', linewidth=1.5, alpha=0.8, label='Mean')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Median')
    
    # Add legend only for first subplot
    if idx == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==== Cell 6 ====
"""
Creates a correlation heatmap for the important variables
"""
important_vars = [
    'emissions_intensity', 'energy_efficiency', 'renewables_share', 'diversity_index', 
    'board_independence', 'revenue_growth', 'cost_of_capital', 'volatility', 'esg_score'
]

# Calculate correlation matrix
corr_matrix = df[important_vars].corr()

# Create heatmap
plt.figure(figsize=(10, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, 
            cmap='RdYlBu_r', center=0, square=True,
            fmt='.3f', cbar_kws={"shrink": .8}, linewidths=0.5)

plt.title('Correlation Matrix - Key ESG and Financial Variables', fontsize=14, fontweight='bold', pad=10)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Numeric features only (exclude targets and categorical label)
feature_cols = [
    "emissions_intensity", "energy_efficiency", "renewables_share",
    "injury_rate", "diversity_index", "training_hours",
    "board_independence", "exec_comp_alignment", "controversies"
]

X = df[feature_cols].values
Y_multi = df[["revenue_growth", "cost_of_capital", "volatility"]].values  # multi-output regression
y_cls   = df["risk_tier"].astype("category").cat.codes  # for LDA classification

# It's recommended to standardize the data *BEFORE* performing PCA,
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)    # To compute the first 5 principal components.

X_pca = pca.fit_transform(X_scaled)  # To project data into the new coordinate system.

# The fraction of the dataset’s variance explained by each component.
# Example: if PC1 = 0.45 and PC2 = 0.25, then together the first two PCs 
# explain 70% of the variance in the original data.
#
explained_var = pca.explained_variance_ratio_
print("Explained Variance Ratio (first 5 PCs):", np.round(explained_var, 5))  # np.round(explained_var[:5], 5))

# "pca.components_" is a matrix where each row = one principal component (PC), 
# each column = contribution of an original feature.
# They show how strongly each original variable contributes to a PC.
#  - Positive loading → feature contributes positively to PCi.
#  - Negative loading → feature contributes in the opposite direction.
#  - Larger absolute value → stronger influence on PCi.
#
components = pca.components_  # loadings for each PC

np.round(components, 4)

# Sort PC1 loadings by 'absolute value' (biggest influence first)
pc1_loadings = list(zip(feature_cols, components[0]))
pc1_sorted = sorted(pc1_loadings, key=lambda x: abs(x[1]), reverse=True)

print("\nPC1 loadings (sorted by absolute contribution):")
for name, loading in pc1_sorted:
    print(f"{name:22s} {loading: .3f}")

# Find cumulative variance explained
#
cum_explained_var = np.cumsum(explained_var)

# --- Cumulative Variance Explained ---
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% threshold')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.show()

# Single chart per cell: "Scree plot"
#
plt.figure(figsize=(7,4))
plt.plot(range(1, len(explained_var)+1), explained_var, marker="o")
plt.title("PCA Scree Plot (Explained Variance Ratio)")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.xticks(range(1, len(explained_var)+1))
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# 2D projection on first two PCs
plt.figure(figsize=(7,4))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA Projection (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# Factor Analysis to identify latent factors in ESG data
#
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 300
df = pd.DataFrame({
    "emissions": np.random.gamma(2., 50., n),
    "renewables": np.random.beta(2, 5, n) * 100,
    "energy_efficiency": np.random.normal(65, 10, n).clip(30, 95),
    "diversity": np.random.normal(55, 12, n).clip(0, 100),
    "training_hours": np.random.normal(20, 5, n).clip(0, 60),
    "board_independence": np.random.normal(60, 15, n).clip(0, 100),
    "exec_alignment": np.random.normal(58, 10, n).clip(0, 100),
})

# Standardize
X_scaled = StandardScaler().fit_transform(df)

# Apply Factor Analysis (assume 3 factors)
fa = FactorAnalysis(n_components=3, random_state=42)
factor_scores = fa.fit_transform(X_scaled)

# Factor loadings
loadings = pd.DataFrame(
    fa.components_.T,   # Transpose of the "Components with maximum variance"
    index=df.columns,
    columns=["Factor1", "Factor2", "Factor3"]
)
print(loadings.round(3))

# Use factor scores for clustering
#
factor_df = pd.DataFrame(factor_scores, columns=["Factor1","Factor2","Factor3"])
kmeans = KMeans(n_clusters=3, n_init=10, random_state=123)
clusters = kmeans.fit_predict(factor_df)
factor_df["Cluster"] = clusters

sil = silhouette_score(factor_df[["Factor1","Factor2","Factor3"]], clusters)
print("\nSilhouette score:", round(sil, 3))

# Visualize clusters in factor space
plt.figure(figsize=(7,4))
for c in sorted(factor_df["Cluster"].unique()):
    mask = factor_df["Cluster"] == c
    plt.scatter(factor_df.loc[mask, "Factor1"], factor_df.loc[mask, "Factor2"],
                label=f"Cluster {c}", alpha=0.75)
plt.xlabel("Factor1")
plt.ylabel("Factor2")
plt.title("Clustering Companies by Factor Scores (ESG Example)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()