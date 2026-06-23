# feature_selection.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. Simulate the ESG Data
# ----------------------------------------------------------------
# Assume this is ESG data for companies, with 10 features (E1-G3) 
# and the target being the overall ESG Score.
np.random.seed(11)
n_samples = 100
features = ['E1_CarbonIntensity', 'E2_WaterUse', 'E3_WasteGen',
            'S1_Diversity', 'S2_Safety', 'S3_Community',
            'G1_BoardIndep', 'G2_AuditQuality', 'G3_ExecComp', 'F1_MarketCap']
X = pd.DataFrame(np.random.rand(n_samples, len(features)) * 100, columns=features)

# Create the Target (ESG_Score) to clearly depend on a few features (e.g., E1, S2, G1)
# and add some noise.
y = (3 * X['E1_CarbonIntensity'] + 5 * X['S2_Safety'] - 2 * X['G1_BoardIndep'] + 
     np.random.randn(n_samples) * 10)
# Scale the score to be between 0 and 100
y = (y - y.min()) / (y.max() - y.min()) * 100  

print(f"Original Feature Count: {X.shape[1]}\n")

# ----------------------------------------------------------------
# 2. Data Pre-processing (Scaling)
# ----------------------------------------------------------------
# Feature Scaling is crucial for methods like Linear Regression and distance-based models.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


## --- Method 1: Filter Method (Statistical) ---
### Example: Using F-test (for Regression) to measure the linear relationship
### of each feature with the target variable.

# Select the top 4 best features (k=4)
k_filter = 4
selector_filter = SelectKBest(score_func=f_regression, k=k_filter)
X_filtered = selector_filter.fit_transform(X_scaled, y)
selected_mask_filter = selector_filter.get_support()
selected_features_filter = X.columns[selected_mask_filter].tolist()

print("## Filter Method (SelectKBest with F-test)")
print(f"Selected Features (Top {k_filter}): {selected_features_filter}")
print("-" * 60)


## --- Method 2: Wrapper Method (Model-Based Search) ---
### Example: Recursive Feature Elimination (RFE)
### Uses an external model (Linear Regression) to recursively eliminate the worst features.

# Select the top 4 best features (n_features_to_select=4)
n_features_rfe = 4
model_rfe = LinearRegression()
rfe = RFE(model_rfe, n_features_to_select=n_features_rfe, step=1)
rfe.fit(X_scaled, y)
selected_mask_rfe = rfe.support_
selected_features_rfe = X.columns[selected_mask_rfe].tolist()

print("## Wrapper Method (RFE with Linear Regression)")
print(f"Selected Features (Top {n_features_rfe}): {selected_features_rfe}")
print("-" * 60)


## --- Method 3: Embedded Method (Model's Internal Mechanism) ---
### Example: Using Random Forest Regressor to extract Feature Importance.

# Instantiate and train the model
model_embedded = RandomForestRegressor(n_estimators=100, random_state=42)
model_embedded.fit(X_scaled, y)

# Retrieve Feature Importance scores
feature_importances = pd.Series(model_embedded.feature_importances_, index=X.columns)

# Select the top 4 most important features
n_features_rf = 4
selected_features_embedded = feature_importances.nlargest(n_features_rf).index.tolist()

print("## Embedded Method (Random Forest Feature Importance)")
print(f"Selected Features (Top {n_features_rf}): {selected_features_embedded}")
print(f"Top {n_features_rf} Importance Scores:\n{feature_importances.nlargest(n_features_rf).to_string()}")
print("-" * 60)