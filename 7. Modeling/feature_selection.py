# feature_selection.py
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=123)
model.fit(X, y)

# 3. Calculate and sort feature importance
importances = pd.Series(model.feature_importances_, index=data.feature_names)
# Sort ascending for horizontal bar chart so the most important features appear at the top
top_10_features = importances.sort_values(ascending=True).tail(10)
print("Top 10 Feature Importances:\n", top_10_features)

# 4. Create the plot with optimized layout
fig, ax = plt.subplots(figsize=(8, 5))
top_10_features.plot(kind='barh', color='skyblue', ax=ax)

# Set titles and adjust font sizes for clarity
ax.set_title('Top 10 Feature Importance from Random Forest', fontsize=14, pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)

# Add vertical gridlines for better readability of scores
ax.xaxis.grid(True, linestyle='--', alpha=0.7)

# Use tight_layout to prevent clipping of long feature names
plt.tight_layout()

plt.show()