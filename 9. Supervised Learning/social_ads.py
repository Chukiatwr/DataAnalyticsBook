# social_ads.py
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- Step 1: Data Loading ---
dataset = pd.read_csv('Social_Network_Ads.csv')

# Drop 'User ID' as it is a unique identifier with no predictive power
df = dataset.drop(['User ID'], axis=1)

# --- Step 2: Feature Engineering (One-Hot Encoding) ---
# Separate features (X_raw) and target (y)
X_raw = df.drop(['Purchased'], axis=1)
y = df['Purchased'].values

# Apply One-Hot Encoding to categorical variables (Gender)
# drop_first=True prevents the 'Dummy Variable Trap' (multicollinearity)
X = pd.get_dummies(X_raw, drop_first=True)

print("--- Features after One-Hot Encoding (X) ---")
print(X.head())

# --- Step 3: Splitting and Scaling ---
# Split the dataset: 80% Training, 20% Testing
# stratify=y ensures the class distribution is preserved in both sets
X_train_raw, X_test_raw, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=101, stratify=y
)

# Initialize MinMaxScaler to scale features between 0 and 1
sc = MinMaxScaler()

# Fit and transform the training set; transform the test set using training parameters
X_train = sc.fit_transform(X_train_raw)
X_test = sc.transform(X_test_raw)

# --- Step 4: Model Training ---

# 1. Logistic Regression (Linear Model)
lr = LogisticRegression(random_state=101)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 2. Decision Tree (Non-Linear Model)
dt = DecisionTreeClassifier(random_state=101)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# --- Step 5: Prediction Results Comparison ---
# Create a DataFrame to compare actual labels with model predictions
comparison_df = df.iloc[indices_test].copy()
comparison_df['LR_Prediction'] = lr_pred
comparison_df['DT_Prediction'] = dt_pred

print("\n--- Actual vs. Predicted Results (Sample of Test Set) ---")
print(comparison_df.head(20))

# Print final accuracy scores
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")

# --- Step 6: Feature Importance Visualization ---
def show_feature_importances(algo_name, column_names, importances):
    # Sort importances in descending order
    sorted_indices = np.argsort(importances)[::-1] 

    plt.rcParams["figure.figsize"] = (8, 4)
    plt.title(f'Feature Importances using {algo_name}', fontsize=15)
    
    # Create the bar chart
    plt.bar(range(len(importances)), importances[sorted_indices], align='center')
    plt.xticks(range(len(importances)), column_names[sorted_indices], rotation=0)
    
    # Annotate bars with numerical values
    for index, data in enumerate(importances[sorted_indices]):
        plt.text(x=index, y=data + .005, s=f"{data:.3f}", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Visualize which features the Decision Tree prioritized
show_feature_importances('Decision Tree', X.columns, dt.feature_importances_)