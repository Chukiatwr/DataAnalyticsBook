# house_price.py
#
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Load a realistic dataset (California Housing)
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target # Target is the median house value in $100,000s

# 2. Data Splitting (Maintain Bias-Variance Trade-off)
# We reserve 20% for testing to evaluate the model's generalization ability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 3. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Model Interpretation: Extracting Intercept and Coefficients
intercept = model.intercept_
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print(f"Model Intercept (beta_0): {intercept:.4f}")
print("\nModel Coefficients (beta_i):")
print(coefficients)

# 5. Model Prediction on unseen data (Test Set)
y_pred = model.predict(X_test)

# 6. Performance Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Root Mean Squared Error for easier interpretation
r2 = r2_score(y_test, y_pred) # Coefficient of Determination

print(f"\n--- Evaluation Metrics ---")
print(f"R-squared Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 7. Residual Analysis Plot
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot: Check for Homoscedasticity')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Errors)')
plt.show()