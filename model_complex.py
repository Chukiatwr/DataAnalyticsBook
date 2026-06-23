import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Generate Synthetic Data
np.random.seed(123)
X = np.sort(np.random.rand(20))
y = np.sin(2 * np.pi * X) + np.random.randn(20) * 0.2

# 2. Track errors across different complexities (Polynomial Degrees)
degrees = np.arange(1, 15)
train_errors, val_errors = [], []

for degree in degrees:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])
    model.fit(X[:, np.newaxis], y)
    
    # Calculate Mean Squared Error
    train_errors.append(mean_squared_error(y, model.predict(X[:, np.newaxis])))
    # (In real cases, validation error is calculated on a separate set)
    val_errors.append(train_errors[-1] + (0.01 * (degree**2))) # Simulated val error for visualization

# 3. Plotting the Trade-off
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'r-o', label='Training Error (Bias Indicator)')
plt.plot(degrees, val_errors, 'b-s', label='Validation Error (Total Error)')
plt.axvline(x=3, color='g', linestyle='--', label='Optimal Complexity')

plt.title('Bias-Variance Trade-off & Model Complexity', fontsize=14)
plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()