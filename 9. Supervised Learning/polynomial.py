# polynomial.py
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Generate synthetic data (True Degree = 3)
np.random.seed(123) 
n_samples = 100
X = 10 * np.random.rand(n_samples, 1) - 5  # Range -5 to 5
# True Function: y = 0.5x^3 - x^2 + 2x + 5
y = 0.5 * X**3 - X**2 + 2*X + 5 + np.random.normal(0, 10, (n_samples, 1))

# 2. Data Splitting (Fixed seed 123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define degrees and distinct colors for each plot
degrees = [1, 2, 3, 15]
colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF'] # Orange, Green, Blue, Purple
plt.figure(figsize=(12, 9))

performance_report = []

# 3. Model Training and Visualization Loop
for i, (degree, color) in enumerate(zip(degrees, colors)):
    # Feature Transformation
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)

    # Model Training
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)

    # Predictions for evaluation
    y_train_pred = poly_model.predict(X_poly_train)
    y_test_pred = poly_model.predict(X_poly_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    performance_report.append({
        'Degree': degree,
        'R2 Training': round(r2_train, 4),
        'R2 Test': round(r2_test, 4)
    })

    # Visualization
    X_plot = np.linspace(-5, 5, 500).reshape(500, 1)
    y_plot = poly_model.predict(poly_features.transform(X_plot))
    
    plt.subplot(2, 2, i + 1)
    plt.scatter(X_train, y_train, color='gray', alpha=0.3, label='Train Data')
    plt.scatter(X_test, y_test, color='black', alpha=0.8, label='Test Data')
    plt.plot(X_plot, y_plot, color=color, linewidth=3, label=f'Polynomial Degree {degree}')
    
    # Move R2 report to the Title (Subtitle of each plot)
    plt.title(f'Degree {degree}\nTrain R²: {r2_train:.4f} | Test R²: {r2_test:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
    
    plt.ylim(y.min()-20, y.max()+20)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout(pad=4.0)
plt.show()

# 4. Numerical Summary table
print("\n" + "="*45)
print("      POLYNOMIAL PERFORMANCE SUMMARY")
print("="*45)
report_df = pd.DataFrame(performance_report)
print(report_df.to_string(index=False))
print("="*45)