# trend_charts.py
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid") # Set Seaborn theme

# Create sample data
np.random.seed(123)  # For reproducibility
dates = pd.date_range(start='2024-01-01', periods=24, freq='M')
sales_product_a = np.cumsum(np.random.randn(24) * 20 + 50) + 500  # แนวโน้มเพิ่มขึ้นเล็กน้อย
sales_product_b = np.cumsum(np.random.randn(24) * 12 + 40) + 400
sales_product_c = np.cumsum(np.random.randn(24) * 5 + 30) + 300

# Combine into a DataFrame
df_sales = pd.DataFrame({
    'Date': dates,
    'Clothes': sales_product_a,
    'Shoes':   sales_product_b,
    'Accessories': sales_product_c
})

# Create Long format for Seaborn (Stacked Area)
df_long = df_sales.melt(id_vars='Date', var_name='Product', value_name='Sales')

################################################################
## 📉 Example of Line Chart (Matplotlib)
plt.figure(figsize=(9, 4))

# Plot each product line
plt.plot(df_sales['Date'], df_sales['Clothes'], label='Clothes', marker='o', linestyle='-')
plt.plot(df_sales['Date'], df_sales['Shoes'], label='Shoes', marker='s', linestyle='--')
plt.plot(df_sales['Date'], df_sales['Accessories'], label='Accessories', marker='^', linestyle=':')

plt.title('Trends of Sales (24 months)', fontsize=16)
plt.xlabel('Period')
plt.ylabel('Cumulative Sales (units)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

################################################################
## 🏔️ Example of Stacked Area Chart (Matplotlib)
plt.figure(figsize=(9, 4))

# Define products list
products = ['Clothes', 'Shoes', 'Accessories']

plt.stackplot(   # for Stacked Area Chart
    df_sales['Date'], 
    df_sales[products].values.T, # Transpose to get correct shape
    labels=products,
    alpha=0.8,
    colors=['turquoise', 'orange', 'lightgreen']
)

plt.title('Sales Components Over Time (Stacked Area)', fontsize=16)
plt.xlabel('Period')
plt.ylabel('Cummulative Sales (units)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

################################################################
## 💯 Example of 100% Stacked Area Chart (Matplotlib)
plt.figure(figsize=(9, 4))

# 1. Calculate total sales per period
df_sales['Total'] = df_sales[products].sum(axis=1)

# 2. Calculate % proportion for each product
percentages = (df_sales[products].T / df_sales['Total'].values).T * 100

plt.stackplot(   # for 100% Stacked Area Chart
    df_sales['Date'], 
    percentages.values.T, 
    labels=products,
    alpha=0.8,
    colors=['turquoise', 'orange', 'lightgreen']
)

plt.title('Sales Proportion Over Time (Stacked Area)', fontsize=16)
plt.xlabel('Period')
plt.ylabel('Proportion (%)')
plt.ylim(0, 100)    # set Y constant from 0 to 100
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()