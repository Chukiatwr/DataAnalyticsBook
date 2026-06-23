import pandas as pd
import numpy as np

# Read the raw data
df = pd.read_csv("dirty_data.csv")
print(df, '\n')
print(df.info())

# Check for duplicate rows

duplicate_rows = df[df.duplicated()]
print("\nDuplicate rows (excluding first occurrence):")
print(duplicate_rows, '\n')

# Count missing values per column
print("Number of missing values of each column:\n", df.isna().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Handling Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())  # เติม median ของ Age
df['JoinDate'] = df['JoinDate'].fillna("2020-01-01")  # เติม default date

df['Salary'] = df['Salary'].replace("not available", np.nan)  # แปลง not available เป็น NaN
df['Salary'] = df['Salary'].astype(float)  # แปลง Salary เป็น float
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())  # เติม mean ของ Salary

print(df)

# Format the strings: remove trailing space and capitalize the words
df['Name'] = df['Name'].str.strip()  # ตัดช่องว่าง
df['Name'] = df['Name'].str.capitalize()  # ทำให้เป็นตัวพิมพ์ใหญ่ตัวแรก
df['Gender'] = df['Gender'].str.capitalize()  # ทำให้เป็นตัวพิมพ์ใหญ่ตัวแรก

# Change Gender to be Category for more efficient further processing
df['Gender'] = df['Gender'].astype('category')

# Handle Outliers (Age > 110 is considered an outlier)
df.loc[df['Age'] > 110, 'Age'] = df['Age'].median()

# Show the cleaned dataset
print("\nCleaned data")
print(df, '\n')
print(df.info())



df.loc[df['Age'] > 30, 'Salary'] = np.NaN
print(df, '\n')
print(df.info())
print("*-" * 30)


# Create dummy variables (one-hot encoding) for the KNN Imputer
# KNN works only with numeric data, so categories must be encoded.
df_encoded = pd.get_dummies(df, columns=['Gender'], prefix='Gender', dummy_na=False)
df_encoded.drop(columns=['Gender_Male'], inplace=True) # Drop one dummy column to avoid multicollinearity

# --- 4. Select Features and Apply KNN Imputation ---

from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler


# Select the columns we want to use for imputation and the column we want to impute (Salary)
# We use Age (after cleaning), and the encoded Gender (Female) as features.
imputation_cols = ['Age', 'Gender_Female', 'Salary']
df_impute = df_encoded[imputation_cols].copy()

# Initialize KNN Imputer (using k=2 since our dataset is very small)
# The imputer finds the K nearest neighbors for a missing value and takes the average of their values.
imputer = KNNImputer(n_neighbors=2)

# Apply the imputer to the selected columns
# The imputer handles NaNs in both the features (Age) and the target (Salary)
imputed_data = imputer.fit_transform(df_impute)

# Convert the results back to a DataFrame
df_imputed = pd.DataFrame(imputed_data, columns=imputation_cols)
print(df_imputed)
print("*-" * 30)

# --- 5. Display Results ---

# Assign the imputed salary back to the original DataFrame structure
df['Salary_Imputed'] = df_imputed['Salary']

print("*-" * 30)
print(df)
