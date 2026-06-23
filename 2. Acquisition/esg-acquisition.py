# esg-acquisition.py
 
import pandas as pd
import seaborn as sns

df = pd.read_csv("ESG-data.csv")

print("=== Preview the first and last few rows ===")
print(df.head())
print(df.tail(7))

print("\n=== Size of data ===")
print(df.shape)

print("\n=== Basic structure of data ===")
print(df.info())

print("\n=== Basic statistics ===")
print(df.describe())

print("\n=== Column names ===")
print(df.columns)

# ---------------------------------------------------
# Quality Check
# ---------------------------------------------------

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Duplicate Records ===")
print(df.duplicated().sum())

print("\n=== Number of Unique Countries ===")
print(df['country'].nunique())
print(df['country'].value_counts())

print("\n=== Check esg_score > 90 ===")
print(df[df["esg_score"] > 90])

print("\n=== Boxplot of ESG Score ===")
sns.boxplot(x=df["esg_score"])