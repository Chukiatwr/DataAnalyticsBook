# deduplication.py

import pandas as pd

# Sample dataset with duplicates
df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003", "C001", "C004", "C002", "C005"],
    "name": ["John Doe", "Jane Smith", "Alice", 
             "John Doe", "Bob", "Jane Smith", "John Henry"],
    "email": ["john@example.com", "jane@example.com", "alice@example.com",
              "john@example.com", "bob@example.com", "jane@example.com", 
              "john@example.com"],
    "purchase": [500, 700, 300, 500, 200, 700, 800]
})

print("Original data:")
print(df)

# --- Step 1: Detect duplicates ---
duplicates = df[df.duplicated()]
print("\nDuplicated rows (exact match across all columns):")
print(duplicates)

# --- Step 2: Remove duplicates ---
df_no_dup = df.drop_duplicates()
print("\nData after removing full duplicates:")
print(df_no_dup)

# --- Step 3: Detect duplicates based on specific column ---
emduplicates = df[df.duplicated('email')]
print("\nDuplicated rows (exact match on only email):")
print(emduplicates)

# --- Step 3: Remove duplicates based on specific column ---
df_no_dup_email = df.drop_duplicates(subset=['email'])
print("\nData after removing duplicates based on ['email']:")
print(df_no_dup_email)