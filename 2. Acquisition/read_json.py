# read_json.py

import pandas as pd
import json

# ------------------------------------------------------------
# 0. เตรียมข้อมูลตัวอย่างและบันทึกเป็น JSON
# ------------------------------------------------------------
data = [
    {"id": 1, "name": "Apple",  "price": 20},
    {"id": 2, "name": "Banana", "price": 12},
    {"id": 3, "name": "Orange", "price": 15}
]

# บันทึกเป็นไฟล์ JSON
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("=== Write to data,json ===")

# ------------------------------------------------------------
# 1. Read basic JSON file
# ------------------------------------------------------------
df = pd.read_json("data.json")
print("\n=== Read data from JSON file.===")
print(df.head())

# ------------------------------------------------------------
# 2. Use orient to format JSON
# ------------------------------------------------------------
# Defalt is orient="columns", but often set to be records
df_records = pd.read_json("data.json", orient="records")
print("\n=== Read JSON file with orient=records ===")
print(df_records.head())

# ------------------------------------------------------------
# 3. Output to be a Series
# ------------------------------------------------------------
# Create JSON with normal key-value pairs
simple_data = {"a": 1, "b": 2, "c": 3}
with open("simple.json", "w", encoding="utf-8") as f:
    json.dump(simple_data, f, ensure_ascii=False, indent=2)

s = pd.read_json("simple.json", typ="series")
print("\n=== Read JSON as Series ===")
print(s)

# ------------------------------------------------------------
# 4. Read JSON from string of disctionary
# ------------------------------------------------------------
data_str = '{"id": [1,2], "name": ["Grape","Mango"], "price": [40,60]}'
df_str = pd.read_json(data_str)
print("\n=== Read JSON from string ===")
print(df_str)

