# join.py

import pandas as pd

# Employees table
employees = pd.DataFrame({
    'emp_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [1, 2, 1, 3]
})

# Departments table
departments = pd.DataFrame({
    'dept_id': [1, 2, 4],
    'dept_name': ['Sales', 'IT', 'HR']
})

print("Employees DataFrame:\n", employees)
print("\nDepartments DataFrame:\n", departments)

# --- Using join() ---
# Step 1: Set 'dept_id' as the index for both DataFrames
employees_indexed = employees.set_index('dept_id')
departments_indexed = departments.set_index('dept_id')

# Step 2: Join on index (dept_id)
result = employees_indexed.join(departments_indexed, how='left')

# Step 3: Reset index if desired
result = result.reset_index()

print("\nJoined DataFrame (using join()):\n", result)

# Step 4: Compare the result using merge()
result_merge = pd.merge(
    employees,            # left DataFrame
    departments,           # right DataFrame
    on='dept_id',          # key column
    how='left'             # join type
)
print("\nMerged DataFrame (using merge()):\n", result_merge)
