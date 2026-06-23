# merge.py

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

# Inner Join - keep only matching keys in both tables
result = pd.merge(employees, departments, on='dept_id', how='inner')
print("\nInner Join Result:\n", result)

# Left Join - keep all keys from the left table (employees)
result = pd.merge(employees, departments, on='dept_id', how='left')
print("\nLeft Join Result:\n", result)

# Right Join - keep all keys from the right table (departments)
result = pd.merge(employees, departments, on='dept_id', how='right')
print("\nRight Join Result:\n", result)

# Outer Join - keep all keys from both tables
result = pd.merge(employees, departments, on='dept_id', how='outer')
print("\nOuter Join Result:\n", result)

# Merge on different column names
# Let's rename dept_id in departments to department_id
departments_renamed = departments.rename(columns={'dept_id': 'department_id'})
result = pd.merge(employees, departments_renamed, left_on='dept_id', right_on='department_id', how='inner')
print("\nMerge on Different Column Names Result:\n", result)
# Note: department_id will be dropped in the result as it's redundant

