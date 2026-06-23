def find_equilibrium_index(arr: list[int]) -> int:
    """
    Finds the first equilibrium index in a list of integers with O(n) complexity.
    The equilibrium index in an array is a position where the sum of elements on 
    the left is equal to the sum of elements on the right. It is useful in various 
    algorithms, particularly in finding balance points in data. The goal is to 
    identify such an index efficiently using a linear time complexity.

    """
    if not arr:
        return -1
    
    total_sum = sum(arr)
    left_sum = 0
    right_sum = total_sum
    
    for i in range(len(arr)):
        current_element = arr[i]
        
        # 1. Update right_sum (Sum to the right of i)
        right_sum -= current_element
        
        # 2. Check for equilibrium
        if left_sum == right_sum:
            return i 
            
        # 3. Update left_sum (Sum to the left of i+1)
        left_sum += current_element
        
    return -1

# --- Additional Test Cases ---
tests = [
    ([10, -5, 5], 0),                # Case 5: Negative numbers, equilibrium at start
    ([-1, 3, -4, 2], 1),             # Case 6: Zero total sum
    ([2, 4, 6, 8, 10], -1),          # Case 7: No equilibrium
    ([4, 0, 0, 0, 0], -1),           # Case 8: Zeroes, but no equilibrium
    ([1, 1, 1, 1, 1], 2),            # Case 9: Multiple/Central equilibrium
    ([0, 0, 0, 0, 0], 0),            # Case 10: All zeroes
    ([-7, 1, 5, 2, -4, 3, 0], 3)     # Case 11: LeetCode/HackerRank classic (Index 3: Left Sum = -1, Right Sum = -1)
]

print("--- Running Additional Tests ---")
for arr, expected in tests:
    result = find_equilibrium_index(arr)
    status = "PASS" if result == expected else "FAIL"
    print(f"Array: {arr}")
    print(f"  Result: {result}, Expected: {expected} -> **{status}**")