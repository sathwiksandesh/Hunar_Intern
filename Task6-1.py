import numpy as np
# user input for the order of the matrix and its values
n = int(input("Enter the order of the square matrix :"))
print(f"\nYou are creating a {n}x{n} matrix.")
print(f"Enter {n*n} elements row by row:")

elements = []
for i in range(n):
    row = []
    for j in range(n):
        value = int(input(f"Element at position [{i+1}][{j+1}]: "))
        row.append(value)
    elements.append(row)

# Create the numpy array
matrix = np.array(elements)

#  Calculate the sum of diagonal elements using a loop
diagonal_sum = 0
for i in range(n):
    diagonal_sum += matrix[i][i]

print("\nThe matrix you entered is:")
print(matrix)
print(f"\nSum of diagonal elements in the {n}x{n} matrix: {diagonal_sum}")
