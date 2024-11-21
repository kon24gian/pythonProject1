from scipy.sparse import diags
import numpy as np

from sympy import Matrix
from sympy.physics.quantum import TensorProduct
from scipy.sparse import diags


def parameterized_matrix(h, n):
    # Create the diagonals for the tridiagonal structure
    sub_diag = (h - 2) * np.ones(n - 1)  # Sub-diagonal (offset -1)
    main_diag = -2 * np.ones(n)  # Main diagonal (offset 0)
    super_diag = (-2 - h) * np.ones(n - 1)  # Super-diagonal (offset 1)

    # Construct the tridiagonal matrix as a sparse matrix
    A = diags([sub_diag, main_diag, super_diag], offsets=[-1, 0, 1]).toarray()
    B = diags([sub_diag, main_diag, super_diag], offsets=[1, 0, -1]).toarray()
    # Modify the first row and last row explicitly
    A[0, :] = np.zeros(n)  # Clear the first row
    A[0, 0] = h ** 2  # Set the first element of the first row to h^2

    A[-1, :] = np.zeros(n)  # Clear the last row
    A[-1, -1] = h ** 2  # Set the last element of the last row to h^2
    B[0, :] = np.zeros(n)  # Clear the first row
    B[0, 0] = h ** 2  # Set the first element of the first row to h^2

    B[-1, :] = np.zeros(n)  # Clear the last row
    B[-1, -1] = h ** 2  # Set the last element of the last row to h^2
    return TensorProduct(A,B)


# Example usage:
h=2
n = 5
result1 = parameterized_matrix(h,n)

print(result1)
