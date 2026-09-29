import numpy as np

def lutx(A):
    """
    Returns one matrix withe L below U on and above it.
    """
    A = np.array(A, dtype=float).copy()
    n = A.shape[0]
    p = np.arange(n)

    for k in range(n - 1):
        # Find index of largest element below diagonal in k-th column
        m = np.argmax(np.abs(A[k:n, k])) + k

        # Skip elimination if column is zero
        if A[m, k] != 0:
            # Swap pivot row[cite: 1]
            if m != k:
                A[[k, m], :] = A[[m, k], :]
                p[[k, m]] = p[[m, k]]

            # Compute multipliers[cite: 1]
            A[k+1:, k] /= A[k, k]

            # Update the remainder of the matrix[cite: 1]
            A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

    #Returns Single
    return A, p

A = np.array([[8,5,-10],
              [5,2,16],
              [-10,-16,5]])
b = np.array([1, 3, -2])

LU, p = lutx(A)
n = A.shape[0]

L = np.tril(LU, -1) + np.eye(n)
U = np.triu(LU)

y = np.linalg.solve(L, b[p])

x = np.linalg.solve(U, y)

x_ref = np.linalg.solve(A, b)

print("x (via LU) =", x)
print("x (A\\b)    =", x_ref)
print("difference =", np.abs(x - x_ref).max())
