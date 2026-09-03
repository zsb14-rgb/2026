import numpy as np

def lutx(A):
    """
    Triangular factorization, textbook version.
    [L, U, p] = lutx(A) produces a unit lower triangular matrix L,
    an upper triangular matrix U, and a 0-indexed permutation vector p,
    so that L @ U == A[p, :].
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

    # Separate result[cite: 1]
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)

    return L, U, p