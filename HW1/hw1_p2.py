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
    sig = 1 #sign of permutation

    for k in range(n - 1):
        # Find index of largest element below diagonal in k-th column
        m = np.argmax(np.abs(A[k:n, k])) + k

        # Skip elimination if column is zero
        if A[m, k] != 0:
            # Swap pivot row[cite: 1]
            if m != k:
                A[[k, m], :] = A[[m, k], :]
                p[[k, m]] = p[[m, k]]
                sig = -sig #intechange flips the sign
            # Compute multipliers[cite: 1]
            A[k+1:, k] /= A[k, k]

            # Update the remainder of the matrix[cite: 1]
            A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])

    # Separate result[cite: 1]
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)

    return L, U, p, sig

#myDet(A)
def myDet(A):
    """
    Computes the determinant of a square matrix A using LU factorization.
    """
    L,U,p,sig = lutx(A)
    return sig*np.prod(np.diag(U))

A = np.array([[8,5,-10],
              [5,2,16],
              [-10,-16,5]])

L, U, p, sig = lutx(A)
print("L =\n", L)
print("U =\n", U)
print("p =", p)
print("sig =", sig)

print("myDet(A)      =", myDet(A))
print("np.linalg.det =", np.linalg.det(A))


