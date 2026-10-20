import numpy as np
import matplotlib.pyplot as plt

# Data from the homework
xval = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])   # dot size (nm)
yval = 25 * (1 + 0.1 * (np.random.rand(9) - 0.5)) * xval**2       # wavelength
m = len(xval)                                                      # number of points

# Plot the data
plt.plot(xval, yval, 'o', mfc='none', label='Data')
plt.xlim(0, 5.25)

#Gradient Descent
c = np.array([3.0, 1.0])     
cN = np.array([0.0, 0.0])    

maxiter = 100000   # max number of iterations
toler_c = 0.001    # tolerance on parameters
toler_j = 0.001    # tolerance on cost
delc = np.array([1000.0, 1000.0])   # change in parameters
delj = 1000.0      # change in cost
jold = 1000.0      # previous cost
it = 0             # iteration count
l_rate = 0.01      # learning rate

# Repeat until the cost and parameters stop changing
while it < maxiter and (np.sqrt(abs(delj)) / np.linalg.norm(yval) > toler_j
                        or np.max(delc) > toler_c):
    it = it + 1
    sum1 = 0.0
    sum2 = 0.0
    jval = 0.0

    for i in range(m):
        err = c[0] + c[1] * xval[i] - yval[i]   # error at this point
        sum1 = sum1 + err                       # for c0 gradient
        sum2 = sum2 + err * xval[i]             # for c1 gradient
        jval = jval + err**2                    # for cost

    sum1 = sum1 / m          # gradient for c0
    sum2 = sum2 / m          # gradient for c1
    jval = jval / (2 * m)    # cost

    delj = jval - jold       # change in cost
    jold = jval

    cN[0] = c[0] - l_rate * sum1   # update c0
    cN[1] = c[1] - l_rate * sum2   # update c1

    delc = np.abs(c - cN)    # change in parameters
    c = cN.copy()

print("Gradient Descent:  c0 =", round(c[0], 4), " c1 =", round(c[1], 4), " iterations =", it)

# Normal Equations
# Each row of amat is [1, x], bvec is the data
amat = np.zeros((m, 2))
bvec = np.zeros(m)
for i in range(m):
    amat[i, 0] = 1.0
    amat[i, 1] = xval[i]
    bvec[i] = yval[i]

# Solve (A^T A) c = A^T b
cval = np.linalg.solve(amat.T @ amat, amat.T @ bvec)

print("Normal Equations:  c0 =", round(cval[0], 4), " c1 =", round(cval[1], 4))

# ---------------- Compare ----------------
print("Difference:        dc0 =", round(c[0] - cval[0], 4), " dc1 =", round(c[1] - cval[1], 4))

# Plot both fits
plt.plot(xval, c[0] + c[1] * xval, '-', label='Gradient Descent')
plt.plot(xval, cval[0] + cval[1] * xval, 's', mfc='none', label='Normal Equations')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('hw3_q1_plot.png', dpi=150)
plt.show()