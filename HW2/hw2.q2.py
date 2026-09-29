import numpy as np
import matplotlib.pyplot as plt

pbar = 1.0      # characteristic pressure
kio = 1.0       # initial flow conductance
nel = 6         # number of elements
nnode = 7       # number of nodes

p = np.linspace(1.0, 0.0, nnode)   # initial guess: straight line (constant-k solution)
p[0] = 1.0                         # BC: P(1) = 1, P(7) = 0
rhs = np.zeros(nnode)
rhs[1] = kio

residual = np.ones(nnode - 2)
toler_r = 1e-5
maxiter = 100
it = 0

while it < maxiter and np.linalg.norm(residual) / np.linalg.norm(rhs) > toler_r:
    it += 1
    amat = np.zeros((nnode, nnode))
    jacob = np.zeros((nnode, nnode))
    for i in range(nel):                        # element i connects node i and i+1
        p_el = (p[i] + p[i+1]) / 2.0
        k_el = kio * np.exp(p_el / pbar)        # NEW: k = kio*exp(P/Pbar)
        dkdp = k_el / (2.0 * pbar)              # dk_el/dp_i (the 1/2 is from the average)
        dp = p[i] - p[i+1]
        amat[i, i]     += k_el
        amat[i, i+1]   -= k_el
        amat[i+1, i]   -= k_el
        amat[i+1, i+1] += k_el
        jacob[i, i]     += k_el + dp * dkdp
        jacob[i, i+1]   += -k_el + dp * dkdp
        jacob[i+1, i]   += -k_el - dp * dkdp
        jacob[i+1, i+1] += k_el - dp * dkdp
        if i == 0:
            rhs[1] = k_el * p[0]
    residual = rhs[1:6] - amat[1:6, 1:6] @ p[1:6]     # interior nodes 2-6
    p_s = np.linalg.solve(jacob[1:6, 1:6], residual)
    p[1:6] += p_s
    print(f"iter {it}: residual norm = {np.linalg.norm(residual):.3e}")

print("\nFinal pressures:")
for n, val in enumerate(p, start=1):
    print(f"  node {n}: {val:.6f}")

nodes = np.arange(1, nnode + 1)
plt.plot(nodes, p, '-ks')
plt.xlabel('Node number')
plt.ylabel('Pressure')
plt.title(r'Single vessel, $k = k_o \exp(P/\bar{P})$')
plt.grid(True)
plt.savefig('hw2_q2_plot.png', dpi=150)
plt.show()