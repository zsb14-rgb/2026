import matplotlib.pyplot as plt
import numpy as np
from euler_int import euler_int
from mid_point import mid_point


# ODE function: rewrite y'' = y as a first-order system [y, y']
def q1_eq(t, y):
    return np.array([y[1], y[0]])


# Setup: x range and step size
to = 0.0
tf = 5.0
tstep = 0.02
x = np.linspace(to, tf, 500)   # x values for the exact solution

# ---------------- Case (a): y = 1, y' = -1 ----------------
y = np.array([1.0, -1.0])

# Run both integrators
solE, timeE = euler_int(q1_eq, to, tf, y, tstep)
solM, timeM = mid_point(q1_eq, to, tf, y, tstep)

# Exact solution: y = e^-x, y' = -e^-x
y_exact = np.exp(-x)
yp_exact = -np.exp(-x)

# Plot Euler and mid-point side by side, with exact solution as lines
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(timeE, solE[0, :], 'o', mfc='none', label='y(x)')
plt.plot(timeE, solE[1, :], 's', mfc='none', label='dy/dx')
plt.plot(x, y_exact, 'y-', lw=2)
plt.plot(x, yp_exact, 'm-', lw=2)
plt.xlabel('x-value')
plt.ylabel('y or dy/dx-value')
plt.title('Euler Integration (a)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(timeM, solM[0, :], 'o', mfc='none', label='y(x)')
plt.plot(timeM, solM[1, :], 's', mfc='none', label='dy/dx')
plt.plot(x, y_exact, 'y-', lw=2)
plt.plot(x, yp_exact, 'm-', lw=2)
plt.xlabel('x-value')
plt.ylabel('y or dy/dx-value')
plt.title('Mid-Point Integration (a)')
plt.legend()
plt.tight_layout()
plt.savefig('hw4_q1a.png', dpi=120)

# ---------------- Case (b): y = 1, y' = 1 ----------------
y = np.array([1.0, 1.0])

# Run both integrators
solE, timeE = euler_int(q1_eq, to, tf, y, tstep)
solM, timeM = mid_point(q1_eq, to, tf, y, tstep)

# Exact solution: y = e^x, y' = e^x
y_exact = np.exp(x)
yp_exact = np.exp(x)

# Plot Euler and mid-point side by side, with exact solution as lines
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(timeE, solE[0, :], 'o', mfc='none', label='y(x)')
plt.plot(timeE, solE[1, :], 's', mfc='none', label='dy/dx')
plt.plot(x, y_exact, 'y-', lw=2)
plt.plot(x, yp_exact, 'm-', lw=2)
plt.xlabel('x-value')
plt.ylabel('y or dy/dx-value')
plt.title('Euler Integration (b)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(timeM, solM[0, :], 'o', mfc='none', label='y(x)')
plt.plot(timeM, solM[1, :], 's', mfc='none', label='dy/dx')
plt.plot(x, y_exact, 'y-', lw=2)
plt.plot(x, yp_exact, 'm-', lw=2)
plt.xlabel('x-value')
plt.ylabel('y or dy/dx-value')
plt.title('Mid-Point Integration (b)')
plt.legend()
plt.tight_layout()
plt.savefig('hw4_q1b.png', dpi=120)

plt.show()