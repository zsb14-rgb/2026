import matplotlib.pyplot as plt
import numpy as np
from mid_point import mid_point


# ODE function: rewrite y'''' = y'' as a first-order system [y, y', y'', y''']
def q2_eq(t, y):
    return np.array([y[1], y[2], y[3], y[2]])


# Setup: x range, step size, and initial conditions y=0, y'=1, y''=0, y'''=-1
to = 0.0
tf = 1.0
tstep = 0.01
y = np.array([0.0, 1.0, 0.0, -1.0])

# Run the mid-point integrator
sol, time = mid_point(q2_eq, to, tf, y, tstep)

# Exact solution: y = A + Bx + Ce^x + De^-x, initial conditions give y = 2x - sinh(x)
x = np.linspace(to, tf, 500)
y_exact = 2 * x - np.sinh(x)

# Plot numerical solution as markers, exact solution as a line
plt.plot(time, sol[0, :], 'o', mfc='none', label='Numerical Solution')
plt.plot(x, y_exact, '-', lw=2, label='Exact Solution')
plt.xlabel('x-value')
plt.ylabel('y-value')
plt.title('Mid-Point Integration')
plt.legend()
plt.savefig('hw4_q2.png', dpi=120)
plt.show()