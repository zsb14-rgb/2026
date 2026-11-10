import numpy as np


def euler_int(fnc, to, tf, yo, tstep):
    # Euler integration: fnc is the derivative function, to/tf are start and
    # end time, yo is the initial condition, tstep is the time step
    t = to
    y = np.array(yo, dtype=float)
    time = [t]
    sol = [y]

    # Step forward until tf, storing time and solution each step
    while t < tf - 1e-12:
        y = y + tstep * fnc(t, y)
        t = t + tstep
        time.append(t)
        sol.append(y)

    return np.array(sol).T, np.array(time)