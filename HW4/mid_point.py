import numpy as np


def mid_point(fnc, to, tf, yo, tstep):
    # Mid-point integration: fnc is the derivative function, to/tf are start
    # and end time, yo is the initial condition, tstep is the time step
    t = to
    y = np.array(yo, dtype=float)
    time = [t]
    sol = [y]

    # Step forward until tf: estimate y at the half step, then use the
    # slope there to take the full step
    while t < tf - 1e-12:
        s1 = fnc(t, y)
        y_half = y + tstep * s1 / 2
        y = y + tstep * fnc(t + tstep / 2, y_half)
        t = t + tstep
        time.append(t)
        sol.append(y)

    return np.array(sol).T, np.array(time)