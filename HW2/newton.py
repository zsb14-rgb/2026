import numpy as np

def newton(f,fprime,xo,kmax=100):
    """Newtons Method: find a zero of f starting from xo. Return (x,k)"""
    eps = np.finfo(float).eps
    k = 0
    x = xo
    xprev = xo + 10
    while abs(x-xprev) > eps * abs(x) and k < kmax:
        xprev = x
        x= x -f(x) / fprime(x)
        k += 1
    return x,k