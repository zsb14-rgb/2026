import numpy as np

def bisect(f,a,b,kmax=200):
        """Bisection:: find a zero of f [a,b]. Returns (x,k)."""
        if np.sign(f(a)) == np.sign(f(b)):
            raise ValueError("f(a) and f(b) must have different signs")
        eps = np.finfo(float).eps
        k = 0
        x = (a+b)/2.0
        while abs(b-a) > eps* abs(b) and k < kmax:
               x = (a+b)/2.0
               if np.sign(f(x)) == np.sign(f(b)):
                   b = x
               else:
                   a = x
               k += 1
        return x,k

        