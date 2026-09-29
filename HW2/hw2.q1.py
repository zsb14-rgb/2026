import numpy as np
from bisect import bisect
from newton import newton

f = lambda x: np.cos(x) * np.exp(-x / 5)
fprime = lambda x: -np.exp(-x / 5) * (np.sin(x) + np.cos(x) / 5)

guesses = [1.5, 4.5, 8.0, 11.0, 14.0]   # Newton starting points, read off a plot of f(x)

print(f"{'n':>2} {'exact':>12} {'bisect':>12} {'k':>3} {'err':>9} {'newton':>12} {'k':>3} {'err':>9}")
for n in range(5):
    exact = np.pi / 2 + n * np.pi
    xb, kb = bisect(f, n * np.pi, (n + 1) * np.pi)   # each [n*pi, (n+1)*pi] brackets one zero
    xn, kn = newton(f, fprime, guesses[n])           # start from eyeballed guess
    print(f"{n+1:>2} {exact:12.8f} {xb:12.8f} {kb:3d} {abs(xb-exact):9.1e} "
          f"{xn:12.8f} {kn:3d} {abs(xn-exact):9.1e}")