import numpy as np
import matplotlib.pyplot as plt
 
 
# Logistic function (same as logistic.m)
def logistic(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y
 
 
# Part (a)
# Grid of points
datax1 = np.arange(0, 2.01, 0.08)
datax2 = np.arange(0, 2.01, 0.08)
 
# C values for part (a)
c0 = -1.0
c1 = 0.0
c2 = 0.0
c3 = 1.0
c4 = 1.0
 
# Plot each point: red if z > 0, blue if z < 0
plt.figure()
for i in range(len(datax1)):
    for j in range(len(datax2)):
        z = c0 + c1*datax1[i] + c2*datax2[j] + c3*datax1[i]**2 + c4*datax2[j]**2
        if z > 0.0:
            plt.plot(datax1[i], datax2[j], 'ro', mfc='none')
        else:
            plt.plot(datax1[i], datax2[j], 'bo', mfc='none')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(a) Blue z<0, Red z>0')
 
# Find y = logistic(z) on a grid
x1 = np.arange(0, 2.01, 0.1)
x2 = np.arange(0, 2.01, 0.1)
cvec = np.array([c0, c1, c2, c3, c4])
z = np.zeros((len(x1), len(x2)))
y = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        z[i, j] = cvec @ np.array([1, x1[i], x2[j], x1[i]**2, x2[j]**2])
        y[i, j] = logistic(z[i, j])
 
# Contour plot of y
plt.figure()
plt.contourf(x1, x2, y.T)
plt.contour(x1, x2, y.T, colors='k')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(a) Contours of y(z)')
 
 
#Part (b)
# Grid of points
datax1 = np.arange(0, 2.01, 0.08)
datax2 = np.arange(0, 2.01, 0.08)
 
# C values for part (b)
c0 = -2.0
c1 = 1.0
c2 = 2.0
c3 = 0.0
c4 = 0.0
 
# Plot each point: red if z > 0, blue if z < 0
plt.figure()
for i in range(len(datax1)):
    for j in range(len(datax2)):
        z = c0 + c1*datax1[i] + c2*datax2[j] + c3*datax1[i]**2 + c4*datax2[j]**2
        if z > 0.0:
            plt.plot(datax1[i], datax2[j], 'ro', mfc='none')
        else:
            plt.plot(datax1[i], datax2[j], 'bo', mfc='none')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(b) Blue z<0, Red z>0')
 
# Find y = logistic(z) on a grid
x1 = np.arange(0, 2.01, 0.1)
x2 = np.arange(0, 2.01, 0.1)
cvec = np.array([c0, c1, c2, c3, c4])
z = np.zeros((len(x1), len(x2)))
y = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        z[i, j] = cvec @ np.array([1, x1[i], x2[j], x1[i]**2, x2[j]**2])
        y[i, j] = logistic(z[i, j])
 
# Contour plot of y
plt.figure()
plt.contourf(x1, x2, y.T)
plt.contour(x1, x2, y.T, colors='k')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(b) Contours of y(z)')
 
 
# Part (c)
# Grid of points
datax1 = np.arange(0, 2.01, 0.08)
datax2 = np.arange(0, 2.01, 0.08)
 
# C values for part (c)
c0 = -0.5
c1 = -1.0
c2 = -0.5
c3 = 1.0
c4 = 0.5
 
# Plot each point: red if z > 0, blue if z < 0
plt.figure()
for i in range(len(datax1)):
    for j in range(len(datax2)):
        z = c0 + c1*datax1[i] + c2*datax2[j] + c3*datax1[i]**2 + c4*datax2[j]**2
        if z > 0.0:
            plt.plot(datax1[i], datax2[j], 'ro', mfc='none')
        else:
            plt.plot(datax1[i], datax2[j], 'bo', mfc='none')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(c) Blue z<0, Red z>0')
 
# Find y = logistic(z) on a grid
x1 = np.arange(0, 2.01, 0.1)
x2 = np.arange(0, 2.01, 0.1)
cvec = np.array([c0, c1, c2, c3, c4])
z = np.zeros((len(x1), len(x2)))
y = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        z[i, j] = cvec @ np.array([1, x1[i], x2[j], x1[i]**2, x2[j]**2])
        y[i, j] = logistic(z[i, j])
 
# Contour plot of y
plt.figure()
plt.contourf(x1, x2, y.T)
plt.contour(x1, x2, y.T, colors='k')
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('(c) Contours of y(z)')
 
# Save all 6 plots to the same folder as this script
import os
outdir = os.path.dirname(os.path.abspath(__file__))
names = ['a_points', 'a_contour', 'b_points', 'b_contour', 'c_points', 'c_contour']
for num, name in zip(plt.get_fignums(), names):
    plt.figure(num).savefig(os.path.join(outdir, 'hw3_q2_' + name + '.png'), dpi=150)

# Show all 6 plots
plt.show()