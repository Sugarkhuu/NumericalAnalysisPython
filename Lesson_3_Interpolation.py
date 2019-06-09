import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([5, -7, 15])

poly = lagrange(x, y)

xpol = np.linspace(x[0], x[2], 5)
ypol = poly(xpol)

plt.scatter(x, y, marker='s', c='r')
plt.plot(xpol, ypol)
