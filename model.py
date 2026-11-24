import numpy as np

def modelU(x, y, m = 1, k = 1, R = 1):
    r = np.sqrt(x**2 + y**2)
    return np.where(r < R, (k / 2 * R) * (3 -  ( r**2 / R**2)), k/r)

def force(x, y, m = 1, k = 1, R = 1):
    r = np.sqrt(x**2 + y**2)
    coeff = k / np.maximum(r, R) ** 3
    return x * coeff, y * coeff


