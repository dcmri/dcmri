"""
Helper functions
"""

import math
import numpy as np

def trapz(t, f):
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def quad(t, K):
    nt = len(t)
    mid = math.floor(nt/2)
    return quadratic(t, t[0], t[mid], t[-1], K[0], K[1], K[2])

def lin(t, K):
    return linear(t, t[0], t[-1], K[0], K[1])

def quadratic(x, x1, x2, x3, y1, y2, y3):
    """returns a quadratic function of x 
    that goes through the three points (xi, yi)"""

    a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
    a /= (x1-x2)*(x1-x3)*(x2-x3)
    b = (y2-y1)/(x2-x1) - a*(x1+x2)
    c = y1-a*x1**2-b*x1
    return a*x**2+b*x+c

def linear(x, x1, x2, y1, y2):
    """returns a linear function of x 
    that goes through the two points (xi, yi)"""

    b = (y2-y1)/(x2-x1)
    c = y1-b*x1
    return b*x+c

