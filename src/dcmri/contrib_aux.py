"""
Helper functions
"""

import math

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

