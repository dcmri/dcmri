"""
Helper functions
"""

import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np

def quad(t: np.ndarray,
         K: np.ndarray
         ) -> np.ndarray:
    """Returns a quadratic function.
    
    Uses the `quadratic` function to calculate the
    quadratic function of t that goes through the
    three points (ti, Ki).

    Args:
        t: numpy array of time series, t.
        K: numpy array of rate constant series, K (= 1/T).

    Returns:
        Quadratic function of t as an np.ndarray.
    """
    nt = len(t)
    mid = math.floor(nt/2)
    return quadratic(t, t[0], t[mid], t[-1], K[0], K[1], K[2])


def lin(t: np.ndarray,
        K: np.ndarray
        ) -> np.ndarray:
    """Returns a linear function. 
    
    Uses the `linear` function to calculate
    the linear function of t that goes through
    the two points (ti, Ki).

    Args:
        t: numpy array of time series, t.
        K: numpy array of rate constant series, K (= 1/T).

    Returns:
        Linear function of t as an np.ndarray.
    """
    return linear(t, t[0], t[-1], K[0], K[1])


def quadratic(x: np.ndarray,
              x1: int,
              x2: int,
              x3: int,
              y1: int,
              y2: int,
              y3: int
              ) -> np.ndarray:
    """Returns a quadratic function.
     
    Calculates the quadratic function of x that
    goes through the three points (xi, yi).

    Args:
        x: numpy array.
        y: numpy array.

    Returns:
        Quadratic function of x as an np.ndarray.
    """
    a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
    a /= (x1-x2)*(x1-x3)*(x2-x3)
    b = (y2-y1)/(x2-x1) - a*(x1+x2)
    c = y1-a*x1**2-b*x1
    return a*x**2+b*x+c


def linear(x: np.ndarray,
           x1: int,
           x2: int,
           y1: int,
           y2: int):
    """Returns a linear function.
     
    Calculates the linear function of x that
    goes through the two points (xi, yi)
    
    Args:
    x: numpy array.
    y: numpy array.

    Returns:
        Linear function of x as an np.ndarray.
    """
    b = (y2-y1)/(x2-x1)
    c = y1-b*x1
    return b*x+c
