import numpy as np
from scipy.stats import rv_histogram


def prop_ddelta(t):
    h = np.zeros(len(t)-1)  
    if t[0] != 0:
        return np.zeros(len(t))
    h[0]=1
    dist = rv_histogram((h,t), density=True)
    # Include a factor 2 because this is a unit for the convolution product as defined by conv() (trapezoidal integration)
    return 2*dist.pdf(t)


def res_ddelta(t):
    if t[0] != 0:
        return np.zeros(len(t))
    h = prop_ddelta(t)
    return 1 - trapz(h, t)


def tarray(J, t=None, dt=1.0):
    if t is None:
        t = dt*np.arange(len(J))
    else:
        t = np.array(t)
        if len(t) != len(J):
            raise ValueError('Time array must have same length as the input.')   
    return t


def trapz(f, t=None, dt=1.0):
    """Trapezoidal integration. 
    Can be replaced by scipy function."""
    f = np.array(f)
    n = len(f)
    t = tarray(f, t=t, dt=dt)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g


def expconv(f, T, t=None, dt=1.0):
    """Convolve a 1D-array with a normalised exponential.

    Uses an efficient and accurate numerical formula to calculate the convolution, as detailed in the appendix of `Flouri et al (2016) <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.25991>`_

    Args:
        a (array_like): the 1D array to be convolved.
        T (float): the characteristic time of the exponential function. time and T must be in the same units.
        t (array_like): the time points where the values of a are defined. tThese do not have to to be equally spaced.
        dt (float): spacing between time points (ignored if t is defined).
        
    Returns:
        np.ndarray: a numpy array of the same shape as ca.

    Notes: 
        by definition, expconv preserves the area under a
        if T=0, expconv returns a copy of a
    """
    if T==0: 
        return f
    f = np.array(f)
    n = len(f)
    t = tarray(f, t=t, dt=dt)
    g = np.zeros(n)
    x = (t[1:n] - t[0:n-1])/T
    df = (f[1:n] - f[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = f[0:n-1]*E0 + df*E1
    for i in range(0,n-1):
        g[i+1] = E[i]*g[i] + add[i]      
    return g


def uconv(f, h, dt):
    n = len(f) 
    g = np.empty(n)
    h = np.flip(h)
    g[0] = 0
    for i in np.arange(1, n):
        g[i] = np.trapz(f[:i+1]*h[-(i+1):], dx=dt)
    return g


def conv(f, h, t=None, dt=1.0):
    if len(f) != len(h):
        raise ValueError('f and h must have the same length.')
    if t is None:
        return uconv(f, h, dt)
    n = len(t)
    g = np.zeros(n)
    for k in range(1, n):
        fk = np.interp(t[k]-t[:k+1], t[:k+1], f[:k+1], left=0, right=0)
        g[k] = np.trapz(h[:k+1]*fk, t[:k+1])
    return g



if __name__ == "__main__":
    
    print('All tools tests passed!!')
