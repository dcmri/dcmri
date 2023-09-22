import numpy as np

def expconv(T, time, a):
    """Convolve a 1D-array with a normalised exponential.

    expconv() uses an efficient and accurate numerical formula to calculate the convolution,
    as detailed in the appendix of Flouri et al., Magn Reson Med, 76 (2016), pp. 998-1006.

    Note (1): by definition, expconv preserves the area under a(time)
    Note (2): if T=0, expconv returns a copy of a

    Arguments
    ---------
    a : numpy array
        the 1D array to be convolved.
    time : numpy array
        the time points where the values of ca are defined
        these do not have to to be equally spaced.
    T : float
        the characteristic time of the the exponential function.
        time and T must be in the same units.

    Returns
    -------
    a numpy array of the same shape as ca.

    Example
    -------
    coming soon..

    """
    if T==0: return a

    n = len(time)
    f = np.zeros(n)
    x = (time[1:n] - time[0:n-1])/T
    da = (a[1:n] - a[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = a[0:n-1]*E0 + da*E1
    for i in range(0,n-1):
        f[i+1] = E[i]*f[i] + add[i]      
    return f

def trapz(t, f):
    n = len(f)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g

def prop_plug(t, J, K):
    if K==0:
        return np.zeros(len(t))
    return np.interp(t-1/K, t, J, left=0) 

def prop_comp(t, J, K):
    if K == 0:
        return np.zeros(len(t))
    return expconv(1/K, t, J)

def res_plug(t, J, K):
    if K==0:
        return trapz(t, J)
    Jo = prop_plug(t, J, 1/K)
    return trapz(t, J-Jo)

def res_comp(t, J, K):
    if K == 0:
        return trapz(t, J)
    return expconv(1/K, t, J)/K