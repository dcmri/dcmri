import numpy as np

def ddist(H, T, t):
    # discrete distribution - T is an array of times with the boundaries of the histogram bins
    h = np.zeros(len(t))
    for k in range(len(T)-1):
        h += H[k]*dstep(T[k], T[k+1], t)
    return h

def dstep(T0, T1, t):
    # Helper function - discrete step
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    n = len(t)
    i = np.where((t > T0)*(t < T1))[0]
    if len(i) == 0:
        return ddelta((T0+T1)/2, t)
    i0, i1 = i[0], i[-1]
    t0, t1 = t[i0], t[i1]
    hi = 0
    if i0 > 0:
        u0 = (t0-T0)/(t0-t[i0-1])
        hi += 0.5*(1+u0)*(t0-t[i0-1])
        if i0 > 1:
            hi += 0.5*u0*(t[i0-1]-t[i0-2])
    hi += t1-t0
    if i1 < n-1:
        u1 = (T1-t1)/(t[i1+1]-t1)
        hi += 0.5*(1+u1)*(t[i1+1]-t1)
        if i1 < n-2:
            hi += 0.5*u1*(t[i1+2]-t[i1+1])
    h = np.zeros(n)
    h[i] = 1/hi
    if i0 > 0:
        h[i0-1] = u0/hi
    if i1 < n-1:
        h[i1+1] = u1/hi
    return h


def ddelta(T, t):
    # Helper function - discrete delta
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    n = len(t)
    h = np.zeros(n)
    if T < t[0]:
        return h
    if T > t[-1]:
        return h
    if T == t[0]:
        h[0] = 2/(t[1]-t[0])
        return h
    if T == t[-1]:
        h[-1] = 2/(t[-1]-t[-2])
        return h
    i = np.where(T >= t)[0][-1]
    u = (T-t[i])/(t[i+1]-t[i])
    if i == 0:
        h[i] = (1-u)*2/(t[i+1]-t[i])
    else:
        h[i] = (1-u)*2/(t[i+1]-t[i-1])
    if i == n-2:
        h[i+1] = u*2/(t[i+1]-t[i])
    else:
        h[i+1] = u*2/(t[i+2]-t[i])
    return h