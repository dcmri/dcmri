import math
import warnings

import numpy as np
from scipy.special import gamma
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


def train(predict, time, signal, pars, free, x=None, reset=False, **kwargs):
    """Optimization logic using normalized parameter values."""

    if free == {}:
        return None, None
    
    if isinstance(signal, tuple):
        signal = np.concatenate(signal)

    p0 = _compute_normalized_pars(pars, free, x)

    def predict_normalized(_, *normalized_pars):
        _update_original_pars(pars, normalized_pars, free, x)
        if x is None:
            ypred = predict(time)
        else:
            ypred = predict(time, x)
        return np.concatenate(ypred) if isinstance(ypred, tuple) else ypred

    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, signal, p0, bounds=(0, 1), **kwargs
        )
        sdev = _sdev(pcov, free)
    except RuntimeError as e:
        warnings.warn(f"Curve fit failed: {e}. Using initial values.")
        fitted_pars, pcov, sdev = p0, None, None

    if reset:
        # Set state to original values
        _update_original_pars(pars, p0, free, x)
    else:
        # Set state to final values
        _update_original_pars(pars, fitted_pars, free, x)
    
    vals = {p: fitted_pars[i] for i, p in enumerate(free)}
    return vals, sdev, pcov


def _compute_normalized_pars(original_pars, free_pars, x=None):
    if x is None:
        return [normalize(original_pars[p], free_pars[p]) for p in free_pars]
    else:
        return [normalize(original_pars[p][x], free_pars[p]) for p in free_pars]

def _update_original_pars(original_pars, normalized_pars, free, x=None):
    for i, p in enumerate(free):
        if x is None:
            original_pars[p] = renormalize(normalized_pars[i], free[p])
        else:
            original_pars[p][x] = renormalize(normalized_pars[i], free[p])

def normalize(v, bounds):
    return (v - bounds[0]) / (bounds[1] - bounds[0])

def renormalize(v, bounds):
    return v * (bounds[1] - bounds[0]) + bounds[0]

def _sdev(pcov, free):
    sdev = {}
    i = 0
    for p in free.keys():
        sdev[p] = renormalize(np.sqrt(pcov[i,i]), free[p])
        i += 1
    return sdev



def interp(y, x, pos=False, floor=False, lower=None, upper=None) -> np.ndarray:
    """Interpolate uniformly sampled data. 

    This function is a convenience wrapper for standard interpolation, used in dcmri for instance to parametrize non-stationat models.

    Args:
        y (array-like): List of values to interpolate. These are assumed to be uniformly distributed over x.
        x (array-like): Interpolate y at these locations. 
        pos (bool, optional): return only positive values. Defaults to False.
        floor (bool, optional): Return only results higher than the lowest value in y. Defaults to False.
        lower (float, optional): Lower bound for interpolation. If provided, values below this will be set to lower. Defaults to None.
        upper (float, optional): Upper bound for interpolation. If provided, values above this will be set to upper. Defaults to None.

    Returns:
        np.ndarray: array of the same length as x, containing the values of y interpolated on x.

    Notes:
        The function uses linear interpolate when y has length 2, quadratic when y has length 3, and cubic spline interpolation otherwise.

    Example:
        >>> import dcmri as dc
        >>> dc.interp([1,3,2], np.arange(10))
        array([1.        , 1.68888889, 2.23333333, 2.63333333, 2.88888889,
               3.        , 2.96666667, 2.78888889, 2.46666667, 2.        ])
    """

    # Interpolate y on x, assuming y-values are uniformly distributed over the x-range
    if np.isscalar(y):
        yi = y*np.ones(len(x))
    elif np.size(y) == 1:
        yi = y[0]*np.ones(len(x))
    elif np.size(y) == 2:
        yi = _lin(x, y)
    elif np.size(y) == 3:
        yi = _quad(x, y)
    else:
        x_y = np.linspace(np.amin(x), np.amax(x), len(y))
        yi = CubicSpline(x_y, y)(x)
    if pos:
        yi[yi < 0] = 0
    if floor:
        y0 = np.amin(y)
        yi[yi < y0] = y0
    if lower is not None:
        yi[yi < lower] = lower
    if upper is not None:
        yi[yi > upper] = upper
    return yi


def _quad(t, K):
    # Helper
    nt = len(t)
    mid = math.floor(nt/2)
    return _quadratic(t, t[0], t[mid], t[-1], K[0], K[1], K[2])


def _lin(t, K):
    # Helper
    return _linear(t, t[0], t[-1], K[0], K[1])


def _linear(x, x1, x2, y1, y2):
    # Helper
    # returns a linear function of x
    # that goes through the two points (xi, yi)
    L1 = (x2-x)/(x2-x1)
    L2 = (x-x1)/(x2-x1)
    return y1*L1 + y2*L2


def _quadratic(x, x0, x1, x2, y0, y1, y2):
    # Helper
    # returns a quadratic function of x
    # that goes through the three points (xi, yi)
    L0 = (x-x1)*(x-x2)/((x0-x1)*(x0-x2))
    L1 = (x-x0)*(x-x2)/((x1-x0)*(x1-x2))
    L2 = (x-x0)*(x-x1)/((x2-x0)*(x2-x1))
    return y0*L0 + y1*L1 + y2*L2


def tarray(n, t=None, dt=1.0):
    # Helper function - generate time array.
    if t is None:
        t = dt*np.arange(n)
    else:
        if not isinstance(t, np.ndarray):
            t = np.array(t)
        if len(t) != n:
            raise ValueError('Time array must have same length as the input.')
    return t


def trapz(f, t=None, dt=1.0):
    # Helper function - perform trapezoidal integration.
    # Replace by scipy.integrate.trapezoid
    f = np.array(f)
    n = len(f)
    t = tarray(n, t=t, dt=dt)
    g = np.empty(n)
    g[0] = 0
    for i in range(n-1):
        g[i+1] = g[i] + (t[i+1]-t[i]) * (f[i+1]+f[i]) / 2
    return g


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


def ddist(H, T, t):
    # discrete distribution - T is an array of times with the boundaries of the histogram bins
    h = np.zeros(len(t))
    for k in range(len(T)-1):
        h += H[k]*dstep(T[k], T[k+1], t)
    return h


def intprod(f, h, t=None, dt=1.0):
    # Helper function
    # Integrate the product of two piecewise linear functions
    # by analytical integration over each interval.
    # Derivation:
    # If f and h are linear between x and y, then we can define slopes:
    # dx = y-x
    # sf(x) = (f(y)-f(x))/dx
    # sh(x) = (h(y)-h(x))/dx
    # With this the integral over the interval becomes:
    # \int_x^y du f(u)h(u)
    # = \int_0^dx du [f(x)+usf(x)] [h(x)+ush(x)]
    # = f(x)*h(x)*dx
    # + [f(x)*sh(x)+sf(x)*h(x)]dx**2/2
    # + sf(x)*sh(x)*dx**3/3
    g = 0
    for l in range(len(f)-1):
        if t is not None:
            dt = t[l+1]-t[l]
        sf = (f[l+1]-f[l])/dt
        sh = (h[l+1]-h[l])/dt
        g += h[l]*f[l]*dt
        g += (sh*f[l]+sf*h[l])*dt**2/2
        g += sh*sf*dt**3/3
    return g


def _intstep(f, h, t=None, dt=1.0):
    g = f*h
    if t is None:
        return 0.5*np.sum(g[1:]+g[:-1])*dt
    else:
        return 0.5*np.sum((g[1:]+g[:-1])*(t[1:]-t[:-1]))


def uconv(f, h, dt=1.0, solver='trap'):
    # Helper function: convolution over uniformly sampled grid.
    n = len(f)
    g = np.zeros(n)
    h = np.flip(h)
    for k in range(1, n):
        if solver == 'trap':
            g[k] = intprod(f[:k+1], h[-(k+1):], dt=dt)
        elif solver == 'step':
            g[k] = _intstep(f[:k+1], h[-(k+1):], dt=dt)
    return g



def conv(f, h, t=None, dt=1.0, solver='step'):
    """Convolve two 1D-arrays.

    This function returns the convolution :math:`f(t)\\otimes h(t)`, using piecewise linear integration to approximate the integrals in the convolution product.

    Args:
        f (array_like): the first 1D array to be convolved.
        h (array_like): the second 1D array to be convolved.
        t (array_like, optional): the time points where the values of f are defined. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Raises:
        ValueError: if f and h have a different length.

    Returns:
        numpy.ndarray: a 1D numpy array of the same length as f and h.

    See Also:
        `expconv`, `biexpconv`, `nexpconv`, `stepconv`

    Notes: 
        The convolution product :math:`f(t)\\otimes h(t)` implemented by `conv` is explicitly defined as:

        .. math::
            g(t) = \\int_0^t du\\, f(u) h(t-u)

        `conv` returns an approximation to this continuous convolution product, calculated by piecewise linear integration. This is not to be confused with other convolution functions, such as `numpy.convolve` which performs discrete convolution. Tracer-kinetic theory is defined by continuous equations and therefore should be performed with `conv` for maximal accuracy, though the difference may be small at high temporal resolution.

        `conv` is generally applicable to any f(t) and h(t), but more accurate formulae for some special cases exists and should be used if available. An example is `expconv`, to be used when either f(t) or h(t) is an exponential.

    Example:
        Import package and create vectors f and h:

        >>> import dcmri as dc
        >>> f = [5,4,3,6]
        >>> h = [2,9,1,3]

        Calculate :math:`g(t) = f(t) \\otimes h(t)` over a uniformly sampled grid of time points with spacing dt=1:

        >>> dc.conv(f, h)
        array([ 0.        , 25.33333333, 41.66666667, 49.        ])

        Calculate the same convolution over a grid of time points with spacing dt=2:

        >>> dc.conv(f, h, dt=2)
        array([ 0.        , 50.66666667, 83.33333333, 98.        ])

        Calculate the same convolution over a non-uniform grid of time points:

        >>> t = [0,1,3,7]
        >>> dc.conv(f, h, t)
        array([  0.        ,  25.33333333,  57.41666667, 108.27083333])
    """
    if len(f) != len(h):
        raise ValueError('f and h must have the same length.')
    if t is None:
        return uconv(f, h, dt, solver=solver)
    n = len(t)
    g = np.zeros(n)
    tf = np.flip(t)
    f = np.flip(f)
    for k in range(1, n):
        tkf = t[k]-tf[-(k+1):]
        tk = np.unique(np.concatenate((t[:k+1], tkf)))
        fk = np.interp(tk, tkf, f[-(k+1):], left=0, right=0)
        hk = np.interp(tk, t[:k+1], h[:k+1], left=0, right=0)
        if solver == 'trap':
            g[k] = intprod(fk, hk, tk)
        elif solver == 'step':
            g[k] = _intstep(fk, hk, tk)
    return g


def inttrap(f, t, t0, t1):
    # Helper function: integrate f from t0 to t1
    ti = t[(t0 < t)*(t < t1)]
    ti = np.concatenate(([t0], ti, [t1]))
    fi = np.interp(ti, t, f, left=0, right=0)
    return trapezoid(fi, ti)


def stepconv(f, T, D, t=None, dt=1.0):
    """Convolve a 1D-array with a normalised step function.

    Args:
        f (array_like): the 1D array to be convolved.
        T (float): the central time point of the step function. 
        D (float): half-width of the step function, as a fraction of T. D must be less or equal to 1.
        t (array_like, optional): the time points where the values of f are defined, in the same units as T. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Raises:
        ValueError: if D > 1.

    Returns:
        numpy.ndarray: a 1D numpy array of the same length as f.

    See Also:
        `conv`, `expconv`, `biexpconv`, `nexpconv`

    Notes: 
        `stepconv` implements the same convolution product as `conv`, but is more accurate and faster in the special case where one of the factors is known to be a step function.

    Example:
        Import package, create a vector f and an array of time points:

        >>> import dcmri as dc
        >>> f = [5,4,3,6]
        >>> t = [0,2,4,7]

        Convolve f with a step function that is centered on t=3 with a half width of 1.5 = 0.5*3: 

        >>> dc.stepconv(f, 3, 0.5, t)
        array([0.        , 0.8125    , 3.64583333, 3.5625    ])
    """
    if D > 1:
        raise ValueError('The dispersion factor D must be <= 1')
    TW = D*T      # Half width of step
    T0 = T-TW     # Initial time point of step
    T1 = T+TW
    n = len(f)
    t = tarray(n, t=t, dt=dt)
    g = np.zeros(n)
    k = len(t[t < T0])
    ti = t[(T0 <= t)*(t <= T1)]
    for tk in ti:
        g[k] = inttrap(f, t, 0, tk-T0)
        k += 1
    ti = t[T1 < t]
    for tk in ti:
        g[k] = inttrap(f, t, tk-T1, tk-T0)
        k += 1
    return g/(2*TW)


def expconv(f, T, t=None, dt=1.0, tol=0):
    """Convolve a 1D-array with a normalised exponential.

    This function returns the convolution :math:`f(t)\\otimes\\exp(-t/T)/T` using an efficient and accurate numerical formula, as detailed in the appendix of `Flouri et al (2016) <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.25991>`_ 

    Args:
        f (array_like): the 1D array to be convolved.
        T (float): the characteristic time of the normalized exponential function. 
        t (array_like, optional): the time points where the values of f are defined, in the same units as T. If t=None, the time points are assumed to be uniformly spaced with spacing dt. Defaults to None.
        dt (float, optional): spacing between time points for uniformly spaced time points. This parameter is ignored if t is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: a 1D numpy array of the same length as f.

    See Also:
        `conv`, `biexpconv`, `nexpconv`, `stepconv`

    Notes: 
        `expconv` implements the same convolution product as `conv`, but is more accurate and faster in the special case where one of the factors is known to be an exponential:

        .. math::
            g(t) = \\frac{e^{-t/T}}{T} \\otimes f(t)

        In code this translates as:

        .. code-block:: python

            g = expconv(f, T, t)

        `expconv` should be used instead of `conv` whenever this applies. Since the transit time distribution of a compartment is exponential, this is an important use case.  

        `expconv` can calculate a convolution between two exponential factors, but in that case an analytical formula can be used which is faster and more accurate. It is implemented in the function `biexpconv`.

    Example:
        Import package and create a vector f:

        >>> import dcmri as dc
        >>> f = [5,4,3,6]

        Calculate :math:`g(t) = f(t) \\otimes \\exp(-t/3)/3` over a uniformly sampled grid of time points with spacing dt=1:

        >>> dc.expconv(f, 3)
        array([0.        , 1.26774952, 1.89266305, 2.6553402 ])

        Calculate the same convolution over a grid of time points with spacing dt=2:

        >>> dc.expconv(f, 3, dt=2)
        array([0.        , 2.16278873, 2.7866186 , 3.70082337])

        Calculate the same convolution over a non-uniform grid of time points:

        >>> t = [0,1,3,7]
        >>> dc.expconv(f, 3, t)
        array([0.        , 1.26774952, 2.32709015, 4.16571645])
    """

    if T == 0:
        return f
    f = np.array(f)
    n = len(f)
    t = tarray(n, t=t, dt=dt)
    x = (t[1:n] - t[0:n-1])/T 
    if 1/x.min() < tol: # very small T
        return f
    if t.max() / T < tol: # very large T
        # Large T - exponential is linear over the interval
        # g(t) = (1/T) * int_0^t du f(u) exp(-(t-u)/T)
        # g(t) = (1/T) * int_0^t du f(u) (1 - (t-u)/T)
        # g(t) = (1/T) * int_0^t du f(u) - (1/T^2) * int_0^t du f(u) (t-u)
        # g(t) = (1/T) * int_0^t du f(u) - (1/T^2) * int_0^t du f(u) t + (1/T^2) * int_0^t du f(u) u
        # g(t) = (1/T) * int_0^t du f(u) - (t/T^2) * int_0^t du f(u) + (1/T^2) * int_0^t du f(u) u
        # g(t) = (1/T)[1 - (t/T)] * int_0^t du f(u) + (1/T^2) * int_0^t du f(u) u
        # Tg = [1 - (t/T)] * int_0^t du f(u) + (1/T) * int_0^t du f(u) u 
        f0 = trapz(f, t=t, dt=dt)
        f1 = trapz(f*t, t=t, dt=dt)
        return f0 / T + (f1 - t * f0) / T**2
    df = (f[1:n] - f[0:n-1])/x
    E = np.exp(-x)
    E0 = 1-E
    E1 = x-E0
    add = f[0:n-1]*E0 + df*E1
    g = np.zeros(n, dtype=E.dtype) 
    for i in range(0, n-1):
        g[i+1] = E[i]*g[i] + add[i]
    return g




def biexpconv(T1, T2, t):
    """Convolve two normalised exponentials analytically.

    Args:
        T1 (float): the characteristic time of the first exponential function. 
        T2 (float): the characteristic time of the second exponential function, in the same units as T1. 
        t (array_like, optional): the time points where the values of f are defined, in the same units as T. 

    Returns:
        numpy.ndarray: The result of the convolution as a 1D array.

    See Also:
        `conv`, `expconv`, `nexpconv`, `stepconv`

    Notes: 
        `biexpconv` returns the exact analytical result of the following convolution:

        .. math::
            g(t) = \\frac{e^{-t/A}}{A} \\otimes \\frac{e^{-t/B}}{B}

        The formula is a biexponential with unit area:

        .. math::
            g(t) = \\frac{Ae^{-t/A}-Be^{-t/B}}{A-B}

        In code this translates as:

        .. code-block:: python

            g = biexpconv(A, B, t)

    Example:
        Import package and create a vector of uniformly sampled time points t with spacing 5.0s:

        >>> import dcmri as dc
        >>> t = 5.0*np.arange(4)

        Calculate the convolution of two normalised exponentials with time constants 10s and 15s:

        >>> g = dc.biexpconv(10, 15, t)
        array([-0.        ,  0.02200013,  0.02910754,  0.02894986])
    """
    if T1 == T2:
        return (t/T1) * np.exp(-t/T1)/T1
    else:
        return (np.exp(-t/T1)-np.exp(-t/T2))/(T1-T2)


def nexpconv(n, T, t):
    """Convolve n identical normalised exponentials analytically

    Args:
        n (float): number of exponentials. Since an analytical formula is used this can also be non-integer.
        T (float): the characteristic time of the exponential. 
        t (array_like, optional): the time points where the values of f are defined, in the same units as T. 

    Returns:
        numpy.ndarray: The result of the convolution as a 1D array.

    Raises:
        ValueError: if n<1 and if T<0

    See Also:
        `conv`, `expconv`, `biexpconv`, `stepconv`

    Notes: 
        `nexpconv` returns the exact analytical result of the following n convolutions:

        .. math::
            g(t) = \\frac{e^{-t/T}}{T} \\otimes \\ldots \\otimes \\frac{e^{-t/T}}{T} 

        The result is a gamma variate function with unit area:

        .. math::
            g(t) = \\frac{1}{\\Gamma(n)}\\left(\\frac{t}{T}\\right)^{n-1} \\frac{e^{-t/T}}{T} 

        Note the gamma-variate function becomes unstable due to numerical 
        overflow at large n and/or short T. 
        This function handles that situation by moving to a numerical 
        solution, but this slows down the computations significantly.
        Callers should if possible avoid this situation by placing suitable 
        bounds on the parameters.

    Example:
        Import package and create a vector of uniformly sampled time points t with spacing 5.0s:

        >>> import dcmri as dc
        >>> t = 5.0*np.arange(4)

        Calculate the convolution of 4 normalised exponentials with time constants 5s:

        >>> g = dc.nexpconv(4, 5, t)
        array([0.        , 0.01226265, 0.03608941, 0.04480836])
    """
    if T < 0:
        raise ValueError('T must be non-negative')
    if n < 1:
        raise ValueError('n cannot be smaller than 1')
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    u = t/T

    # Calculate gamma variate, silencing the warnings
    # as invalid cases are handled properly in the next line.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        g = u**(n-1) * np.exp(-u)/T/gamma(n)

    # Handle overflow/invalid values
    if False in np.isfinite(g):
        # At large n the analytical formula runs into overflow
        # use numerical calculation in this case (slower and less accurate)
        # The caller is responsible for avoiding this situation by setting suitable parameter bounds
        g = np.exp(-t/T)/T
        n0 = int(np.floor(n))
        for _ in range(n0-1):
            g = expconv(g, T, t)
        if n != n0:
            # Interpolate between n0 and n0+1
            g1 = expconv(g, T, t)
            u = n-n0
            g = g*u + g1*(1-u)
    return g





def sample(t, tp, Sp, dt=None) -> np.ndarray:
    """Sample a signal at given time points.

    Args:
        t (array): The 1D time points at which to evaluate the signal.
        tp (array): the 1D time points of the signal to be sampled.
        Sp (array): the 1D or 2D values (n_samples, n_times) of the signal to be sampled.
        dt (float, optional): sampling duration.

    Returns:
        np.ndarray: Signals sampled at times t.
    """
    t = np.asarray(t)
    tp = np.asarray(tp)
    Sp = np.asarray(Sp)
    
    # Handle 1D input by promoting it to 2D (1, n_times)
    is_1d = Sp.ndim == 1
    if is_1d:
        Sp = Sp[np.newaxis, :]

    if t.size == 0:
        return np.array([])

    # 1. With dt=0 this is just interpolation
    if dt is None or dt == 0:
        sig_interp = interp1d(tp, Sp, kind='linear', axis=-1, 
                              bounds_error=False, fill_value=0)
        res = sig_interp(t)
        return res.flatten() if is_1d else res

    # 2. Windowed Trapezoidal Logic
    cum_int = cumulative_trapezoid(Sp, tp, initial=0, axis=-1)
    
    # Use fill_value=(0, 'extrapolate') or a constant to handle boundaries
    # Since we want it to be 0 outside the range, we manually handle the right-side fill
    int_interp = interp1d(tp, cum_int, kind='linear', axis=-1, 
                          bounds_error=False, fill_value=(0, np.nan))
    
    t_start = t - dt/2
    t_end = t + dt/2

    F_start = int_interp(t_start)
    F_end = int_interp(t_end)

    # Correctly handle boundaries for any Sp shape
    tp_min, tp_max = tp[0], tp[-1]
    total_integral = cum_int[:, -1][:, np.newaxis] # Shape (n_samples, 1)

    # If t < tp_min -> 0
    # If t > tp_max -> total_integral
    F_start = np.where(t_start < tp_min, 0, F_start)
    F_start = np.where(t_start > tp_max, total_integral, F_start)

    F_end = np.where(t_end < tp_min, 0, F_end)
    F_end = np.where(t_end > tp_max, total_integral, F_end)

    Ss = (F_end - F_start) / dt
    
    return Ss.flatten() if is_1d else Ss


# def _orig_sample(t, tp, Sp, dt=None) -> np.ndarray:
#     """Sample a signal at given time points.

#     Args:
#         t (array-like): The time points at which to evaluate the signal.
#         tp (array-like): the time points of the signal to be sampled.
#         Sp (array-like): the values of the signal to be sampled. Values that are outside of the range are set to zero.
#         dt (float, optional): sampling duration. If this is not provided, linear interpolation between the data points is used.  Defaults to None.

#     Returns:
#         np.ndarray: Signals sampled at times t.
#     """
#     if len(t) == 0:
#         return np.array([])
#     tmax = max(t)
#     tpmax = max(tp)
#     if tpmax < tmax:
#         raise ValueError(
#             f"Cannot sample until time {tmax}. "
#             f"The largest time point that can be sampled is {tpmax}."  
#         )
#     if dt is None:
#         return np.interp(t, tp, Sp, left=0, right=0)
#     if dt == 0:
#         return np.interp(t, tp, Sp, left=0, right=0)
#     Ss = np.zeros(len(t))
#     for k, tk in enumerate(t):
#         tb = [tk-dt/2, tk+dt/2]
#         Sb = np.interp(tb, tp, Sp)
#         i = (tp > tb[0]) & (tp < tb[1])
#         ti = np.concatenate(([tb[0]], tp[i], [tb[1]]))
#         Si = np.concatenate(([Sb[0]], Sp[i], [Sb[1]]))
#         Ss[k] = trapezoid(Si, ti)/dt
#     return Ss


def add_noise(signal, sdev: float) -> np.ndarray:
    """Add noise to an MRI magnitude signal.

    Args:
        signal (array-like): Signal values.
        sdev (float): Standard deviation of the noise.

    Returns:
        np.ndarray: signal with noise added.
    """
    noise_x = np.random.normal(0, sdev, np.size(signal))
    noise_y = np.random.normal(0, sdev, np.size(signal))
    signal = np.sqrt((signal+noise_x)**2 + noise_y**2)
    return signal


def loss(ypred, ydata, metric='NRMS', nfree=None) -> float:
    """_summary_

    Args:
        ypred (array): predictions
        ydata (array): data
        metric (str, optional): either RMS (Root-mean-square), 
            NRMS (normalised RMS), AIC (Akaike Information Criterion), 
            cAIC (corrected AIC) or BIC (Bayesian Information Criterion). 
            Defaults to 'NRMS'.
        nfree (float, optional): Number of free parameters (required for 
            AIC, cAIC and BIC). Defaults to None.

    Raises:
        ValueError: raised if nfree=None for loss functions that require it

    Returns:
        float: loss value
    """
    if metric == 'RMS':
        loss = np.linalg.norm(ypred - ydata, axis=-1)
    elif metric == 'NRMS':
        ynorm = np.linalg.norm(ydata, axis=-1)
        yerr = np.linalg.norm(ypred - ydata, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            loss = 100*yerr/ynorm
    elif metric == 'AIC':
        rss = np.sum((ypred-ydata)**2, axis=-1)
        n = ydata.shape[-1]
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*2 + n*np.log(rss/n)
    elif metric == 'cAIC':
        rss = np.sum((ypred-ydata)**2)
        n = ydata.shape[-1]
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*2 + n*np.log(rss/n) + 2*nfree*(nfree+1)/(n-nfree-1)
    elif metric == 'BIC':
        rss = np.sum((ypred-ydata)**2, axis=-1)
        n = ydata.shape[-1]
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*np.log(n) + n*np.log(rss/n)
    return loss


# def transfer_rate(y, x, bounds=(-np.inf, np.inf)) -> np.ndarray:
#     """Return a time-varying transfer rate k(t) defined by 

#     .. math::

#         k(x) = k_i \frac{1 + r * (x-x_0)}{1 + h * (x-x_0)}

#     given :math:`k_i` and the values :math:`k_m, k_f` at the middle 
#     and the end of the x-interval, respectively. The function derives 
#     the values for r and h.
        
#     Args:
#         y (array): 3-element array with values :math:`k_i, k_m, k_f`.
#         x (array: x-values where the function is to be defined
#         bounds (tuple, optional): Lower and upper bounds for the result. 
#           The function is clipped to this range. Defaults to (-np.inf, np.inf).

#     Returns:
#         np.ndarray: k(x)
#     """
#     # Linear diurnal variation and MM-effect of drug concentration:

#     # k(t) = k0 * (1 + r * t) / (1 + c(t) / cm)              -- with cm > 0 and 1 + r * t > 0

#     # Linear variation in drug concentration:

#     # k(t) = k0 * (1 + r * t) / (1 + (c0 + s * t) / cm)      -- with c0 + s * t > 0 and c0 > 0

#     # Simplify:

#     # k(t) = k0 * (1 + r * t) / (1 + c0 / cm + (s / cm) * t)
#     # k(t) = [k0 / (1 + c0 / cm)] * (1 + r * t) / (1 + [ (s / cm) / (1 + c0 / cm )] * t)

#     # Model:

#     # k(t) = ki * (1 + r * t) / (1 + h * t)

#     # r quantifies diurnal variation, h=drug dependence

#     # khe: r!=0, h=?
#     # kbh: r==0, h=?

#     # baseline: h=0, kbh: r=0; khe  r != 0
#     # drug visit 

#     # # reparameterize with km (mid) and kf (end)

#     # km = ki * (1 + r * tm) / (1 + h * tm)
#     # kf = ki * (1 + r * tf) / (1 + h * tf)

#     # solve for h, r:

#     # km * (1 + h * tm) = ki * (1 + r * tm)
#     # kf * (1 + h * tf) = ki * (1 + r * tf)

#     # km + h * km * tm = ki + r * ki * tm
#     # kf + h * kf * tf = ki + r * ki * tf

#     # km - ki = r * ki * tm - h * km * tm
#     # kf - ki = r * ki * tf - h * kf * tf

#     # km - ki       ki * tm    - km * tm       r
#     #           =
#     # kf - ki       ki * tf    - kf * tf       h

#     # det = - ki * tm * kf * tf + ki * tf * km * tm 
#     # = (- kf + km) * tm * tf * ki = 0
#     # iff
#     # km = kf

#     # First consider the case km=kf:

#     # kf - ki = r * ki * tm - h * kf * tm
#     # kf - ki = r * ki * tf - h * kf * tf

#     # From the 1st:

#     # h = (r * ki * tm - kf + ki) / (kf * tm)

#     # Insert in the 2nd:

#     # kf - ki = (kf - ki) * tf / tm

#     # Since tf != tm this is only possible if kf=ki, ie. the constant solution

#     # If km=kf => r=0 and h=0
#     # else invert the matrix

#     #y = [ki, km, kf]

#     ki, km, kf = y[0], y[1], y[2]
    
#     tf = 1
#     tm = 0.5

#     if y[1] == y[2]:
#         r = 0
#         h = 0
#     else:
#         mat = np.array([[ki * tf, - kf * tf], [ki * tm, - km * tm]])
#         Y = np.array([kf - ki, km - ki])
#         X = np.linalg.inv(mat).dot(Y)
#         r = X[0]
#         h = X[1]

#     t  =x-x[0]
#     kt = ki * (1 + r * t) / (1 + h * t)

#     kt[kt < bounds[0]] = bounds[0]
#     kt[kt > bounds[1]] = bounds[1]

#     return kt


# if __name__=='__main__':

#     import matplotlib.pyplot as plt

#     y = [1, 2, 1]
#     t = np.linspace(0,1,100)
#     # k = transfer_rate(y, t, bounds=(-np.inf, np.inf))
#     r, h = -100, -10
#     k = y[0] * (1 + r * t) / (1 + h * t)

#     plt.plot(t, k)
#     plt.show()
    

#     #print(k)
