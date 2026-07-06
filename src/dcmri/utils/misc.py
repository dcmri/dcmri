import math

import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import rice
from scipy.optimize import minimize



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


def interp(y, x, pos=False, floor=False, lower=None, upper=None) -> np.ndarray:
    """Interpolate uniformly sampled data. 

    This function is a convenience wrapper for standard interpolation, used in dcmri for instance to parametrize non-stationary models.
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

    tp_min, tp_max = tp[0], tp[-1]
    
    # --- Exception for tp_min == tp_max ---
    if tp_min == tp_max:
        # Broadcast the single column of values across the length of t
        # Sp[:, 0][:, np.newaxis] has shape (n_samples, 1) -> tiles to (n_samples, len(t))
        res = np.tile(Sp[:, 0][:, np.newaxis], (1, t.size))
        return res.flatten() if is_1d else res
    
    # Get boundary values for extrapolation: shape (n_samples, 1)
    sp_left = Sp[:, 0][:, np.newaxis]
    sp_right = Sp[:, -1][:, np.newaxis]

    # --- 1. Pure Interpolation Path ---
    if dt is None or dt == 0:
        # fill_value=(left_val, right_val) extends the edge values constantly
        sig_interp = interp1d(tp, Sp, kind='linear', axis=-1, 
                              bounds_error=False, fill_value=(Sp[:, 0], Sp[:, -1]))
        res = sig_interp(t)
        return res.flatten() if is_1d else res

    # --- 2. Windowed Trapezoidal Logic ---
    cum_int = cumulative_trapezoid(Sp, tp, initial=0, axis=-1)
    
    # Standard interpolation for inside the boundary bounds
    int_interp = interp1d(tp, cum_int, kind='linear', axis=-1, 
                          bounds_error=False, fill_value=np.nan)
    
    t_start = t - dt/2
    t_end = t + dt/2

    F_start = int_interp(t_start)
    F_end = int_interp(t_end)

    # Total integral at the rightmost boundary
    total_integral = cum_int[:, -1][:, np.newaxis] 

    # --- Correctly handle boundaries with constant extension physics ---
    # Left side extrapolation: integral decreases linearly moving backwards from tp_min
    F_start = np.where(t_start < tp_min, 0 - (tp_min - t_start) * sp_left, F_start)
    F_end = np.where(t_end < tp_min, 0 - (tp_min - t_end) * sp_left, F_end)

    # Right side extrapolation: integral increases linearly moving forwards from tp_max
    F_start = np.where(t_start > tp_max, total_integral + (t_start - tp_max) * sp_right, F_start)
    F_end = np.where(t_end > tp_max, total_integral + (t_end - tp_max) * sp_right, F_end)

    # Average value over the window of duration dt
    Ss = (F_end - F_start) / dt
    
    return Ss.flatten() if is_1d else Ss

# def sample_old(t, tp, Sp, dt=None) -> np.ndarray:
#     """Sample a signal at given time points.

#     Args:
#         t (array): The 1D time points at which to evaluate the signal.
#         tp (array): the 1D time points of the signal to be sampled.
#         Sp (array): the 1D or 2D values (n_samples, n_times) of the signal to be sampled.
#         dt (float, optional): sampling duration.

#     Returns:
#         np.ndarray: Signals sampled at times t.
#     """
#     t = np.asarray(t)
#     tp = np.asarray(tp)
#     Sp = np.asarray(Sp)
    
#     # Handle 1D input by promoting it to 2D (1, n_times)
#     is_1d = Sp.ndim == 1
#     if is_1d:
#         Sp = Sp[np.newaxis, :]

#     if t.size == 0:
#         return np.array([])

#     # 1. With dt=0 this is just interpolation
#     if dt is None or dt == 0:
#         sig_interp = interp1d(tp, Sp, kind='linear', axis=-1, 
#                               bounds_error=False, fill_value=0)
#         res = sig_interp(t)
#         return res.flatten() if is_1d else res

#     # 2. Windowed Trapezoidal Logic
#     cum_int = cumulative_trapezoid(Sp, tp, initial=0, axis=-1)
    
#     # Use fill_value=(0, 'extrapolate') or a constant to handle boundaries
#     # Since we want it to be 0 outside the range, we manually handle the right-side fill
#     int_interp = interp1d(tp, cum_int, kind='linear', axis=-1, 
#                           bounds_error=False, fill_value=(0, np.nan))
    
#     t_start = t - dt/2
#     t_end = t + dt/2

#     # Override the FIRST point only (index 0)
#     # We make it start at t[0] and end at t[0] + dt/2
#     if t[0] - dt/2 < tp[0]:  # Only adjust if the first window extends before tp[0]
#         t_start[0] = t[0]
#         t_end[0] = t[0] + dt/2

#     F_start = int_interp(t_start)
#     F_end = int_interp(t_end)

#     # Correctly handle boundaries for any Sp shape
#     tp_min, tp_max = tp[0], tp[-1]
#     total_integral = cum_int[:, -1][:, np.newaxis] # Shape (n_samples, 1)

#     # If t < tp_min -> 0
#     # If t > tp_max -> total_integral
#     F_start = np.where(t_start < tp_min, 0, F_start)
#     F_start = np.where(t_start > tp_max, total_integral, F_start)

#     F_end = np.where(t_end < tp_min, 0, F_end)
#     F_end = np.where(t_end > tp_max, total_integral, F_end)

#     # Define the divisors (dt for most, dt/2 for the first point)
#     divisors = np.full_like(t, dt, dtype=np.float64)
#     if t[0] - dt/2 < tp[0]:
#         divisors[0] = dt/2
#     Ss = (F_end - F_start) / divisors
    
#     return Ss.flatten() if is_1d else Ss


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



def mle_rice(data, fit_loc=False):
    """
    Maximum-likelihood estimate of Rician parameters from 1D array `data`.

    Parameters
    ----------
    data : array-like, shape (n,)
        Observations (must be >= 0 unless you fit loc).
    fit_loc : bool
        If True, estimate loc as well. If False, assume loc == 0.

    Returns
    -------
    dict with keys:
      - 'nu'    : estimated noncentrality parameter nu
      - 'sigma' : estimated scale sigma
      - 'b'     : estimated shape parameter b = nu/sigma
      - 'loc'   : estimated location (0 if fit_loc=False)
      - 'success', 'message' from optimizer
    """

    data = np.asarray(data, dtype=float)
    if not fit_loc and (data < 0).any():
        raise ValueError("data contains negative values but fit_loc=False. "
                         "Either remove negatives or set fit_loc=True.")

    # initial guess using scipy's fit (fast and robust)
    if fit_loc:
        b0, loc0, sigma0 = rice.fit(data)         # returns (shape, loc, scale)
        x0 = np.array([np.log(b0), loc0, np.log(sigma0)])
    else:
        b0, loc0, sigma0 = rice.fit(data, floc=0)
        x0 = np.log([b0, sigma0])  # we optimize in log-space for positivity

    # Negative log-likelihood to minimize (we parametrize to enforce positivity)
    if fit_loc:
        def neglog(x):
            b = np.exp(x[0])
            loc = x[1]
            sigma = np.exp(x[2])
            return -np.sum(rice.logpdf(data, b, loc=loc, scale=sigma))
        bounds = [(None, None), (None, None), (None, None)]
    else:
        def neglog(x):
            b = np.exp(x[0])
            sigma = np.exp(x[1])
            return -np.sum(rice.logpdf(data, b, loc=0.0, scale=sigma))
        bounds = [(None, None), (None, None)]

    res = minimize(neglog, x0, method='L-BFGS-B', bounds=bounds,
                   options={'ftol':1e-12, 'gtol':1e-8})

    if fit_loc:
        b_hat = float(np.exp(res.x[0]))
        loc_hat = float(res.x[1])
        sigma_hat = float(np.exp(res.x[2]))
    else:
        b_hat = float(np.exp(res.x[0]))
        sigma_hat = float(np.exp(res.x[1]))
        loc_hat = 0.0

    nu_hat = b_hat * sigma_hat

    return {
        'nu': nu_hat,
        'sigma': sigma_hat,
        'b': b_hat,
        'loc': loc_hat,
        'success': res.success,
        'message': res.message,
        'nll': float(res.fun)
    }



def describe(data, n0=1, rician=False):
    """Compute descriptive parameter maps for a signal array.

    Args:
        data (numpy.ndarray): array with signal data. Dimensions have 
            to be at least 2, where the last dimension is time.
        n0 (int, optional): Number of baseline points. Defaults to 1.
        rician (bool, optional): Whether to correct for Rician noise in 
            computation of baseline signal and noise (slow). Defaults 
            to False.

    Raises:
        ValueError: if rician=True, n0 needs to be 3 or higher.

    Returns:
        dict: Dictionary with parameter maps.
    """

    maps = {}
    if n0==1:
        maps['Sb'] = data[...,0]
    else:
        maps['Sb'] = np.mean(data[...,:n0], axis=-1)
    if n0 > 2:
        maps['Nb'] = np.std(data[...,:n0], axis=-1)
    maps['SEmax'] = np.max(data, axis=-1) - maps['Sb']
    maps['SEauc'] = np.sum(data, axis=-1) - maps['Sb'] * data.shape[-1]
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        maps['RSEmax'] = np.where(maps['Sb']!=0, maps['SEmax']/maps['Sb'], 0)
        maps['RSEauc'] = np.where(maps['Sb']!=0, maps['SEauc']/maps['Sb'], 0)
    if rician:
        if n0 < 3:
            raise ValueError('Rician correction can only be applied if n0 > 2')
        Sb_rice = np.zeros(maps['Sb'].size)
        Nb_rice = np.zeros(maps['Sb'].size)
        data_xt = data.reshape(-1, data.shape[-1])
        for x in tqdm(range(data_xt.shape[0]), desc='Computing Rician noise', total=data_xt.shape[0]):
            rice = mle_rice(data_xt[x,:n0])
            Sb_rice[x] = rice['nu']
            Nb_rice[x] = rice['sigma']
        maps['Sb_rician'] = Sb_rice.reshape(data.shape[:-1])
        maps['Nb_rician'] = Nb_rice.reshape(data.shape[:-1])
    return maps



