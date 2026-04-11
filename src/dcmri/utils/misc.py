import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

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
