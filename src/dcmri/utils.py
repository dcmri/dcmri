import os
import multiprocessing
import warnings
import math
import numpy as np
from scipy.special import gamma
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


try: 
    num_workers = int(len(os.sched_getaffinity(0)))
except: 
    num_workers = int(os.cpu_count())


class Model:
    # Abstract base class for end-to-end models.                  

    def __init__(self):
        self.free = []
        self.bounds = [-np.inf, np.inf]
        self.pcov = None

    def predict(self, xdata):
        """Predict the data for given x-values.

        Args:
            xdata (tuple or array-like): Either an array with x-values (time points) or a tuple with multiple such arrays

        Raises:
            NotImplementedError: If no predict method is defined for the model

        Returns:
            tuple or array-like: Either an array of predicted y-values (if xdata is an array) or a tuple of such arrays (if xdata is a tuple).
        """
        raise NotImplementedError('No predict function provided')
    
    def train(self, xdata, ydata, **kwargs):
        """Train the free parameters of the model using the data provided.

        After training, the attribute ``pars`` will contain the updated parameter values, and ``popt`` contains the covariance matrix. The function uses the scipy function `scipy.optimize.curve_fit`, but has some additional keywords for convenience during model prototyping. 

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            kwargs: any keyword parameters accepted by `scipy.optimize.curve_fit`.

        Returns:
            Model: A reference to the model instance.
        """
        return train(self, xdata, ydata, **kwargs)


    def plot(self, xdata, ydata, xlim=None, testdata=None, fname=None, show=True):
        """Plot the model fit against data.

        Args:
            xdata (array-like): Array with x-data (time points)
            ydata (array-like): Array with y-data (signal data)
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            testdata (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Path name to save the image. Defaults to None.
            show (bool, optional): If True, the plot is shown; if false it is saved to disk but not shown. If no fname is provided the image is always shown and this keyword is ignored. Defaults to True.

        Raises:
            NotImplementedError: If no plot function is implemented for the model.
        """
        raise NotImplementedError('No plot function implemented for model ' + self.__class__.__name__)
    
    
    def cost(self, xdata, ydata, metric='NRMS')->float:
        """Goodness-of-fit value.

        Args:
            xdata (array-like): Array with x-data (time points).
            ydata (array-like): Array with y-data (signal values)
            metric (str, optional): Which metric to use - options are 'NRMS' (Normalized root-mean-square), 'AIC' (Akaike information criterion) or 'BIC' (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        # Predict data at all xvalues
        y = self.predict(xdata)
        if isinstance(ydata, tuple):
            y = np.concatenate(y)
            ydata = np.concatenate(ydata)

        # Calclulate the loss at the fitted values
        if metric == 'NRMS':
            loss = 100*np.linalg.norm(y - ydata)/np.linalg.norm(ydata)
        elif metric == 'AIC':
            return
        elif metric == 'BIC':
            return
        return loss


    def export_params(self)->list:
        """Free parameters with descriptions.

        Returns:
            list: list with one element for each free parameter. The elements are lists themselves providing, for each parameter, [short name, long name, value, unit].
        """
        # Short name, full name, value, units.
        pars = {}
        for p in self.free:
            pars[p] = [p+' name', getattr(self, p), p+' unit']
        return self._add_sdev(pars)
    

    def print_params(self, round_to=None):
        """Print a summary of the model parameters and their uncertainties.

        Args:
            round_to (int, optional): Round to how many digits. If this is not provided, the values are not rounded. Defaults to None.
            units (str, optional): Which unit system to use in the return values. Defaults to 'standard'.
        """
        pars = self.export_params()
        print('-----------------------------------------')
        print('Free parameters with their errors (stdev)')
        print('-----------------------------------------')
        for par in self.free:
            if par in pars:
                p = pars[par]
                if round_to is None:
                    v = p[1]
                    verr = p[3]
                else:
                    v = round(p[1], round_to)
                    verr = round(p[3], round_to)
                print(p[0] + ' ('+par+'): ' + str(v) + ' (' + str(verr) + ') ' + p[2])
        print('------------------')
        print('Derived parameters')
        print('------------------')
        for par in pars:
            if par not in self.free:
                p = pars[par]
                if round_to is None:
                    v = p[1]
                else:
                    v = round(p[1], round_to)
                print(p[0] + ' ('+par+'): ' + str(v) + ' ' + p[2])


    def get_params(self, *args, round_to=None):
        """Get parameter values.

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is not provided, the values are not rounded. Defaults to None.

        Returns:
            list: values of parameter values
        """
        pars = []
        for a in args:
            v = getattr(self, a)
            if round_to is not None:
                v = round(v, round_to)
            pars.append(v)
        return pars
    
    
    def _getflat(self, attr:np.ndarray=None)->np.ndarray:
        if attr is None:
            attr = self.free
        vals = []
        for a in attr:
            v = getattr(self, a)
            vals = np.append(vals, np.ravel(v))
        return vals
    

    def _setflat(self, vals:np.ndarray, attr:np.ndarray=None):
        if attr is None:
            attr = self.free
        i=0
        for p in attr:
            v = getattr(self, p)
            if np.isscalar(v):
                v = vals[i]
                i+=1
            else:
                n = np.size(v)
                v = np.reshape(vals[i:i+n], np.shape(v))
                i+=n
            setattr(self, p, v)
                

    def _x_scale(self):
        n = len(self.free)
        xscale = np.ones(n)
        for p in range(n):
            if np.isscalar(self.bounds[0]):
                lb = self.bounds[0]
            else:
                lb = self.bounds[0][p]
            if np.isscalar(self.bounds[1]):
                ub = self.bounds[1]
            else:
                ub = self.bounds[1][p]
            if (not np.isinf(lb)) and (not np.isinf(ub)):
                xscale[p] = ub-lb
        return xscale


    def _add_sdev(self, pars):
        for par in pars:
            pars[par].append(0)
        if self.pcov is None:
            perr = np.zeros(np.size(self.free))
        else:
            perr = np.sqrt(np.diag(self.pcov))

        for i, par in enumerate(self.free):
            if par in pars:
                pars[par][-1] = perr[i]
        return pars


def _fit_pixel(xdata, ydata, model, params, kwargs):
    mdl = model(**params)
    mdl.train(xdata, ydata, **kwargs)
    return mdl.export_params()


def fit(xdata, ydata, model, params=None, parallel=True, **kwargs):
    
    # Reshape to (x,t)
    shape = ydata.shape
    ydata = np.reshape(ydata, (-1,shape[-1]))
    if np.size(xdata) == np.size(ydata):
        xdata = np.reshape(xdata, (-1,shape[-1]))
    
    # Perform the fit pixelwise
    if not parallel: # for debugging
        par = []
        for x in range(ydata.shape[0]):
            if params is None:
                params_x = {}
            elif isinstance(params, dict):
                params_x = params
            else:
                params_x = params[x]
            args_x = (xdata[x,:], ydata[x,:], model, params_x, kwargs)
            par_x = _fit_pixel(*args_x)
            par.append(par_x)
    else:
        args = []
        for x in range(ydata.shape[0]):
            if params is None:
                params_x = {}
            elif isinstance(params, dict):
                params_x = params
            else:
                params_x = params[x]
            args_x = (xdata[x,:], ydata[x,:], model, params_x, kwargs)
            args.append(args_x)
        pool = multiprocessing.Pool(processes=num_workers)
        par = pool.map(_fit_pixel, args)
        pool.close()
        pool.join()

    # Create output arrays
    pars = {}
    sdev = {}
    for p in par[0].keys():
        pars[p] = np.array([par_x[p][1] for par_x in par]).reshape(shape[:-1])
        sdev[p] = np.array([par_x[p][3] for par_x in par]).reshape(shape[:-1])
    
    return pars, sdev


def train(model:Model, xdata, ydata, **kwargs):

    if isinstance(ydata, tuple):
        y = np.concatenate(ydata)
    else:
        y = ydata

    def fit_func(_, *p):
        model._setflat(p)
        yp = model.predict(xdata)
        if isinstance(yp, tuple):
            return np.concatenate(yp)
        else:
            return yp

    p0 = model._getflat()
    try:
        pars, model.pcov = curve_fit(
            fit_func, None, y, p0, 
            bounds=model.bounds, #x_scale=model._x_scale(),
            **kwargs)
    except RuntimeError as e:
        msg = 'Runtime error in curve_fit -- \n'
        msg += str(e) + ' Returning initial values.'
        warnings.warn(msg)
        pars = p0
        model.pcov = np.zeros((np.size(p0), np.size(p0)))
    
    model._setflat(pars)
    
    return model
    

def interp(y, x, pos=False, floor=False)->np.ndarray:
    """Interpolate uniformly sampled data. 

    This function is a convenience wrapper for standard interpolation, used in dcmri for instance to parametrize non-stationat models.

    Args:
        y (array-like): List of values to interpolate. These are assumed to be uniformly distributed over x.
        x (array-like): Interpolate y at these locations.
        pos (bool, optional): return only positive values. Defaults to False.
        floor (bool, optional): Return only results higher than the lowest value in y. Defaults to False.

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
    elif np.size(y)==1:
        yi = y[0]*np.ones(len(x))
    elif np.size(y)==2:
        yi = _lin(x,y)
    elif np.size(y)==3:
        yi = _quad(x,y)
    else:
        x_y = np.linspace(np.amin(x), np.amax(x), len(y))
        yi = CubicSpline(x_y, y)(x)
    if pos:
        yi[yi<0] = 0
    if floor:
        y0 = np.amin(y)
        yi[yi<y0] = y0
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
    #returns a linear function of x 
    #that goes through the two points (xi, yi)
    L1 = (x2-x)/(x2-x1)
    L2 = (x-x1)/(x2-x1)
    return y1*L1 + y2*L2

def _quadratic(x, x0, x1, x2, y0, y1, y2):
    # Helper
    #returns a quadratic function of x 
    #that goes through the three points (xi, yi)
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
        t=np.array(t)
    n = len(t)
    h = np.zeros(n)
    if T<t[0]:
        return h
    if T>t[-1]:
        return h
    if T==t[0]:
        h[0]=2/(t[1]-t[0])
        return h
    if T==t[-1]:
        h[-1]=2/(t[-1]-t[-2])
        return h
    i = np.where(T>=t)[0][-1]
    u = (T-t[i])/(t[i+1]-t[i])
    if i==0:
        h[i] = (1-u)*2/(t[i+1]-t[i])
    else:
        h[i] = (1-u)*2/(t[i+1]-t[i-1])
    if i==n-2:
        h[i+1] = u*2/(t[i+1]-t[i])
    else:
        h[i+1] = u*2/(t[i+2]-t[i])
    return h


def dstep(T0, T1, t):
    # Helper function - discrete step
    if not isinstance(t, np.ndarray):
        t=np.array(t)
    n = len(t)
    i = np.where((t>T0)*(t<T1))[0]
    if len(i)==0:
        return ddelta((T0+T1)/2, t)
    i0, i1 = i[0], i[-1]
    t0, t1 = t[i0], t[i1]
    hi = 0
    if i0>0:
        u0 = (t0-T0)/(t0-t[i0-1])
        hi += 0.5*(1+u0)*(t0-t[i0-1])
        if i0>1:
            hi += 0.5*u0*(t[i0-1]-t[i0-2])   
    hi += t1-t0
    if i1<n-1:
        u1 = (T1-t1)/(t[i1+1]-t1)
        hi += 0.5*(1+u1)*(t[i1+1]-t1)
        if i1<n-2:
            hi += 0.5*u1*(t[i1+2]-t[i1+1])
    h = np.zeros(n)
    h[i] = 1/hi
    if i0>0:
        h[i0-1] = u0/hi
    if i1<n-1:
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
        return 0.5*np.sum((g[1:]+g[:-1])*(t[1:]-t[:-1]) )

def uconv(f, h, dt=1.0, solver='trap'):
    # Helper function: convolution over uniformly sampled grid.
    n = len(f) 
    g = np.zeros(n)
    h = np.flip(h)
    for k in range(1, n):
        if solver=='trap':
            g[k] = intprod(f[:k+1], h[-(k+1):], dt=dt)
        elif solver=='step':
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
        if solver=='trap':
            g[k] = intprod(fk, hk, tk)
        elif solver=='step':
            g[k] = _intstep(fk, hk, tk)
    return g


def inttrap(f, t, t0, t1):
    # Helper function: integrate f from t0 to t1
    ti = t[(t0<t)*(t<t1)]
    ti = np.concatenate(([t0],ti,[t1]))
    fi = np.interp(ti, t, f, left=0, right=0)
    return np.trapz(fi,ti)


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
    if D>1:
        raise ValueError('The dispersion factor D must be <= 1')
    TW = D*T      # Half width of step
    T0 = T-TW     # Initial time point of step
    T1 = T+TW
    n = len(f)
    t = tarray(n, t=t, dt=dt)
    g = np.zeros(n)
    k = len(t[t<T0])
    ti = t[(T0<=t)*(t<=T1)]
    for tk in ti:
        g[k] = inttrap(f, t, 0, tk-T0)
        k+=1
    ti = t[T1<t]
    for tk in ti:
        g[k] = inttrap(f, t, tk-T1, tk-T0)
        k+=1
    return g/(2*TW)


def expconv(f, T, t=None, dt=1.0):
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

    if T==0: 
        return f
    f = np.array(f)
    n = len(f)
    t = tarray(n, t=t, dt=dt)
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
    if T1==T2:
        return (t/T1) * np.exp(-t/T1)/T1
    else:
        return (np.exp(-t/T1)-np.exp(-t/T2))/(T1-T2)


def nexpconv(n, T, t):
    """Convolve n identical normalised exponentials analytically.

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
            g(t) = \\frac{1}{\\Gamma(n)}\\left(\\frac{t}{T}\\right) \\frac{e^{-t/T}}{T} 

    Example:
        Import package and create a vector of uniformly sampled time points t with spacing 5.0s:

        >>> import dcmri as dc
        >>> t = 5.0*np.arange(4)

        Calculate the convolution of 4 normalised exponentials with time constants 5s:

        >>> g = dc.nexpconv(4, 5, t)
        array([0.        , 0.01226265, 0.03608941, 0.04480836])
    """
    if T<0:
        raise ValueError('T must be non-negative')
    if n<1:
        raise ValueError('n cannot be smaller than 1')
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    u = t/T
    g = u**(n-1) * np.exp(-u)/T/gamma(n)
    if False in np.isfinite(g):
        # At large n the analytical formula runs into overflow
        # use numerical calculation in this case (slower and less accurate)
        g = np.exp(-t/T)/T
        n0 = int(np.floor(n))
        for _ in range(n0-1):
            g = expconv(g, T, t)
        if n!=n0:
            # Interpolate between n0 and n0+1
            g1 = expconv(g, T, t)
            u = n-n0
            g = g*u + g1*(1-u)
    return g


