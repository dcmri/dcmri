import numpy as np
from scipy import integrate
from scipy.stats import rice
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats


def vfa_nonlinear(signal_intensities, flip_angles_deg, tr, bounds=None, verbose=0):
    """
    Calculates R1 and S0 from VFA data using a NON-LINEAR fit.

    This function fits the data directly to the SPGR signal equation:
    S(a) = S0 * sin(a) * (1 - exp(-TR*R1)) / (1 - cos(a) * exp(-TR*R1))
    
    This method can be more stable and accurate than the linear fit,
    especially in the presence of noise.

    Args:
        signal_intensities (list or np.ndarray): A list or array of measured
                                                  signal intensities.
        flip_angles_deg (list or np.ndarray): A list or array of corresponding flip angles in degrees.
        tr (float): The repetition time (TR) of the sequence.
        bounds (tuple): bounds on (R1, S0) as a tuple ([lower_R1, lower_S0], [upper_R1, upper_S0]). Default is ([0, 0], [np.inf, np.inf])
        verbose (int): if set to 1, warning messages are printed. Defaults to 0.

    Returns:
        tuple: A tuple containing the calculated (R1, S0).
               Returns initial guesses if the non-linear fit fails to converge.
    """
    # Convert inputs to numpy arrays for vectorized operations
    signals = np.array(signal_intensities)
    
    # If the signal intensities are image arrays, loop over the pixels
    if signals.ndim > 1:
        signals_shape = signals.shape
        signals_array = signals.reshape(-1, signals_shape[-1])
        R1_array = np.zeros(signals_array.shape[0])
        S0_array = np.zeros(signals_array.shape[0])
        for x in tqdm(range(signals_array.shape[0]), desc='Performing non-linear VFA fit'):
            R1_array[x], S0_array[x] = vfa_nonlinear(signals_array[x,:], flip_angles_deg, tr, bounds, verbose)
        R1_array = R1_array.reshape(signals_shape[:-1])
        S0_array = S0_array.reshape(signals_shape[:-1])
        return R1_array, S0_array

    # --- 0. Input Validation and Conversion ---
    if len(flip_angles_deg) != len(signals):
        raise ValueError("Input arrays for flip angles and signals must have the same length.")
    
    # Default bounds
    if bounds is None:
        bounds = ([0, 0], [np.inf, np.inf])
    
    # Convert flip angles from degrees to radians for trigonometric functions
    flip_angles_rad = np.deg2rad(flip_angles_deg)

    # --- 1. Define the SPGR signal model for curve_fit ---
    # tr is passed as a fixed argument to the model function
    def spgr_model(alpha_rad, r1, s0):
        if r1 <= 0: # T1 must be positive
            return np.inf
        e1 = np.exp(-tr * r1)
        return s0 * np.sin(alpha_rad) * (1 - e1) / (1 - np.cos(alpha_rad) * e1)

    # --- 2. Provide Initial Guesses and Bounds ---
    # Good initial guesses are important for non-linear fitting.
    # Guess S0 as the maximum signal, and T1 as a typical biological value.
    initial_s0_guess = np.max(signals)
    initial_r1_guess = 1/1.2
    initial_guesses = [initial_r1_guess, initial_s0_guess]
    
    # --- 3. Perform Non-Linear Fit ---
    try:
        popt, pcov = curve_fit(
            spgr_model,
            flip_angles_rad,
            signals,
            p0=initial_guesses,
            bounds=bounds
        )
        calculated_r1, calculated_s0 = popt
        return calculated_r1, calculated_s0
    except RuntimeError:
        print("Warning (Non-Linear Fit): Could not converge to a solution. Returning initial guesses.")
        return initial_r1_guess, initial_s0_guess


def vfa_linear(signal_intensities, flip_angles_deg, tr, bounds=None, verbose=0):
    """
    Calculates R1 and S0 from variable flip angle (VFA) SPGR data.

    This function uses the linearized form of the steady-state spoiled
    gradient-echo (SPGR) signal equation to perform a linear fit and
    extract R1 and S0.

    The linearized equation is:
    S(a)/sin(a) = E1 * S(a)/tan(a) + S0*(1-E1)
    where E1 = exp(-TR * R1). This is a linear equation of the form y = m*x + c.

    Args:
        signal_intensities (list or np.ndarray): A list or array of measured
                                                  signal intensities.
        flip_angles_deg (list or np.ndarray): A list or array of corresponding flip angles in degrees.
        tr (float): The repetition time (TR) of the sequence.
        bounds (tuple): bounds on (R1, S0) as a tuple ([lower_R1, lower_S0], [upper_R1, upper_S0]). Default is ([0, 0], [np.inf, np.inf])
        verbose (int): if set to 1, warning messages are printed. Defaults to 0.

    Returns:
        tuple: A tuple containing the calculated (R1, S0).
               Returns (0, 0) if the calculation is not physically
               plausible (e.g., due to noisy data leading to a slope >= 1).
    """
    # Convert inputs to numpy arrays for vectorized operations
    signals = np.array(signal_intensities)

    # If the signal intensities are image arrays, loop over the pixels
    if signals.ndim > 1:
        signals_shape = signals.shape
        signals_array = signals.reshape(-1, signals_shape[-1])
        R1_array = np.zeros(signals_array.shape[0])
        S0_array = np.zeros(signals_array.shape[0])
        for x in tqdm(range(signals_array.shape[0]), desc='Performing linear VFA fit'):
            R1_array[x], S0_array[x] = vfa_linear(signals_array[x,:], flip_angles_deg, tr, bounds, verbose)
        R1_array = R1_array.reshape(signals_shape[:-1])
        S0_array = S0_array.reshape(signals_shape[:-1])
        return R1_array, S0_array

    # --- 1. Input Validation and Conversion ---
    if len(flip_angles_deg) != len(signals):
        raise ValueError("Input arrays for flip angles and signals must have the same length.")
    
    # Default bounds
    if bounds is None:
        bounds = ([0, 0], [np.inf, np.inf])
    
    # Convert flip angles from degrees to radians for trigonometric functions
    flip_angles_rad = np.deg2rad(flip_angles_deg)

    # --- 2. Data Transformation for Linearization ---
    # Avoid division by zero for tan(90 degrees) if present
    # and for sin(0 degrees). We filter out these data points.
    valid_indices = (np.sin(flip_angles_rad) != 0) & (np.cos(flip_angles_rad) != 0)
    
    if np.sum(valid_indices) < 2:
        if verbose==1:
            print("Warning: Not enough valid data points (<2) for a linear fit. Returning lower bounds.")
        return bounds[0][0], bounds[0][1]
        
    y = signals[valid_indices] / np.sin(flip_angles_rad[valid_indices])
    x = signals[valid_indices] / np.tan(flip_angles_rad[valid_indices])

    if np.array_equal(x,y):
        if verbose==1:
            print("Warning: Equal values for x and y - cannot perform linear fit. Returning lower bounds")
        return bounds[0][0], bounds[0][1]   

    # --- 3. Linear Regression ---
    # Use np.polyfit to find the slope (m) and intercept (c) of the line y = mx + c
    # The degree of the polynomial is 1 for a linear fit.
    #slope, intercept = np.polyfit(x, y, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # --- 4. Calculate T1 and S0 ---
    # The slope corresponds to E1
    e1 = slope
    
    # Check for physically plausible E1 value. E1 must be > 0 and < 1.
    # A slope >= 1 or <= 0 would result in a non-real or negative T1.
    if not (0 < e1 < 1):
        if verbose==1:
            print(f"Warning: Calculated slope (E1 = {e1:.4f}) is outside the valid range (0, 1).")
            print("This may be due to noise or other artifacts. Returning lower bounds.")
        return bounds[0][0], bounds[0][1]

    # Calculate T1 from E1
    # T1 = -TR / ln(E1)
    r1 = - np.log(e1) / tr
    
    # Calculate S0 from the intercept
    # Intercept = S0 * (1 - E1) => S0 = Intercept / (1 - E1)
    s0 = intercept / (1 - e1)

    # Apply bounds
    r1 = bounds[0][0] if r1 < bounds[0][0] else r1
    s0 = bounds[0][1] if s0 < bounds[0][1] else s0
    r1 = bounds[1][0] if r1 > bounds[1][0] else r1
    s0 = bounds[1][1] if s0 > bounds[1][1] else s0

    return r1, s0


def convmat(f:np.ndarray, order=2):
    """Return the convolution matrix

    The convolution product f*g can be computed by a matrix multiplication 
    dt * M(f) # g. This function returns the matrix M(f) for a given f, 
    which can be inverted to perform deconvolution.

    Args:
        f (numpy.ndarray): 1D array to be convolved
        order (int, optional): Order of the integration. Defaults to 2.

    Returns:
        numpy.ndarray: n x n square matrix
    """
    n = len(f)
    mat = np.zeros((n,n))

    if order==1:
        for i in range(0,n):
            for j in range(0,i+1):
                mat[i,j] = f[i-j]
        
    elif order==2:
        for i in range(1,n):
            mat[i,i] = 2*f[0] + f[1]    
            for j in range(1,i):        
                mat[i,i-j] = f[j-1] + 4*f[j] + f[j+1]
            mat[i,0] = f[i-1] + 2*f[i]
        mat = mat/6

    return mat


def invconvmat(f, order=2, tol=1e-15, method='TSVD'):
    mat = convmat(f, order)
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    svmin = tol*np.amax(s)
    if method=='Tikhonov':
        s_inv = s/(s**2 + svmin**2)
    elif method=='TSVD':
        s_inv = np.array([1/x if x > svmin else 0 for x in s])
    else:
        raise ValueError(
            f"Unknown deconvolution method {method}. Possible values "
            "are 'TSVD' (Truncated Singular Value Decomposition) or "
            "'Tikhonov'."
        )
    return np.dot(Vt.T * s_inv, U.T)


def deconv(h:np.ndarray, g:np.ndarray, dt=1.0, order=2, 
           method='TSVD', tol=1e-15) -> np.ndarray:
    """Deconvolve two uniformly sampled 1D functions.

    If and (h,g) are known in h = g*f, this function estimates f = deconv(h, g).

    Args:
        h (numpy array): Result of the convolution. if g has N 
            elements, than h can be an N-element array, or a  
            N x K - element array where N is the length of g. In this 
            case each column is deconvolved indepdently with g.
        g (numpy array): One factor of the convolution (1D array).
        dt (float, optional): Time between samples. Defaults to 1.0.
        order (int, optional): Integration order of the convolution 
            matrix. Defaults to 2.
        method (str, optional): Regularization method. Current options 
            are 'TSVD' (Truncated Singular Value Decomposition) or 
            'Tikhonov'. Defaults to False.
        tol (float, optional): Tolerance for the inversion of the 
            convolution matrix (rgularization parameter). Singular 
            values less than a fraction 'tol' of the largest 
            singular value are ignored. Defaults to 1e-15.

    Returns:
        numpy.ndarray: Estimate of the convolution factor f. This has 
        the same shape as h.
    """
    if g.ndim > 1:
        raise ValueError("g must be 1-dimensional.")
    if h.ndim > 2:
        raise ValueError("h must have 1 or 2 dimensions.")
    if h.shape[0] != len(g):
        raise ValueError(
            "The first dimension of h must have the same length as g."
        )
    ginv = invconvmat(g, order, tol, method)
    return (ginv @ h) / dt


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



# Used in iBEAt - not yet exposed in dcmri
def pixel_2cfm_linfit(imgs: np.ndarray, aif: np.ndarray = None, time: np.ndarray = None, baseline: int = 1, Hct=0.45):

    # Reshape to 2D (x,t)
    shape = np.shape(imgs)
    imgs = imgs.reshape((-1, shape[-1]))

    S0 = np.mean(imgs[:, :baseline], axis=1)
    Sa0 = np.mean(aif[:baseline])
    ca = (aif-Sa0)/(1-Hct)

    A = np.empty((imgs.shape[1], 4))
    A[:, 2], A[:, 3] = _ddint(ca, time)

    fit = np.empty(imgs.shape)
    par = np.empty((imgs.shape[0], 4))
    for x in range(imgs.shape[0]):
        c = imgs[x, :] - S0[x]
        ctii, cti = _ddint(c, time)
        A[:, 0] = -ctii
        A[:, 1] = -cti
        p = np.linalg.lstsq(A, c, rcond=None)[0]
        fit[x, :] = S0[x] + p[0]*A[:, 0] + p[1] * \
            A[:, 1] + p[2]*A[:, 2] + p[3]*A[:, 3]
        par[x, :] = _params_2cfm(p)

    # Apply bounds
    smax = np.amax(imgs)
    fit[fit < 0] = 0
    fit[fit > 2*smax] = 2*smax

    # Return in original shape
    fit = fit.reshape(shape)
    par = par.reshape(shape[:-1] + (4,))

    return fit, par


def _ddint(c, t):
    ci = integrate.cumulative_trapezoid(c, t, initial=0)
    ci = np.insert(ci, 0, 0)
    cii = integrate.cumulative_trapezoid(ci, t, initial=0)
    cii = np.insert(cii, 0, 0)
    return cii, ci


def _params_2cfm(X):

    alpha = X[0]
    beta = X[1]
    gamma = X[2]
    Fp = X[3]

    if alpha == 0:
        if beta == 0:
            return [Fp, 0, 0, 0]
        else:
            return [Fp, 1/beta, 0, 0]

    nom = 2*alpha
    det = beta**2 - 4*alpha
    if det < 0:
        Tp = beta/nom
        Te = Tp
    else:
        root = np.sqrt(det)
        Tp = (beta - root)/nom
        Te = (beta + root)/nom

    if Te == 0:
        PS = 0
    else:
        if Fp == 0:
            PS == 0
        else:
            T = gamma/(alpha*Fp)
            PS = Fp*(T-Tp)/Te

    # Convert to conventional units and apply bounds
    Fp *= 6000
    if Fp < 0:
        Fp = 0
    if Fp > 2000:
        Fp = 2000
    if Tp < 0:
        Tp = 0
    if Tp > 600:
        Tp = 600
    PS *= 6000
    if PS < 0:
        PS = 0
    if PS > 2000:
        PS = 2000
    if Te < 0:
        Te = 0
    if Te > 600:
        Te = 600
    return [Fp, Tp, PS, Te]

