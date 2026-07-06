import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats

def estimate_bat(t, signal, n0=10, threshold_multiplier=1.1, persistence=3):
    """
    Estimates the time point where a noisy 1D signal departs from its initial baseline
    using the maximum deviation from the mean as the noise metric.
    
    Parameters:
    -----------
    t : array_like
        1D array of time points corresponding to the signal data.
    signal : array_like
        1D discrete data vector representing the signal over time.
    n0 : int
        The number of initial data points used to calculate the baseline properties.
    threshold_multiplier : float
        Multiplier applied to the maximum baseline difference to set the threshold.
        Must be > 1.0 (e.g., 1.1 means 10% higher than the maximum baseline noise).
    persistence : int
        Number of consecutive points that must exceed the threshold to trigger detection.
        
    Returns:
    --------
    float or None
        The estimated time point where the change starts, or None if no change is detected.
    """
    t = np.asarray(t)
    signal = np.asarray(signal)
    # (channels, components, times)

    # Convert to magnitude signal
    signal = np.linalg.norm(signal, axis=1)
    # (channels, times)

    # Return an average over channels
    bat = []
    for channel in range(signal.shape[0]):
        bat += [_estimate_bat_channel(t, signal[channel,:], n0=n0, threshold_multiplier=threshold_multiplier, persistence=persistence)]

    return np.mean(bat)

    # if n0==1:
    #     return t[1]
    
    # if len(signal) != len(t):
    #     raise ValueError("The time array 't' and 'signal' must have the same length.")
    # if len(signal) <= n0:
    #     raise ValueError("Signal length must be greater than the baseline length.")
    
    # # 1. Estimate baseline properties
    # baseline = signal[:n0]
    # baseline_mean = np.mean(baseline)
    
    # # Calculate noise level as the maximum absolute difference from the mean
    # max_baseline_diff = np.max(np.abs(baseline - baseline_mean))
    
    # # 2. Define upper and lower thresholds
    # # We scale the maximum observed noise slightly to create a safety boundary
    # threshold_deviation = max_baseline_diff * threshold_multiplier
    # upper_thresh = baseline_mean + threshold_deviation
    # lower_thresh = baseline_mean - threshold_deviation
    
    # # 3. Find where the signal exceeds the threshold bounds
    # out_of_bounds = (signal > upper_thresh) | (signal < lower_thresh)
    
    # # 4. Enforce persistence to avoid false triggers
    # for i in range(n0, len(signal) - persistence + 1):
    #     if np.all(out_of_bounds[i : i + persistence]):
    #         # Return the exact time from the time array 't'
    #         return t[i]
            
    # return t[0]


def _estimate_bat_channel(t, signal, n0=10, threshold_multiplier=1.1, persistence=3):
    # (times)

    if n0==1:
        return t[1]
    
    if len(signal) != len(t):
        raise ValueError("The time array 't' and 'signal' must have the same length.")
    if len(signal) <= n0:
        raise ValueError("Signal length must be greater than the baseline length.")
    
    # 1. Estimate baseline properties
    baseline = signal[:n0]
    baseline_mean = np.mean(baseline)
    
    # Calculate noise level as the maximum absolute difference from the mean
    max_baseline_diff = np.max(np.abs(baseline - baseline_mean))
    
    # 2. Define upper and lower thresholds
    # We scale the maximum observed noise slightly to create a safety boundary
    threshold_deviation = max_baseline_diff * threshold_multiplier
    upper_thresh = baseline_mean + threshold_deviation
    lower_thresh = baseline_mean - threshold_deviation
    
    # 3. Find where the signal exceeds the threshold bounds
    out_of_bounds = (signal > upper_thresh) | (signal < lower_thresh)
    
    # 4. Enforce persistence to avoid false triggers
    for i in range(n0, len(signal) - persistence + 1):
        if np.all(out_of_bounds[i : i + persistence]):
            # Return the exact time from the time array 't'
            return t[i]
            
    return t[0]


def _estimate_bat(t, y):
    """
    Finds the smallest time point corresponding to the half maximum of y 
    using linear interpolation.
    """
    # Convert inputs to numpy arrays just in case
    t = np.asarray(t)
    y = np.asarray(y)

    # For a multi-channel signal, return an average over channels
    if y.ndim > 1:
        est = [estimate_bat(t, y[i,:]) for i in range(y.shape[0])]
        return np.mean(est)

    # Calculate the half-maximum value
    y_min = np.min(y)
    y_max = np.max(y)
    half_max = y_min + (y_max - y_min) / 2.0
    
    # Find the first index where y is greater than or equal to the half-maximum
    # (Excluding the very first point because we need a previous point to interpolate from)
    idx_above = np.where(y[1:] >= half_max)[0]
    
    if len(idx_above) == 0:
        raise ValueError("The signal never reaches the calculated half-maximum.")
        
    # Correct index shift because we sliced from y[1:]
    idx = idx_above[0] + 1
    
    # Points for interpolation
    t0, t1 = t[idx - 1], t[idx]
    # y0, y1 = y[idx - 1], y[idx]
    
    # # Edge case: If the two y-values are identical, avoid division by zero
    # if y1 == y0:
    #     return t0
        
    # Linear interpolation formula solved for t:
    # t = t0 + (half_max - y0) * (t1 - t0) / (y1 - y0)
    # t_half = t0 + (half_max - y0) * (t1 - t0) / (y1 - y0)
    t_half = (t0 + t1) / 2
    
    return t_half


def conc_dce(signal, S, r1=None, n0=None, S0=None, R1b=None, defaults=None):
    
    # Normalize signal
    if S0 is None:
        Sn0 = signal(defaults, R1=R1b, S0=1, R2s=np.ones_like(R1b), TE=0, v=1, Fw=0, me=1)['S'] # Exp factor absorbed in S0
        Sb = np.sum(S[:, :n0], axis=1) / n0
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn_data = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Create lookup table
    c_step = 0.01 * 1e-3
    c_max = 0.01
    c_range = np.arange(0, c_max, c_step)
    R1_min = 0
    R1_lookup = R1_min + r1 * c_range
    Sn_lookup = signal(defaults, R1=R1_lookup, S0=1, R2s=np.ones_like(R1_lookup), TE=0, v=1, Fw=0, me=1)['S']

    # Look up conc values
    R1 = np.interp(Sn_data, Sn_lookup, R1_lookup)

    # Convert R1 to conc
    R1b = np.sum(R1[:, :n0], axis=1) / n0
    R1b = R1b[:, np.newaxis]
    return (R1 - R1b) / r1


def conc_dsc(S, r2=None, TE=None, n0=None) -> np.ndarray:
    # S/Sb = exp(-TE(R2-R2b))
    #   ln(S/Sb) = -TE(R2-R2b)
    #   R2-R2b = -ln(S/Sb)/TE
    # R2 = R2b + r2C
    #   C = (R2-R2b)/r2
    #   C = -ln(S/Sb)/TE/r2

    # 1. Calculate Sb (the baseline)
    Sb = np.mean(S[:, :n0], axis=1)[:, np.newaxis]
    # Reshape Sb to (n_samples, 1) to divide S (n_samples, n_times)
    S_normalized = np.divide(S, Sb, out=np.zeros_like(S, dtype=float), where=Sb != 0)

    # 2. Calculate Concentration
    S_safe = np.clip(S_normalized, 1e-10, None)
    C = -np.log(S_safe) / (TE * r2)
    return C
    

def conc_ss(S, r1=None, FA=None, TR=None, B1corr=None, n0=None, S0=None, R1b=None) -> np.ndarray:
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R1b)) / (1-cFA*exp(-TR*R1b))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn
    FA = np.radians(FA * B1corr)
    cFA = np.cos(FA)
    sFA = np.sin(FA)

    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        E0 = np.exp(-TR * R1b)
        Sn0 = sFA * (1 - E0) / (1 - cFA * E0)
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Invert analytically for R1
    Sn = Sn / sFA
    En = (1 - Sn) / (1 - cFA * Sn)
    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.where(En <= 0, 0, -np.log(En)/TR)

    # Convert R1 to conc
    R1b = np.sum(R1[:, :n0], axis=1) / n0
    R1b = R1b[:, np.newaxis]
    return (R1 - R1b) / r1


def conc_dce_lin(S, r1=None, n0=None, S0=None, R1b=None):
    # S = S0 * R1
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        S0 = Sb / R1b
        S0 = np.divide(Sb, R1b, out=np.zeros_like(Sb, dtype=float), where=R1b > 0)

    S0 = S0[:, np.newaxis] 
    R1 = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Convert R1 to conc
    R1b = np.sum(R1[:, :n0], axis=1) / n0
    R1b = R1b[:, np.newaxis]
    return (R1 - R1b) / r1


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
            fit_results = vfa_nonlinear(signals_array[x,:], flip_angles_deg, tr, bounds, verbose)
            R1_array[x] = fit_results[0]
            S0_array[x] = fit_results[1]
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
        e1 = np.exp(-tr * r1)
        return s0 * np.sin(alpha_rad) * (1 - e1) / (1 - np.cos(alpha_rad) * e1)

    # --- 2. Provide Initial Guesses and Bounds ---
    # Good initial guesses are important for non-linear fitting.
    # Guess S0 as the maximum signal, and T1 as a typical biological value.
    initial_s0_guess = np.max(signals)
    initial_r1_guess = 1/1.2
    initial_guesses = [initial_r1_guess, initial_s0_guess]
    
    # --- 3. Perform Non-Linear Fit ---
    popt, pcov = curve_fit(
        spgr_model,
        flip_angles_rad,
        signals,
        p0=initial_guesses,
        bounds=bounds,
    )
    calculated_r1, calculated_s0 = popt
    return calculated_r1, calculated_s0



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

