import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats



def conc_dce(Sn_model, S, n0=None, R10=None, S0=None, r1=None, R20s=None, **params):
    
    # TE does not affect the concentration
    if R20s is None:
        TE = 0
        R20s = 1
    elif 'TE' in params:
        TE = params['TE']
    else:
        TE = 0

    #Normalize signal
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        if np.isscalar(R20s):
            R20sb = np.full_like(R10, R20s)
        else:
            R20sb = R20s
        Sn0 = Sn_model(R1=R10, TE=TE, R2s=R20sb, S0=1, v=1, Fw=0, me=1, R1i=None, Fi=None) # Baseline R20 absorbed in S0
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn_data = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Create lookup table
    c_step = 0.01 * 1e-3
    c_max = 0.01
    c_range = np.arange(0, c_max, c_step)
    R1_min = 0
    R1_lookup = R1_min + r1 * c_range
    Sn_lookup = Sn_model(R1=R1_lookup, TE=TE, R2s=np.full_like(R1_lookup, R20s), S0=1, v=1, Fw=0, me=1, R1i=None, Fi=None)

    # # Check that lookup values are strictly increasing
    # if not np.all(np.diff(Sn_lookup) > 0):
    #     raise ValueError(
    #         f"Cannot convert signal to concentration directly. "
    #         "The signal values are not monotonously increasing for the given concentration range. \n"
    #         "You can avoid direct inversion by fitting straight to signal data."
    #     )

    # Look up R1 values
    R1 = np.interp(Sn_data, Sn_lookup, R1_lookup)
    
    # Convert to concentration
    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1


def conc_dsc(S, n0, r2, TE) -> np.ndarray:
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
    

def conc_ss(S, n0=None, R10=None, S0=None, R20s=None, r1=None, **p) -> np.ndarray:
    # S = Sinf * (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sb = Sinf * (1-exp(-TR*R10)) / (1-cFA*exp(-TR*R10))
    # Sn = (1-exp(-TR*R1)) / (1-cFA*exp(-TR*R1))
    # Sn * (1-cFA*exp(-TR*R1)) = 1-exp(-TR*R1)
    # exp(-TR*R1) - Sn *cFA*exp(-TR*R1) = 1-Sn
    # (1-Sn*cFA) * exp(-TR*R1) = 1-Sn
    FA = np.radians(p['FA'] * p['B1corr'])
    cFA = np.cos(FA)
    sFA = np.sin(FA)

    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        E0 = np.exp(-p['TR'] * R10)
        ED = 1 if R20s is None else np.exp(-p['TE'] * R20s)
        Sn0 = ED * sFA * (1 - E0) / (1 - cFA * E0)
        S0 = np.divide(Sb, Sn0, out=np.zeros_like(Sb, dtype=float), where=Sn0 > 0)

    S0 = S0[:, np.newaxis]
    Sn = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    # Invert analytically
    Sn = Sn / sFA
    En = (1 - Sn) / (1 - cFA * Sn)
    with np.errstate(divide='ignore', invalid='ignore'):
        R1 = np.where(En <= 0, 0, -np.log(En)/p['TR'])

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1


def conc_dce_lin(S, n0, R10, S0, r1):
    # S = S0 * R1
    if S0 is None:
        Sb = np.sum(S[:, :n0], axis=1) / n0
        S0 = Sb / R10
        S0 = np.divide(Sb, R10, out=np.zeros_like(Sb, dtype=float), where=R10 > 0)

    S0 = S0[:, np.newaxis] 
    R1 = np.divide(S, S0, out=np.zeros_like(S, dtype=float), where=S0 > 0)

    if R10 is None:
        R10 = np.sum(R1[:, :n0], axis=1) / n0

    R10 = R10[:, np.newaxis]
    return (R1 - R10) / r1


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

