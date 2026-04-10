import warnings
from joblib import Parallel, delayed

import numpy as np
from scipy.optimize import curve_fit


def train_batch(predict, time, signal, pars, free, **kwargs):
    def train_pixel(x):
        return train(predict, time, signal[x,...], pars, free, x, **kwargs)
    
    if signal.shape[0]==1:
        results = [train_pixel(0)]
    else:
        results = Parallel(n_jobs=-1)(delayed(train_pixel)(x) for x in range(signal.shape[0]))  

    return results 


def train(predict, time, signal, pars, free, x=None, reset=False, **kwargs):
    """Optimization logic using normalized parameter values."""

    if free == {}:
        return None, None
    
    # Flatten the signal
    if isinstance(signal, tuple):
        signal = np.concatenate(signal)
    signal = signal.reshape(-1)

    # Compute initial values
    p0 = _compute_normalized_pars(pars, free, x)

    # Define prediction function
    def predict_normalized(_, *normalized_pars):
        _update_original_pars(pars, normalized_pars, free, x)
        if x is None:
            ypred = predict(time)
        else:
            ypred = predict(time, x)
        
        # Flatten the signal prediction
        if isinstance(ypred, tuple):
            ypred = np.concatenate(ypred) 
        return ypred.reshape(-1)

    # Perform the optimization
    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, signal, p0, bounds=(0, 1), **kwargs
        )
        sdev = _sdev(pcov, free)
    except RuntimeError as e:
        warnings.warn(f"Curve fit failed: {e}. Using initial values.")
        fitted_pars, pcov, sdev = p0, None, None

    # Reset if needed
    if reset:
        # Rewind state to original values
        _update_original_pars(pars, p0, free, x)
    else:
        # Set state to final values
        _update_original_pars(pars, fitted_pars, free, x)
    
    # Create retur values
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