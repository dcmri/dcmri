import warnings
import logging
from joblib import Parallel, delayed

import numpy as np
from scipy.optimize import curve_fit


def train_batch(predict, time, signal, pars, free, **kwargs):
    def train_pixel(x):
        return train(predict, time, signal[x,...], pars, free, x, **kwargs)
    
    if signal.shape[0]==1:
        results = [train_pixel(0)]
    else:
        #results = [train_pixel(x) for x in range(signal.shape[0])] 
        results = Parallel(n_jobs=-1)(delayed(train_pixel)(x) for x in range(signal.shape[0]))  

    return results 

def format_batch_training(results, free):
    # Format outputs
    vals = {p: [] for p in free}
    sdev = {p: [] for p in free}
    for p in free:
        for r in results:
            if p in r[0]:
                vals[p].append(r[0][p])
            else:
                vals[p].append(np.nan)
            if p in r[1]:
                sdev[p].append(r[1][p])
            else:
                sdev[p].append(np.nan)
        vals[p] = np.array(vals[p])
        sdev[p] = np.array(sdev[p])

    pcov = np.empty(len(results), dtype=object)
    pcov[:] = [r[2] for r in results]

    model = np.empty(len(results), dtype=object)
    if len(results[0]) == 4:
        model[:] = [r[3] for r in results]
    else:
        model[:] = None

    return vals, sdev, pcov, model


def train(predict, time, signal, pars, free, x=None, reset=False, sigma=None, **kwargs):
    """Optimization logic using normalized parameter values."""

    if free == {}:
        return None, None, None
    
    # Flatten the signal
    if isinstance(signal, tuple):
        signal = np.concatenate(signal)
    signal = signal.reshape(-1)

    # Flatten sigma
    if isinstance(sigma, tuple):
        sigma = np.concatenate(sigma)

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
        # return np.ascontiguousarray(ypred, dtype=np.float64).reshape(-1)

    # Perform the optimization
    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, signal, p0, bounds=(0, 1), 
            sigma=sigma, **kwargs
        )
        sdev = _sdev(pcov, free)
    except RuntimeError as e:
        warnings.warn(f"Curve fit failed: {e}. Using initial values.")
        fitted_pars, pcov, sdev = p0, None, None

    # Create return values
    vals = {p: renormalize(fitted_pars[i], free[p]) for i, p in enumerate(free)}  

    # Update state
    if reset:
        # Rewind state to original values
        _update_original_pars(pars, p0, free, x)
    else:
        # Set state to final values
        _update_original_pars(pars, fitted_pars, free, x)
    
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


 # bad solution - replace loss below with a scalar loss
def scalar_loss(ypred, ydata, metric='NRMS', nfree=None) -> float:
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
        loss = np.linalg.norm(ypred - ydata)
    elif metric == 'NRMS':
        ynorm = np.linalg.norm(ydata)
        yerr = np.linalg.norm(ypred - ydata)
        with np.errstate(divide='ignore', invalid='ignore'):
            loss = 100*yerr/ynorm
    elif metric == 'AIC':
        rss = np.sum((ypred-ydata)**2)
        n = ydata.size
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*2 + n*np.log(rss/n)
    elif metric == 'cAIC':
        rss = np.sum((ypred-ydata)**2)
        n = ydata.size
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*2 + n*np.log(rss/n) + 2*nfree*(nfree+1)/(n-nfree-1)
    elif metric == 'BIC':
        rss = np.sum((ypred-ydata)**2)
        n = ydata.size
        if nfree is None:
            raise ValueError('Please specify the number of free parameters.')
        with np.errstate(divide='ignore'):
            loss = nfree*np.log(n) + n*np.log(rss/n)
    else:
        raise ValueError(f"Unknown metric {metric}")
    return loss


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
    # This is not intuitive - loss functions should return a scalar
    ydata = ydata.reshape(ydata.shape[0], -1)
    ypred = ypred.reshape(ypred.shape[0], -1)
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
        rss = np.sum((ypred-ydata)**2, axis=-1)
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
    else:
        raise ValueError(f"Unknown metric {metric}")
    return loss