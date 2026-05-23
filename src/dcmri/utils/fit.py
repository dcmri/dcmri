import warnings
import logging
from joblib import Parallel, delayed

import numpy as np
from scipy.optimize import curve_fit, minimize, differential_evolution
import copy


def train_batch(predict, time, signal, pars, free, **kwargs):
    def train_pixel(x):
        return train(predict, time, signal[x,...], pars, free, x, **kwargs)
    
    if signal.shape[0]==1:
        results = [train_pixel(0)]
    else:
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
        return None, None
    
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


def train_custom(predict, time, signal, pars, free, x=None, reset=False, sigma=None, loss=None, **kwargs):
    """Optimization logic using normalized parameter values and scipy.optimize.minimize."""

    if free == {}:
        return None, None
    
    # Flatten the signal
    if isinstance(signal, tuple):
        signal = np.concatenate(signal)
    signal = signal.reshape(-1)

    # Flatten sigma
    if isinstance(sigma, tuple):
        sigma = np.concatenate(sigma)
    if sigma is not None:
        sigma = sigma.reshape(-1)

    # Compute initial values
    p0 = _compute_normalized_pars(pars, free, x)

    # Helper to calculate model prediction
    def get_prediction(normalized_pars):
        _update_original_pars(pars, normalized_pars, free, x)
        if x is None:
            ypred = predict(time)
        else:
            ypred = predict(time, x)
        
        if isinstance(ypred, tuple):
            ypred = np.concatenate(ypred) 
        return ypred.reshape(-1)

    # Define the custom objective (loss) function
    def objective_function(normalized_pars):
        ypred = get_prediction(normalized_pars)

        if loss is not None:
            return loss(ypred, signal, pars=normalized_pars, sigma=sigma)

        # 1. Base Loss: Weighted or standard Least Squares
        if sigma is not None:
            residuals = (signal - ypred) / sigma
        else:
            residuals = signal - ypred
        
        return np.sum(residuals ** 2) 

    # Set up bounds (mimicking bounds=(0, 1) from your curve_fit)
    bounds = [(0, 1) for _ in p0]

    # Perform the optimization
    try:
        # L-BFGS-B handles bounds effectively and uses gradients efficiently
        res = minimize(
            objective_function, 
            p0, 
            bounds=bounds, 
            method=kwargs.pop('method', 'L-BFGS-B'), 
            **kwargs
        )
        
        if not res.success:
            warnings.warn(f"Minimize did not converge successfully: {res.message}")
            
        fitted_pars = res.x
        
        # Approximate Covariance Matrix using the inverse Hessian (Hessian approximation)
        # Note: Depending on the method, res.hess_inv might be an L-BFGS operator.
        # For formal error propagation with regularization, you may need a custom Hessian.
        try:
            if hasattr(res, 'hess_inv') and res.hess_inv is not None:
                if hasattr(res.hess_inv, 'todense'):
                    pcov = res.hess_inv.todense()
                else:
                    pcov = res.hess_inv
            else:
                pcov = None
        except Exception:
            pcov = None
            
        sdev = _sdev(pcov, free) if pcov is not None else None

    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using initial values.")
        fitted_pars, pcov, sdev = p0, None, None

    # Reset if needed
    if reset:
        _update_original_pars(pars, p0, free, x)
    else:
        _update_original_pars(pars, fitted_pars, free, x)
    
    # Create return values
    vals = {p: fitted_pars[i] for i, p in enumerate(free)}
    return vals, sdev, pcov


def train_diff(predict, time, signal, pars, free, x=None, reset=False, sigma=None, msg=None, **kwargs):
    """Optimization logic using Differential Evolution (No initial guess required)."""

    if free == {}:
        return None, None
    
    # Flatten the signal
    if isinstance(signal, tuple):
        signal = np.concatenate(signal)
    signal = signal.reshape(-1)

    # Flatten sigma (needed for weighted Least Squares)
    if isinstance(sigma, tuple):
        sigma = np.concatenate(sigma)
        sigma = sigma.reshape(-1)
    
    # We still compute p0 to know how many parameters we have 
    # and to provide a fallback if the optimization fails.
    p0 = _compute_normalized_pars(pars, free, x)

    # --- COST FUNCTION FOR DIFFERENTIAL EVOLUTION ---
    def objective_function(normalized_pars):
        # 1. Update the original pars object
        _update_original_pars(pars, normalized_pars, free, x)
        
        # 2. Generate prediction
        if x is None:
            ypred = predict(time)
        else:
            ypred = predict(time, x)
        
        if isinstance(ypred, tuple):
            ypred = np.concatenate(ypred) 
        ypred = ypred.reshape(-1)

        # 3. Calculate Sum of Squared Errors (Weighted if sigma exists)
        if sigma is not None:
            return np.sum(((signal - ypred) / sigma)**2)
        return np.sum((signal - ypred)**2)
    
    # --- DEFINE THE CALLBACK ---
    def my_callback(xk, convergence):
        my_callback.cntr += 1  # Increment the function attribute
        # xk is the current best normalized parameter set
        # convergence is the completion fraction (0.0 to 1.0)
        current_error = objective_function(xk)
        loginfo = f"Generation {my_callback.cntr}: Progress: {convergence:.2%} | Current Best Error: {current_error:.4e}"
        if msg:
            loginfo = f"{msg}: {loginfo}"
        logging.info(loginfo)

        # print(f"Progress: {convergence:.2%} | Current Best Error: {current_error:.4e}")
        # Hint: You can return True here to stop the optimization early!
        
    my_callback.cntr = 0  # Initialize the attribute

    # --- RUN THE OPTIMIZATION ---
    # Since your system uses normalized parameters, bounds are always (0, 1)
    bounds = [(0, 1) for _ in range(len(free))]

    try:
        result = differential_evolution(
            objective_function, 
            bounds, 
            callback=my_callback,
            **kwargs  # Pass things like popsize, tol, or mutation here
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        fitted_pars = result.x
        # Differential Evolution doesn't provide pcov natively like curve_fit
        pcov, sdev = None, None 

    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}. Using initial values.")
        fitted_pars, pcov, sdev = p0, None, None

    # Reset or Update state
    if reset:
        _update_original_pars(pars, p0, free, x)
    else:
        _update_original_pars(pars, fitted_pars, free, x)
    
    # Create return values
    vals = {p: renormalize(fitted_pars[i], free[p]) for i, p in enumerate(free)}
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