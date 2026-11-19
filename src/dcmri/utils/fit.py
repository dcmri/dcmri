import warnings
import logging
import itertools

from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pydmr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from dcmri.core.tools import get_quantity


def train_brute(predict, signal, pars, free, n_steps=3, 
                brute=None, n_jobs=-1, **kwargs):
    
    if free == {}:
        raise ValueError("The dictionary of free parameters can't be empty")
    
    if brute is None:
        return _train(predict, signal, pars, free, **kwargs)

    free_brute = {b for b in brute if b in free}
    if not free_brute:
        return _train(predict, signal, pars, free, **kwargs)

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    names = sorted(free_brute)  # fixed ordering for the grid
    grids = [np.linspace(*free[name], n_steps) for name in names]

    starts = []
    for values in itertools.product(*grids):
        start = dict(pars)  # copy: non-brute params keep their initial values
        start.update(zip(names, values))
        starts.append(start)

    if n_jobs==1:
        results = [
            _train(predict, signal, start, free, **kwargs)
            for start in starts
        ]       
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_train)(predict, signal, start, free, **kwargs)
            for start in starts
        )

    return min(results, key=lambda r: r['loss'])


def _train(predict, signal, pars, free, sigma=None, snorm=True, **kwargs):
    """Optimization logic using normalized parameter values."""

    signal, a, b = _normalize_signal(signal, snorm)
    # if isinstance(signal, tuple):
    #     signal = np.concatenate([s.reshape(-1) for s in signal])
    # signal = signal.reshape(-1)
    
    # Flatten sigma
    if isinstance(sigma, tuple):
        sigma = np.concatenate(sigma)

    # Compute initial values
    p0 = _normalized_pars(pars, free)

    # Define prediction function
    def predict_normalized(_, *normalized_pars):
        _update_pars(pars, normalized_pars, free)
        ypred = predict(pars)
        ypred, _, _ = _normalize_signal(ypred, snorm, a, b)
        return ypred
        # if isinstance(ypred, tuple):
        #     ypred = np.concatenate([y.reshape(-1) for y in ypred]) 
        # return ypred.reshape(-1)

    # Perform the optimization
    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, signal, p0, bounds=(0, 1), 
            sigma=sigma, **kwargs
        )
    except RuntimeError as e:
        # Rewind state to initial values
        _update_pars(pars, p0, free)
        raise RuntimeError(f"Curve fit failed: {e}")
        # fitted_pars, pcov, sdev = p0, None, None

    # Rewind state to initial values
    _update_pars(pars, p0, free)

    # Create return values
    sdev = _sdev(pcov, free)
    vals = _original_pars(fitted_pars, free) 
    cost = 100 * np.linalg.norm(predict_normalized(None, *fitted_pars) - signal) / np.linalg.norm(signal)

    return {'popt': vals, 'psdev': sdev, 'pcov': pcov, 'loss': cost}


def _normalized_pars(pars, free_pars):
    return [normalize(pars[p], free_pars[p]) for p in free_pars]

def _original_pars(normalized_pars, free):
    return {p: renormalize(normalized_pars[i], free[p]) for i, p in enumerate(free)}

def _update_pars(pars, normalized_pars, free):
    for i, p in enumerate(free):
        pars[p] = renormalize(normalized_pars[i], free[p])

def normalize(v, bounds):
    return (v - bounds[0]) / (bounds[1] - bounds[0])

def renormalize(v, bounds):
    return v * (bounds[1] - bounds[0]) + bounds[0]

def _normalize_signal(signal, snorm=False, a=None, b=None):
    if isinstance(signal, tuple):
        signal = [s.reshape(-1) for s in signal]
        if snorm:
            if a is None:
                a = [np.std(s) for s in signal]
                b = [np.mean(s) for s in signal]
            signal = [
                (s - b[i]) / a[i] if a[i] != 0 else s - b[i] 
                for i, s in enumerate(signal)
            ]
        signal = np.concatenate(signal)
    else:
        signal = signal.reshape(-1)
        if snorm:
            if a is None:
                a = np.std(signal)
                b = np.mean(signal)
            signal = (signal - b) / a if a != 0 else signal - b
    return signal, a, b

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


def save_results(result:dict, filepath:str, subj='subject_id', study='study_id'):
    dmr = {
        'data': {},
        'pars': {},
        'columns': ['group', 'dicom_key', 'osipi_key'],
    }
    if result['pcov'] is not None:
        dmr['rois'] = {}
        dmr['sdev'] = {}

    for key, val in result['pars'].items():
        q = get_quantity(key)
        if isinstance(val, str):
            dmr['data'][key] = [q['name'], q['unit'], 'str', q['group'], q['dicom_key'], q['osipi_key']]
        else:
            dmr['data'][key] = [q['name'], q['unit'], 'float', q['group'], q['dicom_key'], q['osipi_key']]
        dmr['pars'][subj, study, key] = val

    for i, key in enumerate(result['popt']):
        q = get_quantity(key)
        if isinstance(val, str):
            dmr['data'][key] = [q['name'], q['unit'], 'str', q['group'], q['dicom_key'], q['osipi_key']]
        else:
            dmr['data'][key] = [q['name'], q['unit'], 'float', q['group'], q['dicom_key'], q['osipi_key']]
        if result['pcov'] is not None:
            dmr['rois'][subj, study, key] = result['pcov'][:,i]
            dmr['sdev'][subj, study, key] = result['psdev'][key]

    pydmr.write(filepath, dmr)


def read_results(filepath:str):
    dmr = pydmr.read(filepath)
    result = {}

    result['pars'] = {}
    for (subj, study, key), val in dmr['pars'].items():
        result['pars'][key] = val

    result['psdev'] = {}
    result['popt'] = {}
    for (subj, study, key), val in dmr['sdev'].items():
        result['psdev'][key] = val
        result['popt'][key] = result['pars'][key]

    pcov = []
    for (subj, study, key), val in dmr['rois'].items():
        pcov.append(val)
    result['pcov'] = np.array(pcov)

    return result, subj, study



def _cluster(pcov, method='average'):
    # Hierarchical clustering on d = 1 - |correlation|.
    # Working with correlations makes the result independent of parameter units.
    sd = np.sqrt(np.abs(np.diag(pcov)))
    sd[sd == 0] = 1.0
    corr = pcov / np.outer(sd, sd)
    dist = np.clip(1.0 - np.abs(corr), 0, None)
    dist = 0.5 * (dist + dist.T)              # remove rounding asymmetry
    np.fill_diagonal(dist, 0)
    return linkage(squareform(dist, checks=False), method=method,
                   optimal_ordering=True)


def plot_pcov(result, fname: str = None, show=True, cluster=True,
              dendrograms=True, vmin=-0.025, vmax=0.025):

    if result['pcov'] is None:
        return

    pcov = np.asarray(result['pcov'])
    labels = list(result['psdev'].keys())
    n = len(labels)

    Z = None
    if cluster and n > 2:
        Z = _cluster(pcov)
        order = dendrogram(Z, no_plot=True)['leaves']
        pcov = pcov[np.ix_(order, order)]
        labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pcov, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.5)
    ax.tick_params(which='minor', length=0)

    # Dendrogram attached to the top of the heatmap
    if Z is not None and dendrograms:
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="12%", pad=0.05)
        dendrogram(Z, ax=ax_top, orientation='top', no_labels=True,
                   color_threshold=0, above_threshold_color='0.4')
        ax_top.set_xlim(0, 10 * n)             # leaves sit at 5, 15, 25, ...
        ax_top.axis('off')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Covariance Magnitude')

    fig.suptitle("Parameter Covariance Matrix Heatmap", fontsize=14)
    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname=fname)
    if show:
        plt.show()
    else:
        plt.close(fig)
        

def plot_pvar(result, fname: str = None, xlim: list = None, show=True):

    if result['pcov'] is None:
        return

    values = result['popt']
    sdevs = result['psdev']

    names = sorted(values.keys(), key=str.lower)
    rel = np.array([sdevs[k] / abs(values[k]) for k in names])  # normalized sdev
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(names) + 1.5))

    ax.axvline(0, color='gray', linewidth=1, zorder=0)
    ax.errorbar(np.zeros(len(names)), y, xerr=rel, fmt='o', color='C0',
                ecolor='C0', elinewidth=1.5, capsize=4)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized sdev (sdev / value)")
    ax.set_title("Parameter uncertainty", fontsize=14, pad=15)

    # Default: symmetric limits around zero; override with xmin / xmax if given
    lim = rel.max() * 1.1
    ax.set_xlim(-lim if xlim is None else xlim[0],
                lim if xlim is None else xlim[1])

    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname=fname)
    if show:
        plt.show()
    else:
        plt.close(fig)


# # Obsolete
# def _orig_train_brute(predict, signal, pars, free, n_steps=3, brute=None, sigma=None, **kwargs):
#     if free == {}:
#         raise ValueError("The dictionary of free parameters can't be empty")
    
#     if brute is None:
#         return _train(predict, signal, pars, free, sigma=sigma, **kwargs)

#     free_brute = {b for b in brute if b in free}
#     if not free_brute:
#         return _train(predict, signal, pars, free, sigma=sigma, **kwargs)

#     if n_steps < 1:
#         raise ValueError("n_steps must be at least 1")

#     names = sorted(free_brute)  # fixed ordering for the grid
#     grids = [np.linspace(*free[name], n_steps) for name in names]

#     best = None
#     for values in itertools.product(*grids):
#         start = dict(pars)
#         start.update(zip(names, values))

#         try:
#             result = _train(predict, signal, start, free, sigma=sigma, **kwargs)
#         except Exception:
#             continue  # skip starting points where the fit fails

#         loss = result['loss']
#         if np.isfinite(loss) and (best is None or loss < best['loss']):
#             best = result

#     if best is None:
#         raise RuntimeError("curvefit failed for every starting point")
    
#     return best

# # Obsolete
# def train_bat(predict, time, signal, pars, free, x=None, reset=False, sigma=None, n_bat=1, bats=None, btol=1e-3, **kwargs):
#     if bats is None:
#         bats = ['BAT']

#     for bat_i in bats:
#         if bat_i in free:
#             if n_bat > 1:
#                 n_bat_i, loss_i = 1, np.inf
#                 while (n_bat_i < n_bat) and (loss_i > btol):

#                     bat_array = np.linspace(free[bat_i][0], free[bat_i][1], n_bat_i)
#                     cost = []
#                     for bat in bat_array:
#                         pars[bat_i] = bat
#                         result = train(predict, time, signal, pars, free, x=x, reset=True, sigma=sigma, **kwargs)
#                         cost += [result['loss']]

#                     n_bat_i += 1
#                     loss_i = min(cost)
                    
#                 pars[bat_i] = bat_array[cost.index(loss_i)]

#     return train(predict, time, signal, pars, free, x=x, reset=reset, sigma=sigma, **kwargs)

# # OBSOLETE PHASE OUT
# def train(predict, time, signal, pars, free, x=None, reset=False, sigma=None, **kwargs):
#     """Optimization logic using normalized parameter values."""

#     if free == {}:
#         return None, None, None
    
#     # Flatten the signal
#     if isinstance(signal, tuple):
#         signal = np.concatenate([s.reshape(-1) for s in signal])
#         # signal = np.concatenate(signal)
#     signal = signal.reshape(-1)

#     # Flatten sigma
#     if isinstance(sigma, tuple):
#         sigma = np.concatenate(sigma)

#     # Compute initial values
#     p0 = _compute_normalized_pars(pars, free, x)

#     # Define prediction function
#     def predict_normalized(_, *normalized_pars):
#         _update_original_pars(pars, normalized_pars, free, x)
#         if x is None:
#             ypred = predict(time)
#         else:
#             ypred = predict(time, x)
        
#         # Flatten the signal prediction
#         if isinstance(ypred, tuple):
#             ypred = np.concatenate(ypred) 
#         return ypred.reshape(-1)
#         # return np.ascontiguousarray(ypred, dtype=np.float64).reshape(-1)

#     # Perform the optimization
#     try:
#         fitted_pars, pcov = curve_fit(
#             predict_normalized, None, signal, p0, bounds=(0, 1), 
#             sigma=sigma, **kwargs
#         )
#         sdev = _sdev(pcov, free)
#     except RuntimeError as e:
#         warnings.warn(f"Curve fit failed: {e}. Using initial values.")
#         fitted_pars, pcov, sdev = p0, None, None

#     # Create return values
#     vals = {p: renormalize(fitted_pars[i], free[p]) for i, p in enumerate(free)}  
#     cost = 100 * np.linalg.norm(predict_normalized(None, *fitted_pars) - signal) / np.linalg.norm(signal)

#     # Update state
#     if reset:
#         # Rewind state to original values
#         _update_original_pars(pars, p0, free, x)
#     else:
#         # Set state to final values
#         _update_original_pars(pars, fitted_pars, free, x)
    
#     return {'popt': vals, 'psdev': sdev, 'pcov': pcov, 'loss': cost}


# def _compute_normalized_pars(original_pars, free_pars, x=None):
#     if x is None:
#         return [normalize(original_pars[p], free_pars[p]) for p in free_pars]
#     else:
#         return [normalize(original_pars[p][x], free_pars[p]) for p in free_pars]

# def _update_original_pars(original_pars, normalized_pars, free, x=None):
#     for i, p in enumerate(free):
#         if x is None:
#             original_pars[p] = renormalize(normalized_pars[i], free[p])
#         else:
#             original_pars[p][x] = renormalize(normalized_pars[i], free[p])




# not in use
def train_batch(predict, signal, pars, free, **kwargs):
    def train_pixel(x):
        return train_brute(predict, signal[x,...], pars, free, x, **kwargs)
    
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
