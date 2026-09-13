import copy
import numpy as np

from dcmri.kinetics import functions_blocks


FLUX_PARAMETERS = {
    '2CX': ['H', 'v_b', 'v_i', 'F_b', 'PS'],
    'HF': ['H', 'v_i', 'PS'],
    'WV': ['H', 'v_i', 'Ktrans'],
    '2CU': ['H', 'v_b', 'F_b', 'PS'],
    'HFU': ['H', 'PS'],
    'FX': ['H', 'v_e', 'F_b'],
    'NX': ['v_b', 'F_b'],
    'NXP': ['v_b', 'F_b'],
    'U': ['F_b'],
}

CONC_PARAMETERS = {
    '2CX': ['H', 'v_b', 'v_i', 'F_b', 'PS'],
    'HF': ['H', 'v_b', 'v_i', 'PS'],
    '2CU': ['H', 'v_b', 'F_b', 'PS'],
    'HFU': ['H', 'v_b', 'PS'],
    'WV': ['H', 'v_i', 'Ktrans'],
    'FX': ['H', 'v_e', 'F_b'],
    'NX': ['v_b', 'F_b'],
    'NXP': ['v_b', 'F_b'],
    'U': ['F_b'],
}

def dpars_tissue(p, H=0.45):

    p = copy.deepcopy(p)

    if {'v_b'}.issubset(p):
        p['v_p'] = (1 - H) * p['v_b']

    if {'v_e', 'v_p'}.issubset(p):
        p['v_i'] = p['v_e'] - p['v_p']

    elif {'v_p', 'v_i'}.issubset(p):
        p['v_e'] = p['v_p'] + p['v_i']

    if {'v_b', 'v_i'}.issubset(p):
        p['vc'] = 1 - p['v_b'] - p['v_i']

    return p

# def _all_pars(kin, wex, seq, p):

#     #pars = _model_pars(kin, wex, seq)
#     p = {par: p[par] for par in pars}

#     try:
#         p['Fp'] = p['F_b'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['v_p'] = p['v_b'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['Ktrans'] = _div(p['Fp'] * p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['v_e'] = p['v_i'] + p['vc']
#     except KeyError:
#         pass
#     try:
#         p['E'] = _div(p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Ti'] = _div(p['v_i'], p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Tp'] = _div(p['v_p'], p['PS'] + p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Tb'] = _div(p['v_p'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Te'] = _div(p['v_e'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Twc'] = _div(1 - p['v_b'] - p['v_i'], p['PSc'])
#     except KeyError:
#         pass
#     try:
#         p['Twi'] = _div(p['v_i'], p['PSc'] + p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['Twb'] = _div(p['v_b'], p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['FAcorr'] = p['B1corr'] * p['FA']
#     except KeyError:
#         pass

#     return p


# def _div(a, b):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, 0, np.divide(a, b))
        

def conc_tissue_u(ca, t=None, dt=1.0, F_b=None):
    """
    Tissue concentration in an uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (1, nt) array.

    See Also
    --------
    flux_tissue_u : Flux out of an uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> F_b = 0.01
    >>> dc.conc_tissue_u(ca, t, F_b=F_b)
    array([[0.   , 0.075, 0.325, 0.775, 1.525]])
    """
    ca = np.array(ca)
    C = functions_blocks.conc_trap(F_b * ca, t=t, dt=dt)
    return C.reshape(1, -1)
    
def conc_tissue_fx(ca, t=None, dt=1.0, H=None, v_e=None, F_b=None):
    """
    Tissue concentration in a fast-exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_e : float
        Extracellular volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (1, nt) array.

    See Also
    --------
    flux_tissue_fx : Flux out of a fast-exchange tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_fx(ca, t, H=0.45, v_e=0.1, F_b=0.01)
    array([[0.        , 0.06657176, 0.23421188, 0.40905712, 0.42647156]])
    """
    ca = np.array(ca)
    if F_b == 0:
        ce = ca*0
    else:
        Fp = (1-H)*F_b
        ce = functions_blocks.flux_comp(ca/(1-H), t=t, dt=dt, T=v_e/Fp)
    return v_e*ce.reshape(1, -1)

def conc_tissue_nx(ca, t=None, dt=1.0, v_b=None, F_b=None):
    """
    Tissue concentration in a no-exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_b : float
        Blood volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (1, nt) array.

    See Also
    --------
    flux_tissue_nx : Flux out of a no-exchange tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_nx(ca, t, v_b=0.1, F_b=0.01)
    array([[0.        , 0.06065307, 0.18552507, 0.27445719, 0.23040206]])
    """
    ca = np.array(ca)
    if F_b == 0:
        Cb = ca*0
    else:
        Cb = functions_blocks.conc_comp(F_b*ca, t=t, dt=dt, T=v_b/F_b)
    return Cb.reshape(1, -1)

def conc_tissue_nxp(ca, t=None, dt=1.0, v_b=None, F_b=None):
    """
    Tissue concentration in a no-exchange plug-flow tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_b : float
        Blood volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (1, nt) array.

    See Also
    --------
    flux_tissue_nxp : Flux out of a no-exchange plug-flow tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_nxp(ca, t, v_b=0.1, F_b=0.01)
    array([[0.   , 0.075, 0.225, 0.3  , 0.25 ]])
    """
    ca = np.array(ca)
    if F_b == 0:
        Cb = ca*0
    else:
        Cb = functions_blocks.conc_plug(F_b*ca, t=t, dt=dt, T=v_b/F_b)
    return Cb.reshape(1, -1)

def conc_tissue_wv(ca, t=None, dt=1.0, H=None, v_i=None, Ktrans=None):
    """
    Tissue concentration in a weakly vascularized tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    Ktrans : float
        Volume transfer constant (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (1, nt) array.

    See Also
    --------
    flux_tissue_wv : Flux out of a no-exchange plug-flow tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_wv(ca, t, H=0.45, v_i=0.1, Ktrans=0.001)
    array([[0.        , 0.01333801, 0.05546861, 0.12371975, 0.20828741]])
    """
    ca = np.array(ca)
    if Ktrans == 0:
        ci = ca*0
    else:
        ci = functions_blocks.flux_comp(ca/(1-H), t=t, dt=dt, T=v_i/Ktrans)
    return v_i * ci.reshape(1, -1)

def conc_tissue_hfu(ca, t=None, dt=1.0, H=None, v_b=None, PS=None):
    """
    Tissue concentration in a high-flow uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt) array.

    See Also
    --------
    flux_tissue_hfu : Flux out of a high-flow uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_hfu(ca, t, H=0.45, v_b=0.1, PS=0.003)
    array([[0.1       , 0.2       , 0.3       , 0.3       , 0.2       ],
           [0.        , 0.04090909, 0.17727273, 0.42272727, 0.83181818]])
    """
    ca = np.array(ca)
    v_p = v_b*(1-H)
    cp = ca/(1-H)
    Ci = functions_blocks.conc_trap(PS*cp, t=t, dt=dt)
    return np.stack((v_p*cp, Ci)) 

def conc_tissue_hf(ca, t=None, dt=1.0, H=None, v_i=None, v_b=None, PS=None):
    """
    Tissue concentration in a high-flow tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt) array.

    See Also
    --------
    flux_tissue_hf : Flux out of a high-flow uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_hf(ca, t, H=0.45, v_i=0.3, v_b=0.1, PS=0.003)
    array([[0.1       , 0.2       , 0.3       , 0.3       , 0.2       ],
           [0.        , 0.04001404, 0.16640584, 0.37115924, 0.62486222]])
    """
    ca = np.array(ca)
    v_p = v_b*(1-H)
    ca = ca/(1-H)
    Cp = v_p*ca
    if PS == 0:
        Ci = 0*ca
    else:
        Ci = functions_blocks.conc_comp(PS*ca, t=t, dt=dt, T=v_i/PS)
    return np.stack((Cp, Ci))

def conc_tissue_2cu(ca, t=None, dt=1.0, H=None, v_b=None, F_b=None, PS=None):
    """
    Tissue concentration in a 2-compartment uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt) array.

    See Also
    --------
    flux_tissue_2cu : Flux out of a 2-compartment uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_2cu(ca, t, H=0.45, v_b=0.1, F_b=0.01, PS=0.003)
    array([[0.        , 0.05446241, 0.14519581, 0.18930117, 0.14318597],
           [0.        , 0.00742669, 0.06187893, 0.19871861, 0.47075354]])
    """
    if np.isinf(F_b):
        return conc_tissue_hfu(ca, t=t, dt=dt, H=H, v_b=v_b, PS=PS)
    ca = np.array(ca)
    v_p = (1 - H) * v_b
    Fp = (1 - H) * F_b
    ca = ca / (1 - H)
    if Fp+PS == 0:
        return np.zeros((2, len(ca)))
    Tp = v_p/(Fp+PS)
    Cp = functions_blocks.conc_comp(Fp*ca, t=t, dt=dt, T=Tp)
    if v_p == 0:
        Ktrans = PS*Fp/(PS+Fp)
        Ci = functions_blocks.conc_trap(Ktrans*ca, t=t, dt=dt)
    else:
        Ci = functions_blocks.conc_trap(PS*Cp/v_p, t=t, dt=dt)
    return np.stack((Cp, Ci))

def conc_tissue_2cx(ca, t=None, dt=1.0, H=None, v_i=None, v_b=None, F_b=None, PS=None):
    """
    Tissue concentration in a 2-compartment exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt) array.

    See Also
    --------
    flux_tissue_2cu : Flux out of a 2-compartment uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_tissue_2cx(ca, t, H=0.45, v_i=0.3, v_b=0.1, F_b=0.01, PS=0.003)
    array([[0.        , 0.05456069, 0.14710382, 0.1980532 , 0.16692313],
           [0.        , 0.00713764, 0.06075431, 0.18960275, 0.40554217]])
    """
    ca = np.array(ca)
    v_p = (1-H)*v_b
    Fp = (1-H)*F_b
    if np.isinf(Fp):
        return conc_tissue_hf(ca, t=t, dt=dt, H=H, v_i=v_i, v_b=v_b, PS=PS)

    ca = ca/(1-H)
    J = Fp*ca

    if Fp+PS == 0:
        Cp = np.zeros(len(ca))
        Ce = np.zeros(len(ca))
        return np.stack((Cp, Ce))

    Tp = v_p/(Fp+PS)
    E = PS/(Fp+PS)

    if PS == 0:
        Cp = functions_blocks.conc_comp(Fp*ca, t=t, dt=dt, T=Tp)
        Ci = np.zeros(len(ca))
        return np.stack((Cp, Ci))

    Ti = v_i/PS
    C = functions_blocks.conc_2cxm(J, t=t, dt=dt, T=[Tp, Ti], E=E)
    return C
    




def flux_tissue_u(ca, t=None, dt=1.0, F_b=None):
    """
    Flux out of an uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 1D array (nt) array.

    See Also
    --------
    conc_tissue_u : Tissue concentration in an uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_u(ca, t, F_b=0.01)
    array([0., 0., 0., 0., 0.])
    """
    ca = np.array(ca)
    return functions_blocks.flux_trap(F_b*ca)

def flux_tissue_fx(ca, t=None, dt=1.0, H=None, v_e=None, F_b=None):
    """
    Flux out of a fast-exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_e : float
        Extracellular volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (1, nt) array.

    See Also
    --------
    conc_tissue_fx : Concentration in a fast-exchange tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_fx(ca, t, H=0.45, v_e=0.1, F_b=0.01)
    array([0.        , 0.00366145, 0.01288165, 0.02249814, 0.02345594])
    """
    ca = np.array(ca)
    if F_b == 0:
        return np.zeros(len(ca))
    Fp = F_b*(1-H)
    return functions_blocks.flux_comp(F_b*ca, t=t, dt=dt, T=v_e/Fp)

def flux_tissue_nx(ca, t=None, dt=1.0, v_b=None, F_b=None):
    """
    Flux out of a no-exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_b : float
        Blood volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (1, nt) array.

    See Also
    --------
    conc_tissue_nx : Flux out of a no-exchange tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_nx(ca, t, v_b=0.1, F_b=0.01)
    array([0.        , 0.00606531, 0.01855251, 0.02744572, 0.02304021])
    """
    ca = np.array(ca)
    if F_b == 0:
        return np.zeros(len(ca))
    return functions_blocks.flux_comp(F_b*ca, t=t, dt=dt, T=v_b/F_b)

def flux_tissue_nxp(ca, t=None, dt=1.0, v_b=None, F_b=None):
    """
    Flux out of a no-exchange plug-flow tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_b : float
        Blood volume fraction (mL/cm3)
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (1, nt) array.

    See Also
    --------
    conc_tissue_nxp : Tissue concentration in a no-exchange plug-flow tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_nxp(ca, t, v_b=0.1, F_b=0.01)
    array([0.        , 0.        , 0.02      , 0.03      , 0.02333333])
    """
    ca = np.array(ca)
    if F_b == 0:
        return np.zeros(len(ca))
    return functions_blocks.flux_plug(F_b*ca, t=t, dt=dt, T=v_b/F_b)

def flux_tissue_wv(ca, t=None, dt=1.0, H=None, v_i=None, Ktrans=None):
    """
    Flux out of a weakly vascularized tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    Ktrans : float
        Volume transfer constant (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 3D array (2, 2, nt) array. 

    See Also
    --------
    conc_tissue_wv : Tissue concentration in a no-exchange plug-flow tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_wv(ca, t, H=0.45, v_i=0.1, Ktrans=0.001)
    array([[[       nan,        nan,        nan,        nan,        nan],
            [0.        , 0.00013338, 0.00055469, 0.0012372 , 0.00208287]],

           [[0.00181818, 0.00363636, 0.00545455, 0.00545455, 0.00363636],
            [0.        , 0.        , 0.        , 0.        , 0.        ]]])
    """
    ca = np.array(ca)
    ca = ca / (1 - H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan # TODO: double-check this
    J[1, 0, :] = Ktrans * ca
    if Ktrans != 0:
        J[0, 1, :] = functions_blocks.flux_comp(Ktrans*ca, t=t, dt=dt, T=v_i/Ktrans)
    return J

def flux_tissue_hfu(ca, t=None, dt=1.0, H=None, PS=None):
    """
    Flux out of a high-flow uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    PS : float
        Permeability-surface area product (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (2, 2, nt) array.

    See Also
    --------
    conc_tissue_hfu : Tissue concentration in a high-flow uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_hfu(ca, t, H=0.45, PS=0.003)
    array([[[       nan,        nan,        nan,        nan,        nan],
            [0.        , 0.        , 0.        , 0.        , 0.        ]],

        [[0.00545455, 0.01090909, 0.01636364, 0.01636364, 0.01090909],
            [0.        , 0.        , 0.        , 0.        , 0.        ]]])
    """
    ca = np.array(ca)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.nan
    J[1, 0, :] = PS*ca/(1-H)
    return J

def flux_tissue_hf(ca, t=None, dt=1.0, H=None, v_i=None, PS=None):
    """
    Flux out of a high-flow tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (2, 2, nt) array.

    See Also
    --------
    conc_tissue_hf : Tissue concentration in a high-flow tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_hf(ca, t, H=0.45, v_i=0.3, PS=0.003)
    array([[[       inf,        inf,        inf,        inf,        inf],
            [0.        , 0.00040014, 0.00166406, 0.00371159, 0.00624862]],

        [[0.00545455, 0.01090909, 0.01636364, 0.01636364, 0.01090909],
            [0.        , 0.        , 0.        , 0.        , 0.        ]]])
    """
    ca = np.array(ca)
    ca = ca/(1-H)
    J = np.zeros(((2, 2, len(ca))))
    J[0, 0, :] = np.inf
    J[1, 0, :] = PS*ca
    if PS == 0:
        J[0, 1, :] = 0*ca
    else:
        J[0, 1, :] = functions_blocks.flux_comp(PS*ca, t=t, dt=dt, T=v_i/PS)
    return J

def flux_tissue_2cu(ca, t=None, dt=1.0, H=None, v_b=None, F_b=None, PS=None):
    """
    Flux out of a 2-compartment uptake tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (2, 2, nt) array.

    See Also
    --------
    conc_tissue_2cu : Tissue concentration in a 2-compartment uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_2cu(ca, t, H=0.45, v_b=0.1, F_b=0.01, PS=0.003)
    array([[[0.        , 0.00299543, 0.00798577, 0.01041156, 0.00787523],
            [0.        , 0.        , 0.        , 0.        , 0.        ]],

        [[0.        , 0.00163387, 0.00435587, 0.00567904, 0.00429558],
            [0.        , 0.        , 0.        , 0.        , 0.        ]]])
    """
    ca = np.array(ca)
    C = conc_tissue_2cu(ca, t=t, dt=dt, H=H, v_b=v_b, F_b=F_b, PS=PS)
    ca = ca/(1-H)
    Fp = F_b*(1-H)
    J = np.zeros(((2, 2, len(ca))))
    if v_b == 0:
        if Fp+PS != 0:
            Ktrans = Fp*PS/(Fp+PS)
            J[0, 0, :] = Fp*ca
            J[1, 0, :] = Ktrans*ca
    else:
        J[0, 0, :] = Fp*C[0, :]/v_b
        J[1, 0, :] = PS*C[0, :]/v_b
    return J


def flux_tissue_2cx(ca, t=None, dt=1.0, H=None, v_b=None, v_i=None, F_b=None, PS=None):
    """
    Flux out of a 2-compartment exchange tissue.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial blood concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    H : float
        Hematocrit
    v_i : float
        Interstitial volume fraction (mL/cm3)
    v_b : float
        Blood volume fraction (mL/cm3)
    PS : float
        Permeability-surface area product (mL/sec/cm3).
    F_b : float
        Tissue blood flow (mL/sec/cm3).

    Returns
    -------
    np.ndarray
        Flux as a 2D array (2, nt) array.

    See Also
    --------
    conc_tissue_2cu : Tissue concentration in a 2-compartment uptake tissue

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.flux_tissue_2cx(ca, t, H=0.45, v_i=0.3, v_b=0.1, F_b=0.01, PS=0.003)
    array([[[0.00000000e+00, 5.45606860e-03, 1.47103815e-02, 1.98053196e-02,
            1.66923134e-02],
            [0.00000000e+00, 7.13764072e-05, 6.07543096e-04, 1.89602751e-03,
            4.05542166e-03]],

        [[0.00000000e+00, 2.97603742e-03, 8.02384446e-03, 1.08029016e-02,
            9.10489819e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00]]])
    """
    ca = np.array(ca)

    if np.isinf(F_b):
        return flux_tissue_hf(ca, t=t, dt=dt, H=H, v_i=v_i, PS=PS)

    if F_b == 0:
        return np.zeros((2, 2, len(ca)))

    Fp = F_b*(1-H)

    if PS == 0:
        Jp = flux_tissue_nx(ca, t=t, dt=dt, v_b=v_b, F_b=F_b)
        J = np.zeros((2, 2, len(ca)))
        J[0, 0, :] = Jp
        return J
    
    C = conc_tissue_2cx(ca, t=t, dt=dt, H=H, v_b=v_b, v_i=v_i, F_b=F_b, PS=PS)
    # Deriv_e standard parameters
    v_p = v_b*(1-H)
    Tp = v_p/(Fp+PS)
    Te = v_i/PS
    E = PS/(Fp+PS)
    # Build the system matrix K
    T = [Tp, Te]
    E = [
        [1-E, 1],
        [E,   0],
    ]
    return functions_blocks._J_ncomp(C, T, E)


# def flux_tissue_2cf(ca, t=None, dt=1.0, v_p=None, Fp=None, PS=None, Te=None):
#     if Fp+PS == 0:
#         return np.zeros((2, 2, len(ca)))
#     # Derive standard parameters
#     Tp = v_p/(Fp+PS)
#     E = PS/(Fp+PS)
#     J = Fp*ca
#     T = [Tp, Te]
#     # Solve the system explicitly
#     t = utils.tarray(len(J), t=t, dt=dt)
#     Jo = np.zeros((2, 2, len(t)))
#     J0 = blocks.flux(J, T[0], t=t, model='comp')
#     J10 = E*J0
#     Jo[1, 0, :] = J10
#     Jo[1, 1, :] = blocks.flux(J10, T[1], t=t, model='comp')
#     Jo[0, 0, :] = (1-E)*J0
#     return Jo
