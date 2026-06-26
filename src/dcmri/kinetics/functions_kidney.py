import copy
import numpy as np

import dcmri.kinetics.functions_blocks as pk


PARAMETERS = {
    '2CF': ['Fp', 'vp', 'FF', 'Tt'],
    '2PF': ['Fp', 'vp', 'FF', 'Tt'],
    'CPF': ['Fp', 'vp', 'FF', 'Tt'],
    '2CFU': ['Fp', 'vp', 'FF'],
    '2PFU': ['Fp', 'vp', 'FF'],
    'FN': ['Fp', 'vp', 'FF', 'ht'],
    'HF': ['vp', 'Ft', 'Tt'],
    'HFU': ['vp', 'Ft'],
}

VASCULAR_MODEL = {
    '2CF': 'comp', 
    '2PF': 'plug', 
    'CPF': 'comp', 
    '2CFU': 'comp', 
    '2PFU': 'plug', 
    'HF': 'pass', 
    'HFU': 'pass', 
    'FN': 'plug',
}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b)
    

def dpars_kidney(p, kinetics='2CF', H=0.45) -> dict:

    p = copy.deepcopy(p)

    if {'Fp'}.issubset(p):
        p['Fb'] = _div(p['Fp'], 1 - H)

    if {'vp', 'Fp', 'Tt'}.issubset(p):
        p['Tp'] = _div(p['vp'], p['Fp']+p['Ft'])

    if {'vp', 'Fp'}.issubset(p):
        p['Tv'] = _div(p['vp'], p['Fp'])

    if {'FF', 'Fp'}.issubset(p):
        p['Ft'] = p['FF'] * p['Fp']
        p['Eg'] = _div(p['Ft'], p['Ft'] + p['Fp'])
        
    if {'Ft', 'vol'}.issubset(p):
        p['GFR'] = p['Ft'] * p['vol']  

    if {'Fp', 'vol'}.issubset(p):
        p['RBF'] = _div(p['Fp'] * p['vol'], 1-H)
        p['RPF'] = p['Fp']*p['vol']

    if {'fc', 'Eg', 'Fp'}.issubset(p):
        p['Fb_med'] = (1 - p['fc']) * (1 - p['Eg']) * p['Fp'] / (1 - H)

    if {'Fb_med', 'vol'}.issubset(p):
        p['SKMBF'] = p['Fb_med'] * p['vol']

    return p



def conc_kidney_2cf(ca, t=None, dt=1.0, Fp=None, vp=None, FF=None, Tt=None):
    """
    Two-compartment filtration model for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a 
    two-compartment arrangement: a vascular (plasma) compartment and a tubular 
    compartment. The tracer enters the plasma space via the arterial input, 
    where a fraction is filtered into the tubules (determined by the filtration fraction) 
    and the remainder exits via the venous outflow.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), representing 
        the ratio of glomerular filtration rate (GFR) to plasma flow (Fp). 
        Defaults to None.
    Tt : float, optional
        Tubular transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        tubular concentration (`Ct`).

    See Also
    --------
    conc_kidney_2pf : Two-compartment parallel filtration model.
    conc_kidney_cpf : Compact parallel filtration model.
    conc_kidney_2cfu : Two-compartment filtration model with uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_2cf(ca, t=t, Fp=0.05, vp=0.15, FF=0.2, Tt=25.0)
    array([[0.        , 0.17904154, 0.34302271, 0.37492074, 0.2604166 ],
           [0.        , 0.02794652, 0.16515698, 0.36177205, 0.46591827]])
    """
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_comp(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_comp(Ft * cp, t=t, dt=dt, T=Tt)
    return np.stack((Cp, Ct))

# def conc_kidney_3cf(ca, t=None, dt=1.0, Fp=None, vgp=None, vpp=None, FF=None, Tt=None):
#     ca = np.array(ca)
#     Ft = FF * Fp # Tubular flow
#     E = Ft / (Ft + Fp)
#     Jgp = Fp * ca # influx in glomerular plasma
    
#     # Concentration at the peritubular inlet
#     Tg = vgp / (Fp - Ft)
#     Cgp = (1 - E) * pk.conc_comp(Jgp, Tg, t=t, dt=dt)
#     cgp = Cgp / vgp 

#     # Peritubular concentration
#     Tpp = vpp / Fp
#     Jpp = (Fp - Ft) * cgp
#     Cpp = pk.conc_comp(Jpp, Tpp, t=t, dt=dt)

#     # Concentration at the tubular inlet
#     Tg = vgp / Ft
#     Cgp = E * pk.conc_comp(Jgp, Tg, t=t, dt=dt)
#     cgp = Cgp / vgp 

#     # Tubular concentration
#     Ct = pk.conc_comp(Ft * cgp, Tt, t=t, dt=dt) 

#     return np.stack((Cgp, Cpp, Ct))

def conc_kidney_2pf(ca, t=None, dt=1.0, Fp=None, vp=None, FF=None, Tt=None):
    """
    Two-plug flow filtration model for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a 
    two-site arrangement where both sites are modeled as ideal 
    plug-flow systems (pure delay systems rather than well-mixed spaces). The 
    tracer enters the vascular plasma space via the arterial input. A fraction 
    of the plasma flow is filtered into the tubular space (determined by the 
    filtration fraction), while the remaining tracer transits through the 
    vasculature to the venous exit.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), representing 
        the ratio of glomerular filtration rate (GFR) to plasma flow (Fp). 
        Defaults to None.
    Tt : float, optional
        Tubular transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        tubular concentration (`Ct`) under plug-flow conditions.

    See Also
    --------
    conc_kidney_2cf : Two-compartment filtration model using well-mixed compartments.
    conc_kidney_cpf : Compact parallel filtration model.
    conc_kidney_2pfu : Two-plug flow model with cellular uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_2pf(ca, t=t, Fp=0.05, vp=0.15, FF=0.2, Tt=25.0)
    array([[0.        , 0.1875    , 0.375     , 0.46875   , 0.40625   ],
           [0.        , 0.03125   , 0.21875   , 0.546875  , 0.77604167]])
    """
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_plug(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_plug(Ft * cp, t=t, dt=dt, T=Tt)
    return np.stack((Cp, Ct))

def conc_kidney_cpf(ca, t=None, dt=1.0, Fp=None, vp=None, FF=None, Tt=None):
    """
    Compact parallel filtration model for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a hybrid 
    two-compartment configuration: a well-mixed vascular (plasma) compartment and an 
    ideal plug-flow (pure delay) tubular compartment. The tracer enters the plasma 
    space via the arterial input. A fraction of this tracer is filtered into the renal 
    tubules (determined by the filtration fraction), while the remaining unfiltered fraction 
    exits via the venous system.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), representing 
        the ratio of glomerular filtration rate (GFR) to plasma flow (Fp). 
        Defaults to None.
    Tt : float, optional
        Tubular transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        tubular concentration (`Ct`) under mixed compartment and plug-flow conditions.

    See Also
    --------
    conc_kidney_2cf : Two-compartment filtration model (both well-mixed compartments).
    conc_kidney_2pf : Two-plug flow filtration model (both plug-flow systems).
    conc_kidney_hfu : High-flow uptake model for the kidney.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_cpf(ca, t=t, Fp=0.05, vp=0.15, FF=0.2, Tt=25.0)
    array([[0.        , 0.17904154, 0.34302271, 0.37492074, 0.2604166 ],
           [0.        , 0.02984026, 0.20386168, 0.47331263, 0.57377171]])
    """
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_comp(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_plug(Ft * cp, t=t, dt=dt, T=Tt)
    return np.stack((Cp, Ct))

def conc_kidney_2pfu(ca, t=None, dt=1.0, Fp=None, vp=None, FF=None):
    """
    Two-plug flow filtration model with uptake for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a 
    combination of an ideal plug-flow (pure delay) vascular compartment and a 
    trapping tubular or cellular compartment. The tracer enters the plasma 
    space via the arterial input. A fraction of the tracer is filtered or 
    taken up into the renal tissue (determined by the filtration fraction `FF`), 
    where it becomes permanently trapped over the duration of the measurement, 
    while the remaining fraction transits to the venous outflow.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), 
        representing the ratio of glomerular filtration (or irreversible tissue 
        uptake) rate to the plasma flow (`Fp`). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        trapped tubular/cellular concentration (`Ct`).

    See Also
    --------
    conc_kidney_2pf : Two-plug flow filtration model with a transit delay (no trap).
    conc_kidney_2cfu : Two-compartment filtration model with uptake using well-mixed spaces.
    conc_kidney_hfu : High-flow uptake model for the kidney.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_2pfu(ca, t=t, Fp=0.05, vp=0.15, FF=0.2)
    array([[0.      , 0.1875  , 0.375   , 0.46875 , 0.40625 ],
           [0.      , 0.03125 , 0.21875 , 0.640625, 1.515625]])
    """
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_plug(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_trap(Ft * cp, t=t, dt=dt)
    return np.stack((Cp, Ct))

def conc_kidney_2cfu(ca, t=None, dt=1.0, Fp=None, vp=None, FF=None):
    """
    Two-compartment filtration model with uptake for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a 
    combination of a well-mixed vascular (plasma) compartment and a trapping 
    tubular or cellular compartment. The tracer enters the plasma space via 
    the arterial input. A fraction of the tracer is filtered or taken up into 
    the renal tissue (determined by the filtration fraction `FF`), where it 
    becomes permanently trapped over the duration of the measurement, while the 
    remaining fraction transits out to the venous outflow.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), 
        representing the ratio of glomerular filtration (or irreversible tissue 
        uptake) rate to the plasma flow (`Fp`). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        trapped tubular/cellular concentration (`Ct`).

    See Also
    --------
    conc_kidney_2cf : Two-compartment filtration model with a transit delay (no trap).
    conc_kidney_2pfu : Two-plug flow filtration model with uptake.
    conc_kidney_hfu : High-flow uptake model for the kidney.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_2cfu(ca, t=t, Fp=0.05, vp=0.15, FF=0.2)
    array([[0.        , 0.17904154, 0.34302271, 0.37492074, 0.2604166 ],
           [0.        , 0.02984026, 0.20386168, 0.5628334 , 1.19817074]])
    """
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_comp(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_trap(Ft * cp, t=t, dt=dt)
    return np.stack((Cp, Ct))

def conc_kidney_hf(ca, t=None, dt=1.0, vp=None, Ft=None, Tt=None):
    """
    High-flow filtration model for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma under a 
    high-flow approximation, where blood plasma flow is assumed to be non-limiting. 
    As a result, the vascular plasma concentration (`Cp`) instantly equilibrates 
    with the incoming arterial input (`ca`). The tracer filtered or entering the 
    tubular space (`Ct`) is modeled as a well-mixed compartment with a specific 
    tubular transit time (`Tt`).

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    Ft : float, optional
        Tubular clearance rate or flow (mL/sec/cm3). Corresponds to the 
        filtration flow from the blood plasma into the tubular space. 
        Defaults to None.
    Tt : float, optional
        Tubular transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        tubular concentration (`Ct`) under high-flow assumptions.

    See Also
    --------
    conc_kidney_2cf : Two-compartment filtration model with flow-limited plasma.
    conc_kidney_hfu : High-flow model with irreversible tubular trapping (uptake).
    conc_kidney_fn : Nephron filtration model.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_hf(ca, t=t, vp=0.15, Ft=0.01, Tt=25.0)
    array([[0.15      , 0.3       , 0.45      , 0.45      , 0.3       ],
           [0.        , 0.06873075, 0.25486161, 0.47826229, 0.56373871]])
    """
    ca = np.array(ca)
    Cp = vp * ca
    Ct = pk.conc_comp(Ft * ca, t=t, dt=dt, T=Tt)
    return np.stack((Cp, Ct))

def conc_kidney_hfu(ca, t=None, dt=1.0, vp=None, Ft=None):
    """
    High-flow filtration model with uptake for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma under a 
    high-flow approximation where plasma flow is non-limiting, meaning the 
    vascular plasma concentration (`Cp`) instantly matches the incoming 
    arterial profile (`ca`). The tracer filtered or taken up into the tubular 
    or cellular space (`Ct`) is assumed to be permanently trapped over the 
    duration of the measurement.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    Ft : float, optional
        Tubular clearance or uptake rate (mL/sec/cm3). Corresponds to the 
        filtration or extraction flow of tracer that becomes permanently 
        trapped in the tissue space. Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        trapped tubular/cellular concentration (`Ct`) under high-flow assumptions.

    See Also
    --------
    conc_kidney_hf : High-flow filtration model with transit delay (no trap).
    conc_kidney_2cfu : Two-compartment filtration model with uptake (flow-limited).
    conc_kidney_2pfu : Two-plug flow filtration model with uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_kidney_hfu(ca, t=t, vp=0.15, Ft=0.01)
    array([[0.15 , 0.3  , 0.45 , 0.45 , 0.3  ],
           [0.   , 0.075, 0.325, 0.775, 1.525]])
    """
    ca = np.array(ca)
    Cp = vp * ca
    Ct = pk.conc_trap(Ft*ca, t=t, dt=dt)
    return np.stack((Cp, Ct))

def conc_kidney_fn(ca, t=None, dt=1.0, ht=None, TT=None, Fp=None, vp=None, FF=None):
    """
    Free nephron filtration model for kidney tissue concentration.

    This model tracks a filterable tracer through the renal parenchyma using a 
    combination of a well-mixed vascular compartment and a tubular/nephron 
    system characterized by a model-independent ('free') distribution of transit 
    times. Rather than assuming a single fixed transit time or a well-mixed space, 
    the tubular transit characteristics are parameterized directly via a histogram (`ht`).
    The vascular plasma concentration (`Cp`) is modeled as an ideal plug flow, 
    from which a fraction is filtered into the nephron network based on the 
    filtration fraction (`FF`).

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    TT : array_like, optional
        Time boundaries defining the bins of the transit time histogram (sec). 
        If None, it is automatically generated as a linear space spanning from 
        0 to the maximum time point, containing one more element than `ht`. 
        Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    vp : float, optional
        Plasma volume fraction. Defaults to None.
    FF : float, optional
        Filtration fraction (dimensionless fraction between 0.0 and 1.0), representing 
        the ratio of glomerular filtration rate (GFR) to plasma flow (Fp). 
        Defaults to None.
    ht : array_like, optional
        The histogram heights representing the relative distribution of transit 
        times across the nephron population. Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the vascular plasma concentration (`Cp`) and the second row is the 
        tubular/nephron concentration (`Ct`) distributed over the specified transit times.

    See Also
    --------
    conc_kidney_2pf : Two-plug flow filtration model (single discrete transit time).
    conc_kidney_cpf : Compact parallel filtration model.
    conc_free : Base model-independent building block for free distributions.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> ht = [0.1, 0.4, 0.3, 0.2] # Histogram of tubular transit times
    >>> dc.conc_kidney_fn(ca, t=t, Fp=0.05, vp=0.15, FF=0.2, ht=ht)
    array([[0.        , 0.1875    , 0.375     , 0.46875   , 0.40625   ],
           [0.        , 0.00625   , 0.021875  , 0.040625  , 0.05260417]])
    """
    ht = np.atleast_1d(ht)
    if TT is None:
        if t is None:
            tmax = dt*np.size(ca)
        else:
            tmax = np.amax(t)
        nTT = 1 + np.size(ht)
        TT = np.linspace(0, tmax, nTT)
    ca = np.array(ca)
    Ft = FF * Fp
    Tp = vp / (Fp + Ft)
    Cp = pk.conc_comp(Fp * ca, t=t, dt=dt, T=Tp)
    cp = Cp/vp
    Ct = pk.conc_free(Ft * cp, dt=dt, h=ht, TT=TT, solver='step')
    return np.stack((Cp, Ct))



def conc_kidney_cm9(ca, t=None, dt=1.0, Fp=None, Eg=None, fc=None, Tglom=None, Tv=None, Tpt=None, Tlh=None, Tdt=None, Tcd=None):
    """
    Cortico-medullary model for regional kidney tissue concentrations.

    This model separates the renal parenchyma into distinct cortical and medullary 
    regions, tracking the transit of a filterable tracer through a 9-compartment 
    network of vasculature and tubuli. It accounts for glomerular filtration, 
    venous transit, and sequential passage through the nephron segments, splitting 
    signals geographically based on the known anatomy of the renal cortex and medulla.

    Parameters
    ----------
    ca : array_like
        Inlet concentration in arterial blood (M).
    t : array_like, optional
        The time points of the inlet concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    Fp : float, optional
        Total renal plasma flow (mL/sec/cm3). Defaults to None.
    Eg : float, optional
        Glomerular extraction fraction (dimensionless fraction between 0.0 and 1.0), 
        representing the fraction of total plasma flow that is filtered into 
        the tubuli. Defaults to None.
    fc : float, optional
        Cortical fraction of the peritubular/venous vasculature (dimensionless fraction 
        between 0.0 and 1.0). Determines how the post-glomerular capillary volume 
        is split between the cortex (`fc`) and the medulla (`1-fc`). Defaults to None.
    Tglom : float, optional
        Transit time of the arterial tree and glomeruli (sec). Defaults to None.
    Tv : float, optional
        Total transit time of the peritubular capillaries and venous system (sec). 
        Defaults to None.
    Tpt : float, optional
        Transit time of the proximal tubuli located in the cortex (sec). Defaults to None.
    Tlh : float, optional
        Transit time of the loops of Henle descending into the medulla (sec). 
        Defaults to None.
    Tdt : float, optional
        Transit time of the distal tubuli located in the cortex (sec). Defaults to None.
    Tcd : float, optional
        Transit time of the collecting ducts passing through the medulla (sec). 
        Defaults to None.

    Returns
    -------
    Ccor : np.ndarray
        Cortical component tissue concentrations as a 2D array of shape (4, nt). 
        The rows correspond to:
        - Row 0: Arteries and glomeruli concentration (`Cg`)
        - Row 1: Cortical peritubular capillaries concentration (`Cv_cor`)
        - Row 2: Proximal tubuli concentration (`Cpt`)
        - Row 3: Distal tubuli concentration (`Cdt`)
    Cmed : np.ndarray
        Medullary component tissue concentrations as a 2D array of shape (3, nt). 
        The rows correspond to:
        - Row 0: Medullary peritubular capillaries/vasa recta concentration (`Cv_med`)
        - Row 1: Loops of Henle concentration (`Clh`)
        - Row 2: Collecting ducts concentration (`Ccd`)

    See Also
    --------
    conc_kidney_2cf : Simple two-compartment whole-kidney filtration model.
    conc_kidney_hf : High-flow whole-kidney filtration model.

    Examples
    --------
    >>> import dcmri as dc
    >>> import numpy as np
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> params = {
    ...     'Fp': 0.05, 'Eg': 0.2, 'fc': 0.8,
    ...     'Tglom': 2.0, 'Tv': 5.0, 'Tpt': 10.0,
    ...     'Tlh': 15.0, 'Tdt': 8.0, 'Tcd': 20.0
    ... }
    >>> Ccor, Cmed = dc.conc_kidney_cm9(ca, t=t, **params)
    >>> Ccor
    array([[0.        , 0.1550749 , 0.27983206, 0.29998885, 0.20666666],
           [0.        , 0.09127819, 0.34020659, 0.46441375, 0.3554523 ],
           [0.        , 0.03304046, 0.15607653, 0.26193579, 0.23433076],
           [0.        , 0.00101391, 0.01921791, 0.08406038, 0.16895006]])
    >>> Cmed
    array([[0.00000000e+00, 2.28195470e-02, 8.50516479e-02, 1.16103438e-01, 8.88630755e-02],
           [0.00000000e+00, 7.41397911e-03, 7.77746518e-02, 2.35015638e-01, 3.48030139e-01],
           [0.00000000e+00, 2.92013799e-04, 1.08708909e-02, 7.85477287e-02, 2.83096784e-01]])
    """
    ca = np.array(ca)
    
    # Flux out of the glomeruli and arterial tree
    Jg = pk.flux_comp(Fp*ca, t=t, dt=dt, T=Tglom)

    # Flux out of the peritubular capillaries and venous system
    Jv = pk.flux_comp((1-Eg)*Jg, t=t, dt=dt, T=Tv)

    # Flux out of the proximal tubuli
    Jpt = pk.flux_comp(Eg*Jg, t=t, dt=dt, T=Tpt)

    # Flux out of the lis of Henle
    Jlh = pk.flux_comp(Jpt, t=t, dt=dt, T=Tlh)

    # Flux out of the distal tubuli
    Jdt = pk.flux_comp(Jlh, t=t, dt=dt, T=Tdt)

    # Flux out of the collecting ducts
    Jcd = pk.flux_comp(Jdt, t=t, dt=dt, T=Tcd)

    # Build cortical concentrations
    Cg = Tglom*Jg      # arteries/glomeruli
    Cv = fc*Tv*Jv   # part of the peritubular capillaries
    Cpt = Tpt*Jpt   # proximal tubuli
    Cdt = Tdt*Jdt   # distal tubuli
    Ccor = np.stack((Cg, Cv, Cpt, Cdt))

    # Build medullary concentrations
    Cv = (1-fc)*Tv*Jv   # part of the peritubular capillaries
    Clh = Tlh*Jlh       # Lis of Henle
    Ccd = Tcd*Jcd       # collecting ducts
    Cmed = np.stack((Cv, Clh, Ccd))

    return Ccor, Cmed

