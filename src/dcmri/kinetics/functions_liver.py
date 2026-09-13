import copy
from typing import Optional

import numpy as np

import dcmri.kinetics.functions_blocks as pk
from dcmri.utils.misc import tarray


PARAMETERS = {
    ('2I-EC', None): ['ffa', 'v_e', 'F_p'],
    ('2I-EC-HF', None): ['ffa', 'v_e'],

    ('1I-EC', None): ['v_e', 'F_p'],
    ('1I-EC-HF', None): ['v_e'],

    ('2I-IC', None): ['ffa', 'v_e', 'F_p', 'E', 'T_h'],
    ('2I-IC', 'U'): ['ffa', 'v_e', 'F_p', 'Ei', 'Ef', 'T_h'],
    ('2I-IC', 'E'): ['ffa', 'v_e', 'F_p', 'E', 'Ti_h', 'Tf_h'],
    ('2I-IC', 'UE'): ['ffa', 'v_e', 'F_p', 'Ei', 'Ef', 'Ti_h', 'Tf_h'], 

    ('2I-IC-HF', None): ['ffa', 'v_e', 'k_e2h', 'T_h'],
    ('2I-IC-HF', 'U'): ['ffa', 'v_e', 'ki_e2h', 'kf_e2h', 'T_h'],
    ('2I-IC-HF', 'E'): ['ffa', 'v_e', 'k_e2h', 'Ti_h', 'Tf_h'],
    ('2I-IC-HF', 'UE'): ['ffa', 'v_e', 'ki_e2h', 'kf_e2h', 'Ti_h', 'Tf_h'],

    ('2I-IC-U', None): ['ffa', 'v_e', 'F_p', 'E'],
    ('2I-IC-U', 'U'): ['ffa', 'v_e', 'F_p', 'Ei', 'Ef'],

    ('1I-IC', None): ['v_e', 'F_p', 'E', 'T_h'],
    ('1I-IC', 'U'): ['v_e', 'F_p', 'Ei', 'Ef', 'T_h'],
    ('1I-IC', 'E'): ['v_e', 'F_p', 'E', 'Ti_h', 'Tf_h'],
    ('1I-IC', 'UE'): ['v_e', 'F_p', 'Ei', 'Ef', 'Ti_h', 'Tf_h'],

    ('1I-IC-HF', None): ['v_e', 'k_e2h', 'T_h'],
    ('1I-IC-HF', 'U'): ['v_e', 'ki_e2h', 'kf_e2h', 'T_h'],
    ('1I-IC-HF', 'E'): ['v_e', 'k_e2h', 'Ti_h', 'Tf_h'],
    ('1I-IC-HF', 'UE'): ['v_e', 'ki_e2h', 'kf_e2h', 'Ti_h', 'Tf_h'],

    # ('1I-IC-HFD', None): ['T_g', 'Dg', 'v_e', 'k_e2h', 'T_h'],
    # ('1I-IC-HFD', 'U'): ['T_g', 'Dg', 'v_e', 'ki_e2h', 'kf_e2h', 'T_h'],
    # ('1I-IC-HFD', 'E'): ['T_g', 'Dg', 'v_e', 'k_e2h', 'Ti_h', 'Tf_h'],
    # ('1I-IC-HFD', 'UE'): ['T_g', 'Dg', 'v_e', 'ki_e2h', 'kf_e2h', 'Ti_h', 'Tf_h'],
    # ('1I-IC-HFDU', None): ['T_g', 'Dg', 'v_e', 'k_e2h'],
    # ('1I-IC-HFDU', 'U'): ['T_g', 'Dg', 'v_e', 'ki_e2h', 'kf_e2h'],
}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b)


def dpars_liver(p, kinetics=None) -> dict:

    H = p['H'] if 'H' in p else 0.45
    
    p = copy.deepcopy(p)
        
    # Non-stationary options

    if {'Ei', 'Ef'}.issubset(p):
        p['E'] = np.mean([p['Ei'], p['Ef']])

    if {'ki_e2h', 'kf_e2h'}.issubset(p):
        p['k_e2h'] = np.mean([p['ki_e2h'], p['kf_e2h']])

    if {'Ti_h', 'Tf_h'}.issubset(p):
        p['T_h'] = np.mean([p['Ti_h'], p['Tf_h']])
    
    if {'Ti_h', 'Tf_h', 'v_e'}.issubset(p):
        v_h = 1 - p['v_e'] / (1 - H)
        p['ki_h2b'] = _div(v_h, p['Ti_h'])
        p['kf_h2b'] = _div(v_h, p['Tf_h'])

    # Dual-inlet models
    if {'F_p', 'ffa'}.issubset(p):
        p['F_ar'] = p['F_p'] * p['ffa']
        p['F_pv'] = p['F_p'] * (1 - p['ffa'])

    # Kinetic models
    
    if kinetics in ['1I-EC', '2I-EC']:
        p['T_e'] = _div(p['v_e'], p['F_p'])

    if kinetics in ['1I-IC', '2I-IC']:
        p['Ktrans'] = p['E'] * p['F_p']
        p['k_e2h'] = _div(p['F_p'] * p['E'], 1 - p['E'])
        p['T_e'] = _div(p['v_e'], p['F_p'] + p['k_e2h'])
        p['K_e2h'] = _div(p['k_e2h'], p['v_e'])
        p['v_h'] = 1 - p['v_e'] / (1 - H)
        p['k_h2b'] = _div(p['v_h'], p['T_h']) 
        p['K_h2b'] = _div(1, p['T_h'])
        
    if kinetics in ['1I-IC-HF', '1I-IC-D', '2I-IC-HF']:
        p['v_h'] = 1 - p['v_e'] / (1 - H)
        p['k_h2b'] = _div(p['v_h'], p['T_h']) 
        p['K_h2b'] = _div(1, p['T_h'])

    if kinetics in ['1I-IC-HF', '2I-IC-HF']: #, '1I-IC-HFD']:
        p['K_e2h'] = _div(p['k_e2h'], p['v_e'])
        p['v_h'] = 1 - p['v_e'] / (1 - H)
        p['k_h2b'] = _div(p['v_h'], p['T_h'])
        p['K_h2b'] = _div(1, p['T_h'])

    # if kinetics in ['1I-IC-HFDU']:
    #     p['K_e2h'] = _div(p['k_e2h'], p['v_e'])
    #     p['v_h'] = 1 - p['v_e'] / (1 - H)
        
    if kinetics == '2I-IC-U':
        p['v_h'] = 1 - p['v_e'] / (1 - H)
        p['Ktrans'] = p['E'] * p['F_p']
        p['k_e2h'] = _div(p['F_p'] * p['E'], 1 - p['E'])
        p['K_e2h'] = _div(p['k_e2h'], p['v_e'])
        p['T_e'] = _div(p['v_e'], p['F_p'] + p['k_e2h'])

    if kinetics in ['2I-EC', '2I-IC', '2I-IC-U']:
        p['F_la'] = p['ffa'] * p['F_p']
        p['F_pv'] = (1 - p['ffa']) * p['F_p']

    if {'k_e2h', 'vol_l'}.issubset(p):
        p['CL'] = p['k_e2h'] * p['vol_l']

    return p
    





# --- Liver kinetic models ---



def conc_liver_2i_ec(ci, t=None, dt=1.0, ffa=None, 
                     v_e=None, F_p=None):
    """
    Dual-inlet extracellular agent liver concentration.

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt) array.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ec((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01)
    array([[0.        , 0.04373231, 0.17143928, 0.34444778, 0.41242799],
           [0.        , 0.        , 0.        , 0.        , 0.        ]])
    """
    ca, cv = ci
    T_e = v_e / F_p
    return _conc_liver(
        ca, cv=cv, v_e_app=v_e, ffa=ffa, T_e=T_e, t=t, dt=dt, 
    )

def conc_liver_2i_ec_hf(ci, t=None, dt=1.0, ffa=None, v_e=None):
    """
    Dual-inlet extracellular agent liver concentration (High-Flow approximation).

    This model assumes a high-flow limit where 
    the tracer instantly equilibrates within the extracellular space.

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        concentration (zeros for an extracellular-only agent).

    See Also
    --------
    conc_liver_2i_ec : Dual-inlet extracellular agent liver concentration with finite flow.
    conc_liver_2i_ic_hf : Dual-inlet intracellular agent liver concentration (High-Flow).

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ec_hf((ca, cv), t=t, ffa=0.3, v_e=0.2)
    array([[0.13 , 0.26 , 0.46 , 0.53 , 0.372],
           [0.   , 0.   , 0.   , 0.   , 0.   ]])
    """
    ca, cv = ci
    return _conc_liver(ca, cv=cv, v_e_app=v_e, ffa=ffa, t=t, dt=dt)

def conc_liver_1i_ec(ca, t=None, dt=1.0, v_e=None, F_p=None):
    """
    Single-inlet extracellular agent liver concentration.

    This model handles a single arterial input and models the portal venous 
    input implicitly using a gut transit time (`Tg`) delay.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        concentration (zeros for an extracellular-only agent).

    See Also
    --------
    conc_liver_2i_ec : Dual-inlet extracellular agent liver concentration.
    conc_liver_1i_ic : Single-inlet intracellular agent liver concentration.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ec(ca, t=t, v_e=0.2, F_p=0.01)
    array([[0.        , 0.06728047, 0.24080767, 0.43032956, 0.46572405],
           [0.        , 0.        , 0.        , 0.        , 0.        ]])
    """
    T_e = v_e / F_p
    return _conc_liver(ca, v_e_app=v_e, T_e=T_e, t=t, dt=dt)

def conc_liver_1i_ec_hf(ca, t=None, dt=1.0, v_e=None):
    """
    Single-inlet extracellular agent liver concentration (High-Flow approximation).

    This model handles a single arterial input and models the portal venous 
    input implicitly using a gut transit time (`Tg`) delay. It assumes a 
    high-flow limit where tracer equilibration across the extracellular space 
    is instantaneous.

    Parameters
    ----------
    ca : array_like
        Concentration in arterial blood (M).
    t : array_like, optional
        The time points of the arterial concentration (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        concentration (zeros for an extracellular-only agent).

    See Also
    --------
    conc_liver_1i_ec : Single-inlet extracellular agent liver concentration with finite flow.
    conc_liver_2i_ec_hf : Dual-inlet extracellular agent liver concentration (High-Flow).

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ec_hf(ca, t=t, v_e=0.2)
    array([[0.2, 0.4, 0.6, 0.6, 0.4],
           [0. , 0. , 0. , 0. , 0. ]])
    """
    return _conc_liver(ca, v_e_app=v_e, t=t, dt=dt)

# def conc_liver_1i_ec_d(ca, t=None, dt=1.0, v_e=None, 
#                        T_e=None, De=None):
#     return _conc_liver(ca, v_e_app=v_e, T_e=T_e, De=De, t=t, dt=dt)



def conc_liver_2i_ic(ci, t=None, dt=1.0, ffa=None, v_e=None, F_p=None, E=None, T_h=None):
    """
    Dual-inlet intracellular agent liver concentration.

    This model tracks a hepatocyte-specific tracer that undergoes extraction 
    from the extracellular space into the liver cells (hepatocytes) before 
    being excreted into the biliary system.

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration.

    See Also
    --------
    conc_liver_2i_ec : Dual-inlet extracellular agent liver concentration.
    conc_liver_2i_ic_hf : Dual-inlet intracellular agent liver concentration (High-Flow).

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01, E=0.15, T_h=30.0)
    array([[0.        , 0.04292407, 0.16359685, 0.31686986, 0.35638917],
           [0.        , 0.00089637, 0.00864157, 0.03092497, 0.06824562]])
    """
    ca, cv = ci
    k_e2h = F_p * E / (1 - E)
    T_e = v_e / (F_p + k_e2h)
    v_e_app = v_e * (1 - E)
    Ktrans = F_p * E
    return _conc_liver(
        ca, v_e_app=v_e_app, cv=cv, ffa=ffa, Ktrans=Ktrans, 
        T_h=T_h, T_e=T_e, t=t, dt=dt, 
    )

def conc_liver_2i_ic_nse(ci, t=None, dt=1.0, ffa=None, 
                         F_p=None, v_e=None, E=None, 
                         Ti_h=None, Tf_h=None):
    """
    Dual-inlet intracellular agent liver concentration with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer with a hepatocyte transit 
    time (`Th`) that varies dynamically over time between an initial and 
    final value.

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time (sec) at the start of the time series. 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time (sec) at the end of the time series. 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary excretion conditions.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration with stationary excretion.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_nse((ca, cv), t=t, ffa=0.3, F_p=0.01, v_e=0.2, E=0.15, Ti_h=45.0, Tf_h=15.0)
    array([[0.        , 0.04292407, 0.16359685, 0.31686986, 0.35638917],
           [0.        , 0.00094685, 0.00982136, 0.0372519 , 0.05770381]])
    """
    ca, cv = ci
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_2i_ic(ci, t=t, dt=dt, F_p=F_p, v_e=v_e, 
                            E=E, ffa=ffa, T_h=T_h)

def conc_liver_2i_ic_nsu(ci, t=None, dt=1.0, ffa=None, 
                         v_e=None, F_p=None, Ei=None, Ef=None, T_h=None):
    """
    Dual-inlet intracellular agent liver concentration with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer with a hepatocyte extraction 
    fraction (`E`) that varies dynamically over time between an initial and 
    final value (e.g., due to physiological or metabolic changes during the scan).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    Ei : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    Ef : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake conditions.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration with stationary uptake.
    conc_liver_2i_ic_nse : Dual-inlet intracellular agent liver concentration with non-stationary excretion.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_nsu((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01, Ei=0.30, Ef=0.05, T_h=30.0)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00223418, 0.01908304, 0.05398519, 0.06316002]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [Ei, Ef])
    return conc_liver_2i_ic(ci, t=t, dt=dt, ffa=ffa, 
                            v_e=v_e, F_p=F_p, E=E, T_h=T_h)

def conc_liver_2i_ic_nsue(ci, t=None, dt=1.0, ffa=None, 
                          v_e=None, F_p=None, Ei=None, Ef=None, 
                          Ti_h=None, Tf_h=None):
    """
    Dual-inlet intracellular agent liver concentration with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer where both the hepatocyte extraction 
    fraction (`E`) and the hepatocyte transit time (`Th`) vary dynamically over time 
    between initial and final values (e.g., due to acute physiological, metabolic, or 
    transporter-mediated changes during the scan).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    Ei : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    Ef : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake and efflux conditions.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration with stationary parameters.
    conc_liver_2i_ic_nsu : Dual-inlet intracellular agent liver concentration with non-stationary uptake only.
    conc_liver_2i_ic_nse : Dual-inlet intracellular agent liver concentration with non-stationary excretion only.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_nsue((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01, Ei=0.30, Ef=0.05, Ti_h=30.0, Tf_h=15.0)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00236001, 0.02147516, 0.06152976, 0.03394174]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [Ei, Ef])
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_2i_ic(ci, t=t, dt=dt, F_p=F_p, v_e=v_e, 
                            E=E, ffa=ffa, T_h=T_h)



def conc_liver_2i_ic_hf(ci, t=None, dt=1.0, ffa=None, 
                        v_e=None, k_e2h=None, T_h=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit).

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. Tissue uptake is instead 
    characterized directly by the hepatocyte clearance/uptake rate constant (`k_e2h`) 
    and hepatocyte transit time (`Th`).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    k_e2h : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration using flow and extraction parameters.
    conc_liver_2i_ic_nsu : Dual-inlet model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_hf((ca, cv), t=t, ffa=0.3, v_e=0.2, k_e2h=0.003, T_h=30.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.0135959 , 0.05637118, 0.12235044, 0.16961473]])
    """
    ca, cv = ci
    return _conc_liver(
        ca, v_e_app=v_e, cv=cv, ffa=ffa, 
        Ktrans=k_e2h, T_h=T_h, t=t, dt=dt, 
    )

def conc_liver_2i_ic_hf_nse(ci, t=None, dt=1.0, ffa=None, 
                            v_e=None, k_e2h=None, Ti_h=None, Tf_h=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. The hepatocyte uptake rate constant 
    (`k_e2h`) remains stationary, while the hepatocyte transit time (`Th`) varies 
    dynamically over time between an initial and final value (e.g., due to changes 
    in biliary excretion or transporter activity during the scan).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    k_e2h : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary excretion conditions.

    See Also
    --------
    conc_liver_2i_ic_hf : Dual-inlet high-flow model with stationary excretion.
    conc_liver_2i_ic_hf_nsu : Dual-inlet high-flow model with non-stationary uptake.
    conc_liver_2i_ic_nse : Dual-inlet flow-limited model with non-stationary excretion.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_hf_nse((ca, cv), t=t, ffa=0.3, v_e=0.2, k_e2h=0.003, Ti_h=30.0, Tf_h=15.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.014625  , 0.06330682, 0.13572378, 0.10896284]])
    """
    ca, _ = ci
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, ffa=ffa, 
                               v_e=v_e, k_e2h=k_e2h, T_h=T_h)

def conc_liver_2i_ic_hf_nsu(ci, t=None, dt=1.0, ffa=None, 
                            v_e=None, ki_e2h=None, kf_e2h=None, T_h=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. The hepatocyte transit time 
    (`Th`) remains stationary, while the hepatocyte uptake rate constant (`k_e2h`) 
    varies dynamically over time between an initial and final value (e.g., due to 
    acute metabolic changes or competitive transporter inhibition during the scan).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    ki_e2h : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    kf_e2h : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake conditions.

    See Also
    --------
    conc_liver_2i_ic_hf : Dual-inlet high-flow model with stationary uptake.
    conc_liver_2i_ic_hf_nse : Dual-inlet high-flow model with non-stationary excretion.
    conc_liver_2i_ic_nsu : Dual-inlet flow-limited model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_hf_nsu((ca, cv), t=t, ffa=0.3, v_e=0.2, ki_e2h=0.003, kf_e2h=0.0005, T_h=30.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.01295492, 0.04837293, 0.08854618, 0.0796007 ]])
    """
    ca, _ = ci
    k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, ffa=ffa, 
                               v_e=v_e, k_e2h=k_e2h, T_h=T_h)

def conc_liver_2i_ic_hf_nsue(ci, t=None, dt=1.0, ffa=None, 
                             v_e=None, ki_e2h=None, kf_e2h=None, 
                             Ti_h=None, Tf_h=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. Both the hepatocyte uptake rate 
    constant (`k_e2h`) and the hepatocyte transit time (`Th`) vary dynamically over 
    time between their respectiv_e initial and final values (e.g., due to complex, 
    concurrent changes in both sinusoidal influx and biliary excretion transporters 
    during the scan).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    ki_e2h : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    kf_e2h : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake and efflux conditions.

    See Also
    --------
    conc_liver_2i_ic_hf : Dual-inlet high-flow model with stationary parameters.
    conc_liver_2i_ic_hf_nsu : Dual-inlet high-flow model with non-stationary uptake only.
    conc_liver_2i_ic_hf_nse : Dual-inlet high-flow model with non-stationary excretion only.
    conc_liver_2i_ic_nsue : Dual-inlet flow-limited model with non-stationary uptake and efflux.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_hf_nsue((ca, cv), t=t, ffa=0.3, v_e=0.2, ki_e2h=0.003, kf_e2h=0.0005, Ti_h=30.0, Tf_h=15.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.01394792, 0.05433428, 0.0966478 , 0.03696622]])
    """
    ca, cv = ci
    k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, ffa=ffa, 
                               v_e=v_e, k_e2h=k_e2h, T_h=T_h)



def conc_liver_2i_ic_u(ci, t=None, dt=1.0, ffa=None, 
                       v_e=None, F_p=None, E=None):
    """
    Dual-inlet intracellular agent liver concentration (Uptake-only model).

    This model tracks a hepatocyte-specific tracer under stationary conditions 
    where the agent is taken up into the hepatocytes but undergoes no biliary 
    excretion or efflux back into the extracellular space during the scan period.

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under uptake-only conditions.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration with both uptake and excretion.
    conc_liver_2i_ic_nsu : Dual-inlet model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_u((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01, E=0.30)
    array([[0.        , 0.04180636, 0.15331954, 0.28315496, 0.29565219],
           [0.        , 0.00223963, 0.02314597, 0.09329366, 0.27933882]])
    """
    ca, cv = ci
    k_e2h = F_p * E / (1 - E)
    T_e = v_e / (F_p + k_e2h)
    v_e_app = v_e * (1 - E)
    Ktrans = F_p * E
    return _conc_liver(
        ca, v_e_app=v_e_app, cv=cv, ffa=ffa, Ktrans=Ktrans, T_e=T_e,
        t=t, dt=dt, 
    )

def conc_liver_2i_ic_u_nsu(ci, t=None, dt=1.0, ffa=None, 
                           F_p=None, v_e=None, Ei=None, Ef=None):
    """
    Dual-inlet intracellular agent liver concentration (Uptake-only model) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under an uptake-only condition 
    (no biliary excretion or efflux back into blood occurs during the scan period). 
    The hepatocyte extraction fraction (`E`) varies dynamically over time between an 
    initial and final value (e.g., due to acute metabolic shifts or competitiv_e transporter 
    inhibition during the time series).

    Parameters
    ----------
    ci : tuple or list of array_like
        Inlet concentrations `(ca, cv)`, where `ca` is the concentration 
        in arterial blood (M) and `cv` is the concentration in portal venous blood (M).
    t : array_like, optional
        The time points of the inlet concentrations (sec). If None, the time 
        points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing between time points for uniformly spaced data (sec). This 
        parameter is ignored if `t` is explicitly provided. Defaults to 1.0.
    ffa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    Ei : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    Ef : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake-only conditions.

    See Also
    --------
    conc_liver_2i_ic_u : Dual-inlet uptake-only model with stationary parameters.
    conc_liver_2i_ic_nsu : Dual-inlet flow-limited model with non-stationary uptake and active efflux.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> cv = [0.5, 1, 2, 2.5, 1.8]
    >>> dc.conc_liver_2i_ic_u_nsu((ca, cv), t=t, ffa=0.3, v_e=0.2, F_p=0.01, Ei=0.30, Ef=0.05)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00236001, 0.02233335, 0.07560343, 0.15157707]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [Ei, Ef])
    return conc_liver_2i_ic_u(ci, t=t, dt=dt, F_p=F_p, v_e=v_e, 
                              E=E, ffa=ffa)


def conc_liver_1i_ic(ca, t=None, dt=1.0, 
                     v_e=None, F_p=None, E=None, T_h=None):
    """
    Single-inlet intracellular agent liver concentration.

    This model tracks a hepatocyte-specific tracer under stationary conditions 
    using a single blood inlet (arterial input only, with an optional gut transit 
    delay parameter to approximate portal venous delivery).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration.

    See Also
    --------
    conc_liver_2i_ic : Dual-inlet intracellular agent liver concentration model.
    conc_liver_1i_ic_nsu : Single-inlet model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic(ca, t=t, v_e=0.2, F_p=0.01, E=0.30, T_h=30.0)
    array([[0.        , 0.06431748, 0.21436459, 0.34956599, 0.32940522],
           [0.        , 0.00326188, 0.02848746, 0.09001914, 0.17039905]])
    """
    v_e_app = v_e * (1 - E)
    Ktrans = F_p * E
    T_e = v_e_app / F_p
    return _conc_liver(
        ca, v_e_app=v_e_app, Ktrans=Ktrans, T_h=T_h, T_e=T_e, 
        t=t, dt=dt
    )

def conc_liver_1i_ic_nsu(ca, t=None, dt=1.0, 
                         v_e=None, F_p=None, Ei=None, Ef=None, 
                         T_h=None):
    """
    Single-inlet intracellular agent liver concentration with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte extraction fraction (`E`) 
    varies dynamically over time between an initial and final value (e.g., due 
    to acute physiological or metabolic changes during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    Ei : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    Ef : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake conditions.

    See Also
    --------
    conc_liver_1i_ic : Single-inlet intracellular agent liver concentration with stationary parameters.
    conc_liver_2i_ic_nsu : Dual-inlet model with non-stationary uptake.
    conc_liver_1i_ic_nse : Single-inlet model with non-stationary excretion.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_nsu(ca, t=t, v_e=0.2, F_p=0.01, Ei=0.30, Ef=0.05, T_h=30.0)
    array([[0.        , 0.075     , 0.2744382 , 0.46512652, 0.42386628],
           [0.        , 0.0034372 , 0.0274768 , 0.07132961, 0.07765741]])
    """
    E = _interp_params(ca, t, dt, [Ei, Ef])
    return conc_liver_1i_ic(ca, t=t, dt=dt, F_p=F_p, 
                            v_e=v_e, E=E, T_h=T_h)

def conc_liver_1i_ic_nse(ca, t=None, dt=1.0, 
                         v_e=None, F_p=None, E=None, Ti_h=None, Tf_h=None):
    """
    Single-inlet intracellular agent liver concentration with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte extraction fraction (`E`) 
    remains stationary, while the hepatocyte transit time (`Th`) varies 
    dynamically over time between an initial and final value (e.g., due to changes 
    in biliary excretion or transporter activity during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary excretion conditions.

    See Also
    --------
    conc_liver_1i_ic : Single-inlet intracellular agent liver concentration with stationary parameters.
    conc_liver_2i_ic_nse : Dual-inlet model with non-stationary excretion.
    conc_liver_1i_ic_nsu : Single-inlet model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_nse(ca, t=t, v_e=0.2, F_p=0.01, E=0.30, Ti_h=30.0, Tf_h=15.0)
    array([[0.        , 0.06431748, 0.21436459, 0.34956599, 0.32940522],
           [0.        , 0.00344558, 0.03205143, 0.10295917, 0.12292478]])
    """
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_1i_ic(ca, t=t, dt=dt, F_p=F_p, v_e=v_e, E=E, T_h=T_h)

def conc_liver_1i_ic_nsue(ca, t=None, dt=1.0, 
                          v_e=None, F_p=None, Ei=None, Ef=None, 
                          Ti_h=None, Tf_h=None):
    """
    Single-inlet intracellular agent liver concentration with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). Both the hepatocyte extraction fraction 
    (`E`) and the hepatocyte transit time (`Th`) vary dynamically over time 
    between their respective initial and final values (e.g., due to complex, 
    concurrent changes in both sinusoidal influx and biliary excretion transporters 
    during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    F_p : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    Ei : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    Ef : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake and efflux conditions.

    See Also
    --------
    conc_liver_1i_ic : Single-inlet intracellular agent liver concentration with stationary parameters.
    conc_liver_1i_ic_nsu : Single-inlet model with non-stationary uptake only.
    conc_liver_1i_ic_nse : Single-inlet model with non-stationary excretion only.
    conc_liver_2i_ic_nsue : Dual-inlet model with non-stationary uptake and efflux.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_nsue(ca, t=t, v_e=0.2, F_p=0.01, Ei=0.30, Ef=0.05, Ti_h=30.0, Tf_h=15.0)
    array([[0.        , 0.075     , 0.2744382 , 0.46512652, 0.42386628],
           [0.        , 0.00363078, 0.03094225, 0.08095485, 0.04013242]])
    """
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    E = _interp_params(ca, t, dt, [Ei, Ef])
    return conc_liver_1i_ic(ca, t=t, dt=dt, F_p=F_p, v_e=v_e, E=E, T_h=T_h)

def conc_liver_1i_ic_hf(ca, t=None, dt=1.0, v_e=None, k_e2h=None, T_h=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit).

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). Tissue uptake is characterized directly 
    by the hepatocyte uptake rate constant (`k_e2h`) and hepatocyte transit time (`Th`).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    k_e2h : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration.

    See Also
    --------
    conc_liver_1i_ic : Single-inlet intracellular agent liver concentration using flow and extraction parameters.
    conc_liver_2i_ic_hf : Dual-inlet high-flow model.
    conc_liver_1i_ic_hf_nsu : Single-inlet high-flow model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_hf(ca, t=t, v_e=0.2, k_e2h=0.003, T_h=30.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.02091678, 0.07947534, 0.15444095, 0.19437905]])
    """
    return _conc_liver(ca, v_e_app=v_e, Ktrans=k_e2h, T_h=T_h, t=t, dt=dt)

def conc_liver_1i_ic_hf_nsu(ca, t=None, dt=1.0, v_e=None, ki_e2h=None, kf_e2h=None, T_h=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte transit time (`Th`) remains 
    stationary, while the hepatocyte uptake rate constant (`k_e2h`) varies dynamically 
    over time between an initial and final value (e.g., due to acute metabolic changes 
    or competitive transporter inhibition during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    ki_e2h : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    kf_e2h : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    T_h : float, optional
        Hepatocyte transit time (sec). Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake conditions.

    See Also
    --------
    conc_liver_1i_ic_hf : Single-inlet high-flow model with stationary uptake.
    conc_liver_1i_ic_hf_nse : Single-inlet high-flow model with non-stationary excretion.
    conc_liver_2i_ic_hf_nsu : Dual-inlet high-flow model with non-stationary uptake.
    conc_liver_1i_ic_nsu : Single-inlet flow-limited model with non-stationary uptake.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_hf_nsu(ca, t=t, v_e=0.2, ki_e2h=0.003, kf_e2h=0.0005, T_h=30.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.01993065, 0.06868066, 0.1137763 , 0.09451032]])
    """
    k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

def conc_liver_1i_ic_hf_nse(ca, t=None, dt=1.0,
                            v_e=None, k_e2h=None, Ti_h=None, Tf_h=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte uptake rate constant 
    (`k_e2h`) remains stationary, while the hepatocyte transit time (`Th`) varies 
    dynamically over time between an initial and final value (e.g., due to changes 
    in biliary excretion or transporter activity during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    k_e2h : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary excretion conditions.

    See Also
    --------
    conc_liver_1i_ic_hf : Single-inlet high-flow model with stationary excretion.
    conc_liver_1i_ic_hf_nsu : Single-inlet high-flow model with non-stationary uptake.
    conc_liver_2i_ic_hf_nse : Dual-inlet high-flow model with non-stationary excretion.
    conc_liver_1i_ic_nse : Single-inlet flow-limited model with non-stationary excretion.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_hf_nse(ca, t=t, v_e=0.2, k_e2h=0.003, Ti_h=30.0, Tf_h=15.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.0225    , 0.08931818, 0.16935315, 0.12013191]])
    """
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

def conc_liver_1i_ic_hf_nsue(ca, t=None, dt=1.0, 
                             v_e=None, ki_e2h=None, kf_e2h=None, Ti_h=None, Tf_h=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). Both the hepatocyte uptake rate constant 
    (`k_e2h`) and the hepatocyte transit time (`Th`) vary dynamically over time 
    between their respectiv_e initial and final values (e.g., due to complex, 
    concurrent changes in both sinusoidal influx and biliary excretion transporters 
    during the scan).

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
    v_e : float, optional
        Extracellular volume fraction. Defaults to None.
    ki_e2h : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    kf_e2h : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Ti_h : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Tf_h : float, optional
        Final hepatocyte transit time at the end of the time series (sec). 
        Defaults to None.

    Returns
    -------
    np.ndarray
        Tissue concentration as a 2D array (2, nt), where the first row is 
        the extracellular concentration and the second row is the intracellular 
        (hepatocyte) concentration under non-stationary uptake and efflux conditions.

    See Also
    --------
    conc_liver_1i_ic_hf : Single-inlet high-flow model with stationary parameters.
    conc_liver_1i_ic_hf_nsu : Single-inlet high-flow model with non-stationary uptake only.
    conc_liver_1i_ic_hf_nse : Single-inlet high-flow model with non-stationary excretion only.
    conc_liver_2i_ic_hf_nsue : Dual-inlet high-flow model with non-stationary uptake and efflux.

    Examples
    --------
    >>> import dcmri as dc
    >>> t = [0, 5, 15, 30, 60]
    >>> ca = [1, 2, 3, 3, 2]
    >>> dc.conc_liver_1i_ic_hf_nsue(ca, t=t, v_e=0.2, ki_e2h=0.003, kf_e2h=0.0005, Ti_h=30.0, Tf_h=15.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.02145833, 0.07719697, 0.12250364, 0.0416289 ]])
    """
    k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
    T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

# def conc_liver_1i_ic_hfd(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, k_e2h=None, T_h=None):
#     return _conc_liver( # approx 1 - E = 1
#         ca, v_e=v_e, Ktrans=k_e2h, T_h=T_h,
#         Tg=Tg, Dg=Dg, t=t, dt=dt, 
#     )

# def conc_liver_1i_ic_hfd_nsu(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, ki_e2h=None, kf_e2h=None, T_h=None):
#     k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

# def conc_liver_1i_ic_hfd_nse(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, k_e2h=None, Ti_h=None, Tf_h=None):
#     T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

# def conc_liver_1i_ic_hfd_nsue(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, ki_e2h=None, kf_e2h=None, Ti_h=None, Tf_h=None):
#     k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
#     T_h = _interp_params(ca, t, dt, [Ti_h, Tf_h])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, v_e=v_e, k_e2h=k_e2h, T_h=T_h)

# def conc_liver_1i_ic_hfdu(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, k_e2h=None):
#     return _conc_liver(
#         ca, v_e=v_e, Ktrans=k_e2h, Tg=Tg, Dg=Dg, t=t, dt=dt, 
#     )

# def conc_liver_1i_ic_hfdu_nsu(ca, t=None, dt=1.0, Tg=None, Dg=None, v_e=None, ki_e2h=None, kf_e2h=None):
#     k_e2h = _interp_params(ca, t, dt, [ki_e2h, kf_e2h])
#     return conc_liver_1i_ic_hfdu(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, v_e=v_e, k_e2h=k_e2h)



def _interp_params(ca: np.ndarray, t: Optional[np.ndarray], dt: float, p):
    ti = tarray(np.size(ca), t=t, dt=dt)
    return p[0] + (p[1] - p[0]) * ti / ti.max()



def _conc_liver(
    ca: np.ndarray,
    cv: np.ndarray = None,
    ffa: float = None,
    v_e_app: float = None, # (1 - E) ve
    T_e: float = None,   
    Ktrans: float = None, # (1 - E) kep
    T_h: float = None,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    
    # v_e ce' = F_p ca - k_e2h ce - F_p ce
    # v_h ch' = k_e2h ce - k_h2b ch
    # ce = F_p/v_e ca * exp( -t (k_e2h + F_p) / ve)
    # Ce = F_p ca * exp(-t/T_e)
    # ch = k_e2h/v_h ce * exp(-t k_h2b/v_h) 
    # Ch = k_e2h/v_e Ce * exp(-t/T_h) 
    #    = E/T_e Ce * exp(-t/T_h) 

    # T_e = p['v_e'] * (1 - p['E']) / p['F_p']
    # Ce = pk.conc_comp(ca * p['F_p'], T_e, t=t, dt=dt)
    # Ch = pk.conc_comp(Ce * p['E']/T_e, T_e, t=t, dt=dt)
    # return np.stack((Ce, Ch))

    ca = np.array(ca)

    # If a portal venous concentration is provided, use it
    if cv is not None:
        ca = ffa * np.array(ca) + (1 - ffa) * np.array(cv)

    # Propagate through the extracellular space
    if T_e is None:
        ce = ca
    elif np.isscalar(T_e):
        ce = pk.flux_comp(ca, t=t, dt=dt, T=T_e)
    else:
        ce = pk.flux_nscomp(ca, t=t, dt=dt, T=T_e)

    # else:
    #     ec_model, ec_pars = 'pfcomp', (T_e, De,)

    # Tissue concentration in the extracellular space
    Ce = v_e_app * ce

    # Tissue concentration in the hepatocytes
    if Ktrans is None:
        Ch = np.zeros(len(ce))
    elif T_h is None:
        Ch = pk.conc_trap(Ktrans * ce, t=t, dt=dt)
    elif np.isscalar(T_h):
        Ch = pk.conc_comp(Ktrans * ce, t=t, dt=dt, T=T_h)
    else:
        Ch = pk.conc_nscomp(Ktrans * ce, t=t, dt=dt, T=T_h)

    return np.stack((Ce, Ch))
    
