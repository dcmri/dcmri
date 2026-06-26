import copy
from typing import Optional

import numpy as np

import dcmri.kinetics.functions_blocks as pk
from dcmri.utils.misc import tarray


PARAMETERS = {
    ('2I-EC', None): ['fa', 've', 'Fp'],
    ('2I-EC-HF', None): ['fa', 've'],

    ('1I-EC', None): ['ve', 'Fp'],
    ('1I-EC-HF', None): ['ve'],
    # ('1I-EC-D', None): ['ve', 'Te', 'De'],

    ('2I-IC', None): ['fa', 've', 'Fp', 'E', 'Th'],
    ('2I-IC', 'U'): ['fa', 've', 'Fp', 'E_i', 'E_f', 'Th'],
    ('2I-IC', 'E'): ['fa', 've', 'Fp', 'E', 'Th_i', 'Th_f'],
    ('2I-IC', 'UE'): ['fa', 've', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f'], 

    ('2I-IC-HF', None): ['fa', 've', 'khe', 'Th'],
    ('2I-IC-HF', 'U'): ['fa', 've', 'khe_i', 'khe_f', 'Th'],
    ('2I-IC-HF', 'E'): ['fa', 've', 'khe', 'Th_i', 'Th_f'],
    ('2I-IC-HF', 'UE'): ['fa', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],

    ('2I-IC-U', None): ['fa', 've', 'Fp', 'E'],
    ('2I-IC-U', 'U'): ['fa', 've', 'Fp', 'E_i', 'E_f'],

    ('1I-IC', None): ['ve', 'Fp', 'E', 'Th'],
    ('1I-IC', 'U'): ['ve', 'Fp', 'E_i', 'E_f', 'Th'],
    ('1I-IC', 'E'): ['ve', 'Fp', 'E', 'Th_i', 'Th_f'],
    ('1I-IC', 'UE'): ['ve', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f'],

    ('1I-IC-HF', None): ['ve', 'khe', 'Th'],
    ('1I-IC-HF', 'U'): ['ve', 'khe_i', 'khe_f', 'Th'],
    ('1I-IC-HF', 'E'): ['ve', 'khe', 'Th_i', 'Th_f'],
    ('1I-IC-HF', 'UE'): ['ve', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],

    # ('1I-IC-HFD', None): ['Tg', 'Dg', 've', 'khe', 'Th'],
    # ('1I-IC-HFD', 'U'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th'],
    # ('1I-IC-HFD', 'E'): ['Tg', 'Dg', 've', 'khe', 'Th_i', 'Th_f'],
    # ('1I-IC-HFD', 'UE'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],
    # ('1I-IC-HFDU', None): ['Tg', 'Dg', 've', 'khe'],
    # ('1I-IC-HFDU', 'U'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f'],
}


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b)


def dpars_liver(p, kinetics=None) -> dict:

    H = p['H'] if 'H' in p else 0.45
    
    p = copy.deepcopy(p)
        
    # Non-stationary options

    if {'E_i', 'E_f'}.issubset(p):
        p['E'] = np.mean([p['E_i'], p['E_f']])

    if {'khe_i', 'khe_f'}.issubset(p):
        p['khe'] = np.mean([p['khe_i'], p['khe_f']])

    if {'Th_i', 'Th_f'}.issubset(p):
        p['Th'] = np.mean([p['Th_i'], p['Th_f']])
    
    if {'Th_i', 'Th_f', 've'}.issubset(p):
        vh = 1 - p['ve'] / (1 - H)
        p['kbh_i'] = _div(vh, p['Th_i'])
        p['kbh_f'] = _div(vh, p['Th_f'])

    # Dual-inlet models
    if {'Fp', 'fa'}.issubset(p):
        p['Fa'] = p['Fp'] * p['fa']
        p['Fv'] = p['Fp'] * (1 - p['fa'])

    # Kinetic models
    
    if kinetics in ['1I-EC', '2I-EC']:
        p['Te'] = _div(p['ve'], p['Fp'])

    if kinetics in ['1I-IC', '2I-IC']:
        p['Ktrans'] = p['E'] * p['Fp']
        p['khe'] = _div(p['Fp'] * p['E'], 1 - p['E'])
        p['Te'] = _div(p['ve'], p['Fp'] + p['khe'])
        p['Khe'] = _div(p['khe'], p['ve'])
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['kbh'] = _div(p['vh'], p['Th']) 
        p['Kbh'] = _div(1, p['Th'])
        
    if kinetics in ['1I-IC-HF', '1I-IC-D', '2I-IC-HF']:
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['kbh'] = _div(p['vh'], p['Th']) 
        p['Kbh'] = _div(1, p['Th'])

    if kinetics in ['1I-IC-HF', '2I-IC-HF']: #, '1I-IC-HFD']:
        p['Khe'] = _div(p['khe'], p['ve'])
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['kbh'] = _div(p['vh'], p['Th'])
        p['Kbh'] = _div(1, p['Th'])

    # if kinetics in ['1I-IC-HFDU']:
    #     p['Khe'] = _div(p['khe'], p['ve'])
    #     p['vh'] = 1 - p['ve'] / (1 - H)
        
    if kinetics == '2I-IC-U':
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['Ktrans'] = p['E'] * p['Fp']
        p['khe'] = _div(p['Fp'] * p['E'], 1 - p['E'])
        p['Khe'] = _div(p['khe'], p['ve'])
        p['Te'] = _div(p['ve'], p['Fp'] + p['khe'])

    if kinetics in ['2I-EC', '2I-IC', '2I-IC-U']:
        p['Fa'] = p['fa'] * p['Fp']
        p['Fv'] = (1 - p['fa']) * p['Fp']

    if {'khe', 'vol_l'}.issubset(p):
        p['CL'] = p['khe'] * p['vol_l']

    return p
    





# --- Liver kinetic models ---



def conc_liver_2i_ec(ci, t=None, dt=1.0, fa=None, 
                     ve=None, Fp=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
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
    >>> dc.conc_liver_2i_ec((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01)
    array([[0.        , 0.04373231, 0.17143928, 0.34444778, 0.41242799],
           [0.        , 0.        , 0.        , 0.        , 0.        ]])
    """
    ca, cv = ci
    Te = ve / Fp
    return _conc_liver(
        ca, cv=cv, ve_app=ve, fa=fa, Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ec_hf(ci, t=None, dt=1.0, fa=None, ve=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
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
    >>> dc.conc_liver_2i_ec_hf((ca, cv), t=t, fa=0.3, ve=0.2)
    array([[0.13 , 0.26 , 0.46 , 0.53 , 0.372],
           [0.   , 0.   , 0.   , 0.   , 0.   ]])
    """
    ca, cv = ci
    return _conc_liver(ca, cv=cv, ve_app=ve, fa=fa, t=t, dt=dt)

def conc_liver_1i_ec(ca, t=None, dt=1.0, ve=None, Fp=None):
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
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
    >>> dc.conc_liver_1i_ec(ca, t=t, ve=0.2, Fp=0.01)
    array([[0.        , 0.06728047, 0.24080767, 0.43032956, 0.46572405],
           [0.        , 0.        , 0.        , 0.        , 0.        ]])
    """
    Te = ve / Fp
    return _conc_liver(ca, ve_app=ve, Te=Te, t=t, dt=dt)

def conc_liver_1i_ec_hf(ca, t=None, dt=1.0, ve=None):
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
    ve : float, optional
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
    >>> dc.conc_liver_1i_ec_hf(ca, t=t, ve=0.2)
    array([[0.2, 0.4, 0.6, 0.6, 0.4],
           [0. , 0. , 0. , 0. , 0. ]])
    """
    return _conc_liver(ca, ve_app=ve, t=t, dt=dt)

# def conc_liver_1i_ec_d(ca, t=None, dt=1.0, ve=None, 
#                        Te=None, De=None):
#     return _conc_liver(ca, ve_app=ve, Te=Te, De=De, t=t, dt=dt)



def conc_liver_2i_ic(ci, t=None, dt=1.0, fa=None, ve=None, Fp=None, E=None, Th=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_2i_ic((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01, E=0.15, Th=30.0)
    array([[0.        , 0.04292407, 0.16359685, 0.31686986, 0.35638917],
           [0.        , 0.00089637, 0.00864157, 0.03092497, 0.06824562]])
    """
    ca, cv = ci
    khe = Fp * E / (1 - E)
    Te = ve / (Fp + khe)
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    return _conc_liver(
        ca, ve_app=ve_app, cv=cv, fa=fa, Ktrans=Ktrans, 
        Th=Th, Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ic_nse(ci, t=None, dt=1.0, fa=None, 
                         Fp=None, ve=None, E=None, 
                         Th_i=None, Th_f=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time (sec) at the start of the time series. 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_2i_ic_nse((ca, cv), t=t, fa=0.3, Fp=0.01, ve=0.2, E=0.15, Th_i=45.0, Th_f=15.0)
    array([[0.        , 0.04292407, 0.16359685, 0.31686986, 0.35638917],
           [0.        , 0.00094685, 0.00982136, 0.0372519 , 0.05770381]])
    """
    ca, cv = ci
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                            E=E, fa=fa, Th=Th)

def conc_liver_2i_ic_nsu(ci, t=None, dt=1.0, fa=None, 
                         ve=None, Fp=None, E_i=None, E_f=None, Th=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E_i : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    E_f : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_2i_ic_nsu((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01, E_i=0.30, E_f=0.05, Th=30.0)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00223418, 0.01908304, 0.05398519, 0.06316002]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, fa=fa, 
                            ve=ve, Fp=Fp, E=E, Th=Th)

def conc_liver_2i_ic_nsue(ci, t=None, dt=1.0, fa=None, 
                          ve=None, Fp=None, E_i=None, E_f=None, 
                          Th_i=None, Th_f=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E_i : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    E_f : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_2i_ic_nsue((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01, E_i=0.30, E_f=0.05, Th_i=30.0, Th_f=15.0)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00236001, 0.02147516, 0.06152976, 0.03394174]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                            E=E, fa=fa, Th=Th)



def conc_liver_2i_ic_hf(ci, t=None, dt=1.0, fa=None, 
                        ve=None, khe=None, Th=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit).

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. Tissue uptake is instead 
    characterized directly by the hepatocyte clearance/uptake rate constant (`khe`) 
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_2i_ic_hf((ca, cv), t=t, fa=0.3, ve=0.2, khe=0.003, Th=30.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.0135959 , 0.05637118, 0.12235044, 0.16961473]])
    """
    ca, cv = ci
    return _conc_liver(
        ca, ve_app=ve, cv=cv, fa=fa, 
        Ktrans=khe, Th=Th, t=t, dt=dt, 
    )

def conc_liver_2i_ic_hf_nse(ci, t=None, dt=1.0, fa=None, 
                            ve=None, khe=None, Th_i=None, Th_f=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. The hepatocyte uptake rate constant 
    (`khe`) remains stationary, while the hepatocyte transit time (`Th`) varies 
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_2i_ic_hf_nse((ca, cv), t=t, fa=0.3, ve=0.2, khe=0.003, Th_i=30.0, Th_f=15.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.014625  , 0.06330682, 0.13572378, 0.10896284]])
    """
    ca, _ = ci
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, fa=fa, 
                               ve=ve, khe=khe, Th=Th)

def conc_liver_2i_ic_hf_nsu(ci, t=None, dt=1.0, fa=None, 
                            ve=None, khe_i=None, khe_f=None, Th=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. The hepatocyte transit time 
    (`Th`) remains stationary, while the hepatocyte uptake rate constant (`khe`) 
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe_i : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    khe_f : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_2i_ic_hf_nsu((ca, cv), t=t, fa=0.3, ve=0.2, khe_i=0.003, khe_f=0.0005, Th=30.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.01295492, 0.04837293, 0.08854618, 0.0796007 ]])
    """
    ca, _ = ci
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, fa=fa, 
                               ve=ve, khe=khe, Th=Th)

def conc_liver_2i_ic_hf_nsue(ci, t=None, dt=1.0, fa=None, 
                             ve=None, khe_i=None, khe_f=None, 
                             Th_i=None, Th_f=None):
    """
    Dual-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting. Both the hepatocyte uptake rate 
    constant (`khe`) and the hepatocyte transit time (`Th`) vary dynamically over 
    time between their respective initial and final values (e.g., due to complex, 
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe_i : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    khe_f : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_2i_ic_hf_nsue((ca, cv), t=t, fa=0.3, ve=0.2, khe_i=0.003, khe_f=0.0005, Th_i=30.0, Th_f=15.0)
    array([[0.13      , 0.26      , 0.46      , 0.53      , 0.372     ],
           [0.        , 0.01394792, 0.05433428, 0.0966478 , 0.03696622]])
    """
    ca, cv = ci
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, fa=fa, 
                               ve=ve, khe=khe, Th=Th)



def conc_liver_2i_ic_u(ci, t=None, dt=1.0, fa=None, 
                       ve=None, Fp=None, E=None):
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
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
    >>> dc.conc_liver_2i_ic_u((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01, E=0.30)
    array([[0.        , 0.04180636, 0.15331954, 0.28315496, 0.29565219],
           [0.        , 0.00223963, 0.02314597, 0.09329366, 0.27933882]])
    """
    ca, cv = ci
    khe = Fp * E / (1 - E)
    Te = ve / (Fp + khe)
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    return _conc_liver(
        ca, ve_app=ve_app, cv=cv, fa=fa, Ktrans=Ktrans, Te=Te,
        t=t, dt=dt, 
    )

def conc_liver_2i_ic_u_nsu(ci, t=None, dt=1.0, fa=None, 
                           Fp=None, ve=None, E_i=None, E_f=None):
    """
    Dual-inlet intracellular agent liver concentration (Uptake-only model) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under an uptake-only condition 
    (no biliary excretion or efflux back into blood occurs during the scan period). 
    The hepatocyte extraction fraction (`E`) varies dynamically over time between an 
    initial and final value (e.g., due to acute metabolic shifts or competitive transporter 
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
    fa : float, optional
        Arterial fraction of liver blood inflow. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    E_i : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    E_f : float, optional
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
    >>> dc.conc_liver_2i_ic_u_nsu((ca, cv), t=t, fa=0.3, ve=0.2, Fp=0.01, E_i=0.30, E_f=0.05)
    array([[0.        , 0.04875   , 0.19588483, 0.3820468 , 0.38490133],
           [0.        , 0.00236001, 0.02233335, 0.07560343, 0.15157707]])
    """
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_2i_ic_u(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                              E=E, fa=fa)


def conc_liver_1i_ic(ca, t=None, dt=1.0, 
                     ve=None, Fp=None, E=None, Th=None):
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_1i_ic(ca, t=t, ve=0.2, Fp=0.01, E=0.30, Th=30.0)
    array([[0.        , 0.06431748, 0.21436459, 0.34956599, 0.32940522],
           [0.        , 0.00326188, 0.02848746, 0.09001914, 0.17039905]])
    """
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    Te = ve_app / Fp
    return _conc_liver(
        ca, ve_app=ve_app, Ktrans=Ktrans, Th=Th, Te=Te, 
        t=t, dt=dt
    )

def conc_liver_1i_ic_nsu(ca, t=None, dt=1.0, 
                         ve=None, Fp=None, E_i=None, E_f=None, 
                         Th=None):
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E_i : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    E_f : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_1i_ic_nsu(ca, t=t, ve=0.2, Fp=0.01, E_i=0.30, E_f=0.05, Th=30.0)
    array([[0.        , 0.075     , 0.2744382 , 0.46512652, 0.42386628],
           [0.        , 0.0034372 , 0.0274768 , 0.07132961, 0.07765741]])
    """
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, Fp=Fp, 
                            ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_nse(ca, t=None, dt=1.0, 
                         ve=None, Fp=None, E=None, Th_i=None, Th_f=None):
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E : float, optional
        Hepatocyte extraction fraction. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_1i_ic_nse(ca, t=t, ve=0.2, Fp=0.01, E=0.30, Th_i=30.0, Th_f=15.0)
    array([[0.        , 0.06431748, 0.21436459, 0.34956599, 0.32940522],
           [0.        , 0.00344558, 0.03205143, 0.10295917, 0.12292478]])
    """
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, Fp=Fp, ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_nsue(ca, t=None, dt=1.0, 
                          ve=None, Fp=None, E_i=None, E_f=None, 
                          Th_i=None, Th_f=None):
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    Fp : float, optional
        Plasma flow (mL/sec/cm3). Defaults to None.
    E_i : float, optional
        Initial hepatocyte extraction fraction at the start of the time series. 
        Defaults to None.
    E_f : float, optional
        Final hepatocyte extraction fraction at the end of the time series. 
        Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_1i_ic_nsue(ca, t=t, ve=0.2, Fp=0.01, E_i=0.30, E_f=0.05, Th_i=30.0, Th_f=15.0)
    array([[0.        , 0.075     , 0.2744382 , 0.46512652, 0.42386628],
           [0.        , 0.00363078, 0.03094225, 0.08095485, 0.04013242]])
    """
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, Fp=Fp, ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_hf(ca, t=None, dt=1.0, ve=None, khe=None, Th=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit).

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). Tissue uptake is characterized directly 
    by the hepatocyte uptake rate constant (`khe`) and hepatocyte transit time (`Th`).

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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_1i_ic_hf(ca, t=t, ve=0.2, khe=0.003, Th=30.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.02091678, 0.07947534, 0.15444095, 0.19437905]])
    """
    return _conc_liver(ca, ve_app=ve, Ktrans=khe, Th=Th, t=t, dt=dt)

def conc_liver_1i_ic_hf_nsu(ca, t=None, dt=1.0, ve=None, khe_i=None, khe_f=None, Th=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte transit time (`Th`) remains 
    stationary, while the hepatocyte uptake rate constant (`khe`) varies dynamically 
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe_i : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    khe_f : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Th : float, optional
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
    >>> dc.conc_liver_1i_ic_hf_nsu(ca, t=t, ve=0.2, khe_i=0.003, khe_f=0.0005, Th=30.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.01993065, 0.06868066, 0.1137763 , 0.09451032]])
    """
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, ve=ve, khe=khe, Th=Th)

def conc_liver_1i_ic_hf_nse(ca, t=None, dt=1.0,
                            ve=None, khe=None, Th_i=None, Th_f=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary excretion.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). The hepatocyte uptake rate constant 
    (`khe`) remains stationary, while the hepatocyte transit time (`Th`) varies 
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe : float, optional
        Hepatocyte uptake rate constant (mL/sec/cm3). Corresponds to the 
        transfer constant from the extracellular space into hepatocytes. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_1i_ic_hf_nse(ca, t=t, ve=0.2, khe=0.003, Th_i=30.0, Th_f=15.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.0225    , 0.08931818, 0.16935315, 0.12013191]])
    """
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, ve=ve, khe=khe, Th=Th)

def conc_liver_1i_ic_hf_nsue(ca, t=None, dt=1.0, 
                             ve=None, khe_i=None, khe_f=None, Th_i=None, Th_f=None):
    """
    Single-inlet intracellular agent liver concentration (High Flow limit) with non-stationary uptake and efflux.

    This model tracks a hepatocyte-specific tracer under a high-flow approximation, 
    where blood flow is assumed to be non-limiting, using a single blood inlet 
    (arterial input only, with an optional gut transit delay parameter to 
    approximate portal venous delivery). Both the hepatocyte uptake rate constant 
    (`khe`) and the hepatocyte transit time (`Th`) vary dynamically over time 
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
    ve : float, optional
        Extracellular volume fraction. Defaults to None.
    khe_i : float, optional
        Initial hepatocyte uptake rate constant at the start of the time series 
        (mL/sec/cm3). Corresponds to the initial transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    khe_f : float, optional
        Final hepatocyte uptake rate constant at the end of the time series 
        (mL/sec/cm3). Corresponds to the final transfer constant from the 
        extracellular space into hepatocytes. Defaults to None.
    Th_i : float, optional
        Initial hepatocyte transit time at the start of the time series (sec). 
        Defaults to None.
    Th_f : float, optional
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
    >>> dc.conc_liver_1i_ic_hf_nsue(ca, t=t, ve=0.2, khe_i=0.003, khe_f=0.0005, Th_i=30.0, Th_f=15.0)
    array([[0.2       , 0.4       , 0.6       , 0.6       , 0.4       ],
           [0.        , 0.02145833, 0.07719697, 0.12250364, 0.0416289 ]])
    """
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, ve=ve, khe=khe, Th=Th)

# def conc_liver_1i_ic_hfd(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe=None, Th=None):
#     return _conc_liver( # approx 1 - E = 1
#         ca, ve=ve, Ktrans=khe, Th=Th,
#         Tg=Tg, Dg=Dg, t=t, dt=dt, 
#     )

# def conc_liver_1i_ic_hfd_nsu(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe_i=None, khe_f=None, Th=None):
#     khe = _interp_params(ca, t, dt, [khe_i, khe_f])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, ve=ve, khe=khe, Th=Th)

# def conc_liver_1i_ic_hfd_nse(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe=None, Th_i=None, Th_f=None):
#     Th = _interp_params(ca, t, dt, [Th_i, Th_f])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, ve=ve, khe=khe, Th=Th)

# def conc_liver_1i_ic_hfd_nsue(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe_i=None, khe_f=None, Th_i=None, Th_f=None):
#     khe = _interp_params(ca, t, dt, [khe_i, khe_f])
#     Th = _interp_params(ca, t, dt, [Th_i, Th_f])
#     return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, ve=ve, khe=khe, Th=Th)

# def conc_liver_1i_ic_hfdu(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe=None):
#     return _conc_liver(
#         ca, ve=ve, Ktrans=khe, Tg=Tg, Dg=Dg, t=t, dt=dt, 
#     )

# def conc_liver_1i_ic_hfdu_nsu(ca, t=None, dt=1.0, Tg=None, Dg=None, ve=None, khe_i=None, khe_f=None):
#     khe = _interp_params(ca, t, dt, [khe_i, khe_f])
#     return conc_liver_1i_ic_hfdu(ca, t=t, dt=dt, Tg=Tg, Dg=Dg, ve=ve, khe=khe)



def _interp_params(ca: np.ndarray, t: Optional[np.ndarray], dt: float, p):
    ti = tarray(np.size(ca), t=t, dt=dt)
    return p[0] + (p[1] - p[0]) * ti / ti.max()



def _conc_liver(
    ca: np.ndarray,
    cv: np.ndarray = None,
    fa: float = None,
    ve_app: float = None, # (1 - E) ve
    Te: float = None,   
    Ktrans: float = None, # (1 - E) kep
    Th: float = None,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    
    # ve ce' = Fp ca - khe ce - Fp ce
    # vh ch' = khe ce - kbh ch
    # ce = Fp/ve ca * exp( -t (khe + Fp) / ve)
    # Ce = Fp ca * exp(-t/Te)
    # ch = khe/vh ce * exp(-t kbh/vh) 
    # Ch = khe/ve Ce * exp(-t/Th) 
    #    = E/Te Ce * exp(-t/Th) 

    # Te = p['ve'] * (1 - p['E']) / p['Fp']
    # Ce = pk.conc_comp(ca * p['Fp'], Te, t=t, dt=dt)
    # Ch = pk.conc_comp(Ce * p['E']/Te, Te, t=t, dt=dt)
    # return np.stack((Ce, Ch))

    ca = np.array(ca)

    # If a portal venous concentration is provided, use it
    if cv is not None:
        ca = fa * np.array(ca) + (1 - fa) * np.array(cv)

    # Propagate through the extracellular space
    if Te is None:
        ce = ca
    elif np.isscalar(Te):
        ce = pk.flux_comp(ca, t=t, dt=dt, T=Te)
    else:
        ce = pk.flux_nscomp(ca, t=t, dt=dt, T=Te)

    # else:
    #     ec_model, ec_pars = 'pfcomp', (Te, De,)

    # Tissue concentration in the extracellular space
    Ce = ve_app * ce

    # Tissue concentration in the hepatocytes
    if Ktrans is None:
        Ch = np.zeros(len(ce))
    elif Th is None:
        Ch = pk.conc_trap(Ktrans * ce, t=t, dt=dt)
    elif np.isscalar(Th):
        Ch = pk.conc_comp(Ktrans * ce, t=t, dt=dt, T=Th)
    else:
        Ch = pk.conc_nscomp(Ktrans * ce, t=t, dt=dt, T=Th)

    return np.stack((Ce, Ch))
    
