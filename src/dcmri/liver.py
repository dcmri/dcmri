import copy
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np

import dcmri.pk as pk
import dcmri.utils as utils


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(a, b)

PARAMS_LIVER = {
    'Te': {'init': 30.0, 'bounds': [0.1, 60], 'name': 'Extracellular mean transit time', 'unit': 'sec'},
    'De': {'init': 0.85, 'bounds': [0, 1], 'name': 'Extracellular dispersion', 'unit': ''},
    've': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Liver extracellular volume fraction', 'unit': 'mL/cm3'},
    'vh': {'init': 0.6, 'bounds': [0.1, 1.0], 'name': 'Hepatocellular volume fraction', 'unit': 'mL/cm3'},
    've_app': {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Apparent liver extracellular volume fraction', 'unit': 'mL/cm3'},
    'T_a': {'init': 2, 'bounds': [0, 30], 'name': 'Arterial mean transit time', 'unit': 'sec'},
    'Tg': {'init': 30, 'bounds': [0.1, 60], 'name': 'Gut mean transit time', 'unit': 'sec'},
    'Dg': {'init': 0.85, 'bounds': [0, 1], 'name': 'Gut dispersion', 'unit': ''},
    'Fp': {'init': 0.008, 'bounds': [0, 1], 'name': 'Liver plasma flow', 'unit': 'mL/sec/cm3'},
    'fa': {'init': 0.2, 'bounds': [0, 1], 'name': 'Arterial flow fraction', 'unit': ''},
    'Ktrans': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Hepatic plasma clearance', 'unit': 'mL/sec/cm3'},
    'Ktrans_i': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Initial hepatic plasma clearance', 'unit': 'mL/sec/cm3'},
    'Ktrans_f': {'init': 0.015, 'bounds': [0.0, 0.1], 'name': 'Final hepatic plasma clearance', 'unit': 'mL/sec/cm3'},
    'khe': {'init': 0.003, 'bounds': [0.0, 0.1], 'name': 'Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'Dkhe': {'init': 0.000, 'bounds': [-1.5e-7, +1.5e-7], 'name': 'Rate of change in hepatocellular uptake rate', 'unit': 'mL/sec/cm3/sec'},
    'khe_i': {'init': 0.002, 'bounds': [0.0, 0.1], 'name': 'Initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'khe_f': {'init': 0.002, 'bounds': [0.0, 0.1], 'name': 'Final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'kbh': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_i': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_f': {'init': 0.0004, 'bounds': [0.0, 0.001], 'name': 'Final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'E': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Liver extraction fraction', 'unit': ''},
    'E_i': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Initial liver extraction fraction', 'unit': ''},
    'E_f': {'init': 0.1, 'bounds': [0.0, 1.0], 'name': 'Final liver extraction fraction', 'unit': ''},
    'Th': {'init': 1800, 'bounds': [600, 36000], 'name': 'Hepatocellular mean transit time', 'unit': 'sec'},
    'DTh': {'init': 0, 'bounds': [-0.25, 0.25], 'name': 'Rate of change in hepatocellular mean transit time', 'unit': ''},
    'Th_i': {'init': 1800, 'bounds': [600, 36000], 'name': 'Initial hepatocellular mean transit time', 'unit': 'sec'},
    'Th_f': {'init': 1800, 'bounds': [600, 36000], 'name': 'Final hepatocellular mean transit time', 'unit': 'sec'},
    'vol': {'init': 1000, 'bounds': [0, 10000], 'name': 'Liver volume', 'unit': 'cm3'},
    'Kbh': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_i': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Initial biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_f': {'init': 0.0001, 'bounds': [0.0, 0.001], 'name': 'Final biliary tissue excretion rate', 'unit': '/sec'},
    'Fa': {'name': 'Arterial plasma flow', 'unit': 'mL/sec/cm3'},
    'Fv': {'name': 'Venous plasma flow', 'unit': 'mL/sec/cm3'},
    'Khe': {'name': 'Hepatocellular tissue uptake rate', 'unit': '/sec'},
    'CL': {'name': 'Liver plasma clearance', 'unit': 'mL/sec'},
}



def params_liver(kinetics='2I-EC', non_stationary=None) -> dict:
    """Parameters characterizing a liver tissue. 

    See section :ref:`liver-tissues` for background and 
    :ref:`table-liver-models` for the full list of parameter options.

    Args:
        kinetics (str, optional): Tracer-kinetic regime. Defaults to '2C-EC'.
        non_stationary (str, optional): For models with an intracellular agent, 
           set to 'U' if uptake kinetics is non-stationary, 
           'E' if excretion is non-stationary, and 'UE' for both. Default is None 
           (all transport stationary).

    Returns: 
        list: parameters for the given model

    Raises:
        ValueError: if the configuration is not recognized.

    Example:

        Print the parameters of a liver tissue:

        >>> import dcmri as dc
        >>> dc.params_liver('2I-EC')
        ['fa', 'T_a', 've', 'Fp']
    """

    pars = None

    # --- Extracellular Models ---

    if kinetics == '1I-EC-D':
        if non_stationary is not None:
            raise ValueError(f"There are no non-stationary EC models. Please set non_stationary=None")
        pars = ['ve', 'Te', 'De']
    
    elif kinetics == '1I-EC':
        if non_stationary is not None:
            raise ValueError(f"There are no non-stationary EC models. Please set non_stationary=None")
        pars = ['fa', 'T_a', 'Tg', 've', 'Fp']
    
    elif kinetics == '2I-EC-HF':
        if non_stationary is not None:
            raise ValueError(f"There are no non-stationary EC models. Please set non_stationary=None")
        pars = ['fa', 'T_a', 've']
    
    elif kinetics == '2I-EC':
        if non_stationary is not None:
            raise ValueError(f"There are no non-stationary EC models. Please set non_stationary=None")
        pars = ['fa', 'T_a', 've', 'Fp']
    
    # --- Intracellular Models ---

    elif kinetics == '1I-IC':

        if non_stationary is None:
            pars = ['ve', 'Fp', 'E', 'Th']
        elif non_stationary == 'U':
            pars = ['ve', 'Fp', 'E_i', 'E_f', 'Th']
        elif non_stationary == 'E':
            pars = ['ve', 'Fp', 'E', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            pars = ['ve', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f']

    elif kinetics == '1I-IC-HF': # Note E=0 at high Fp

        if non_stationary is None:
            pars = ['ve', 'khe', 'Th']
        elif non_stationary == 'U':
            pars = ['ve', 'khe_i', 'khe_f', 'Th']
        elif non_stationary == 'E':
            pars = ['ve', 'khe', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            pars = ['ve', 'khe_i', 'khe_f', 'Th_i', 'Th_f']

    elif kinetics == '1I-IC-HFD':

        if non_stationary is None:
            pars = ['Tg', 'Dg', 've', 'khe', 'Th']
        elif non_stationary == 'U':
            pars = ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th']
        elif non_stationary == 'E':
            pars = ['Tg', 'Dg', 've', 'khe', 'Th_i', 'Th_f']   
        elif non_stationary == 'UE':
            pars = ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f']  
            
    elif kinetics == '1I-IC-HFDU':

        if non_stationary is None:
            pars = ['Tg', 'Dg', 've', 'khe']
        elif non_stationary == 'U':
            pars = ['Tg', 'Dg', 've', 'khe_i', 'khe_f']
        
    elif kinetics == '2I-IC-HF':

        if non_stationary is None:
            pars = ['fa', 'T_a', 've', 'khe', 'Th']
        elif non_stationary == 'U':
            pars = ['fa', 'T_a', 've', 'khe_i', 'khe_f', 'Th']
        elif non_stationary == 'E':
            pars = ['fa', 'T_a', 've', 'khe', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            pars = ['fa', 'T_a', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
        
    elif kinetics == '2I-IC':

        if non_stationary is None:
            pars = ['fa', 'T_a', 've', 'Fp', 'E', 'Th']
        elif non_stationary == 'U':
            pars = ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f', 'Th']
        elif non_stationary == 'E':
            pars = ['fa', 'T_a', 've', 'Fp', 'E', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            pars = ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f']
        
    elif kinetics == '2I-IC-U':

        if non_stationary is None:
            pars = ['fa', 'T_a', 've', 'Fp', 'E']
        elif non_stationary == 'U':
            pars = ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f']
        
    if pars is not None:
        return {p:PARAMS_LIVER[p] for p in pars}   

    raise ValueError(
        f"The model kinetics={kinetics}, non-stationary={non_stationary} "
        f"is not a recognised liver model."
    )


def derived_params_liver(p, kinetics, H=0.45) -> dict:

    p = copy.deepcopy(p)
        
    # Non-stationary options

    if {'E_i', 'E_f'} <= p.keys():
        p['E'] = np.mean([p['E_i'], p['E_f']])

    if {'khe_i', 'khe_f'} <= p.keys():
        p['khe'] = np.mean([p['khe_i'], p['khe_f']])

    if {'Th_i', 'Th_f'} <= p.keys():
        p['Th'] = np.mean([p['Th_i'], p['Th_f']])
    
    if {'Th_i', 'Th_f', 've'} <= p.keys():
        vh = 1 - p['ve'] / (1 - H)
        p['kbh_i'] = _div(vh, p['Th_i'])
        p['kbh_f'] = _div(vh, p['Th_f'])

    # Dual-inlet models

    if {'Fp', 'fa'} <= p.keys():
        p['Fa'] = p['Fp'] * p['fa']
        p['Fv'] = p['Fp'] * (1 - p['fa'])

    # Kinetic models
    
    if kinetics == '1I-EC':
        p['Te'] = _div(p['ve'], p['Fp'])
    
    if kinetics == '2I-EC':
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

    if kinetics in ['1I-IC-HF', '2I-IC-HF', '1I-IC-HFD']:
        p['Khe'] = _div(p['khe'], p['ve'])
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['kbh'] = _div(p['vh'], p['Th'])
        p['Kbh'] = _div(1, p['Th'])

    if kinetics in ['1I-IC-HFDU']:
        p['Khe'] = _div(p['khe'], p['ve'])
        p['vh'] = 1 - p['ve'] / (1 - H)
        
    if kinetics == '2I-IC-U':
        p['vh'] = 1 - p['ve'] / (1 - H)
        p['Ktrans'] = p['E'] * p['Fp']
        p['khe'] = _div(p['Fp'] * p['E'], 1 - p['E'])
        p['Khe'] = _div(p['khe'], p['ve'])
        p['Te'] = _div(p['ve'], p['Fp'] + p['khe'])

    if kinetics in ['2I-EC', '2I-IC', '2I-IC-U']:
        p['Fa'] = p['fa'] * p['Fp']
        p['Fv'] = (1 - p['fa']) * p['Fp']

    if {'khe', 'vol'} <= p.keys():
        p['CL'] = p['khe'] * p['vol']

    return p
    

def conc_liver(
    ci: np.ndarray,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
    kinetics = '2I-EC',
    non_stationary = None,
    sum: bool = True,
    **params: Dict[str, Any]
) -> np.ndarray:
    """
    Compute concentration in liver tissue for a variety of liver models.

    See section :ref:`liver-tissues` for background and 
    :ref:`table-liver-models` for the full list of parameter options.

    Parameters
    ----------
    ci : np.ndarray or tuple of np.ndarray
        Plasma concentration in the arterial input, or, for a dual-inlet 
        tissue, a tuple with arterial and portal-venous inlet concentrations
    t : np.ndarray, optional
        Time points in seconds of the input function `ca`. If not provided, 
        the time points are assumed to be uniformly spaced with spacing `dt`. 
        Defaults to None.
    dt : float, optional
        Spacing in seconds between uniformly spaced time points. Ignored 
        if `t` is provided. Defaults to 1.0.
    kinetics (str, optional): Tracer-kinetic regime. Defaults to '2C-EC'.
    non_stationary (str, optional): For models with an intracellular agent, 
        set to 'U' if uptake kinetics is non-stationary, 
        'E' if excretion is non-stationary, and 'UE' for both. Default is None 
        (all transport stationary).
    sum : bool, optional
        For two-compartment tissues: if True, return the total tissue 
        concentration; if False, return separate compartment concentrations. 
        Defaults to True.
    **params : dict
        Model parameters, specified as keyword arguments. The accepted set 
        of parameters determines which liver model is used.

    Returns
    -------
    np.ndarray
        If `sum=True`: a 1D array with total concentration at each time point.  
        If `sum=False`: a 2D array with concentration in each compartment, 
        shape (2, k), where k is the number of time points. Concentrations 
        are returned in units of M.

    Examples
    --------
    Plot concentration in cortex and medulla for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 30*60, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Generate extracellular and hepatocyte liver tissue 
        concentrations tissue:

        >>> C = dc.conc_liver(
        >>>     ca, 
        >>>     t, 
        >>>     kinetics = '1I-IC',
        >>>     sum = False, 
        >>>     ve = 0.2, 
        >>>     Fp = 0.01, 
        >>>     E = 0.2, 
        >>>     Th = 20 * 60,
        >>> )

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Liver concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, 
        >>>         color='darkred', label='Extracellular')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, 
        >>>         color='darkblue', label='Hepatocytes')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, 
        >>>         color='grey', label='Whole liver')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    
    # Check the model exists and all parameters are provided
    required_pars = params_liver(kinetics, non_stationary)
    if not (set(required_pars) <= set(params.keys())):
        raise ValueError(
            f"Not all required parameters ({required_pars}) are "
            f"provided in the call to conc_liver()."
        )
    
    # Define model function
    conc = '_conc_' + kinetics.lower().replace('-', '_')
    if non_stationary is not None:
        conc += '__' + non_stationary.lower()      

    # Apply model function
    if '-IC' in kinetics:
        return globals()[conc](ci, t=t, dt=dt, sum=sum, **params)
    else:
        if non_stationary is not None:
            raise ValueError("For extracellular models non_stationary must be None")
        return globals()[conc](ci, t=t, dt=dt, **params)



# --- Liver kinetic models ---

def _conc_1i_ec_d(ca, t=None, dt=1.0, **p):
    return _conc_liver(
        ca, p['ve'], Te=p['Te'], De=p['De'], t=t, dt=dt, sum=True,
    )

def _conc_1i_ec(ca, t=None, dt=1.0, **p):
    Te = p['ve'] / p['Fp']
    return _conc_liver(
        ca, p['ve'], Ta=p['T_a'], Tg=p['Tg'],
        fa=p['fa'], Te=Te, t=t, dt=dt, sum=True,
    )

def _conc_2i_ec_hf(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    return _conc_liver(
        ca, p['ve'], Ta=p['T_a'], cv=cv,
        fa=p['fa'], t=t, dt=dt, sum=True,
    )

def _conc_2i_ec(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    Te = p['ve'] / p['Fp']
    return _conc_liver(
        ca, p['ve'], cv=cv, Ta=p['T_a'], 
        fa=p['fa'], Te=Te, t=t, dt=dt, sum=True,
    )


def _conc_1i_ic(ca, t=None, dt=1.0, sum=True, **p):
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    Te = ve_app / p['Fp']
    return _conc_liver(
        ca, ve_app, Ktrans=Ktrans, Th=p['Th'], Te=Te, 
        t=t, dt=dt, sum=sum, 
    )

def _conc_1i_ic__u(ca, t=None, dt=1.0, sum=True, **p):
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return _conc_1i_ic(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic__e(ca, t=None, dt=1.0, sum=True, **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_1i_ic(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic__ue(ca, t=None, dt=1.0, sum=True, **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return _conc_1i_ic(ca, t=t, dt=dt, sum=sum, **p)


def _conc_1i_ic_hf(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver( # approx 1 - E = 1
        ca, p['ve'], Ktrans=p['khe'], Th=p['Th'], 
        t=t, dt=dt, sum=sum, 
    )

def _conc_1i_ic_hf__u(ca, t=None, dt=1.0, sum=True, **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return _conc_1i_ic_hf(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic_hf__e(ca, t=None, dt=1.0, sum=True, **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_1i_ic_hf(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic_hf__ue(ca, t=None, dt=1.0, sum=True, **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_1i_ic_hf(ca, t=t, dt=dt, sum=sum, **p)


def _conc_1i_ic_hfd(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver( # approx 1 - E = 1
        ca, p['ve'], Ktrans=p['khe'], Th=p['Th'],
        Tg=p['Tg'], Dg=p['Dg'], t=t, dt=dt, sum=sum,
    )

def _conc_1i_ic_hfd__u(ca, t=None, dt=1.0, sum=True, **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return _conc_1i_ic_hfd(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic_hfd__e(ca, t=None, dt=1.0, sum=True, **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_1i_ic_hfd(ca, t=t, dt=dt, sum=sum, **p)

def _conc_1i_ic_hfd__ue(ca, t=None, dt=1.0, sum=True, **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_1i_ic_hfd(ca, t=t, dt=dt, sum=sum, **p)


def _conc_1i_ic_hfdu(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver(
        ca, p['ve'], Ktrans=p['khe'], Tg=p['Tg'], Dg=p['Dg'], 
        t=t, dt=dt, sum=sum,
    )

def _conc_1i_ic_hfdu__u(ca, t=None, dt=1.0, sum=True, **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return _conc_1i_ic_hfdu(ca, t=t, dt=dt, sum=sum, **p)


def _conc_2i_ic_hf(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    return _conc_liver(
        ca, p['ve'], cv=cv, Ta=p['T_a'], fa=p['fa'], 
        Ktrans=p['khe'], Th=p['Th'], t=t, dt=dt, sum=sum,
    )

def _conc_2i_ic_hf__e(ci, t=None, dt=1.0, sum=True, **p):
    ca, _ = ci
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_2i_ic_hf(ci, t=t, dt=dt, sum=sum, **p)

def _conc_2i_ic_hf__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, _ = ci
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return _conc_2i_ic_hf(ci, t=t, dt=dt, sum=sum, **p)

def _conc_2i_ic_hf__ue(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_2i_ic_hf(ci, t=t, dt=dt, sum=sum, **p)


def _conc_2i_ic(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    khe = p['Fp'] * p['E'] / (1 - p['E'])
    Te = p['ve'] / (p['Fp'] + khe)
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    return _conc_liver(
        ca, ve_app, cv=cv, Ta=p['T_a'], fa=p['fa'], Ktrans=Ktrans, 
        Th=p['Th'], Te=Te, t=t, dt=dt, sum=sum,
    )

def _conc_2i_ic__e(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_2i_ic(ci, t=t, dt=dt, sum=sum, **p)

def _conc_2i_ic__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return _conc_2i_ic(ci, t=t, dt=dt, sum=sum, **p)

def _conc_2i_ic__ue(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_2i_ic(ci, t=t, dt=dt, sum=sum, **p)


def _conc_2i_ic_u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    khe = p['Fp'] * p['E'] / (1 - p['E'])
    Te = p['ve'] / (p['Fp'] + khe)
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    return _conc_liver(
        ca, ve_app, cv=cv, Ta=p['T_a'], fa=p['fa'], Ktrans=Ktrans, Te=Te,
        t=t, dt=dt, sum=sum,
    )

def _conc_2i_ic_u__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return _conc_2i_ic_u(ci, t=t, dt=dt, sum=sum, **p)


def _interp_params(ca: np.ndarray, t: Optional[np.ndarray], dt: float, p, lower_t=False):
    tarr = utils.tarray(np.size(ca), t=t, dt=dt)
    if lower_t:
        lower = tarr[1] - tarr[0]
    else:
        lower = None
    return utils.interp(p, tarr, lower=lower)



def _conc_liver(
    ca: np.ndarray,
    ve_app: float,
    cv: np.ndarray = None,
    Ta: float = None,
    fa: float = None,
    Ktrans: float = None,
    Th: float = None,
    Te: float = None,
    De: float = None,
    Tg: float = None,
    Dg: float = None,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
    sum: bool = True,
) -> np.ndarray:
    
    # Extracellular space
    if Te is None:
        ec_model, ec_pars = 'pass', ()
    elif De is None:
        if np.isscalar(Te):
            ec_model, ec_pars = 'comp', (Te,)
        else:
            ec_model, ec_pars = 'nscomp', (Te,)
    else:
        ec_model, ec_pars = 'pfcomp', (Te, De,)

    # Hepatocytes
    if Th is None:
        hep_model, hep_pars = "trap", ()
    elif np.isscalar(Th):
        hep_model, hep_pars = "comp", (Th,)
    else:
        hep_model, hep_pars = "nscomp", (Th,)

    # Propagate through arterial tree
    if Ta is not None:
        ca = pk.flux(ca, Ta, t=t, dt=dt, model='plug')

    # Propagate through gut
    if Tg is not None:
        if Dg is not None:
            ca = pk.flux_pfcomp(ca, Tg, Dg, t=t, dt=dt)
            # ca = pk.flux_chain(ca, Tg, Dg, t=t, dt=dt)
        else:
            ca_prop = pk.flux_comp(ca, Tg, t=t, dt=dt)
            ca = fa * ca + (1 - fa) * ca_prop

    # Determine combined inlet concentration (arterial + venous)
    if cv is not None:
        ca = fa * ca + (1 - fa) * cv

    # Propagate through the extracellular space
    ca = pk.flux(ca, *ec_pars, t=t, dt=dt, model=ec_model)

    # Tissue concentration in the extracellular space
    Ce = ve_app * ca

    # Tissue concentration in the hepatocytes
    if Ktrans is None:
        Ch = np.zeros(len(ca))
    else:
        Ch = pk.conc(Ktrans * ca, *hep_pars, t=t, dt=dt, model=hep_model)

    if sum:
        return Ce + Ch
    else:
        return np.stack((Ce, Ch))
    
