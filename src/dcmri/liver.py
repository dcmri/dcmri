import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

import dcmri.pk as pk
import dcmri.utils as utils


def params_liver(kinetics='2I-EC', non_stationary=None) -> list:
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
        ['ve', 'Fp', 'fa', 'Ta']
    """

    # --- Extracellular Models ---

    if kinetics == '1I-EC-D':
        return ['ve', 'Te', 'De']
    
    if kinetics == '1I-EC':
        return ['ve', 'Fp', 'fa', 'Ta', 'Tg']
    
    if kinetics == '2I-EC-HF':
        return ['ve', 'fa', 'Ta']
    
    if kinetics == '2I-EC':
        return ['ve', 'Fp', 'fa', 'Ta']
    
    # --- Intracellular Models ---

    if kinetics == '1I-IC-HF':

        if non_stationary is None:
            return ['ve_app', 'Ktrans', 'Th']
        elif non_stationary == 'U':
            return ['ve_app', 'Ktrans_i', 'Ktrans_f', 'Th']
        elif non_stationary == 'E':
            return ['ve_app', 'Ktrans', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            return ['ve_app', 'Ktrans_i', 'Ktrans_f', 'Th_i', 'Th_f']

    if kinetics == '1I-IC-D':

        if non_stationary is None:
            return ['ve_app', 'Te', 'De', 'Ktrans', 'Th']
        elif non_stationary == 'U':
            return ['ve_app', 'Te', 'De', 'Ktrans_i', 'Ktrans_f', 'Th']
        elif non_stationary == 'E':
            return ['ve_app', 'Te', 'De', 'Ktrans', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            return ['ve_app', 'Te', 'De', 'Ktrans_i', 'Ktrans_f', 'Th_i', 'Th_f']

    if kinetics == '1I-IC-DU':

        if non_stationary is None:
            return ['ve_app', 'Te', 'De', 'Ktrans']
        elif non_stationary == 'U':
            return ['ve_app', 'Te', 'De', 'Ktrans_i', 'Ktrans_f']
        
    if kinetics == '2I-IC-HF':

        if non_stationary is None:
            return ['ve_app', 'fa', 'Ta', 'Ktrans', 'Th']
        elif non_stationary == 'U':
            return ['ve_app', 'fa', 'Ta', 'Ktrans_i', 'Ktrans_f', 'Th']
        elif non_stationary == 'E':
            return ['ve_app', 'fa', 'Ta', 'Ktrans', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            return ['ve_app', 'fa', 'Ta', 'Ktrans_i', 'Ktrans_f', 'Th_i', 'Th_f']
        
    if kinetics == '2I-IC':

        if non_stationary is None:
            return ['ve', 'Fp', 'fa', 'Ta', 'khe', 'Th']
        elif non_stationary == 'U':
            return ['ve', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th']
        elif non_stationary == 'E':
            return ['ve', 'Fp', 'fa', 'Ta', 'khe', 'Th_i', 'Th_f']
        elif non_stationary == 'UE':
            return ['ve', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
        
    if kinetics == '2I-IC-U':

        if non_stationary is None:
            return ['ve', 'Fp', 'fa', 'Ta', 'khe']
        elif non_stationary == 'U':
            return ['ve', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f']

    raise ValueError(
        f"The model kinetics={kinetics}, non-stationary={non_stationary} "
        f"is not a recognised liver model."
    )


def derived_params_liver(p):
        
    def _div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.divide(a, b)
        
    if {'Th_i', 'Th_f'} <= p.keys():
        p['Th'] = np.mean([p['Th_i'], p['Th_f']])

    if {'khe_i', 'khe_f'} <= p.keys():
        p['khe'] = np.mean([p['khe_i'], p['khe_f']])

    if {'Th_i', 'Th_f', 've'} <= p.keys():
        p['kbh_i'] = _div(1 - p['ve'], p['Th_i'])
        p['kbh_f'] = _div(1 - p['ve'], p['Th_f'])

    if {'Fp', 'fa'} <= p.keys():
        p['Fa'] = p['Fp'] * p['fa']
        p['Fv'] = p['Fp']*(1-p['fa'])

    if {'ve', 'Fp', 'khe'} <= p.keys():
        p['Te'] = _div(p['ve'], p['Fp'] + p['khe'])

    if {'Th'} <= p.keys():
        p['Kbh'] = _div(1, p['Th'])

    if {'khe', 've'} <= p.keys():
        p['Khe'] = _div(p['khe'], p['ve'])

    if {'ve', 'Th'} <= p.keys():
        p['kbh'] = _div(1-p['ve'], p['Th'])

    if {'Ktrans', 'Fp'} <= p.keys():
        p['E'] = p['Ktrans'] / p['Fp']

    if {'khe', 'Fp'} <= p.keys():
        p['E'] = p['khe'] / (p['khe'] + p['Fp'])
        p['Ktrans'] = p['khe'] * p['Fp'] / (p['khe'] + p['Fp'])

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

        Generate plasma and tubular tissue 
        concentrations with a non-stationary model:

        >>> C = dc.conc_liver(ca, t, sum=False, 
        >>>     H = 0.45, ve = 0.2, Te = 30, De = 0.5, 
        >>>     khe = [0.003, 0.01], Th = [180, 600],
        >>> )

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
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
        return globals()[conc](ci, t=t, dt=dt, **params)



# --- Liver kinetic models ---

def _conc_1i_ec_d(ca, t=None, dt=1.0, **p):
    ce = pk.flux(ca, p['Te'], p['De'], t=t, dt=dt, model='pfcomp')
    return p['ve'] * ce

def _conc_1i_ec(ca, t=None, dt=1.0, **p):
    cv = pk.flux_comp(ca, p['Tg'], t=t, dt=dt)
    return _conc_liver_2i_ec(ca, cv, p['Ta'], p['fa'], p['Fp'], p['ve'], t=t, dt=dt)

def _conc_2i_ec_hf(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    ca_delayed = pk.flux(ca, p['Ta'], t=t, dt=dt, model='plug')
    ce = p['fa'] * ca_delayed + (1 - p['fa']) * cv
    return p['ve'] * ce

def _conc_2i_ec(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    return _conc_liver_2i_ec(ca, cv, p['Ta'], p['fa'], p['Fp'], p['ve'], t=t, dt=dt)

def _conc_1i_ic_hf(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver_1i_ic(
        ca, p['ve_app'], p['Ktrans'], t=t, dt=dt, sum=sum, 
        extracellular=['pass', ()], 
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_1i_ic_hf__u(ca, t=None, dt=1.0, sum=True, **p):
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], Ktrans, t=t, dt=dt, sum=sum, 
        extracellular=['pass', ()], 
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_1i_ic_hf__e(ca, t=None, dt=1.0, sum=True, **p):
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], p['Ktrans'], t=t, dt=dt, sum=sum, 
        extracellular=['pass', ()], 
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_1i_ic_hf__ue(ca, t=None, dt=1.0, sum=True, **p):
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], Ktrans, t=t, dt=dt, sum=sum, 
        extracellular=['pass', ()], 
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_1i_ic_d(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver_1i_ic(
        ca, p['ve_app'], p['Ktrans'], t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])], 
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_1i_ic_d__u(ca, t=None, dt=1.0, sum=True, **p):
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], Ktrans, t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])], 
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_1i_ic_d__e(ca, t=None, dt=1.0, sum=True, **p):
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], p['Ktrans'], t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])], 
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_1i_ic_d__ue(ca, t=None, dt=1.0, sum=True, **p):
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], Ktrans, t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])], 
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_1i_ic_du(ca, t=None, dt=1.0, sum=True, **p):
    return _conc_liver_1i_ic(
        ca, p['ve_app'], p['Ktrans'], t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])],
        hepatocytes=['trap', ()]
    )

def _conc_1i_ic_du__u(ca, t=None, dt=1.0, sum=True, **p):
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    return _conc_liver_1i_ic(
        ca, p['ve_app'], Ktrans, t=t, dt=dt, sum=sum,
        extracellular=['pfcomp', (p['Te'], p['De'])],
        hepatocytes=['trap', ()]
    )

def _conc_2i_ic_hf(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve_app'], p['Ktrans'],
        t=t, dt=dt, sum=sum,
        extracellular=['pass', ()],
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_2i_ic_hf__e(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve_app'], p['Ktrans'],
        t=t, dt=dt, sum=sum,
        extracellular=['pass', ()],
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_2i_ic_hf__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve_app'], Ktrans,
        t=t, dt=dt, sum=sum,
        extracellular=['pass', ()],
        hepatocytes=["comp", (p['Th'],)]
    )

def _conc_2i_ic_hf__ue(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Ktrans = _interp_params(ca, t, dt, [p['Ktrans_i'], p['Ktrans_f']])
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve_app'], Ktrans,
        t=t, dt=dt, sum=sum,
        extracellular=['pass', ()],
        hepatocytes=["nscomp", (Th,)]
    )

def _conc_2i_ic(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Te = p['ve'] / (p['Fp'] + p['khe'])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], p['khe'], t=t, dt=dt, sum=sum,
        extracellular=['comp', (Te,)],
        hepatocytes=["comp", (p['Th'],)],
    )

def _conc_2i_ic__e(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Te = p['ve'] / (p['Fp'] + p['khe'])
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], p['khe'], t=t, dt=dt, sum=sum,
        extracellular=['comp', (Te,)],
        hepatocytes=["nscomp", (Th,)],
    )

def _conc_2i_ic__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    khe = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    Te = p['ve'] / (p['Fp'] + khe)
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], khe, t=t, dt=dt, sum=sum,
        extracellular=['nscomp', (Te,)],
        hepatocytes=["comp", (p['Th'],)],
    )

def _conc_2i_ic__ue(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    khe = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    Te = p['ve'] / (p['Fp'] + khe)
    Th = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], khe, t=t, dt=dt, sum=sum,
        extracellular=['nscomp', (Te,)],
        hepatocytes=["nscomp", (Th,)],
    )

def _conc_2i_ic_u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    Te = p['ve'] / (p['Fp'] + p['khe'])
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], p['khe'],
        t=t, dt=dt, sum=sum,
        extracellular=['comp', (Te,)],
        hepatocytes=['trap', ()]
    )

def _conc_2i_ic_u__u(ci, t=None, dt=1.0, sum=True, **p):
    ca, cv = ci
    khe = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    Te = p['ve'] / (p['Fp'] + khe)
    return _conc_liver_2i_ic(
        ca, cv, p['Ta'], p['fa'], p['ve'], khe,
        t=t, dt=dt, sum=sum,
        extracellular=['nscomp', (Te,)],
        hepatocytes=['trap', ()]
    )


def _interp_params(ca: np.ndarray, t: Optional[np.ndarray], dt: float, p):
    tarr = utils.tarray(np.size(ca), t=t, dt=dt)
    return utils.interp(p, tarr)



def _conc_liver_2i_ec(
    ca: np.ndarray,
    cv: np.ndarray,
    Ta: float,
    af: float,
    Fp: float,
    ve: float,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute liver tissue concentration for an extracellular agent with a dual inlet.

    Parameters
    ----------
    ca : np.ndarray
        Concentration in arterial plasma.
    cv : np.ndarray
        Concentration in venous plasma.
    Ta : float
        Arterial transit time.
    af : float
        Arterial flow fraction.
    Fp : float
        Plasma flow.
    ve : float
        Extracellular volume fraction.
    t : np.ndarray, optional
        Array of time points. Defaults to None.
    dt : float, optional
        Time step for uniform sampling. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        Liver tissue concentration.
    """
    # Propagate arterial input through the arterial tree
    ca_propagated = pk.flux(ca, Ta, t=t, dt=dt, model='plug')
    
    # Determine combined inlet concentration (arterial + venous)
    cp = af * ca_propagated + (1 - af) * cv
    
    # Tissue concentration in the extracellular space
    Te = ve / Fp
    Ce = pk.conc_comp(Fp * cp, Te, t=t, dt=dt)
    
    return Ce


def _conc_liver_1i_ic(
    ca: np.ndarray,
    ve_app: float,
    Ktrans: float,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
    sum: bool = True,
    extracellular: Optional[List[Union[str, Tuple]]] = None,
    hepatocytes: Optional[List[Union[str, Tuple]]] = None
) -> np.ndarray:
    """
    Compute liver tissue concentration for an intracellular agent with hepatocyte uptake.

    Parameters
    ----------
    ca : np.ndarray
        Concentration in arterial plasma.
    ve_app : float
        Apparent extracellular volume fraction.
    Ktrans : float
        Hepatocyte transfer constant.
    t : np.ndarray, optional
        Array of time points. Defaults to None.
    dt : float, optional
        Time step for uniform sampling. Defaults to 1.0.
    sum : bool, optional
        If True, return total tissue concentration (Ce + Ch). 
        If False, return a stack of [Ce, Ch]. Defaults to True.
    extracellular : list, optional
        Extracellular model specification: [model_name, params]. Defaults to ['pfcomp', (30, 0.85)].
    hepatocytes : list, optional
        Hepatocyte model specification: [model_name, params]. Defaults to ['comp', (1800,)].

    Returns
    -------
    np.ndarray
        Liver tissue concentration. Either total (Ce + Ch) or stacked (Ce, Ch) depending on `sum`.
    """

    # Propagate through the extracellular space
    ca_prop = pk.flux(ca, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    
    # Tissue concentration in the extracellular space
    Ce = ve_app * ca_prop
    
    # Tissue concentration in the hepatocytes
    Ch = pk.conc(Ktrans * ca_prop, *hepatocytes[1], t=t, dt=dt, model=hepatocytes[0])
    
    if sum:
        return Ce + Ch
    else:
        return np.stack((Ce, Ch))


def _conc_liver_2i_ic(
    ca: np.ndarray,
    cv: np.ndarray,
    Ta: float,
    af: float,
    ve_app: float,
    Ktrans: float,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
    sum: bool = True,
    extracellular: Optional[List[Union[str, Tuple]]] = None,
    hepatocytes: Optional[List[Union[str, Tuple]]] = None
) -> np.ndarray:
    """
    Compute liver tissue concentration for an extracellular agent with hepatocyte uptake 
    and dual inlet (arterial + venous).

    Parameters
    ----------
    ca : np.ndarray
        Concentration in arterial plasma.
    cv : np.ndarray
        Concentration in venous plasma.
    Ta : float
        Arterial transit time.
    af : float
        Arterial flow fraction.
    ve_app : float
        Apparent extracellular volume fraction.
    Ktrans : float
        Hepatocyte transfer constant.
    t : np.ndarray, optional
        Array of time points. Defaults to None.
    dt : float, optional
        Time step for uniform sampling. Defaults to 1.0.
    sum : bool, optional
        If True, return total tissue concentration (Ce + Ch). 
        If False, return a stack of [Ce, Ch]. Defaults to True.
    extracellular : list, optional
        Extracellular model specification: [model_name, params]. Defaults to ['pfcomp', (30, 0.85)].
    hepatocytes : list, optional
        Hepatocyte model specification: [model_name, params]. Defaults to ['comp', (1800,)].

    Returns
    -------
    np.ndarray
        Liver tissue concentration. Either total (Ce + Ch) or stacked (Ce, Ch) depending on `sum`.
    """

    # Propagate through arterial tree
    ca_propagated = pk.flux(ca, Ta, t=t, dt=dt, model='plug')

    # Determine combined inlet concentration (arterial + venous)
    cp = af * ca_propagated + (1 - af) * cv

    # Propagate through the extracellular space
    ca_ext = pk.flux(cp, *extracellular[1], t=t, dt=dt, model=extracellular[0])

    # Tissue concentration in the extracellular space
    Ce = ve_app * ca_ext

    # Tissue concentration in the hepatocytes
    Ch = pk.conc(Ktrans * ca_ext, *hepatocytes[1], t=t, dt=dt, model=hepatocytes[0])

    if sum:
        return Ce + Ch
    else:
        return np.stack((Ce, Ch))
