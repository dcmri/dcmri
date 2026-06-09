import copy
from typing import Optional

import numpy as np

import dcmri.kinetics.lib.blocks as pk
from dcmri.utils.misc import tarray


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
    Fb : float
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
    >>> Fb = 0.01
    >>> dc.conc_tissue_u(ca, t, Fb=Fb)
    array([[0.   , 0.075, 0.325, 0.775, 1.525]])
    """

def conc_liver_2i_ec(ci, t=None, dt=1.0, T_a=None, fa=None, 
                     ve=None, Fp=None):
    ca, cv = ci
    Te = ve / Fp
    return _conc_liver(
        ca, cv=cv, ve_app=ve, Ta=T_a, fa=fa, Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ec_hf(ci, t=None, dt=1.0, T_a=None, fa=None, 
                        ve=None):
    ca, cv = ci
    return _conc_liver(
        ca, cv=cv, ve_app=ve, Ta=T_a, fa=fa, t=t, dt=dt, 
    )

def conc_liver_1i_ec(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                     ve=None, Fp=None):
    Te = ve / Fp
    return _conc_liver(
        ca, Ta=T_a, Tg=Tg, ve_app=ve, Te=Te, t=t, dt=dt, 
    )

def conc_liver_1i_ec_hf(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                     ve=None):
    return _conc_liver(
        ca, Ta=T_a, Tg=Tg, ve_app=ve, t=t, dt=dt, 
    )

# def conc_liver_1i_ec_d(ca, t=None, dt=1.0, ve=None, 
#                        Te=None, De=None):
#     return _conc_liver(ca, ve_app=ve, Te=Te, De=De, t=t, dt=dt)



def conc_liver_2i_ic(ci, t=None, dt=1.0, T_a=None, fa=None, ve=None, 
                     Fp=None, E=None, Th=None):
    ca, cv = ci
    khe = Fp * E / (1 - E)
    Te = ve / (Fp + khe)
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    return _conc_liver(
        ca, ve_app=ve_app, cv=cv, Ta=T_a, fa=fa, Ktrans=Ktrans, 
        Th=Th, Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ic_nse(ci, t=None, dt=1.0, T_a=None, fa=None, 
                         Fp=None, ve=None, E=None, 
                         Th_i=None, Th_f=None):
    ca, cv = ci
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                            E=E, T_a=T_a, fa=fa, Th=Th)

def conc_liver_2i_ic_nsu(ci, t=None, dt=1.0, T_a=None, fa=None, 
                         ve=None, Fp=None, E_i=None, E_f=None, Th=None):
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, T_a=T_a, fa=fa, 
                            ve=ve, Fp=Fp, E=E, Th=Th)

def conc_liver_2i_ic_nsue(ci, t=None, dt=1.0, T_a=None, fa=None, 
                          ve=None, Fp=None, E_i=None, E_f=None, 
                          Th_i=None, Th_f=None):
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                            E=E, T_a=T_a, fa=fa, Th=Th)



def conc_liver_2i_ic_hf(ci, t=None, dt=1.0, T_a=None, fa=None, 
                        ve=None, khe=None, Th=None):
    ca, cv = ci
    return _conc_liver(
        ca, ve_app=ve, cv=cv, Ta=T_a, fa=fa, 
        Ktrans=khe, Th=Th, t=t, dt=dt, 
    )

def conc_liver_2i_ic_hf_nse(ci, t=None, dt=1.0, T_a=None, fa=None, 
                            ve=None, khe=None, Th_i=None, Th_f=None):
    ca, _ = ci
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, T_a=T_a, fa=fa, 
                               ve=ve, khe=khe, Th=Th)

def conc_liver_2i_ic_hf_nsu(ci, t=None, dt=1.0, T_a=None, fa=None, 
                            ve=None, khe_i=None, khe_f=None, Th=None):
    ca, _ = ci
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, T_a=T_a, fa=fa, 
                               ve=ve, khe=khe, Th=Th)

def conc_liver_2i_ic_hf_nsue(ci, t=None, dt=1.0, T_a=None, fa=None, 
                             ve=None, khe_i=None, khe_f=None, 
                             Th_i=None, Th_f=None):
    ca, cv = ci
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt, T_a=T_a, fa=fa, 
                               ve=ve, khe=khe, Th=Th)



def conc_liver_2i_ic_u(ci, t=None, dt=1.0, T_a=None, fa=None, 
                       ve=None, Fp=None, E=None):
    ca, cv = ci
    khe = Fp * E / (1 - E)
    Te = ve / (Fp + khe)
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    return _conc_liver(
        ca, ve_app=ve_app, cv=cv, Ta=T_a, fa=fa, Ktrans=Ktrans, Te=Te,
        t=t, dt=dt, 
    )

def conc_liver_2i_ic_u_nsu(ci, t=None, dt=1.0, T_a=None, fa=None, 
                           Fp=None, ve=None, E_i=None, E_f=None):
    ca, cv = ci
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_2i_ic_u(ci, t=t, dt=dt, Fp=Fp, ve=ve, 
                              E=E, T_a=T_a, fa=fa)


def conc_liver_1i_ic(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                     ve=None, Fp=None, E=None, Th=None):
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

    # This is the same:
    ve_app = ve * (1 - E)
    Ktrans = Fp * E
    Te = ve_app / Fp
    return _conc_liver(
        ca, Ta=T_a, Tg=Tg, ve_app=ve_app, Ktrans=Ktrans, Th=Th, Te=Te, 
        t=t, dt=dt
    )

def conc_liver_1i_ic_nsu(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                         ve=None, Fp=None, E_i=None, E_f=None, 
                         Th=None):
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, Fp=Fp, 
                            ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_nse(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                         ve=None, Fp=None, E=None, Th_i=None, Th_f=None):
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, Fp=Fp, 
                            ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_nsue(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                          ve=None, Fp=None, E_i=None, E_f=None, 
                          Th_i=None, Th_f=None):
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    E = _interp_params(ca, t, dt, [E_i, E_f])
    return conc_liver_1i_ic(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, Fp=Fp, 
                            ve=ve, E=E, Th=Th)

def conc_liver_1i_ic_hf(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                        ve=None, khe=None, Th=None):
    return _conc_liver(
        ca, Ta=T_a, Tg=Tg, ve_app=ve, Ktrans=khe, Th=Th, t=t, dt=dt,  
    )

def conc_liver_1i_ic_hf_nsu(ca, t=None, dt=1.0, T_a=None, Tg=None,
                            ve=None, khe_i=None, khe_f=None, Th=None):
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, ve=ve, khe=khe, Th=Th)

def conc_liver_1i_ic_hf_nse(ca, t=None, dt=1.0, T_a=None, Tg=None,
                            ve=None, khe=None, Th_i=None, Th_f=None):
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, ve=ve, khe=khe, Th=Th)

def conc_liver_1i_ic_hf_nsue(ca, t=None, dt=1.0, T_a=None, Tg=None, 
                             ve=None, khe_i=None, khe_f=None, Th_i=None, Th_f=None):
    khe = _interp_params(ca, t, dt, [khe_i, khe_f])
    Th = _interp_params(ca, t, dt, [Th_i, Th_f])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt, T_a=T_a, Tg=Tg, ve=ve, khe=khe, Th=Th)

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
    Ta: float = None,
    Tg: float = None,
    Dg: float = None,
    fa: float = None,
    ve_app: float = None, # (1 - E) ve
    Te: float = None,
    De: float = None,    
    Ktrans: float = None, # (1 - E) kep
    Th: float = None,
    t: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    
    # Propagate through arterial tree
    if Ta is not None:
        ca = pk.flux(ca, Ta, t=t, dt=dt, model='plug')

    # If a portal venous concentration is provided, use it
    if cv is not None:
        ca = fa * ca + (1 - fa) * cv

    # Otherwise see if it can be derived by propagation through the gut
    elif Tg is not None:
        if Dg is None:
            ca = pk.flux_comp(ca, Tg, t=t, dt=dt)
        # else: # No case for this
        #     ca = pk.flux_pfcomp(ca, Tg, Dg, t=t, dt=dt)

    # Propagate through the extracellular space
    if Te is None:
        ec_model, ec_pars = 'pass', ()
    elif De is None:
        if np.isscalar(Te):
            ec_model, ec_pars = 'comp', (Te,)
        else:
            ec_model, ec_pars = 'nscomp', (Te,)
    # else:
    #     ec_model, ec_pars = 'pfcomp', (Te, De,)

    ce = pk.flux(ca, *ec_pars, t=t, dt=dt, model=ec_model)

    # Tissue concentration in the extracellular space
    Ce = ve_app * ce

    # Tissue concentration in the hepatocytes
    if Th is None:
        hep_model, hep_pars = "trap", ()
    elif np.isscalar(Th):
        hep_model, hep_pars = "comp", (Th,)
    else:
        hep_model, hep_pars = "nscomp", (Th,)

    if Ktrans is None:
        Ch = np.zeros(len(ce))
    else:
        Ch = pk.conc(Ktrans * ce, *hep_pars, t=t, dt=dt, model=hep_model)

    return np.stack((Ce, Ch))
    
