import copy
from typing import Optional

import numpy as np

import dcmri.kinetics.lib.blocks as pk
from dcmri.utils.misc import tarray, interp


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

    if {'khe', 'vol_l'}.issubset(p):
        p['CL'] = p['khe'] * p['vol_l']

    return p
    





# --- Liver kinetic models ---

def conc_liver_1i_ec_d(ca, t=None, dt=1.0, **p):
    return _conc_liver(
        ca, p['ve'], Te=p['Te'], De=p['De'], t=t, dt=dt, 
    )

def conc_liver_1i_ec(ca, t=None, dt=1.0, **p):
    Te = p['ve'] / p['Fp']
    return _conc_liver(
        ca, p['ve'], Ta=p['T_a'], Tg=p['Tg'],
        fa=p['fa'], Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ec_hf(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    return _conc_liver(
        ca, p['ve'], Ta=p['T_a'], cv=cv,
        fa=p['fa'], t=t, dt=dt, 
    )

def conc_liver_2i_ec(ci, t=None, dt=1.0, **p):
    ca, cv = ci
    Te = p['ve'] / p['Fp']
    return _conc_liver(
        ca, p['ve'], cv=cv, Ta=p['T_a'], 
        fa=p['fa'], Te=Te, t=t, dt=dt, 
    )


def conc_liver_1i_ic(ca, t=None, dt=1.0,  **p):
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
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    Te = ve_app / p['Fp']
    return _conc_liver(
        ca, ve_app, Ktrans=Ktrans, Th=p['Th'], Te=Te, 
        t=t, dt=dt
    )

def conc_liver_1i_ic__u(ca, t=None, dt=1.0,  **p):
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return conc_liver_1i_ic(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic__e(ca, t=None, dt=1.0,  **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_1i_ic(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic__ue(ca, t=None, dt=1.0,  **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return conc_liver_1i_ic(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hf(ca, t=None, dt=1.0,  **p):
    return _conc_liver( # approx 1 - E = 1
        ca, p['ve'], Ktrans=p['khe'], Th=p['Th'], 
        t=t, dt=dt,  
    )

def conc_liver_1i_ic_hf__u(ca, t=None, dt=1.0,  **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hf__e(ca, t=None, dt=1.0,  **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hf__ue(ca, t=None, dt=1.0,  **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_1i_ic_hf(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hfd(ca, t=None, dt=1.0,  **p):
    return _conc_liver( # approx 1 - E = 1
        ca, p['ve'], Ktrans=p['khe'], Th=p['Th'],
        Tg=p['Tg'], Dg=p['Dg'], t=t, dt=dt, 
    )

def conc_liver_1i_ic_hfd__u(ca, t=None, dt=1.0,  **p):
    ti = tarray(np.size(ca), t=t, dt=dt)
    p['khe'] = p['khe_i'] + (p['khe_i'] - p['khe_i']) * ti / ti.max()
    # p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_i']])
    return conc_liver_1i_ic_hfd(ca, t=t, dt=dt, **p)

def conc_liver_1i_ic_hfd__e(ca, t=None, dt=1.0,  **p):
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_1i_ic_hfd(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hfd__ue(ca, t=None, dt=1.0,  **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_1i_ic_hfd(ca, t=t, dt=dt,  **p)

def conc_liver_1i_ic_hfdu(ca, t=None, dt=1.0,  **p):
    return _conc_liver(
        ca, p['ve'], Ktrans=p['khe'], Tg=p['Tg'], Dg=p['Dg'], 
        t=t, dt=dt, 
    )

def conc_liver_1i_ic_hfdu__u(ca, t=None, dt=1.0,  **p):
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return conc_liver_1i_ic_hfdu(ca, t=t, dt=dt,  **p)

def conc_liver_2i_ic_hf(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    return _conc_liver(
        ca, p['ve'], cv=cv, Ta=p['T_a'], fa=p['fa'], 
        Ktrans=p['khe'], Th=p['Th'], t=t, dt=dt, 
    )

def conc_liver_2i_ic_hf__e(ci, t=None, dt=1.0,  **p):
    ca, _ = ci
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic_hf__u(ci, t=None, dt=1.0,  **p):
    ca, _ = ci
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic_hf__ue(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    p['khe'] = _interp_params(ca, t, dt, [p['khe_i'], p['khe_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_2i_ic_hf(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    khe = p['Fp'] * p['E'] / (1 - p['E'])
    Te = p['ve'] / (p['Fp'] + khe)
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    return _conc_liver(
        ca, ve_app, cv=cv, Ta=p['T_a'], fa=p['fa'], Ktrans=Ktrans, 
        Th=p['Th'], Te=Te, t=t, dt=dt, 
    )

def conc_liver_2i_ic__e(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_2i_ic(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic__u(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return conc_liver_2i_ic(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic__ue(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    p['Th'] = _interp_params(ca, t, dt, [p['Th_i'], p['Th_f']])
    return conc_liver_2i_ic(ci, t=t, dt=dt,  **p)

def conc_liver_2i_ic_u(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    khe = p['Fp'] * p['E'] / (1 - p['E'])
    Te = p['ve'] / (p['Fp'] + khe)
    ve_app = p['ve'] * (1 - p['E'])
    Ktrans = p['Fp'] * p['E']
    return _conc_liver(
        ca, ve_app, cv=cv, Ta=p['T_a'], fa=p['fa'], Ktrans=Ktrans, Te=Te,
        t=t, dt=dt, 
    )

def conc_liver_2i_ic_u__u(ci, t=None, dt=1.0,  **p):
    ca, cv = ci
    p['E'] = _interp_params(ca, t, dt, [p['E_i'], p['E_f']])
    return conc_liver_2i_ic_u(ci, t=t, dt=dt,  **p)


def _interp_params(ca: np.ndarray, t: Optional[np.ndarray], dt: float, p, lower_t=False):
    tarr = tarray(np.size(ca), t=t, dt=dt)
    lower = None
    # if lower_t:
    #     lower = tarr[1] - tarr[0]
    # else:
    #     lower = None
    return interp(p, tarr, lower=lower)



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

    return np.stack((Ce, Ch))
    
