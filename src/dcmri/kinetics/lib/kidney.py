import copy
import numpy as np

import dcmri.kinetics.lib as pk


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

    if {'Ft', 'Fp'}.issubset(p):
        p['Eg'] = _div(p['Ft'], p['Ft'] + p['Fp'])
        p['FF'] = _div(p['Ft'], p['Fp'])

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



def conc_kidney_2cf(ca, t=None, dt=1.0, Fp=None, vp=None, Ft=None, Tt=None):
    Tp = vp/(Fp+Ft)
    Cp = pk.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    cp = Cp/vp
    Ct = pk.conc_comp(Ft*cp, Tt, t=t, dt=dt)
    return np.stack((Cp, Ct))

def conc_kidney_hf(ca, t=None, dt=1.0, vp=None, Ft=None, Tt=None):
    Cp = vp*ca
    Ct = pk.conc_comp(Ft*ca, Tt, t=t, dt=dt)
    return np.stack((Cp, Ct))

def conc_kidney_fn(ca, t=None, dt=1.0, TT=None, Fp=None, Tp=None, Ft=None, ht=None):
    if TT is None:
        if t is None:
            tmax = dt*np.size(ca)
        else:
            tmax = np.amax(t)
        nTT = 1 + np.size(ht)
        TT = np.linspace(0, tmax, nTT)
    vp = Tp*(Fp+Ft)
    Cp = pk.conc_plug(Fp*ca, Tp, t=t, dt=dt)
    cp = Cp/vp
    Ct = pk.conc_free(Ft*cp, ht, dt=dt, TT=TT, solver='step')
    return np.stack((Cp, Ct))



def conc_kidney_cm9(ca, t=None, dt=1.0, Fp=None, Eg=None, fc=None, Tglom=None, Tv=None, Tpt=None, Tlh=None, Tdt=None, Tcd=None):
    """Concentration in kidney cortex and medulla tissues.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input 
          function *ca*. If *t* is not provided, the time points are assumed 
          to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          explicity provided. Defaults to 1.0.
        sum (bool, optional): For two-compartment tissues, set to True to 
          return the total tissue concentration. Defaults to True.
        kinetics (str, optional): Kinetics of the tissue, currently only '7C' 
          available - see below for detail. Defaults to '7F'. 

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: If sum=True, each return value 
        is a 1D array with the total concentration at each time point, in 
        cortex and medulla, respectively. If sum=False each return value is 
        the concentration in each compartment, and at each time point, of 
        cortex and medulla as a 2D array with dimensions *(n,k)*, where n is 
        the number of compartments and *k* is the number of time points in 
        *ca*. The concentration is returned in units of M.


    Notes:
        Currently implemented kinetic models are: 

        - '7CF': 7-compartment model. 
          params = (Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd,). 
          Cortico-medullary model with 4 cortical compartments (glomeruli, 
          peritubular capillaries & veins, proximal tubuli and distal tubuli) 
          and 3 medullary compartments (peritubular capillaries & veins, 
          list of Henle and collecting ducts). 

        The 9 model parameters are:

        - **Fp** (float): Plasma flow into the tissue, in units of mL 
          plasma per sec and per mL tissue (mL/sec/mL).
        - **Eg** (float): Glomerular extraction fraction
        - **fc** (float): Cortical flow fraction
        - **Tg** (float): Glomerular mean transit time in sec
        - **Tv** (float): Peritubular & venous mean transit time in sec
        - **Tpt** (float): Proximal tubuli mean transit time in sec
        - **Tlh** (float): Lis of Henle mean transit time in sec
        - **Tdt** (float): Distal tubuli mean transit time in sec
        - **Tcd** (float): Collecting duct mean transit time in sec

    Example:

        Plot concentration in cortex and medulla for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif.parker(t, BAT=20)

        Use the function to generate total cortex and medulla tissue concentrations:

        >>> Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd = 0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30
        >>> Cc, Cm = dc.conc_kidney_cm(ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, t=t, kinetics='7C')

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*Cc, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
        >>> ax.plot(t/60, 1000*Cm, linestyle='-', linewidth=3.0, color='darkgreen', label='Medulla')
        >>> ax.plot(t/60, 1000*(Cc+Cm), linestyle='-', linewidth=3.0, color='darkgrey', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    # Flux out of the glomeruli and arterial tree
    Jg = pk.flux(Fp*ca, Tglom, t=t, dt=dt, model='comp')

    # Flux out of the peritubular capillaries and venous system
    Jv = pk.flux((1-Eg)*Jg, Tv, t=t, dt=dt, model='comp')

    # Flux out of the proximal tubuli
    Jpt = pk.flux(Eg*Jg, Tpt, t=t, dt=dt, model='comp')

    # Flux out of the lis of Henle
    Jlh = pk.flux(Jpt, Tlh, t=t, dt=dt, model='comp')

    # Flux out of the distal tubuli
    Jdt = pk.flux(Jlh, Tdt, t=t, dt=dt, model='comp')

    # Flux out of the collecting ducts
    Jcd = pk.flux(Jdt, Tcd, t=t, dt=dt, model='comp')

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



# def _deriv_params(p):

#     # Kidneys
#     if 'FF' not in p:
#         p['FF'] = _div(p['Eb'], 1-p['Eb'])
#     if {'RPF', 'FF'}.issubset(p):   
#         p['GFR'] =  p['RPF'] * p['FF']
#     if {'DRPF', 'RPF'}.issubset(p): 
#         p['RPF_lk'] = p['DRPF'] * p['RPF']
#         p['RPF_rk'] = (1 - p['DRPF']) * p['RPF']
#     if {'DRF', 'GFR'}.issubset(p):
#         p['GFR_lk'] = p['DRF'] * p['GFR']
#         p['GFR_rk'] = (1 - p['DRF']) * p['GFR']

#     # Kidney LK
#     if {'RPF_lk', 'vol_lk'}.issubset(p):
#         p['Fp_lk'] = _div(p['RPF_lk'], p['vol_lk'])
#     if {'RPF_lk', 'GFR_lk', 'vp_lk', 'vol_lk'}.issubset(p):
#         p['Tp_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk']+p['GFR_lk'])
#     if {'RPF_lk', 'vp_lk', 'vol_lk'}.issubset(p):
#         p['Tv_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk'])
#     if {'GFR_lk', 'vol_lk'}.issubset(p):
#         p['Ft_lk'] = _div(p['GFR_lk'], p['vol_lk'])
#     if {'GFR_lk', 'RPF_lk'}.issubset(p):
#         p['FF_lk'] = _div(p['GFR_lk'], p['RPF_lk'])
#         p['E_lk'] = _div(p['GFR_lk'], p['GFR_lk']+p['RPF_lk'])

#     # Kidney RK
#     if {'RPF_rk', 'vol_rk'}.issubset(p):
#         p['Fp_rk'] = _div(p['RPF_rk'], p['vol_rk'])
#     if {'RPF_rk', 'GFR_rk', 'vp_rk', 'vol_rk'}.issubset(p):
#         p['Tp_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk']+p['GFR_rk'])
#     if {'RPF_rk', 'vp_rk', 'vol_rk'}.issubset(p):
#         p['Tv_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk'])
#     if {'GFR_rk', 'vol_rk'}.issubset(p):
#         p['Ft_rk'] = _div(p['GFR_rk'], p['vol_rk'])
#     if {'GFR_rk', 'RPF_rk'}.issubset(p):
#         p['FF_rk'] = _div(p['GFR_rk'], p['RPF_rk'])
#         p['E_rk'] = _div(p['GFR_rk'], p['GFR_rk']+p['RPF_rk'])

#     return p


# def _div(a, b):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, 0, np.divide(a, b))
