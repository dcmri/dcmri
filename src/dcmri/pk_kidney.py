import numpy as np
import dcmri as dc


def conc_kidney(ca:np.ndarray, *params, t=None, dt=1.0, sum=True, kinetics='2CF', **kwargs)->np.ndarray:
    """Concentration in kidney tissues.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetics of the tissue, either '2CF', 'FN' - see below for detail. Defaults to '2CF'. 
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration. Defaults to True.
        kwargs (dict, optional): any optional keyword parameters required by the kinetic model - see below for detail.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where *k* is the number of time points in *ca*. The concentration is returned in units of M.

    Notes:
        Currently implemented kinetic models are: 
        
        - '2CF': two-compartment filtration model. params = (Fp, Tp, Ft, Tt,)
        - 'FN': free nephron model. params = (Fp, Tp, Ft, h, ). 

        The model parameters are:

        - **Fp** (float, mL/sec/mL): Plasma flow.
        - **Tp** (float, sec): plasma mean transit time.
        - **Ft** (float, mL/sec/mL): tubular flow.
        - **Tt** (float, sec): tubular mean transit time.
        - **hh** (array-like, 1/sec): frequences of transit time histogram. The boundaries of the transit time bins can be provided as an array in a keyword parameter TT, which has to have one more element than h. If TT is not provided, the transit time bins are equally space in the range [0, tmax], where tmax is the largest acquisition time.
        

    Example:

        Plot concentration in cortex and medulla for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 300, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define some parameters and generate plasma and tubular tissue concentrations with a 2-compartment filtration model:

        >>> Fp, Tp, Ft, Tt = 0.05, 10, 0.01, 120
        >>> C = dc.conc_kidney(ca, Fp, Tp, Ft, Tt, t=t, sum=False, kinetics='2CF')

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Tubuli')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()

        Use generate plasma and tubular tissue concentrations using the free nephron model for comparison. We assume 4 transit time bins with the following boundaries (in units of seconds):
        
        >>> TT = [0, 15, 30, 60, 120]

        with longest transit times most likely (note the frequences to not have to add up to 1):
        
        >>> h = [1, 2, 3, 4]
        >>> C = dc.conc_kidney(ca, Fp, Tp, Ft, h, t=t, sum=False, kinetics='FN', TT=TT)

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Plasma')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Tubuli')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()

    """
    if kinetics == '2CF':
        return _conc_kidney_2cf(ca, *params, t=t, dt=dt, sum=sum)
    elif kinetics == 'FN':
        # TT = [15,30,60,90,150,300,600]
        return _conc_kidney_fn(ca, *params, t=t, dt=dt, sum=sum, **kwargs)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')


def _conc_kidney_2cf(ca, Fp, Tp, Ft, Tt, t=None, dt=1.0, sum=True):
    vp = Tp*(Fp+Ft)
    Cp = dc.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    cp = Cp/vp
    Ct = dc.conc_comp(Ft*cp, Tt, t=t, dt=dt)   
    if sum:
        return Cp+Ct
    else:
        return np.stack((Cp,Ct))


def _conc_kidney_fn(ca, Fp, Tp, Ft, h, t=None, dt=1.0, sum=True, TT=None):
    if TT is None:
        if t is None:
            tmax = dt*np.size(ca)
        else:
            tmax = np.amax(t)
        nTT = 1+np.size(h)
        TT = np.linspace(0, tmax, nTT)
    vp = Tp*(Fp+Ft)
    Cp = dc.conc_plug(Fp*ca, Tp, t=t, dt=dt) 
    cp = Cp/vp
    Ct = dc.conc_free(Ft*cp, h, dt=dt, TT=TT, solver='step')  
    if sum:
        return Cp+Ct
    else:
        return np.stack((Cp,Ct)) 


def conc_kidney_cortex_medulla(ca:np.ndarray, *params, t=None, dt=1.0, sum=True, kinetics='7C'):
    """Concentration in kidney cortex and medulla tissues.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration. Defaults to True.
        kinetics (str, optional): Kinetics of the tissue, currently only '7C' available - see below for detail. Defaults to '7F'. 
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: If sum=True, each return value is a 1D array with the total concentration at each time point, in cortex and medulla, respectively. If sum=False each return value is the concentration in each compartment, and at each time point, of cortex and medulla as a 2D array with dimensions *(n,k)*, where n is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.


    Notes:
        Currently implemented kinetic models are: 
        
        - '7CF': 7-compartment model. params = (Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd,). Cortico-medullary model with 4 cortical compartments (glomeruli, peritubular capillaries & veins, proximal tubuli and distal tubuli) and 3 medullary compartments (peritubular capillaries & veins, list of Henle and collecting ducts). 

        The 9 model parameters are:

        - **Fp** (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL).
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
        >>> ca = dc.aif_parker(t, BAT=20)

        Use the function to generate total cortex and medulla tissue concentrations:

        >>> Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd = 0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30
        >>> Cc, Cm = dc.conc_kidney_cortex_medulla(ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, t=t, kinetics='7C')

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
    if kinetics == '7C':
        return _conc_kidney_cm9(ca, *params, t=t, dt=dt, sum=sum)
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')

    

def _conc_kidney_cm9(ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, t=None, dt=1.0, sum=True):

    # Flux out of the glomeruli and arterial tree
    Jg = dc.flux(Fp*ca, Tg, t=t, dt=dt, kinetics='comp')

    # Flux out of the peritubular capillaries and venous system
    Jv = dc.flux((1-Eg)*Jg, Tv, t=t, dt=dt, kinetics='comp')

    # Flux out of the proximal tubuli
    Jpt = dc.flux(Eg*Jg, Tpt, t=t, dt=dt, kinetics='comp')

    # Flux out of the lis of Henle
    Jlh = dc.flux(Jpt, Tlh, t=t, dt=dt, kinetics='comp')

    # Flux out of the distal tubuli
    Jdt = dc.flux(Jlh, Tdt, t=t, dt=dt, kinetics='comp')

    # Flux out of the collecting ducts
    Jcd = dc.flux(Jdt, Tcd, t=t, dt=dt, kinetics='comp')

    # Build cortical concentrations
    Cg = Tg*Jg      # arteries/glomeruli
    Cv = fc*Tv*Jv   # part of the peritubular capillaries
    Cpt = Tpt*Jpt   # proximal tubuli 
    Cdt = Tdt*Jdt   # distal tubuli 
    if sum:
        Ccor = Cg + Cv + Cpt + Cdt
    else:
        Ccor = np.stack((Cg,Cv,Cpt,Cdt))

    # Build medullary concentrations 
    Cv = (1-fc)*Tv*Jv   # part of the peritubular capillaries
    Clh = Tlh*Jlh       # Lis of Henle
    Ccd = Tcd*Jcd       # collecting ducts
    if sum:
        Cmed = Cv + Clh + Ccd
    else:
        Cmed = np.stack((Cv,Clh,Ccd))

    return Ccor, Cmed 
    