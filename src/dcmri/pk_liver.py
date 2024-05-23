import numpy as np
import dcmri as dc


def kidney_conc_2cm(ca, Fp, Tp, Ft, Tt, t=None, dt=1.0, sum=True):
    """Two-compartment filtration model.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL).
        Tp (float): Plasma mean transit time (sec)
        Ft (float): Tubular flow (mL/sec/mL)
        Tt (float): Tubular mean transit time (sec)
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `kidney_conc_pf`, `kidney_conc_cm9`

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

        Use the function to generate cortex and medulla tissue concentrations:

        >>> C = dc.kidney_conc_2cm(ca, 0.05, 10, 0.01, 120, t, sum=False)

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkgreen', label='Medulla')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-.', linewidth=3.0, color='darkviolet', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    vp = Tp*(Fp+Ft)
    Cp = dc.conc_comp(Fp*ca, Tp, t=t, dt=dt)
    cp = Cp/vp
    Ct = dc.conc_comp(Ft*cp, Tt, t=t, dt=dt)   
    if sum:
        return Cp+Ct
    else:
        return np.stack((Cp,Ct))


def kidney_conc_pf(ca, 
        Fp, Tp, Ft, h, t=None, dt=1.0,
        TT = [15,30,60,90,150,300,600], sum=True):
    """Modelling the glomeruli as a plug-flow system and the nephron as a free transit time distribution with 6 steps.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL).
        Tp (float): Plasma mean transit time (sec)
        Ft (float): Tubular flow (mL/sec/mL)
        h (array-like): Array of 6 frequencies (1/sec) providing the probability of a transit time being in the bins defined by TT. The frequencies do not have to be normalized.
        TT (array-like): Array with 7 transit times (sec) determining the boundaries of bins in the transit time histogram. Defaults to [15,30,60,90,150,300,600]. 
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `kidney_conc_2cm`, `kidney_conc_cm9`

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

        Use the function to generate cortex and medulla tissue concentrations:

        >>> C = dc.kidney_conc_pf(ca, 0.05, 10, 0.01, [1,1,1,1,1,1], t, sum=False)

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkgreen', label='Medulla')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-.', linewidth=3.0, color='darkviolet', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    vp = Tp*(Fp+Ft)
    Cp = dc.conc_plug(Fp*ca, Tp, t=t, dt=dt) 
    cp = Cp/vp
    Ct = dc.conc_free(Ft*cp, h, dt=dt, TT=TT, solver='step')  
    if sum:
        return Cp+Ct
    else:
        return np.stack((Cp,Ct)) 


def kidney_conc_cm9(ca, Fp, Eg, fc, Tg, Tv, Tpt, Tlh, Tdt, Tcd, t=None, dt=1.0, sum=True):
    """Concentrations derived from a cortico-medullary model with 4 cortical compartments (glomeruli, peritubular capillaries & veins, proximal tubuli and distal tubuli) and 3 medullary compartments (peritubular capillaries & veins, list of Henle and collecting ducts). 

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Fp (float): Plasma flow into the tissue, in units of mL plasma per sec and per mL tissue (mL/sec/mL).
        Eg (float): Glomerular extraction fraction
        fc (float): Cortical flow fraction
        Tg (float): Glomerular mean transit time in sec
        Tv (float): Peritubular & venous mean transit time in sec
        Tpt (float): Proximal tubuli mean transit time in sec
        Tlh (float): Lis of Henle mean transit time in sec
        Tdt (float): Distal tubuli mean transit time in sec
        Tcd (float): Collecting duct mean transit time in sec
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: If sum=True, each return value is a 1D array with the total concentration at each time point, in cortex and medulla, respectively. If sum=False each return value is the concentration in each compartment, and at each time point, of cortex and medulla as a 2D array with dimensions *(n,k)*, where n is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

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

        Use the function to generate cortex and medulla tissue concentrations:

        >>> Cc, Cm = dc.kidney_conc_cm9(ca, 0.03, 0.15, 0.8, 4, 10, 60, 60, 30, 30, t)

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*Cc, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex')
        >>> ax.plot(t/60, 1000*Cm, linestyle='--', linewidth=3.0, color='darkgreen', label='Medulla')
        >>> ax.plot(t/60, 1000*(Cc+Cm), linestyle='-.', linewidth=3.0, color='darkviolet', label='Whole kidney')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    # Flux out of the glomeruli and arterial tree
    Jg = dc.flux_comp(Fp*ca, Tg, t=t, dt=dt)

    # Flux out of the peritubular capillaries and venous system
    Jv = dc.flux_comp((1-Eg)*Jg, Tv, t=t, dt=dt)

    # Flux out of the proximal tubuli
    Jpt = dc.flux_comp(Eg*Jg, Tpt, t=t, dt=dt)

    # Flux out of the lis of Henle
    Jlh = dc.flux_comp(Jpt, Tlh, t=t, dt=dt)

    # Flux out of the distal tubuli
    Jdt = dc.flux_comp(Jlh, Tdt, t=t, dt=dt)

    # Flux out of the collecting ducts
    Jcd = dc.flux_comp(Jdt, Tcd, t=t, dt=dt)

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
    