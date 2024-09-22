import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils


def conc_liver(ca, *params, t=None, dt=1.0, kinetics='EC', sum=True, cv=None):
    """Concentration in liver tissue.

    Args:
        ca (array-like): concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        kinetics (str, optional): Kinetics of the tissue, either 'EC', 'IC', 'ICNSU' or 'ICNS'- see below for detail. Defaults to 'EC'. 
        sum (bool, optional): For two-compartment tissues, set to True to return the total tissue concentration. Defaults to True.
        cv (array-like, optional): portal venous concentration for dual-inlet models. Defaults to None.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where *k* is the number of time points in *ca*. The concentration is returned in units of M.

    Notes:
        Currently implemented kinetic models are: 
        
        - 'EC': model for extracellular contrast agent. params = (ve, Te, De,) if cv=None (single-inlet model) and params = (Ta, af, Fp, ve,) otherwise (dual-inlet model).
        - 'IC': model for intracellular contrast agent. params = (ve, Te, De, khe, Th, ) if cv=None (single-inlet model and params = (Ta, af, Fp, ve, khe, Th,) otherwise (dual-inlet model). 
        - 'ICNS': non-stationary model for intracellular contrast agent. params = (ve, Te, De, khe, Th, ) if cv=None (single-inlet model and params = (Ta, af, Fp, ve, khe, Th,) otherwise (dual-inlet model). In this case khe and Th are arrays with 2 or more values.
        - 'ICNSU': non-stationary uptake model for intracellular contrast agent. params = (ve, Te, De, khe, Th, ) if cv=None (single-inlet model and params = (Ta, af, Fp, ve, khe, Th,) otherwise (dual-inlet model). In this case khe is an array with 2 or more values.

        The model parameters are:

        - **Ta** (float, sec): Arterial transit time
        - **af** (float): Arterial flow fraction
        - **Fp** (float, mL/sec/mL): Plasma flow.
        - **ve** (float, mL/mL): liver extracellular volume fraction.
        - **De** (float): Extracellular dispersion in the range [0,1].
        - **Te** (float, sec): Extracellular mean transit time.
        - **khe** (float, mL/sec/mL): Intracellular uptake rate.
        - **Th** (float, sec): hepatocyte mean transit time.

    Example:

        Plot concentration in cortex and medulla for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 30*60, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Define some parameters and generate plasma and tubular tissue concentrations with a non-stationary model:

        >>> ve, Te, De, khe, Th = 0.2, 30, 0.5, [0.003, 0.01], [180, 600]
        >>> C = dc.conc_liver(ca, ve, Te, De, khe, Th, t=t, sum=False, kinetics='ICNS')

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Kidney concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='--', linewidth=3.0, color='darkred', label='Extracellular')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkblue', label='Hepatocytes')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-', linewidth=3.0, color='grey', label='Whole liver')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    if kinetics=='EC':
        if cv is None:
            ve, Te, De = params
            cp = ca
            return _conc_liver(cp, ve, extracellular=['pfcomp', (Te, De)], t=t, dt=dt)
        else:
            Ta, af, Fp, ve  = params
            return _conc_liverav(ca, cv, Ta, af, Fp, ve, t=t, dt=dt)
        
    elif kinetics=='IC':
        if cv is None:
            ve, Te, De, khe, Th = params
            return _conc_liver_hep(ca, ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['comp', (Th,)])
        else:
            Ta, af, ve, Te, De, khe, Th = params
            return _conc_liverav_hep(ca, cv, 
                Ta, af, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['comp', (Th,)])
        
    elif kinetics=='IC-HF':
        if cv is None:
            ve, khe, Th = params
            return _conc_liver_hep(ca, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pass', ()],
                hepatocytes = ['comp', (Th,)])
        else:
            Ta, af, ve, khe, Th = params
            return _conc_liverav_hep(ca, cv, 
                Ta, af, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pass', ()],
                hepatocytes = ['nscomp', (Th,)])
        
    elif kinetics=='ICNSU':
        tarr = utils.tarray(np.size(ca), t=t, dt=dt)
        if cv is None:
            ve, Te, De, khe, Th = params
            if np.size(khe) != np.size(tarr):
                khe = utils.interp(khe, tarr)
            return _conc_liver_hep(ca, ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['comp', (Th,)])
        else:
            Ta, af, ve, Te, De, khe, Th = params
            if np.size(khe) != np.size(tarr):
                khe = utils.interp(khe, tarr)
            return _conc_liverav_hep(ca, cv, 
                Ta, af, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['nscomp', (Th,)])
        
    elif kinetics=='ICNS':
        tarr = utils.tarray(np.size(ca), t=t, dt=dt)
        if cv is None:
            ve, Te, De, khe, Th = params
            if np.size(khe) != np.size(tarr):
                khe = utils.interp(khe, tarr)
            if np.size(Th) != np.size(tarr):
                Th = utils.interp(Th, tarr)
            return _conc_liver_hep(ca, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['nscomp', (Th,)])
        else:
            Ta, af, ve, Te, De, khe, Th = params
            if np.size(khe) != np.size(tarr):
                khe = utils.interp(khe, tarr)
            if np.size(Th) != np.size(tarr):
                Th = utils.interp(Th, tarr)
            return _conc_liverav_hep(ca, cv, 
                Ta, af, 
                ve, khe, 
                t=t, dt=dt, sum=sum,
                extracellular = ['pfcomp', (Te, De)],
                hepatocytes = ['nscomp', (Th,)])
    else:
        raise ValueError('Kinetic model ' + kinetics + ' is not currently implemented.')


def _conc_liver(ca, ve, 
        extracellular = ['pfcomp', (30, 0.85)],
        t=None, dt=1.0):

    """Single-inlet liver model modelling the extracellular space (gut and liver) as a plug-flow compartment.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Te (float, sec): mean transit time of the extracellular space.
        De (float): Transit time dispersion of the extracellular space, in the range [0,1].
        ve (float): volume faction of the extracellular space.
        t (array_like, sec, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, sec, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    Example:

        Plot concentration in the liver for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 600, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        Use the function to generate liver tissue concentrations:

        >>> C = dc._conc_liver(ca, 0.3, extracellular=['pfcomp', (20, 0.5)], t=t)

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Liver concentrations')
        >>> ax.plot(t/60, 1000*C, linestyle='-', linewidth=3.0, color='darkviolet', label='Whole liver')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """
    # Propagate through the extracellular space
    ce = pk.flux(ca, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    # Tissue concentration in the extracellular space
    Ce = ve*ce
    return Ce


def _conc_liverav(ca, cv, Ta:float, af, Fp, ve, t=None, dt=1.0):
    """Dual-inlet liver model for extracellular agents."""

    # Propagate through arterial tree
    ca = pk.flux(ca, Ta, t=t, dt=dt, model='plug')
    # Determine inlet concentration
    cp = af*ca + (1-af)*cv
    # Tissue concentration in the extracellular space
    Te = ve/Fp
    Ce = pk.conc_comp(Fp*cp, Te, t=t, dt=dt)
    return Ce


def _conc_liver_hep(ca, ve, khe, t=None, dt=1.0, sum=True,
        extracellular = ['pfcomp', (30, 0.85)],
        hepatocytes = ['comp', (30*60,)]):

    """Single-inlet liver model modelling the extracellular space (gut and liver) as a plug-flow compartment and the hepatocytes as a non-stationary compartment.

    Args:
        ca (array_like): the indicator concentration in the plasma of the feeding artery, as a 1D array, in units of M.
        Te (float, sec): mean transit time of the extracellular space.
        De (float): Transit time dispersion of the extracellular space, in the range [0,1].
        ve (float): volume faction of the extracellular space.
        khe (array-like, mL/sec/mL): array of rate constants for indicator transport from extracellular space to hepatocytes. 
        Th (array-like, sec): array of mean transit times of the hepatocytes.
        t (array_like, sec, optional): the time points in sec of the input function *ca*. If *t* is not provided, the time points are assumed to be uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, sec, optional): spacing in seconds between time points for uniformly spaced time points. This parameter is ignored if *t* is explicity provided. Defaults to 1.0.
        sum (bool, optional): if set to True, the total concentration is returned. If set to False, the concentration in the compartments is returned separately. Defaults to True.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total concentration at each time point. If sum=False this is the concentration in each compartment, and at each time point, as a 2D array with dimensions *(2,k)*, where 2 is the number of compartments and *k* is the number of time points in *ca*. The concentration is returned in units of M.

    See Also:
        `liver_conc_pcc`

    Example:

        Plot concentration in the liver for typical values:

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import dcmri as dc

        Generate a population-average input function:

        >>> t = np.arange(0, 1200, 1.5)
        >>> ca = dc.aif_parker(t, BAT=20)

        In this case we can allow the function to change over the duration of the scan. Lets assume that the uptake and excretion functions are both inhibited at the start of the scan, and recover to normal values towards the end:

        >>> khe = dc.interp([0.01, 0.2], t)
        >>> Th = dc.interp([1800, 300], t)

        Use the function to generate liver tissue concentrations:

        >>> C = dc._conc_liver_hep(ca, 0.3, khe, t, sum=False,
        ...     extracellular = ['pfcomp', (20, 0.5)],
        ...     hepatocytes = ['nscomp', Th],
        ... )

        Plot all concentrations:

        >>> fig, ax = plt.subplots(1,1,figsize=(6,5))
        >>> ax.set_title('Liver concentrations')
        >>> ax.plot(t/60, 1000*C[0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Extracellular')
        >>> ax.plot(t/60, 1000*C[1,:], linestyle='--', linewidth=3.0, color='darkgreen', label='Hepatocytes')
        >>> ax.plot(t/60, 1000*(C[0,:]+C[1,:]), linestyle='-.', linewidth=3.0, color='darkviolet', label='Whole liver')
        >>> ax.set_xlabel('Time (min)')
        >>> ax.set_ylabel('Tissue concentration (mM)')
        >>> ax.legend()
        >>> plt.show()
    """

    # Propagate through the extracellular space
    ce = pk.flux(ca, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    # Tissue concentration in the extracellular space
    Ce = ve*ce
    # Tissue concentration in the hepatocytes
    Ch = pk.conc(khe*ce, *hepatocytes[1], t=t, dt=dt, model=hepatocytes[0])
    if sum:
        return Ce+Ch
    else:
        return np.stack((Ce,Ch))
    

def _conc_liverav_hep(ca, cv, Ta, af, ve, khe, t=None, dt=1.0, sum=True,
        extracellular = ['pfcomp', (30, 0.85)],
        hepatocytes = ['comp', (30*60,)]):

    # Propagate through arterial tree
    ca = pk.flux(ca, Ta, t=t, dt=dt, model='plug')
    # Determine inlet concentration
    cp = af*ca + (1-af)*cv
    # Propagate through the extracellular space
    ce = pk.flux(cp, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    # Tissue concentration in the extracellular space
    Ce = ve*ce
    # Tissue concentration in the hepatocytes
    Ch = pk.conc(khe*ce, *hepatocytes[1], t=t, dt=dt, model=hepatocytes[0])
    if sum:
        return Ce+Ch
    else:
        return np.stack((Ce,Ch))