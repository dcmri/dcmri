import numpy as np
import dcmri.pk as pk
import dcmri.utils as utils

def params_liver(kinetics='2I-EC', stationary='UE') -> list:
    """Parameters characterizing a liver tissue. 
    For more detail see :ref:`liver-tissues`.

    Args:
        kinetics (str, optional): Tracer-kinetic regime. Defaults to '2C-EC'.
        stationary (bool, optional): Stationary kinetics or not. 
          Defaults to False.

    Returns: 
        list: liver parameters

    Raises:
        ValueError: if the configuration is not recognized.

    Example:

        Print the parameters of a liver tissue:

        >>> import dcmri as dc
        >>> dc.params_liver('2I-EC', True)
        ['ve', 'Fp', 'fa', 'Ta']
    """

    if kinetics == '2I-EC':
        return ['H', 've', 'Fp', 'fa', 'Ta']
    if kinetics == '2I-EC-HF':
        return ['H', 've', 'fa', 'Ta']
    if kinetics == '1I-EC':
        return ['H', 've', 'Fp', 'fa', 'Ta', 'Tg']
    if kinetics == '1I-EC-D':
        return ['H', 've', 'Te', 'De']
    if kinetics == '2I-IC':
        if stationary == 'UE':
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe', 'Th']
        elif stationary == 'E':
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th']
        elif stationary == 'U':
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe', 'Th_i', 'Th_f']
        elif stationary is None:
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
    if kinetics == '2I-IC-U':
        if stationary == 'U':
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe']
        elif stationary == None:
            return ['H', 've', 'Fp', 'fa', 'Ta', 'khe_i', 'khe_f']
    if kinetics == '2I-IC-HF':
        if stationary == 'UE':
            return ['H', 've', 'fa', 'Ta', 'khe', 'Th']
        elif stationary == 'E':
            return ['H', 've', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th']
        elif stationary == 'U':
            return ['H', 've', 'fa', 'Ta', 'khe', 'Th_i', 'Th_f']
        elif stationary is None:
            return ['H', 've', 'fa', 'Ta', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
    if kinetics == '1I-IC-HF':
        if stationary == 'UE':
            return ['H', 've', 'khe', 'Th']
        elif stationary == 'E':
            return ['H', 've', 'khe_i', 'khe_f', 'Th']
        elif stationary == 'U':
            return ['H', 've', 'khe', 'Th_i', 'Th_f']
        elif stationary is None:
            return ['H', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
    if kinetics == '1I-IC-D':
        if stationary == 'UE':
            return ['H', 've', 'Te', 'De', 'khe', 'Th']
        elif stationary == 'E':
            return ['H', 've', 'Te', 'De', 'khe_i', 'khe_f', 'Th']
        elif stationary == 'U':
            return ['H', 've', 'Te', 'De', 'khe', 'Th_i', 'Th_f']
        elif stationary is None:
            return ['H', 've', 'Te', 'De', 'khe_i', 'khe_f', 'Th_i', 'Th_f']
    if kinetics == '1I-IC-DU':
        if stationary == 'U':
            return ['H', 've', 'Te', 'De', 'khe']
        elif stationary == None:
            return ['H', 've', 'Te', 'De', 'khe_i', 'khe_f']
        
    raise ValueError(
        "The model " + str(kinetics) + ", " + str(stationary) + " is "
        "not recognised."
    )
    


def conc_liver(ca, t=None, dt=1.0, sum=True, cv=None, **params):
    """Concentration in liver tissue.

    See section :ref:`liver-tissues` for background.

    Args:
        ca (array-like): blood concentration in the arterial input.
        params (tuple): free model parameters.
        t (array_like, optional): the time points in sec of the input function 
          *ca*. If *t* is not provided, the time points are assumed to be 
          uniformly spaced with spacing *dt*. Defaults to None.
        dt (float, optional): spacing in seconds between time points for 
          uniformly spaced time points. This parameter is ignored if *t* is 
          explicity provided. Defaults to 1.0.
        sum (bool, optional): For two-compartment tissues, set to True to 
          return the total tissue concentration. Defaults to True.
        cv (array-like, optional): portal venous concentration for dual-inlet 
          models. Defaults to None.
        params (dict): the model parameters as keyword arguments. See table 
          :ref:`table-liver-models` for possible options.

    Returns:
        numpy.ndarray: If sum=True, this is a 1D array with the total 
          concentration at each time point. If sum=False this is the 
          concentration in each compartment, and at each time point, as a 
          2D array with dimensions *(2,k)*, where *k* is the number of time 
          points in *ca*. The concentration is returned in units of M.

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

    # Combine non-stationary parameters in a single array
    if 'khe_i' in params:
        params['khe'] = [params['khe_i'], params['khe_f']]
        del params['khe_i']
        del params['khe_f']
    if 'Th_i' in params:
        params['Th'] = [params['Th_i'], params['Th_f']]
        del params['Th_i']
        del params['Th_f']

    model = set(params.keys())

    # Extracellular - Single inlet - Dispersion
    if model == {'H', 've', 'Te', 'De'}:
        ca = ca / (1-params['H'])
        return _conc_liver(
            ca, params['ve'], t=t, dt=dt,
            extracellular = ['pfcomp', (params['Te'], params['De'])])
    
    # Extracellular - Single inlet - Compartment
    elif model == {'H', 've', 'Fp', 'fa', 'Ta', 'Tg'}:
        ca = ca / (1-params['H'])
        cv = pk.flux_comp(ca, params['Tg'], t=t, dt=dt)
        return _conc_liverav(
            ca, cv, 
            params['Ta'], params['fa'], params['Fp'], params['ve'], 
            t=t, dt=dt)
    
    # Extracellular - Dual inlet - High flow
    elif model == {'H', 've', 'fa', 'Ta'}:
        ca = ca / (1-params['H'])
        cv = cv / (1-params['H'])
        ca = pk.flux(ca, params['Ta'], t=t, dt=dt, model='plug')
        ce = params['fa']*ca + (1-params['fa'])*cv
        return params['ve']*ce
    
    # Extracellular - Dual inlet - Compartment
    elif model == {'H', 've', 'Fp', 'fa', 'Ta'}:
        ca = ca / (1-params['H'])
        cv = cv / (1-params['H'])
        return _conc_liverav(
            ca, cv, 
            params['Ta'], params['fa'], params['Fp'], params['ve'], 
            t=t, dt=dt)
    
    # Intracellular - Single inlet - High flow
    elif model == {'H', 've', 'khe', 'Th'}:
        ca = ca / (1-params['H'])
        khe, Th = params['khe'], params['Th']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        if np.isscalar(Th):
            hep = 'comp'
        else:
            hep = 'nscomp'
            if np.size(Th) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                Th = utils.interp(Th, tarr)
        return _conc_liver_hep(
            ca, params['ve'], khe*(1-params['H']), t=t, dt=dt, sum=sum,
            extracellular = ['pass', ()],
            hepatocytes = [hep, (Th,)])
    
    # Intracellular - Single inlet - Dispersion
    elif model == {'H', 've', 'Te', 'De', 'khe', 'Th'}:
        ca = ca / (1-params['H'])
        khe, Th = params['khe'], params['Th']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        if np.isscalar(Th):
            hep = 'comp'
        else:
            hep = 'nscomp'
            if np.size(Th) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                Th = utils.interp(Th, tarr)
        return _conc_liver_hep(
            ca, params['ve'], khe*(1-params['H']), t=t, dt=dt, sum=sum,
            extracellular = ['pfcomp', (params['Te'], params['De'])],
            hepatocytes = [hep, (Th,)])
    
    # Intracellular - Single inlet - Dispersion uptake
    elif model == {'H', 've', 'Te', 'De', 'khe'}:
        ca = ca / (1-params['H'])
        khe = params['khe']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        return _conc_liver_hep(
            ca, params['ve'], khe*(1-params['H']), t=t, dt=dt, sum=sum,
            extracellular = ['pfcomp', (params['Te'], params['De'])],
            hepatocytes = ['trap', ()])

    # Intracellular - Dual inlet - High flow
    elif model == {'H', 've', 'fa', 'Ta', 'khe', 'Th'}: 
        ca = ca / (1-params['H'])
        cv = cv / (1-params['H'])
        khe, Th = params['khe'], params['Th']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        if np.isscalar(Th):
            hep = 'comp'
        else:
            hep = 'nscomp'
            if np.size(Th) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                Th = utils.interp(Th, tarr)
        return _conc_liverav_hep(
            ca, cv, 
            params['Ta'], params['fa'], params['ve'], khe*(1-params['H']), 
            t=t, dt=dt, sum=sum,
            extracellular = ['pass', ()],
            hepatocytes = [hep, (Th,)])
    
    # Intracellular - Dual inlet - Compartment
    elif model == {'H', 've', 'Fp', 'fa', 'Ta', 'khe', 'Th'}:
        ca = ca / (1-params['H'])
        cv = cv / (1-params['H'])
        Te = params['ve']/params['Fp']
        khe, Th = params['khe'], params['Th']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        if np.isscalar(Th):
            hep = 'comp'
        else:
            hep = 'nscomp'
            if np.size(Th) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                Th = utils.interp(Th, tarr)
        return _conc_liverav_hep(
            ca, cv, 
            params['Ta'], params['fa'], params['ve'], khe*(1-params['H']), 
            t=t, dt=dt, sum=sum,
            extracellular=['comp', (Te,)],
            hepatocytes=[hep, (Th,)])
    
    # Intracellular - Dual inlet - Uptake
    elif model == {'H', 've', 'Fp', 'fa', 'Ta', 'khe'}:
        ca = ca / (1-params['H'])
        cv = cv / (1-params['H'])
        Te = params['ve']/params['Fp']
        khe = params['khe']
        if not np.isscalar(khe):
            if np.size(khe) != np.size(ca):
                tarr = utils.tarray(np.size(ca), t=t, dt=dt)
                khe = utils.interp(khe, tarr)
        return _conc_liverav_hep(
            ca, cv, 
            params['Ta'], params['fa'], params['ve'], khe*(1-params['H']), 
            t=t, dt=dt, sum=sum,
            extracellular=['comp', (Te,)],
            hepatocytes=['trap', ()])
   
    else:

        raise ValueError(
            'There is no liver model with parameters ' + str(model) + '.')



def _conc_liver(ca, ve,
                extracellular=['pfcomp', (30, 0.85)],
                t=None, dt=1.0):
    
    # Propagate through the extracellular space
    ce = pk.flux(ca, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    # Tissue concentration in the extracellular space
    Ce = ve*ce
    return Ce


def _conc_liverav(ca, cv, Ta: float, af, Fp, ve, t=None, dt=1.0):

    # Propagate through arterial tree
    ca = pk.flux(ca, Ta, t=t, dt=dt, model='plug')
    # Determine inlet concentration
    cp = af*ca + (1-af)*cv
    # Tissue concentration in the extracellular space
    Te = ve/Fp
    Ce = pk.conc_comp(Fp*cp, Te, t=t, dt=dt)
    return Ce


def _conc_liver_hep(ca, ve, khe, t=None, dt=1.0, sum=True,
                    extracellular=['pfcomp', (30, 0.85)],
                    hepatocytes=['comp', (30*60,)]):

    # Propagate through the extracellular space
    ce = pk.flux(ca, *extracellular[1], t=t, dt=dt, model=extracellular[0])
    # Tissue concentration in the extracellular space
    Ce = ve*ce
    # Tissue concentration in the hepatocytes
    Ch = pk.conc(khe*ce, *hepatocytes[1], t=t, dt=dt, model=hepatocytes[0])
    if sum:
        return Ce+Ch
    else:
        return np.stack((Ce, Ch))


def _conc_liverav_hep(ca, cv, Ta, af, ve, khe, t=None, dt=1.0, sum=True,
                      extracellular=['pfcomp', (30, 0.85)],
                      hepatocytes=['comp', (30*60,)]):

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
        return np.stack((Ce, Ch))
