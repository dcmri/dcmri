import copy
import numpy as np

from dcmri import pk, const
from dcmri.core import SuperFunc
from dcmri.lexicon import QUANTITIES


class ConcAorta(SuperFunc):
    """Whole-body model for indicator flux through the aorta.

    See section :ref:`whole-body-tissues` for a more detailed description of 
    this model.

    Args:
        J_vena (np.ndarray): Indicator influx (mmol/sec) into the veins. 
        t (np.ndarray, optional): Array of time points (sec), must be of 
          equal size as J_vena. If not provided, the time points are uniformly 
          sampled with interval dt. Defaults to None.
        dt (float, optional): Sampling interval in sec. Defaults to 1.0.
        E (float, optional): Body extraction fraction. Defaults to 0.1.
        FFkl (float, optional): Fraction of the cardiac output that passes 
          through kidney and liver. Set FFkl=0 for a whole body model without 
          explicit kidney and liver spaces. Defaults to 0.0.
        FFk (float, optional): Kidney fraction of the flow to kidney and liver. 
          With FFk=0, only the liver is modelled. With FFk=1, only the kidneys 
          are modelled. Defaults to 0.5.
        heartlung (list): 3-element list specifying the model to use for the 
          heart-lung system (see notes for detail).
        organs (list): 3-element list specifying the model to use for the 
          organs (see notes for detail). 
        kidneys (list): 3-element list specifying the model to use for the 
          kidneys (see notes for detail). This keyword is ignored if FFkl=0 
          or FFk=0.
        liver (list): 3-element list specifying the model to use for the 
          liver (see notes for detail). This keyword is ignored if FFkl=0 
          or FFk=1.
        tol (float, optional): Dose tolerance in the solution. The solution 
          propagates the input through the system, until the dose that is 
          left in the system is given by tol*dose0, where dose0 is the 
          initial dose. Defaults to 0.001.
        max_it (int, optional): Maximum number of iterations.

    Returns:
        tuple: Indicator fluxes (mmol/sec) through the vena cava and aorta.

    Notes:

        The lists specifying each organ system consist of 3 elements: the 
        model (str), its parameters (tuple) and any keyword parameters (dict). 
        Any of the basic pharmacokinetic blocks can be used. For instance,
        *chain*, *plug-flow compartment*, and *compartment* would be specified 
        as follows:

          - **chain**: ['chain', (Thl, Dhl), {'solver':'step'}]
          - **plug-flow compartment**: ['pfcomp', (Thl, Dhl), {'solver':'interp'}}
          - **compartment**: ['comp', Thl]
          - **2cxm**: ['2cxm', ([To, Te], Eo)]


    Example:

        Generate flux through aorta:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import dcmri as dc

        # Generate a stepwise injection:
        t = np.arange(0, 120, 2.0)
        Ji = dc.ca_injection(t, 70, 0.5, 0.2, 3, 30)

        # Calculate the fluxes in mmol/sec:
        Ja = dc.flux_aorta(Ji, t)

        # Plot the fluxes:
        plt.plot(t/60, Ja, 'r-', label='Aorta')
        plt.xlabel('Time (min)')
        plt.ylabel('Indicator flux (mmol/sec)')
        plt.legend()
        plt.show()
    """
    configs = {
        'heartlung': ['comp', 'pfcomp', 'chain'],
        'organs': ['comp', '2cxm'],
        
    }
    def __init__(self, heartlung='pfcomp', organs='comp', **params):
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self, select=None):
        organs = {
            'comp': ['To'],
            '2cxm': ['To', 'To_e', 'Eo']
        }[self._cnfg['organs']]

        heartlung = {
            'comp': ['Thl'],
            'pfcomp': ['Thl', 'Dhl'],
            'chain': ['Thl', 'Dhl'],
        }[self._cnfg['heartlung']]

        body = ['BAT', 'CO', 'Eb']
        const = [
            'dt', 'tmax', 'dose_tolerance', 'field_strength',
            'agent', 'weight', 'dose', 'rate', 
        ]
        if select is None:
            return heartlung + organs + body + const 
        if select=='body':
            return heartlung + organs + body 
    
    def __call__(self, **params) -> np.ndarray:
        p = self._update_pars(**params)
        t = np.arange(0, p['tmax'], p['dt'])

        hl, orgs = self._cnfg['heartlung'], self._cnfg['organs']

        if hl=='comp':
            heartlung = ['comp', (p['Thl'],)]
        elif hl=='pfcomp':
            heartlung = ['pfcomp', (p['Thl'], p['Dhl'])]
        elif hl=='chain':
            heartlung = ['chain', (p['Thl'], p['Dhl'])]

        if orgs=='comp':
            organs = ['comp', (p['To'],)]
        elif orgs=='2cxm':
            organs = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]

        conc = const.ca_conc(p['agent'])
        Ji = pk.ca_injection(
            t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = pk.flux_aorta(
            Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=heartlung, organs=organs,
        )
        return Jb / p['CO']
    


# Liver-specific defaults
QUANTITIES_LIVER = QUANTITIES | {
    'T_a': {'init': 2, 'bounds': [0, 30], 'name': 'Arterial mean transit time', 'unit': 'sec'},
    'Fp': {'init': 0.008, 'bounds': [0, 1], 'name': 'Liver plasma flow', 'unit': 'mL/sec/cm3'},
}

class ConcLiver(SuperFunc):
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
        >>> ca = dc.aif.parker(t, BAT=20)

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

    _params_dict = {
        ('1I-EC-D', None): ['ve', 'Te', 'De'],
        ('1I-EC', None): ['fa', 'T_a', 'Tg', 've', 'Fp'],
        ('2I-EC-HF', None): ['fa', 'T_a', 've'],
        ('2I-EC', None): ['fa', 'T_a', 've', 'Fp'],
        ('1I-IC', None): ['ve', 'Fp', 'E', 'Th'],
        ('1I-IC', 'U'): ['ve', 'Fp', 'E_i', 'E_f', 'Th'],
        ('1I-IC', 'E'): ['ve', 'Fp', 'E', 'Th_i', 'Th_f'],
        ('1I-IC', 'UE'): ['ve', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f'],
        ('1I-IC-HF', None): ['ve', 'khe', 'Th'],
        ('1I-IC-HF', 'U'): ['ve', 'khe_i', 'khe_f', 'Th'],
        ('1I-IC-HF', 'E'): ['ve', 'khe', 'Th_i', 'Th_f'],
        ('1I-IC-HF', 'UE'): ['ve', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],
        ('1I-IC-HFD', None): ['Tg', 'Dg', 've', 'khe', 'Th'],
        ('1I-IC-HFD', 'U'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th'],
        ('1I-IC-HFD', 'E'): ['Tg', 'Dg', 've', 'khe', 'Th_i', 'Th_f'],
        ('1I-IC-HFD', 'UE'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],
        ('1I-IC-HFDU', None): ['Tg', 'Dg', 've', 'khe'],
        ('1I-IC-HFDU', 'U'): ['Tg', 'Dg', 've', 'khe_i', 'khe_f'],
        ('2I-IC-HF', None): ['fa', 'T_a', 've', 'khe', 'Th'],
        ('2I-IC-HF', 'U'): ['fa', 'T_a', 've', 'khe_i', 'khe_f', 'Th'],
        ('2I-IC-HF', 'E'): ['fa', 'T_a', 've', 'khe', 'Th_i', 'Th_f'],
        ('2I-IC-HF', 'UE'): ['fa', 'T_a', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f'],
        ('2I-IC', None): ['fa', 'T_a', 've', 'Fp', 'E', 'Th'],
        ('2I-IC', 'U'): ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f', 'Th'],
        ('2I-IC', 'E'): ['fa', 'T_a', 've', 'Fp', 'E', 'Th_i', 'Th_f'],
        ('2I-IC', 'UE'): ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f', 'Th_i', 'Th_f'],
        ('2I-IC-U', None): ['fa', 'T_a', 've', 'Fp', 'E'],
        ('2I-IC-U', 'U'): ['fa', 'T_a', 've', 'Fp', 'E_i', 'E_f'],
    }

    configs = {
        'kinetics': ['1I-EC-D', '1I-EC', '2I-EC-HF', '2I-EC', '1I-IC', '1I-IC-HF', '1I-IC-HFD', '1I-IC-HFDU', '2I-IC-HF', '2I-IC', '2I-IC-U'],
        'non_stationary': [None, 'U', 'E', 'UE'],
    }

    def __init__(self, kinetics='2I-EC', non_stationary=None, **params):
        cnfg = {'kinetics': kinetics, 'non_stationary': non_stationary}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(QUANTITIES_LIVER, **params)

    def _params(self):
        model = (self._cnfg['kinetics'], self._cnfg['non_stationary'])
        return copy.deepcopy(self._params_dict[model])
    
    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)
        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        
        # Define model function
        conc = 'conc_liver_' + kin.lower().replace('-', '_')
        if ns != None:
            conc += '__' + ns.lower()    

        model_func = getattr(pk, conc)  

        # Apply model function
        if '-IC' in kin:
            return model_func(ca, t=t, dt=dt, **p)
        else:
            if ns != None:
                raise ValueError("For extracellular models non_stationary must be None")
            return model_func(ca, t=t, dt=dt, **p)
        



    
# Kidney-specific defaults
QUANTITIES_KIDNEY = QUANTITIES | {
    'ht': {'init': np.ones(5) / 5, 'bounds': [0, 100], 'name': 'Tubular transit time distribution', 'unit': '1/sec'},
    'Tv': {'name': 'Vascular mean transit time', 'unit': 'sec'},
    'RBF': {'name': 'Renal blood flow', 'unit': 'mL/sec'},
}

class ConcKidney(SuperFunc):
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
        >>> ca = dc.aif.parker(t, BAT=20)

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

    _params_dict = {
        '2CF': ['T_a', 'Fp', 'vp', 'Ft', 'Tt'],
        'HF': ['T_a', 'vp', 'Ft', 'Tt'],
        'FN': ['T_a', 'Fp', 'Tp', 'Ft', 'ht'],
    }
    configs = {
        'kinetics': ['2CF', 'HF', 'FN'],
    }
    def __init__(self, kinetics='2CF', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(QUANTITIES_KIDNEY, **params)

    def _params(self):
        model = self._cnfg['kinetics']
        return copy.deepcopy(self._params_dict[model])
    
    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)
        kin = self._cnfg['kinetics']

        ca = pk.flux(ca, p['T_a'], dt=dt, model='plug')
        p = {k: v for k, v in p.items() if k != 'T_a'}

        if kin == '2CF':
            return pk.conc_kidney_2cf(ca, t=t, dt=dt, **p)
        if kin == 'HF':
            return pk.conc_kidney_hf(ca, t=t, dt=dt, **p)
        if kin == 'FN':
            return pk.conc_kidney_fn(ca, t=t, dt=dt, **p)
        


class ConcCortMed(SuperFunc):

    _params_dict = {
        '7C': ['T_a', 'Fp', 'Eg', 'fc', 'Tglom', 'Tv', 'Tpt', 'Tlh', 'Tdt', 'Tcd'],
    }
    configs = {
        'kinetics': ['7C'],
    }
    def __init__(self, kinetics='7C', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        model = self._cnfg['kinetics']
        return copy.deepcopy(self._params_dict[model])
    
    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)
        kin = self._cnfg['kinetics']

        ca = pk.flux(ca, p['T_a'], dt=dt, model='plug')
        p = {k: v for k, v in p.items() if k != 'T_a'}

        if kin == '7C':
            return pk.conc_kidney_cm9(ca, t=t, dt=dt, **p)
        


class ConcTissueX(SuperFunc):

    _params_dict = {
        '2CX': ['T_a', 'H', 'vb', 'vi', 'Fb', 'PS'],
        'HF': ['T_a', 'H', 'vb', 'vi', 'PS'],
        'WV': ['T_a', 'H', 'vi', 'Ktrans'],
        '2CU': ['T_a', 'H', 'vb', 'Fb', 'PS'],
        'HFU': ['T_a', 'H', 'vb', 'PS'],
        'FX': ['T_a', 'H', 've', 'Fb'],
        'NX': ['T_a', 'vb', 'Fb'],
        'NXP': ['T_a', 'vb', 'Fb'],
        'U': ['T_a', 'Fb'],
    }

    configs = {'kinetics': copy.deepcopy(list(_params_dict.keys()))}

    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        return copy.deepcopy(self._params_dict[self._cnfg['kinetics']])

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params):
        p = self._update_pars(**params)

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}
        
        kinetics = self._cnfg['kinetics']
        if kinetics == 'U': return pk.conc_tissue_u(ca, t=t, dt=dt, **params)
        if kinetics == 'FX': return pk.conc_tissue_fx(ca, t=t, dt=dt, **params)
        if kinetics == 'NX': return pk.conc_tissue_nx(ca, t=t, dt=dt, **params)
        if kinetics == 'NXP': return pk.conc_tissue_nxp(ca, t=t, dt=dt, **params)
        if kinetics == 'WV': return pk.conc_tissue_wv(ca, t=t, dt=dt, **params)
        if kinetics == 'HFU': return pk.conc_tissue_hfu(ca, t=t, dt=dt, **params)
        if kinetics == 'HF': return pk.conc_tissue_hf(ca, t=t, dt=dt, **params)
        if kinetics == '2CU': return pk.conc_tissue_2cu(ca, t=t, dt=dt, **params)
        if kinetics == '2CX': return pk.conc_tissue_2cx(ca, t=t, dt=dt, **params)


class FluxTissueX(SuperFunc):

    _params_dict = {
        '2CX': ['T_a', 'H', 'vb', 'vi', 'Fb', 'PS'],
        'HF': ['T_a', 'H', 'vi', 'PS'],
        'WV': ['T_a', 'H', 'vi', 'Ktrans'],
        '2CU': ['T_a', 'H', 'vb', 'Fb', 'PS'],
        'HFU': ['T_a', 'H', 'PS'],
        'FX': ['T_a', 'H', 've', 'Fb'],
        'NX': ['T_a', 'vb', 'Fb'],
        'NXP': ['T_a', 'vb', 'Fb'],
        'U': ['T_a', 'Fb'],
    }

    configs = {'kinetics': copy.deepcopy(list(_params_dict.keys()))}
    
    def __init__(self, kinetics='2CX', **params):
        cnfg = {'kinetics': kinetics}       
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self):
        return copy.deepcopy(self._params_dict[self._cnfg['kinetics']])

    def __call__(self, ca: np.ndarray, t=None, dt=1.0, **params) -> np.ndarray:
        p = self._update_pars(**params)

        ca = pk.flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}

        kinetics = self._cnfg['kinetics']
        if kinetics == 'U': return pk.flux_tissue_u(ca, **params)
        if kinetics == 'NX': return pk.flux_tissue_nx(ca, t=t, dt=dt, **params)
        if kinetics == 'NXP': return pk.flux_tissue_nxp(ca, t=t, dt=dt, **params)
        if kinetics == 'FX': return pk.flux_tissue_fx(ca, t=t, dt=dt, **params)
        if kinetics == 'WV': return pk.flux_tissue_wv(ca, t=t, dt=dt, **params)
        if kinetics == 'HFU': return pk.flux_tissue_hfu(ca, **params)
        if kinetics == 'HF': return pk.flux_tissue_hf(ca, t=t, dt=dt, **params)
        if kinetics == '2CU': return pk.flux_tissue_2cu(ca, t=t, dt=dt, **params)
        if kinetics == '2CX': return pk.flux_tissue_2cx(ca, t=t, dt=dt, **params)



