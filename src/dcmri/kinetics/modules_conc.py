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

import numpy as np

from dcmri.core.module import Module
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.kinetics.functions_blocks import flux_plug
from dcmri.kinetics.modules_flux import FluxAorta
import dcmri.kinetics.functions_kidney as pk_kidney
import dcmri.kinetics.functions_tissue as pk_tissue
import dcmri.kinetics.functions_liver as pk_liver
import dcmri.kinetics.functions_blocks as blocks

# Helper function
def divC(C, v):
    if np.isscalar(v):
        if v==0:
            return C * 0 # In this case the result does not matter
        else:
            return C / v
    else:
        return np.vstack([divC(C[k], v[k]) for k in range(v.size)])

    
class Conc(Module):
    """Concentration in a generic building block.
    """
    configs = {
        'block': set(blocks.CONC_PARAMETERS.keys()),
    }
    defaults = {
        'block': 'comp',
    }
    def inputs(self):
        inputs = set(blocks.CONC_PARAMETERS[self.config['block']])
        inputs |= {'J'}
        return inputs
    
    def outputs(self):
        return {'C'}
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        model_func = getattr(blocks, f"conc_{self.config['block']}")
        results = {'C': model_func(**p)}

        return self.map_results(results)


class ConcAorta(Module):
    """Whole-body model for indicator concentration in the aorta.

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        **params: override parameter defaults.
    """
    configs = FluxAorta.configs
    defaults = FluxAorta.defaults

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._flux = FluxAorta(**config)
        self._conc = Conc(block='plug')
        self.map_io(imap, omap)
        
    def inputs(self):
        inputs = {'vol_a', 'CO'}
        inputs |= self._flux.mapped_inputs()
        inputs |= self._conc.mapped_inputs() 
        return inputs - {'T', 'J'}
    
    def outputs(self):
        return {'t', 'C_a', 'v_a', 'c_a', 'Fi_a', 'ci_a'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        v_a = 1 # assume no partial volume effect: v_a = 1mL/cm3
        flux = self._flux(p)
        Ta = v_a * p['vol_a'] / p['CO']
        conc = self._conc(p, T=Ta, J=flux['Ja']) 
        C = conc['C'].reshape(1, -1) / p['vol_a']
        results = {
            't': flux['t'], 
            'C_a': C,                                     # (nc, nt)
            'v_a': np.array([v_a]),                       # (nc, )
            'c_a': divC(C, v_a),                          # (nc, nt)
            'Fi_a': np.array([p['CO'] / p['vol_a']]),     # (nc, )
            'ci_a': flux['Ja'].reshape(1, -1) / p['CO'],  # (nc, nt)
        }
        return self.map_results(results)


class ConcLiver(Module):
    """
    Concentration in liver tissue for a variety of liver models.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        params (dict, optional): override parameter defaults.
    """

    configs = {
        'kinetics': [
            '2I-EC',
            '2I-EC-HF', 
            
            '1I-EC', 
            '1I-EC-HF',

            '2I-IC',
            '2I-IC-HF', 
            '2I-IC-U',

            '1I-IC', 
            '1I-IC-HF',
        ],
        'non_stationary': [
            None, 
            'U', 
            'E', 
            'UE'
        ],
    }
    defaults = {
        'kinetics': '2I-EC', 
        'non_stationary': None,
    }

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        if (self.config['kinetics'], self.config['non_stationary']) not in pk_liver.PARAMETERS.keys():
            raise InvalidConfiguration('For extracellular tracers the non-stationary configuration is invalid.')
        self.map_io(imap, omap)
            
    def inputs(self):
        model = (self.config['kinetics'], self.config['non_stationary'])
        inputs = set(pk_liver.PARAMETERS[model])
        inputs |= {'ci_l'}
        return inputs 

    def outputs(self):
        return {'C_l'}

    def lexicon_data(self, qvalues):
        p = {'ci_l': qvalues['ci']}
        return self.update_data(p)
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
        
        # Model function
        kin, ns = self.config['kinetics'], self.config['non_stationary']
        func = 'conc_liver_' + kin.lower().replace('-', '_')
        if ns != None:
            func += '_ns' + ns.lower()    
        liver_conc = getattr(pk_liver, func)

        phys = {k: v for k, v in p.items() if k not in ['ci_l']}
        results = {'C_l': liver_conc(p['ci_l'], **phys)}
        return self.map_results(results)

    def deriv(self, parameter, data):
        p = self.map_data(data, {}, all=False)
        if parameter=='El':
            if 'EC' in self.config['kinetics']:
                return 0
            if 'HF' in self.config['kinetics']:
                return None
            if self.config['non_stationary'] in [None, 'E']:
                return p['E']
            return np.mean([p['E_i'], p['E_f']])


class ConcAortaLiver(Module):
    """Concentration in aorta and liver.
    """
    configs = {
        'bolus': FluxAorta.configs['bolus'],
        'heartlung': FluxAorta.configs['heartlung'],
        'organs': FluxAorta.configs['organs'],
        'lagut': {'pass', 'comp', 'plucom'},
        'liver': {'1I-EC', '1I-IC'},
        'non_stationary': ConcLiver.configs['non_stationary'],
    }
    defaults = {
        'bolus': 'single',
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'lagut': 'comp',
        'liver': '1I-EC',
        'non_stationary': None,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        config_aorta = self.config | {'liver': 'comp'}
        config_liver = self.config | {'kinetics': self.config['liver']}
        
        self._flux_aorta = FluxAorta(**config_aorta)
        self._conc_liver = ConcLiver(**config_liver)

        self.map_io(imap, omap)

    def outputs(self):
        outputs = self._flux_aorta.outputs()
        for roi in ['a', 'l']:
            outputs |= {f'C_{roi}', f'v_{roi}', f'c_{roi}', f'Fi_{roi}', f'ci_{roi}'}
        return outputs

    def inputs(self):
        inputs = {'fCO_l', 'CO'}
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= self._conc_liver.mapped_inputs()
        inputs |= {'H', 'GFR', 'vol_l', 'vol_a'}
        inputs -= {'vr_l', 'vr_o', 'Te_l', 'Fp', 'ci_l'}
        return inputs
        
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        # Kidney extraction fraction       
        PF = (1 - p['fCO_l']) * p[f'CO'] * (1 - p['H'])
        Ek = p['GFR'] / (p['GFR'] + PF)

        # Liver extraction fraction
        Fp = p['fCO_l'] * p['CO'] * (1 - p['H']) / p['vol_l']
        El = self._conc_liver.deriv('El', p)
        
        # Compute aorta flux
        p |= {
            'vr_l': p['fCO_l'] * (1 - El),
            'vr_o': (1 - p['fCO_l']) * (1 - Ek),
            'Te_l': p['ve'] / Fp,
        }
        p |= self._flux_aorta(p)
   
        # Compute liver concentrations
        p |= {
            'Fp': Fp,
            'ci_l': p['Jl'] / p['CO'] / (1 - p['H'])
        }
        p |= self._conc_liver(p)

        # Build output
        C_a = p['Ja'].reshape(1, -1) / p['CO']
        C_l = p['C_l']
        v_l = np.array([p['ve'], 1 - p['ve']])

        p |= {
            'v_a': np.array([1]),
            'Fi_a': np.array([p['CO'] / p['vol_a']]),
            'C_a': C_a, 
            'c_a': C_a,
            'ci_a': C_a,

            'v_l': v_l,
            'Fi_l': np.array([p['fCO_l'] * p['CO'] / p['vol_l'], np.nan]),
            'C_l': C_l,
            'c_l': divC(C_l, v_l),
            'ci_l': np.stack([p['Jl'] / p['CO'], np.full_like(p['t'], np.nan)]),
        } 
        return self.map_results(p)


class ConcAortaPortalLiver(Module):
    """Concentration in aorta, portal vein, liver artery and liver.
    """
    configs = {k:v for k, v in ConcAortaLiver.configs.items() if k != 'lagut'}
    defaults = {k:v for k, v in ConcAortaLiver.defaults.items() if k != 'lagut'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._conc_aol = ConcAortaLiver(lagut='plucom', **self.config)
        self.map_io(imap, omap)

    def outputs(self):
        outputs = self._conc_aol.outputs()
        for roi in ['la', 'pv']:
            outputs |= {f'C_{roi}', f'v_{roi}', f'c_{roi}', f'Fi_{roi}', f'ci_{roi}'}
        return outputs

    def inputs(self):
        inputs = self._conc_aol.inputs()
        inputs |= {'vol_pv', 'vol_la'}
        return inputs
        
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
    
        p |= self._conc_aol(p)

        fCO_la = p['fa'] * p['fCO_l']
        fCO_pv = (1 - p['fa']) * p['fCO_l']

        C_la = p['Jla'].reshape(1, -1) / p['CO']
        C_pv = p['Jpv'].reshape(1, -1) / p['CO']

        p |= {
            'v_la': np.array([1]),
            'Fi_la': np.array([fCO_la * p['CO'] / p['vol_la']]),
            'C_la': C_la,
            'c_la': C_la,
            'ci_la': C_la,

            'v_pv': np.array([1]),
            'Fi_pv': np.array([fCO_pv * p['CO'] / p['vol_pv']]),
            'C_pv': C_pv,
            'c_pv': C_pv,
            'ci_pv': C_pv,
        }              
        return self.map_results(p)



class ConcKidney(Module):
    """Concentration in kidney tissue.
    """
    configs = {
        'kinetics': set(pk_kidney.PARAMETERS.keys()),
    }
    defaults = {
        'kinetics': '2CF',
    }
    def inputs(self, group=None):
        inputs = set(pk_kidney.PARAMETERS[self.config['kinetics']])
        inputs |= {'ca', 'Ta', 'dt'}
        if group=='phys':
            inputs -= {'ca', 'dt'}
        return inputs
    
    def outputs(self):
        return {'Ck'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        ca = flux_plug(p['ca'], T=p['Ta'], dt=p['dt'])
        func = 'conc_kidney_' + self.config['kinetics'].lower()   
        kidney_conc = getattr(pk_kidney, func)
        p = {k: v for k, v in p.items() if k not in ['ca', 'Ta']} 
        results = {'Ck': kidney_conc(ca, **p)}
        return self.map_results(results)

class ConcAortaKidneys(Module):
    """Concentration in aorta and both kidneys.
    """
    configs = {
        'bolus': FluxAorta.configs['bolus'],
        'heartlung': FluxAorta.configs['heartlung'],
        'organs': FluxAorta.configs['organs'],
        'kidneys': {v for v in ConcKidney.configs['kinetics'] if v not in ['HF', 'HFU']},
    }
    defaults = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'kidneys': '2CF',
        'bolus': 'single',
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Setup aorta module
        kidney_vasc = pk_kidney.VASCULAR_MODEL[self.config['kidneys']]
        config_aorta = self.config | {'kidneys': kidney_vasc}
        self._flux_aorta = FluxAorta(**config_aorta)

        # Setup Kidney modules
        imap = {k: f'{k}_lk' for k in {'Fp', 'vp', 'FF', 'Tt', 'ht'}}
        self._conc_lk = ConcKidney(imap=imap, kinetics=self.config['kidneys'])

        imap = {k: f'{k}_rk' for k in {'Fp', 'vp', 'FF', 'Tt', 'ht'}}
        self._conc_rk = ConcKidney(imap=imap, kinetics=self.config['kidneys'])

        self.map_io(imap, omap)

    def outputs(self):
        outputs = self._flux_aorta.outputs()
        for roi in ['a', 'lk', 'rk']:
            outputs |= {f'C_{roi}', f'v_{roi}', f'c_{roi}', f'Fi_{roi}', f'ci_{roi}'}
        return outputs
    
    def inputs(self):
        inputs = {'H', 'fCO_k', 'DRPF', 'CO', 'vol_a'}
        inputs |= self._conc_lk.mapped_inputs()
        inputs |= self._conc_rk.mapped_inputs()
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= {'El', 'vol_lk', 'vol_rk', 'vt_lk', 'vt_rk'}
        return inputs
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        # Derived parameters
        fco_lk = p['fCO_k'] * p['DRPF']
        fco_rk = p['fCO_k'] * (1 - p['DRPF'])
        
        E_lk = p['FF_lk'] / (1 + p['FF_lk'])
        E_rk = p['FF_rk'] / (1 + p['FF_rk'])
    
        Fp_lk = fco_lk * p['CO'] * (1 - p['H']) / p['vol_lk']
        Fp_rk = fco_rk * p['CO'] * (1 - p['H']) / p['vol_rk']

        # Extend data dictionary
        p |= {
            'Fp_lk': Fp_lk,
            'Fp_rk': Fp_rk,
            'Tp_lk': p['vp_lk'] / Fp_lk,
            'Tp_rk': p['vp_rk'] / Fp_rk,
            'vr_lk': fco_lk * (1 - E_lk),
            'vr_rk': fco_rk * (1 - E_rk),
            'vr_o': (1 - p['fCO_k']) * (1 - p['El']),
        }

        # Aorta flux
        p |= self._flux_aorta(p)

        # Kidney concentration
        cp = p['Ja'] / p['CO'] / (1 - p['H'])
        lk = self._conc_lk(p, ca=cp)
        rk = self._conc_rk(p, ca=cp)

        # Save output
        C_a = p['Ja'].reshape(1, -1) / p['CO']

        C_lk = np.pad(lk['Ck'], ((0, 1), (0, 0))) # append zeros for tissue compartment
        vb_lk = p['vp_lk'] / (1 - p['H'])
        ve_lk = 1 - vb_lk - p['vt_lk']
        if ve_lk < 0: ve_lk = 0
        v_lk = np.array([vb_lk, p['vt_lk'], ve_lk])

        C_rk = np.pad(rk['Ck'], ((0, 1), (0, 0))) # append zeros for tissue compartment
        vb_rk = p['vp_rk'] / (1 - p['H'])
        ve_rk = 1 - vb_rk - p['vt_rk']
        if ve_rk < 0: ve_rk = 0
        v_rk = np.array([vb_rk, p['vt_rk'], ve_rk])

        cnan = np.full_like(p['t'], np.nan)

        p |= {
            'v_a': np.array([1]),
            'Fi_a': np.array([p['CO'] / p['vol_a']]),
            'C_a': C_a, 
            'c_a': C_a,
            'ci_a': C_a,

            'v_lk': v_lk,
            'Fi_lk': np.array([fco_lk * p['CO'] / p['vol_lk'], np.nan, np.nan]),
            'C_lk': C_lk, 
            'c_lk': divC(C_lk, v_lk),
            'ci_lk': np.stack([p['Jlk'] / p['CO'], cnan, cnan]),

            'v_rk': v_rk,
            'Fi_rk': np.array([fco_rk * p['CO'] / p['vol_rk'], np.nan, np.nan]),
            'C_rk': C_rk,
            'c_rk': divC(C_rk, v_rk),
            'ci_rk': np.stack([p['Jrk'] / p['CO'], cnan, cnan]),
        }
        return self.map_results(p)
    
    
class ConcCortMed(Module):
    """Concentration in kidney cortex and medulla.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        params (dict, optional): override parameter defaults.

    """
    _params_dict = {
        '7C': {'Ta', 'Fp', 'Eg', 'fc', 'Tglom', 'Tv', 'Tpt', 'Tlh', 'Tdt', 'Tcd'}
    }
    configs = {
        'kinetics': {'7C'},
    }
    defaults = {
        'kinetics': '7C'
    }

    def inputs(self):
        model = self.config['kinetics']
        inputs = self._params_dict[model]
        inputs |= {'ca', 'dt'}
        return inputs
    
    def outputs(self):
        return {'Cc', 'Cm'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        ca = flux_plug(p['ca'], dt=p['dt'], T=p['Ta'])
        p = {k: v for k, v in p.items() if k not in ['ca', 'Ta']}

        if self.config['kinetics'] == '7C':
            Ccor, Cmed = pk_kidney.conc_kidney_cm9(ca, **p)

        results = {'Cc': Ccor, 'Cm': Cmed}
        return self.map_results(results)


class ConcTissueX(Module):
    """Concentration in vascular-interstitial tissue.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        params (dict, optional): override parameter defaults.
    """
    configs = {
        'kinetics': set(pk_tissue.CONC_PARAMETERS.keys())
    }
    defaults = {
        'kinetics': '2CX', 
    }
    def inputs(self):
        model = self.config['kinetics']
        inputs = set(pk_tissue.CONC_PARAMETERS[model])
        inputs |= {'ca', 'dt'}
        return inputs
    
    def outputs(self):
        return {'C'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        ca = flux_plug(p['ca'], dt=p['dt'], T=p['Ta'])
        p = {k: v for k, v in p.items() if k not in ['ca','Ta']}
        
        conc = 'conc_tissue_' + self.config['kinetics'].lower()   
        model_func = getattr(pk_tissue, conc)  
        results = {'C': model_func(ca, **p)}
        return self.map_results(results)