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
from dcmri.kinetics.functions_blocks import flux_plug
from dcmri.kinetics.modules_flux import FluxAorta
import dcmri.kinetics.functions_kidney as pk_kidney
import dcmri.kinetics.functions_tissue as pk_tissue
import dcmri.kinetics.functions_liver as pk_liver
import dcmri.kinetics.functions_blocks as blocks


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
    
    def __call__(self, data):
        p = self.map_data(data)
        model_func = getattr(blocks, f"conc_{self.config['block']}") 
        return {'C': model_func(**p)}


class ConcAorta(Module):
    """Whole-body model for indicator concentration in the aorta.

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        **params: override parameter defaults.
    """
    configs = FluxAorta.configs
    defaults = FluxAorta.defaults

    def __init__(self, config=None, imap=None):
        self.set_config(config)
        self._flux_aorta = FluxAorta(self.config)
        self.map_inputs(imap)
        
    def inputs(self):
        inputs = self._flux_aorta.mapped_inputs()
        inputs |= {'CO'}
        return inputs
    
    def outputs(self):
        return {'C'}
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)
        flux = self._flux_aorta(p)
        return {'C': flux['J'] / p['CO']}
    

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
            return [i for i in inputs if i not in {'ca', 'dt'}]
        return inputs
    
    def outputs(self):
        return {'C'}
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)

        ca = flux_plug(p['ca'], T=p['Ta'], dt=p['dt'])
        func = 'conc_kidney_' + self.config['kinetics'].lower()   
        kidney_conc = getattr(pk_kidney, func)
        p = {k: v for k, v in p.items() if k not in ['ca', 'Ta']} 
        return {'C': kidney_conc(ca, **p)}
    

# TODO: This should NOT return dimension (2, n) for EC models. Just (1, n) or (n,)

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

    # TODO: models as tuples from pk_liver.MODELS so there are no incompatible configs

    def __init__(self, config=None, imap=None):
        self.set_config(config)
        if (self.config['kinetics'], self.config['non_stationary']) not in pk_liver.PARAMETERS.keys():
            raise ValueError('For extracellular tracers the non-stationary configuration is invalid.')
        self.map_inputs(imap)
            
    def inputs(self):
        model = (self.config['kinetics'], self.config['non_stationary'])
        inputs = set(pk_liver.PARAMETERS[model])
        inputs |= {'ci'}
        return inputs 

    def outputs(self):
        return {'C'}
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(p)
        kin, ns = self.config['kinetics'], self.config['non_stationary']
        
        # Model function
        func = 'conc_liver_' + kin.lower().replace('-', '_')
        if ns != None:
            func += '_ns' + ns.lower()    
        liver_conc = getattr(pk_liver, func)

        phys = {k: v for k, v in p.items() if k not in ['ci']}
        return {'C': liver_conc(p['ci'], **phys)}

    def deriv(self, parameter, data):
        p = self.map_data(data)
        if parameter=='El':
            if 'EC' in self.config['kinetics']:
                return 0
            if 'HF' in self.config['kinetics']:
                return None
            if self.config['non_stationary'] in [None, 'E']:
                return p['E']
            return np.mean([p['E_i'], p['E_f']])
    

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
    def __init__(self, config=None, imap=None):
        self.set_config(config)

        # Setup aorta module
        kidney_vasc = pk_kidney.VASCULAR_MODEL[self.config['kidneys']]
        self._flux_aorta = FluxAorta(self.config | {'kidneys': kidney_vasc})

        # Setup Kidney modules
        self._conc_lk = ConcKidney({'kinetics': self.config['kidneys']})
        self._conc_rk = ConcKidney({'kinetics': self.config['kidneys']})

        self._conc_lk.map_inputs({k: f'{k}_lk' for k in self._conc_lk.inputs(group='phys')})
        self._conc_rk.map_inputs({k: f'{k}_rk' for k in self._conc_rk.inputs(group='phys')})

        self.map_inputs(imap)

    def outputs(self):
        return {'Ca', 'Clk', 'Crk'}
    
    def inputs(self):
        inputs = {'fCO_k', 'DRPF', 'CO'}
        inputs |= self._conc_lk.mapped_inputs()
        inputs |= self._conc_rk.mapped_inputs()
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= {'El', 'vol_lk', 'vol_rk', 'H'}
        return inputs
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)

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

        # Compute aorta flux
        flux = self._flux_aorta(p)

        # Compute concentrations
        Ca = flux['J'] / p['CO']

        # Extend data dictionary
        p |= {'ci': Ca / (1 - p['H'])}

        # Kidney concentration
        lk = self._conc_lk(p)
        rk = self._conc_rk(p)

        # Save output
        return {'Ca': Ca, 'Clk': lk['C'], 'Crk': rk['C']}
    

class ConcAortaLiver(Module):
    """Concentration in aorta and liver.
    """
    configs = {
        'bolus': FluxAorta.configs['bolus'],
        'heartlung': FluxAorta.configs['heartlung'],
        'organs': FluxAorta.configs['organs'],
        'lagut': {'pass', 'comp', 'plucom'},
        'liver': {'1I-IC', '1I-EC'},
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
    def __init__(self, config=None, imap=None):
        self.set_config(config)

        config_aorta = self.config | {'liver': 'comp'}
        config_liver = self.config | {'kinetics': self.config['liver']}
        
        self._flux_aorta = FluxAorta(config_aorta)
        self._conc_liver = ConcLiver(config_liver)

        self.map_inputs(imap)

    def outputs(self):
        outputs = {'ca', 'ci', 'Cl'}
        if self.config['lagut'] == 'plucom':
            outputs |= {'cla', 'cpv'}
        return outputs

    def inputs(self):
        inputs = {'fCO_l', 'CO'}
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= self._conc_liver.mapped_inputs()
        inputs |= {'H', 'GFR', 'vol_l'}
        return inputs
        
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)

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
        aorta = self._flux_aorta(p)
    
        # Compute liver concentrations
        p |= {
            'Fp': Fp,
            'ci': aorta['Jl'] / p['CO'] / (1 - p['H'])
        }
        conc = self._conc_liver(p)

        # Build output
        results = {
            'ca': aorta['J'] / p['CO'],
            'ci': aorta['Jl'] / p['CO'],
            'Cl': conc['C']
        }
        if self.config['lagut']=='plucom':
            results['cla'] = aorta['Jla'] / p['CO']
            results['cpv'] = aorta['Jpv'] / p['CO']
        return results
    
    
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
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)

        ca = flux_plug(p['ca'], dt=p['dt'], T=p['Ta'])
        p = {k: v for k, v in p.items() if k not in ['ca', 'Ta']}

        if self.config['kinetics'] == '7C':
            Ccor, Cmed = pk_kidney.conc_kidney_cm9(ca, **p)

        return {'Cc': Ccor, 'Cm': Cmed}
        


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
    
    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)

        ca = flux_plug(p['ca'], dt=p['dt'], T=p['Ta'])
        p = {k: v for k, v in p.items() if k not in ['ca','Ta']}
        
        conc = 'conc_tissue_' + self.config['kinetics'].lower()   
        model_func = getattr(pk_tissue, conc)  
        return {'C': model_func(ca, **p)}


