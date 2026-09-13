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

    def dummy_data(self, nt=5, nc=2):
        data = self.init_data()
        data['J'] = np.ones(nt)
        data['h'] = [1]
        data['TT'] = [0, 1]
        if self.config['block'] in ['bicomp', '2cxm']:
            data['T'] = [1, 1]
        if self.config['block'] == 'ncomp':
            data['T'] = np.ones(nc)
            data['J'] = np.ones((nc, nt))
            data['E'] = np.ones((nc, nc))
        if self.config['block'] == 'nscomp':
            data['T'] = np.ones(nt)
        return data


class ConcAorta(Module):
    """Whole-body model for indicator concentration in the aorta.

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        **params: override parameter defaults.
    """
    configs = FluxAorta.configs
    defaults = FluxAorta.defaults

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self._flux = FluxAorta(**config)
        self._conc = Conc(block='plug')
        self.map_io(imap, omap, iomap)
        
    def inputs(self):
        inputs = {'vol_ao', 'CO'}
        inputs |= self._flux.mapped_inputs()
        inputs |= self._conc.mapped_inputs() 
        return inputs - {'T', 'J'}
    
    def outputs(self):
        return {'tC', 'F_b_ao', 'ci_ao', 'C_ao'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        v_a = 1 # assume no partial volume effect: v_a = 1mL/cm3
        flux = self._flux(p)
        Ta = v_a * p['vol_ao'] / p['CO']
        conc = self._conc(p, T=Ta, J=flux['J_ao']) 
        C = conc['C'].reshape(1, -1) / p['vol_ao']
        results = {
            'tC': flux['tC'], 
            
            'F_b_ao': p['CO'] / p['vol_ao'], 
            'ci_ao': flux['J_ao'].reshape(1, -1) / p['CO'],  # (nc, nt)
            'C_ao': C,                                     # (nc, nt)
        }
        return self.map_results(results)
    
# +--------------------------------------------------------------------------------------------------+
# |                                 ConcLiver - all configs (n = 2)                                  |
# +----------------+-----------------------------------------------------------------------+---------+
# | Key            | Values                                                                | Default |
# +----------------+-----------------------------------------------------------------------+---------+
# | kinetics       | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF, 2I-EC, 2I-EC-HF, 2I-IC, 2I-IC-HF,   | 2I-EC   |
# |                | 2I-IC-U                                                               |         |
# | non_stationary | E, None, U, UE                                                        | None    |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                           ConcLiver - all inputs (n = 14)                                                           |
# +--------+------------+----------------------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | Key    | Unit       | Name                                                                 | Group           | Init  | Bounds       | DICOM | OSIPI |
# +--------+------------+----------------------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | ci_li  | mmol/mL    | inlet concentration in the liver                                     | Indicator       | 0.005 |              |       |       |
# +--------+------------+----------------------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | E_li   |            | extraction fraction in the liver                                     | Physiological   | 0.1   | (0.0, 1.0)   |       |       |
# | Ef_li  |            | final extraction fraction in the liver                               | Physiological   | 0.1   | (0.0, 1.0)   |       |       |
# | Ei_li  |            | initial extraction fraction in the liver                             | Physiological   | 0.1   | (0.0, 1.0)   |       |       |
# | F_p_li | mL/sec/cm3 | flow per unit tissue in plasma of the liver                          | Physiological   | 0.01  | (0, 0.05)    |       |       |
# | T_h    | sec        | mean transit time in hepatocytes                                     | Physiological   | 1800  | (600, 36000) |       |       |
# | Tf_h   | sec        | final mean transit time in hepatocytes                               | Physiological   | 1800  | (600, 36000) |       |       |
# | Ti_h   | sec        | initial mean transit time in hepatocytes                             | Physiological   | 1800  | (600, 36000) |       |       |
# | ffa    |            | arterial flow fraction                                               | Physiological   | 0.2   | (0, 1)       |       |       |
# | k_e2h  | mL/sec/cm3 | tissue transfer rate from extracellular space to hepatocytes         | Physiological   | 0.003 | (0.0, 0.1)   |       |       |
# | kf_e2h | mL/sec/cm3 | final tissue transfer rate from extracellular space to hepatocytes   | Physiological   | 0.003 | (0.0, 0.1)   |       |       |
# | ki_e2h | mL/sec/cm3 | initial tissue transfer rate from extracellular space to hepatocytes | Physiological   | 0.003 | (0.0, 0.1)   |       |       |
# | v_e_li | mL/cm3     | volume fraction in extracellular space of the liver                  | Physiological   | 0.3   | (0.01, 0.6)  |       |       |
# +--------+------------+----------------------------------------------------------------------+-----------------+-------+--------------+-------+-------+
# | dt     | sec        | pseudo-continuous time step                                          | Hyperparameters | 0.5   |              |       |       |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------------------------------------------+
# |                                       ConcLiver - all outputs (n = 3)                                        |
# +-------+----------+----------------------------------------+-----------------+-------+--------+-------+-------+
# | Key   | Unit     | Name                                   | Group           | Init  | Bounds | DICOM | OSIPI |
# +-------+----------+----------------------------------------+-----------------+-------+--------+-------+-------+
# | C_li  | mmol/cm3 | tissue concentration in the liver      | Indicator       | 0.005 | (0, 1) |       |       |
# | ci_li | mmol/mL  | inlet concentration in the liver       | Indicator       | 0.005 |        |       |       |
# | tC_li | sec      | concentration time points in the liver | Indicator       | 0.0   |        |       |       |
# +--------------------------------------------------------------------------------------------------------------+

class ConcLiver(Module):
    """
    Concentration in liver tissue.
    """

    configs = {
        'kinetics': {k[0] for k in pk_liver.PARAMETERS.keys()},
        'non_stationary': {k[1] for k in pk_liver.PARAMETERS.keys()},
    }
    defaults = {
        'kinetics': '2I-EC', 
        'non_stationary': None,
    }

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        if (self.config['kinetics'], self.config['non_stationary']) not in pk_liver.PARAMETERS.keys():
            raise InvalidConfiguration('For extracellular tracers the non-stationary configuration is invalid.')
        self.map_io(imap, omap, iomap)
            
    def inputs(self):
        imap = {'F_p':'F_p_li', 'v_e': 'v_e_li', 'E':'E_li', 'Ei':'Ei_li', 'Ef':'Ef_li'}
        model = (self.config['kinetics'], self.config['non_stationary'])
        inputs = {imap.get(i, i) for i in pk_liver.PARAMETERS[model]}
        inputs |= {'dt', 'ci_li'}
        return inputs 

    def outputs(self):
        return {'tC_li', 'C_li', 'ci_li'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
        
        # Model function
        kin, ns = self.config['kinetics'], self.config['non_stationary']
        func = 'conc_liver_' + kin.lower().replace('-', '_')
        if ns != None:
            func += '_ns' + ns.lower()    
        liver_conc = getattr(pk_liver, func)

        imap = {'F_p_li':'F_p', 'v_e_li':'v_e', 'E_li':'E', 'Ei_li':'Ei', 'Ef_li':'Ef'}
        phys = {imap.get(k, k): v for k, v in p.items() if k not in ['ci_li']}

        C_li = liver_conc(p['ci_li'], **phys)
        tC_li = p['dt'] * np.arange(C_li.shape[1])
        if '2I' in kin:
            ci = p['ffa'] * p['ci_li'][0] + (1 - p['ffa']) * p['ci_li'][1]
        else:
            ci = p['ci_li']

        results = {'tC_li': tC_li, 'C_li': C_li, 'ci_li': ci}
        return self.map_results(results)

    def dummy_data(self, nt=5):
        data = self.init_data()
        ci = np.ones(nt)
        data['ci_li'] = ci if '1I' in self.config['kinetics'] else (ci, ci)
        return data

# +--------------------------------------------------------------------------------------------------+
# |                               ConcAortaLiver - all configs (n = 6)                               |
# +----------------+-----------------------------------------------------------------------+---------+
# | Key            | Values                                                                | Default |
# +----------------+-----------------------------------------------------------------------+---------+
# | bolus          | dual, single                                                          | single  |
# | heartlung      | chain, comp, pfcomp                                                   | pfcomp  |
# | organs         | 2cxm, comp                                                            | comp    |
# | lagut          | comp, pass, plucom                                                    | comp    |
# | liver          | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF                                      | 1I-EC   |
# | non_stationary | E, None, U, UE                                                        | None    |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                     ConcAortaLiver - all inputs (n = 38)                                                     |
# +----------------+------------+--------------------------------------------------+-----------------+------------+--------------+-------+-------+
# | Key            | Unit       | Name                                             | Group           | Init       | Bounds       | DICOM | OSIPI |
# +----------------+------------+--------------------------------------------------+-----------------+------------+--------------+-------+-------+
# | BAT            | sec        | bolus arrival time                               | Indicator       | 30         | (-30, 30)    |       |       |
# | BAT_1          | sec        | 1st bolus arrival time                           | Indicator       | 30         | (-30, 30)    |       |       |
# | BAT_2          | sec        | 2nd bolus arrival time                           | Indicator       | 30         | (-30, 30)    |       |       |
# | agent          |            | contrast agent generic name                      | Indicator       | gadoterate |              |       |       |
# | dose           | mL/kg      | contrast agent dose                              | Indicator       | 0.1        | (0, 0.2)     |       |       |
# | dose_1         | mL/kg      | 1st contrast agent dose                          | Indicator       | 0.1        | (0, 0.2)     |       |       |
# | dose_2         | mL/kg      | 2nd contrast agent dose                          | Indicator       | 0.1        | (0, 0.2)     |       |       |
# | rate           | mL/s       | injection rate                                   | Indicator       | 1          | (0, 10)      |       |       |
# | rate_1         | mL/s       | 1st injection rate                               | Indicator       | 1          | (0, 10)      |       |       |
# | rate_2         | mL/s       | 2nd injection rate                               | Indicator       | 1          | (0, 10)      |       |       |
# +----------------+------------+--------------------------------------------------+-----------------+------------+--------------+-------+-------+
# | CO             | mL/sec     | cardiac output                                   | Physiological   | 100        | (0, 500)     |       |       |
# | D_hl           |            | transit time dispersion in the heart and Lungs   | Physiological   | 0.2        | (0.01, 0.99) |       |       |
# | E_li           |            | extraction fraction in the liver                 | Physiological   | 0.1        | (0.0, 1.0)   |       |       |
# | E_or           |            | extraction fraction in the organs                | Physiological   | 0.15       | (0, 0.5)     |       |       |
# | Ef_li          |            | final extraction fraction in the liver           | Physiological   | 0.1        | (0.0, 1.0)   |       |       |
# | Ei_li          |            | initial extraction fraction in the liver         | Physiological   | 0.1        | (0.0, 1.0)   |       |       |
# | GFR            | mL/sec     | glomerular filtration rate                       | Physiological   | 2          | (0, 10)      |       |       |
# | H              |            | hematocrit                                       | Physiological   | 0.45       | (0, 1)       |       |       |
# | T_b_or         | sec        | mean transit time in blood of the organs         | Physiological   | 20         | (0, 60)      |       |       |
# | T_e_or         | sec        | mean transit time in extracellular of the organs | Physiological   | 120        | (0, 800)     |       |       |
# | T_gu           | sec        | mean transit time in the gut                     | Physiological   | 30         | (0.1, 60)    |       |       |
# | T_h            | sec        | mean transit time in hepatocytes                 | Physiological   | 1800       | (600, 36000) |       |       |
# | T_hl           | sec        | mean transit time in the heart and Lungs         | Physiological   | 10         | (0, 30)      |       |       |
# | T_la           | sec        | mean transit time in the liver artery            | Physiological   | 30         | (0.1, 60)    |       |       |
# | Tf_h           | sec        | final mean transit time in hepatocytes           | Physiological   | 1800       | (600, 36000) |       |       |
# | Ti_h           | sec        | initial mean transit time in hepatocytes         | Physiological   | 1800       | (600, 36000) |       |       |
# | fCO_li         |            | fraction of the cardiac output in the liver      | Physiological   | 0.1        | (0, 0.5)     |       |       |
# | ffa            |            | arterial flow fraction                           | Physiological   | 0.2        | (0, 1)       |       |       |
# | k_e2h          | mL/sec/cm3 | hepatocellular uptake rate                       | Physiological   | 0.003      | (0.0, 0.1)   |       |       |
# | kf_e2h         | mL/sec/cm3 | final hepatocellular uptake rate                 | Physiological   | 0.003      | (0.0, 0.1)   |       |       |
# | ki_e2h         | mL/sec/cm3 | initial hepatocellular uptake rate               | Physiological   | 0.003      | (0.0, 0.1)   |       |       |
# | v_e_li         | mL/cm3     | volume fraction in extracellular of the liver    | Physiological   | 0.3        | (0.01, 0.6)  |       |       |
# +----------------+------------+--------------------------------------------------+-----------------+------------+--------------+-------+-------+
# | dose_tolerance |            | dose tolerance                                   | Hyperparameters | 0.1        |              |       |       |
# | dt             | sec        | pseudo-continuous time step                      | Hyperparameters | 0.5        |              |       |       |
# | tmax           | sec        | maximum time point                               | Hyperparameters | 240        |              |       |       |
# +----------------+------------+--------------------------------------------------+-----------------+------------+--------------+-------+-------+
# | vol_ao         | cm3        | ROI volume in the aorta                          | Whole-body      | 10         | (0.0, 1000)  |       |       |
# | vol_li         | cm3        | ROI volume in the liver                          | Whole-body      | 1000       | (0, 10000)   |       |       |
# | weight         | kg         | body weight                                      | Whole-body      | 70         | (0, 300)     |       |       |
# +----------------------------------------------------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------+
# |                                        ConcAortaLiver - all outputs (n = 14)                                         |
# +--------+------------+--------------------------------------------+-----------------+-------+---------+-------+-------+
# | Key    | Unit       | Name                                       | Group           | Init  | Bounds  | DICOM | OSIPI |
# +--------+------------+--------------------------------------------+-----------------+-------+---------+-------+-------+
# | C_ao   | mmol/cm3   | tissue concentration in the aorta          | Indicator       | 0.005 | (0, 1)  |       |       |
# | C_li   | mmol/cm3   | tissue concentration in the liver          | Indicator       | 0.005 | (0, 1)  |       |       |
# | J_ao   | mmol/sec   | indicator flux in the aorta                | Indicator       | 1     | (0, 10) |       |       |
# | J_la   | mmol/sec   | indicator flux in the liver artery         | Indicator       | 1     | (0, 10) |       |       |
# | J_lag  | mmol/sec   | indicator flux in the liver artery and gut | Indicator       | 1     | (0, 10) |       |       |
# | J_li   | mmol/sec   | indicator flux in the liver                | Indicator       | 1     | (0, 10) |       |       |
# | J_or   | mmol/sec   | indicator flux in the organs               | Indicator       | 1     | (0, 10) |       |       |
# | J_pv   | mmol/sec   | indicator flux in the portal vein          | Indicator       | 1     | (0, 10) |       |       |
# | J_ve   | mmol/sec   | indicator flux in the vein                 | Indicator       | 1     | (0, 10) |       |       |
# | ci_ao  | mmol/mL    | inlet concentration in the aorta           | Indicator       | 0.005 |         |       |       |
# | ci_li  | mmol/mL    | inlet concentration in the liver           | Indicator       | 0.005 |         |       |       |
# | tC     | sec        | concentration time points                  | Indicator       | 0.0   |         |       |       |
# +--------+------------+--------------------------------------------+-----------------+-------+---------+-------+-------+
# | F_b_ao | mL/sec/cm3 | flow per unit tissue in blood of the aorta | Physiological   | 0.02  | (0, 1)  |       |       |
# | F_b_li | mL/sec/cm3 | flow per unit tissue in blood of the liver | Physiological   | 0.02  | (0, 1)  |       |       |
# +----------------------------------------------------------------------------------------------------------------------+

class ConcAortaLiver(Module):
    """Concentration in aorta and liver.
    """
    configs = {
        'bolus': FluxAorta.configs['bolus'],
        'heartlung': FluxAorta.configs['heartlung'],
        'organs': FluxAorta.configs['organs'],
        'lagut': {'pass', 'comp', 'plucom'},
        'liver': {k for k in ConcLiver.configs['kinetics'] if '1I' in k}, # {'1I-EC', '1I-IC'}, 
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

    _all_inputs = None 
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)

        config_aorta = self.config | {'liver': 'comp'}
        config_liver = self.config | {'kinetics': self.config['liver']}
        
        self._flux_aorta = FluxAorta(**config_aorta)
        self._conc_liver = ConcLiver(**config_liver)

        self.map_io(imap, omap, iomap)

    def outputs(self):
        outputs = self._flux_aorta.outputs()
        for roi in ['ao', 'li']:
            outputs |= {f'F_b_{roi}', f'ci_{roi}', f'C_{roi}'}
        return outputs

    def inputs(self):
        inputs = {'fCO_li', 'CO', 'H', 'GFR', 'vol_li', 'vol_ao'}
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= self._conc_liver.mapped_inputs()
        inputs -= {'vr_li', 'vr_or', 'T_e_li', 'F_p_li', 'ci_li'}
        return inputs

    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        # Kidney extraction fraction       
        PF = (1 - p['fCO_li']) * p[f'CO'] * (1 - p['H'])
        Ek = p['GFR'] / (p['GFR'] + PF)

        # Liver extraction fraction
        Fp = p['fCO_li'] * p['CO'] * (1 - p['H']) / p['vol_li']
        El = self.deriv('E_li', p | {'F_p_li': Fp})    
        
        # Compute aorta flux
        p |= {
            'vr_li': p['fCO_li'] * (1 - El),
            'vr_or': (1 - p['fCO_li']) * (1 - Ek),
            'T_e_li': p['v_e_li'] / Fp,
        }
        p |= self._flux_aorta(p)
   
        # Compute liver concentrations
        p |= {
            'F_p_li': Fp,
            'ci_li': p['J_lag'] / p['CO'] / (1 - p['H'])
        }
        p |= self._conc_liver(p)

        # Build output
        C_a = p['J_ao'].reshape(1, -1) / p['CO']
        C_l = p['C_li']

        p |= {
            'F_b_ao': p['CO'] / p['vol_ao'],
            'ci_ao': C_a,
            'C_ao': C_a, 
            
            'F_b_li': p['fCO_li'] * p['CO'] / p['vol_li'],
            'ci_li': p['J_lag'].reshape(1, -1) / p['CO'],
            'C_li': C_l,
        } 
        return self.map_results(p)

    def deriv(self, parameter, p): # Helper
        if parameter=='E_li':
            if 'EC' in self.config['liver']:
                return 0
            if 'HF' in self.config['liver']:
                if self.config['non_stationary'] in ['U', 'UE']:
                    khe = np.mean([p['ki_e2h'], p['kf_e2h']])
                else:
                    khe = p['k_e2h']
                return p['F_p_li'] / (p['F_p_li'] + khe)
            if self.config['non_stationary'] in [None, 'E']:
                return p['E_li']
            return np.mean([p['Ei_li'], p['Ef_li']])


class ConcAortaPortalLiver(Module):
    """Concentration in aorta, portal vein, liver artery and liver.
    """
    configs = {k:v for k, v in ConcAortaLiver.configs.items() if k != 'lagut'}
    defaults = {k:v for k, v in ConcAortaLiver.defaults.items() if k != 'lagut'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self._conc_aol = ConcAortaLiver(lagut='plucom', **self.config)
        self.map_io(imap, omap, iomap)

    def outputs(self):
        outputs = self._conc_aol.outputs()
        for roi in ['la', 'pv']:
            outputs |= {f'F_b_{roi}', f'ci_{roi}', f'C_{roi}'}
        return outputs

    def inputs(self):
        inputs = self._conc_aol.inputs()
        inputs |= {'vol_pv', 'vol_la'}
        return inputs
        
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)
    
        p |= self._conc_aol(p)

        fCO_la = p['ffa'] * p['fCO_li']
        fCO_pv = (1 - p['ffa']) * p['fCO_li']

        C_la = p['J_la'].reshape(1, -1) / p['CO']
        C_pv = p['J_pv'].reshape(1, -1) / p['CO']

        p |= {
            'F_b_la': fCO_la * p['CO'] / p['vol_la'],
            'ci_la': p['J_ao'].reshape(1, -1) / p['CO'],
            'C_la': C_la,
            
            'F_b_pv': fCO_pv * p['CO'] / p['vol_pv'],
            'ci_pv': p['J_ao'].reshape(1, -1) / p['CO'],
            'C_pv': C_pv,
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
    def inputs(self):
        imap = {'F_p':'F_p_ki', 'v_p': 'v_p_ki'}
        inputs = {imap.get(i, i) for i in pk_kidney.PARAMETERS[self.config['kinetics']]}
        inputs |= {'c_ar', 'T_ar', 'dt'}
        return inputs
    
    def outputs(self):
        return {'tC_ki', 'C_ki', 'ci_ki', 'F_u'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        func = 'conc_kidney_' + self.config['kinetics'].lower()   
        kidney_conc = getattr(pk_kidney, func)

        imap = {'F_p_ki':'F_p', 'v_p_ki': 'v_p'}
        pk = {imap.get(k, k): v for k, v in p.items() if k not in ['c_ar', 'T_ar']} 

        ci_k = flux_plug(p['c_ar'], T=p['T_ar'], dt=p['dt'])
        C_ki = kidney_conc(ci_k, **pk)
        tC_ki = p['dt'] * np.arange(C_ki.shape[1])
        results = {'tC_ki':tC_ki, 'ci_ki':ci_k, 'C_ki':C_ki}

        results['F_u'] = p['F_u'] if 'F_u' in p else p['FF'] * p['F_p_ki']

        return self.map_results(results)

    def dummy_data(self, nt=5):
        data = self.init_data()
        data['c_ar'] *= np.ones(nt)
        return data


class ConcAortaKidneys(Module):
    """Concentration in aorta and both kidneys.
    """
    configs = {
        'bolus': FluxAorta.configs['bolus'],
        'heartlung': FluxAorta.configs['heartlung'],
        'organs': FluxAorta.configs['organs'],
        'kidneys': ConcKidney.configs['kinetics'],
    }
    defaults = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'kidneys': '2CF',
        'bolus': 'single',
    }
    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)

        # Setup aorta module
        kidney_vasc = pk_kidney.VASCULAR_MODEL[self.config['kidneys']]
        config_aorta = self.config | {'kidneys': kidney_vasc}
        self._flux_aorta = FluxAorta(**config_aorta)

        # Setup Kidney modules
        imap = {'F_p_ki':'F_p_lk', 'v_p_ki':'v_p_lk', 'F_u':'F_u_lk', 'FF':'FF_lk', 'T_u':'T_u_lk', 'h_u':'h_u_lk'}
        self._conc_lk = ConcKidney(imap=imap, kinetics=self.config['kidneys'])

        imap = {'F_p_ki':'F_p_rk', 'v_p_ki':'v_p_rk', 'F_u':'F_u_rk', 'FF':'FF_rk', 'T_u':'T_u_rk', 'h_u':'h_u_rk'}
        self._conc_rk = ConcKidney(imap=imap, kinetics=self.config['kidneys'])

        self.map_io(imap, omap, iomap)

    def inputs(self):
        inputs = {'H', 'fCO_ki', 'DRPF', 'CO', 'vol_ao'}
        inputs |= self._conc_lk.mapped_inputs()
        inputs |= self._conc_rk.mapped_inputs()
        inputs |= self._flux_aorta.mapped_inputs()
        inputs |= {'E_li', 'vol_lk', 'vol_rk'}
        inputs -= {'c_ar', 'F_p_lk', 'T_p_lk', 'vr_lk', 'F_p_rk', 'T_p_rk', 'vr_rk', 'vr_or'}
        return inputs

    def outputs(self):
        outputs = self._flux_aorta.outputs()
        outputs |= {'F_u_lk', 'F_u_rk'}
        for roi in ['ao', 'lk', 'rk']:
            outputs |= {f'F_b_{roi}', f'ci_{roi}', f'C_{roi}'}
        return outputs
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        # Derived parameters
        fco_lk = p['fCO_ki'] * p['DRPF']
        fco_rk = p['fCO_ki'] * (1 - p['DRPF'])
        
        Fp_lk = fco_lk * p['CO'] * (1 - p['H']) / p['vol_lk']
        Fp_rk = fco_rk * p['CO'] * (1 - p['H']) / p['vol_rk']

        FF_lk = p['F_u_lk'] / Fp_lk if 'HF' in self.config['kidneys'] else p['FF_lk']
        FF_rk = p['F_u_rk'] / Fp_rk if 'HF' in self.config['kidneys'] else p['FF_rk']

        E_lk = FF_lk / (1 + FF_lk)
        E_rk = FF_rk / (1 + FF_rk)

        Fu_lk = p['F_u_lk'] if 'HF' in self.config['kidneys'] else FF_lk * Fp_lk
        Fu_rk = p['F_u_rk'] if 'HF' in self.config['kidneys'] else FF_rk * Fp_rk

        # Extend data dictionary
        p |= {
            'F_u_lk': Fu_lk,
            'F_u_rk': Fu_rk,
            'F_p_lk': Fp_lk,
            'F_p_rk': Fp_rk,
            'T_p_lk': p['v_p_lk'] / Fp_lk,
            'T_p_rk': p['v_p_rk'] / Fp_rk,
            'vr_lk': fco_lk * (1 - E_lk),
            'vr_rk': fco_rk * (1 - E_rk),
            'vr_or': (1 - p['fCO_ki']) * (1 - p['E_li']),
        }

        # Aorta flux
        p |= self._flux_aorta(p)

        # Kidney concentration
        cb = p['J_ao'] / p['CO']
        p['c_ar'] = cb / (1 - p['H'])
        lk = self._conc_lk(p)
        rk = self._conc_rk(p)

        # Save output
        C_a = p['J_ao'].reshape(1, -1) / p['CO']

        p |= {
            'F_b_ao': p['CO'] / p['vol_ao'],
            'ci_ao': C_a,
            'C_ao': C_a, 
            
            'F_b_lk': fco_lk * p['CO'] / p['vol_lk'],
            'ci_lk': cb.reshape(1, -1),
            'C_lk': lk['C_ki'], 
            
            'F_b_rk': fco_rk * p['CO'] / p['vol_rk'],
            'ci_rk': cb.reshape(1, -1),
            'C_rk': rk['C_ki'],
        }
        return self.map_results(p)

    
    
class ConcCortMed(Module):
    """Concentration in kidney cortex and medulla.
    """
    configs = {
        'kinetics': set(pk_kidney.CM_PARAMETERS.keys()),
    }
    defaults = {
        'kinetics': '7C'
    }
    def inputs(self):
        imap = {'F_p':'F_p_ki', 'E':'E_ki'}
        inputs = {imap.get(i, i) for i in pk_kidney.CM_PARAMETERS[self.config['kinetics']]}
        inputs |= {'c_ar', 'T_ar', 'dt'}
        return inputs
    
    def outputs(self):
        return {'tC', 'ci_ki', 'C_kc', 'C_km'}
    
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        imap = {'F_p_ki':'F_p', 'E_ki':'E'}
        pk = {imap.get(k, k): v for k, v in p.items() if k not in ['c_ar', 'T_ar']} 

        ci_k = flux_plug(p['c_ar'], T=p['T_ar'], dt=p['dt'])
        if self.config['kinetics'] == '7C':
            Ccor, Cmed = pk_kidney.conc_kidney_cm9(ci_k, **pk)

        tC = p['dt'] * np.arange(ci_k.size)
        results = {'tC':tC, 'ci_ki':ci_k, 'C_kc': Ccor, 'C_km': Cmed}
        
        return self.map_results(results)

    def dummy_data(self, nt=5):
        data = self.init_data()
        data['c_ar'] *= np.ones(nt)
        return data


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
    def __call__(self, data: dict=None, **kwargs):
        p = self.map_data(data, kwargs)

        func = 'conc_tissue_' + self.config['kinetics'].lower()   
        conc = getattr(pk_tissue, func)  

        px = {k: v for k, v in p.items() if k not in ['c_ar','T_ar']}

        ci = flux_plug(p['c_ar'], T=p['T_ar'], dt=p['dt'])
        C = conc(ci, **px)
        tC = p['dt'] * np.arange(p['c_ar'].size)

        results = {'tC':tC, 'ci':ci, 'C': C}

        return self.map_results(results)

    def inputs(self):
        model = self.config['kinetics']
        inputs = set(pk_tissue.CONC_PARAMETERS[model])
        inputs |= {'c_ar', 'T_ar', 'dt'}
        return inputs
    
    def outputs(self):
        return {'tC', 'ci', 'C'}

    def dummy_data(self, nt=5):
        data = self.init_data()
        data['c_ar'] *= np.ones(nt)
        return data