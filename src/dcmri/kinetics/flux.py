import copy
import numpy as np

from dcmri.utils import const
from dcmri.kinetics.input import ca_injection
import dcmri.kinetics.tissue as pk_tissue
import dcmri.kinetics.blocks as blocks
from dcmri.kinetics.blocks import flux_plug
from dcmri.core.layer import LayerFunction
from dcmri.core.function import Function


class FluxBlock(Function):
    configs = {
        'block': [
            'trap', 
            'pass', 
            'comp', 
            'bicomp', 
            'plug', 
            'chain', 
            'step', 
            'free', 
            'ncomp', 
            'nscomp', 
            'mmcomp', 
            '2cxm',
        ],
    }
    def __init__(self, block='comp', **defaults):
        cnfg = {
            'block': block, 
        }
        self._set_config(**cnfg)
        self._set_params(**defaults)

    def _param_names(self):
        params = {
            'trap': [], 
            'pass': [], 
            'comp': ['T'], 
            'bicomp': ['T'], 
            'plug': ['T'], 
            'chain': ['T', 'D'], 
            'step': ['T', 'D'], 
            'pfcomp': ['T', 'D'], 
            'free': ['h', 'TT', 'TTmin', 'TTmax'], 
            'ncomp': ['T', 'E', 'solver', 'dt_prop'], 
            'nscomp': ['T'], 
            'mmcomp': ['Vmax', 'Km', 'solver'], 
            '2cxm': ['T', 'E'],
        }[self._cnfg['block']]

        return params
    
    def __call__(self, J, t=None, dt=1.0, **kwargs) -> np.ndarray:
        """Aorta indicator concentration.

        Args: 
            **params: override parameter defaults.

        Returns:
            np.ndarray: Aorta blood concentration.
        """
        p = self._update_params(**kwargs)
        model = self._cnfg['block']
        model_func = getattr(blocks, f"flux_{model}")  
        return model_func(J=J, t=t, dt=dt, **p)


class FluxTissueX(LayerFunction):
    """Flux out of vascular-interstitial tissue.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        params (dict, optional): override parameter defaults.
    """

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
        """Flux out of tissue.

        Args:
            ca (np.ndarray): concentrations in arterial blood.
            t (np.ndarray): time points of ca.
            dt (float): time interval (if uniform).
            params (dict, optional): override parameter defaults.

        Returns:
            np.ndarray: Tissue flux.
        """
        p = self._update_pars(**params)

        ca = flux_plug(ca, dt=dt, T=p['T_a'])
        params = {k: v for k, v in p.items() if k != 'T_a'}

        kin = self._cnfg['kinetics']
        conc = 'flux_tissue_' + kin.lower()   
        model_func = getattr(pk_tissue, conc)  
        return model_func(ca, t=t, dt=dt, **params)
    

# class FluxAortaHLO(LayerFunction):
#     configs = {
#         'heartlung': ['comp', 'pfcomp', 'chain'],
#         'organs': ['comp', '2cxm'],
#     }
#     def __init__(self, heartlung='pfcomp', organs='comp', **params):
#         cnfg = {
#             'heartlung': heartlung, 
#             'organs': organs, 
#         }
#         self._cnfg = self._set_config(**cnfg)
#         self._pars = self._set_pars(**params)

#     def _params(self, select=None):
#         organs = {
#             'comp': ['To'],
#             '2cxm': ['To', 'To_e', 'Eo']
#         }[self._cnfg['organs']]

#         heartlung = {
#             'comp': ['Thl'],
#             'pfcomp': ['Thl', 'Dhl'],
#             'chain': ['Thl', 'Dhl'],
#         }[self._cnfg['heartlung']]

#         body = ['BAT', 'CO', 'Eb']
#         const = [
#             'dt', 'tmax', 'dose_tolerance', 'field_strength',
#             'agent', 'weight', 'dose', 'rate', 
#         ]
#         if select is None:
#             return heartlung + organs + body + const 
#         if select=='body':
#             return heartlung + organs + body 
    
#     def __call__(self, **params) -> np.ndarray:
#         """Aorta indicator concentration.

#         Args: 
#             **params: override parameter defaults.

#         Returns:
#             np.ndarray: Aorta blood concentration.
#         """
#         p = self._update_pars(**params)
#         t = np.arange(0, p['tmax'], p['dt'])

#         hl, orgs = self._cnfg['heartlung'], self._cnfg['organs']

#         if hl=='comp':
#             heartlung = ['comp', (p['Thl'],)]
#         elif hl=='pfcomp':
#             heartlung = ['pfcomp', (p['Thl'], p['Dhl'])]
#         elif hl=='chain':
#             heartlung = ['chain', (p['Thl'], p['Dhl'])]

#         if orgs=='comp':
#             organs = ['comp', (p['To'],)]
#         elif orgs=='2cxm':
#             organs = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]

#         conc = const.ca_conc(p['agent'])
#         Ji = ca_injection(
#             t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
#         )
#         Jb = flux_aorta(
#             Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
#             heartlung=heartlung, organs=organs,
#         )
#         return Jb / p['CO']