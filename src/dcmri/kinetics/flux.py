import copy
import numpy as np

import dcmri.kinetics.lib.tissue as pk_tissue
from dcmri.kinetics.lib.blocks import flux_plug
from dcmri.core.layer import LayerFunction

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

        ca = flux_plug(ca, p['T_a'], dt=dt)
        params = {k: v for k, v in p.items() if k != 'T_a'}

        kin = self._cnfg['kinetics']
        conc = 'flux_tissue_' + kin.lower()   
        model_func = getattr(pk_tissue, conc)  
        return model_func(ca, t=t, dt=dt, **params)



