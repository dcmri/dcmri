from copy import deepcopy

import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train
from dcmri.forward.liver import ForwardLiver as Forward

configs = deepcopy(Forward.configs) 
defaults = deepcopy(Forward.defaults)

class InverseLiver(Module):

    configs = configs
    defaults = defaults

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        nt = len(time)
        return pred['S_li'][:, :, :nt].reshape(-1)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  

        if self.config['calibrate']:
            p['Scal_li'] = p['S_li'][..., :p['nb']]
            p['iScal_li'] = np.arange(p['nb'])

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = data['tS_li']
        signal = data['S_li']
        p |= train(self._predict, time, signal, p, p['pfree'], **kwargs)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'S_li', 'nb'}
            inputs -= {'Scal_li', 'iScal_li'}
        inputs |= {'tS_li', 'S_li', 'pfree'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'loss'}
        return outputs
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'v_e_li': (0, 1), 'F_p_li': (0, 1)}),
            'tS_li': pred['tS_li'],
            'S_li': pred['S_li'],
        }
        return self.input_data(p, data)

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        return {p: get_quantity(p)['bounds'] for p in pfree}