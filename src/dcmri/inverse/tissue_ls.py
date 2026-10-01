
from copy import deepcopy

import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.functions_tissue_ls import irf_ls
from dcmri.inverse.sig2conc import SignalToConc
from dcmri.bloch.functions_sequences import channels

configs = deepcopy(SignalToConc.configs) 
defaults = deepcopy(SignalToConc.defaults)

class InverseTissueLS(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'tstart', 'TD', 'Nph', 'TA', 'r1', 'PA', 'S0', 'r2s', 'nb', 'tacq', 'S', 'Nk0', 'TE', 'NSR', 'R1b', 'dt', 'TP', 'TE1', 'B1corr', 'FA', 'tS', 'TR', 'TE2', 'iz', 'ci', 'r2'}
    _all_outputs = {'C', 'irf', 'tC'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._signal_to_conc = SignalToConc(**self.config)
        self.map_io(imap, omap)

    def __call__(self, data: dict=None, tol=1e-2) -> dict:
        p = self.map_data(data)  

        C = self._signal_to_conc(p, S=p['S'][0,0,:])

        t = p['dt'] * np.arange(p['ci'].size)
        C = np.interp(t, p['tS'], C['C'])

        irf = irf_ls(p['ci'], C, p['dt'], tol=tol)

        return self.map_results({'irf': irf, 'tC': t, 'C': C.reshape(1, -1)})

    def inputs(self) -> set:
        inputs = self._signal_to_conc.mapped_inputs()
        inputs |= {'tS', 'S', 'dt', 'ci'}
        return inputs  
    
    def outputs(self):
        outputs = {'irf', 'tC', 'C'}
        return outputs
    
    def dummy_data(self, data: dict=None): 
        nt = 180
        p = self.init_data()

        n_channels = channels(self.config['sequence'])
        components = 1
        S = np.ones((n_channels, components, nt))
        S[:, :, :30] = 0
        S += 1

        ci = p['ci'] * np.ones(nt)
        ci[:30] = 0

        p |= {
            'nb': 5,
            'ci': ci,
            'tS': p['dt'] * np.arange(nt),
            'S': S,
        }
        return self.input_data(p, data)