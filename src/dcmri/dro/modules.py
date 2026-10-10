from copy import deepcopy

import numpy as np

from dcmri.core.module import Module
from dcmri.dro.functions_aif import parker
from dcmri.signal.modules_tissue import ConcToSignal

PARKER_INPUTS = {
    'dt': 0.1,
    'BAT': 30,
    'H': 0.45, # Estimate - not reported in the original paper

    'agent': 'gadodiamide',
    'dose': 0.2,
    'rate': 3,
    'field_strength': 1.5,
    'vol_ao': 1, # Not reported in the paper - small value assures no dispersion
    'weight': 70, # Not reported in the paper

    # Signal group
    'NSR': 0, # Not known so let's ignore the effect
    'S0': 100, # Not know but can be arbitrarily chosen

    # Sequence group
    'FA': 20,
    'Nk0': 64, # CHECK
    'Nph': 128, # CHECK
    'TE': 0.00082,
    'TR': 0.004,
    'tacq': 375,

    # Electromagnetic group
    'R1_b': 1 / 1.441, # T1 blood at 1.5T
    'R2s_b': 1 / 0.2,
    'r1': 4.3 * 1e3, # r1 of gadodiamide
    'r2s': 10 * 1e3,
}

PARKER_OUTPUTS = {
    'agent': 'gadodiamide',
    'dose': 0.2,
    'rate': 3,
    'field_strength': 1.5,
    'vol_ao': 1, # Not reported in the paper - small value assures no dispersion
    'weight': 70, # Not reported in the paper
}

class Parker(Module):
    def __init__(self, imap: dict=None, omap: dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self._conc2sig = ConcToSignal()
        self.map_io(imap, omap, iomap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        # --- Initialise with values from the paper
        p = deepcopy(PARKER_INPUTS | PARKER_OUTPUTS)

        # --- Override any user-defined inputs
        p |= self.map_data(data, kwargs, all=False)

        # --- Compute plasma concentration
        t = np.arange(0, p['tacq'] + p['dt'], p['dt'])
        cp = parker(t, BAT=p['BAT'])

        # --- Convert to blood concentration
        cb = (1 - p['H']) * cp 

        # --- Compute signals
        p |= {'tC': t, 'C': cb}
        p |= self._conc2sig(p, R1b=p['R1_b'], R2sb=p['R2s_b'])
        
        return self.map_results(p) 

    def inputs(self) -> set:
        return set(PARKER_INPUTS.keys())

    def outputs(self) -> set:
        outputs = set(PARKER_INPUTS.keys()) 
        outputs |= set(PARKER_OUTPUTS.keys()) 
        outputs |= {'tC', 'C'}
        outputs |= self._conc2sig.mapped_outputs()   
        return outputs   

    def test_data(self, data: dict=None): 
        p = self.init_data()
        return self.input_data(p, data)