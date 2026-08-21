import numpy as np

from dcmri.core.quantities import QVALUES
from dcmri.core.module import Module
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAortaPortalLiver
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.relaxivity.modules_water_exchange import WaterExchangeFX, WaterExchangeLiver
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal, ConcToSignal


roi_quantities = {
    'v', 'fx', 'Fi', 'Fw', 
    'c', 'ci', 
    'R1b', 'R2b', 'R2sb', 'R1ib', 'R1', 'R2', 'R2s', 'R1i', 
    'tM', 'M', 
    'B1corr', 'Sb', 'S0', 'tS', 'S' 
}
iomap = {
    roi: {k:f'{k}_{roi}' for k in roi_quantities}
    for roi in ['a', 'l', 'pv', 'la']
}


class AortaPortalLiverModel(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = ConcAortaPortalLiver.configs | WaterExchangeLiver.configs | Relax.configs | Magnetization.configs | {
        't2s_relaxation': {None, 'lin', 'quad'},
        'magnitude': Signal.configs['magnitude'],
        'calibrate': [True, False],
    }
    defaults = ConcAortaPortalLiver.defaults | WaterExchangeLiver.defaults | Relax.defaults | Magnetization.defaults | {
        'magnitude': Signal.defaults['magnitude'],
        'calibrate': False,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._conc = ConcAortaPortalLiver(**self.config)
        self._wex = {
            'a': WaterExchangeFX(imap=iomap['a'], omap=iomap['a'], **self.config),
            'pv': WaterExchangeFX(imap=iomap['pv'], omap=iomap['pv'], **self.config),
            'la': WaterExchangeFX(imap=iomap['la'], omap=iomap['la'], **self.config),
            'l': WaterExchangeLiver(**self.config), 
        }
        self._signal = {
            roi: ConcToSignal(
                imap=iomap[roi] | {'tR':'t'},
                omap=iomap[roi],
                **self.config,
            ) for roi in ['a', 'pv', 'la', 'l']
        }
        self.map_io(imap, omap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent'}
        inputs |= self._conc.mapped_inputs()
        for roi in ['a', 'pv', 'la', 'l']:
            inputs |= self._wex[roi].mapped_inputs() 
            inputs |= self._signal[roi].mapped_inputs()

        inputs -= {'r1', 'r2', 'r2s', 'r1i'}
        inputs -= {f'R1ib_{roi}' for roi in ['a', 'pv', 'la', 'l']} 
        inputs -= self._conc.mapped_outputs()
        for roi in ['a', 'pv', 'la', 'l']:
            inputs -= self._wex[roi].mapped_outputs()
            inputs -= (self._signal[roi].mapped_outputs() - self._signal[roi].mapped_inputs())
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in ['a', 'pv', 'la', 'l']:
            outputs |= self._signal[roi].mapped_outputs() 
        return outputs
    
    def lexicon_data(self): 
        p = {'v_a': 1, 'v_pv': 1, 'v_la': 1, 'PSw': 1}
        for roi in ['a', 'pv', 'la', 'l']:
            nc=2 if roi=='l' else 1
            p |= self._signal[roi].lexicon_data(QVALUES, nc=nc)
        return self.update_data(QVALUES | p)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Inlet R1
        if 'R1b_a' in p:
            p['R1ib_a'] = p['R1b_a']
            p['R1ib_pv'] = p['R1b_a']
            p['R1ib_la'] = p['R1b_a']
            p['R1ib_l'] = [p['R1b_a'][0], np.nan]

        # Relaxivities
        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = const.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        rel = {
            'a': {'r1': rb['r1'], 'r2': rb['r2'], 'r2s': rb['r2s'], 'r1i': rb['r1']},
            'pv': {'r1': rb['r1'], 'r2': rb['r2'], 'r2s': rb['r2s'], 'r1i': rb['r1']},
            'la': {'r1': rb['r1'], 'r2': rb['r2'], 'r2s': rb['r2s'], 'r1i': rb['r1']},
            'l': {'r1': [rb['r1'], rh['r1']], 'r2': [rb['r2'], rh['r2']], 'r2s': rb['r2s'], 'r1i': [rb['r1'], np.nan]}
        }

        # Compute
        p |= self._conc(p)
        for roi in ['a', 'pv', 'la', 'l']:
            p |= self._wex[roi](p) 
            p |= self._signal[roi](p, **rel[roi])

        return self.map_results(p)