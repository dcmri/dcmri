from dcmri.core.quantities import QVALUES
from dcmri.core.module import Module
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.relaxivity.modules_water_exchange import WaterExchangeFX
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal, ConcToSignal


roi_quantities = {
    'v', 'fx', 'Fi', 'Fw', 
    'c', 'ci', 
    'R1b', 'R2b', 'R2sb', 'R1ib', 'R1', 'R2', 'R2s', 'R1i', 
    'tM', 'M', 
    'B1corr', 'Sb', 'S0', 'tS', 'S' 
}
iomap = {k:f'{k}_a' for k in roi_quantities}


class AortaModel(Module):
    """Whole-body model for the aorta signal."""

    configs = ConcAorta.configs | WaterExchangeFX.configs | Relax.configs | Magnetization.configs | {
        't2s_relaxation': {None, 'lin', 'quad'},
        'magnitude': Signal.configs['magnitude'],
        'calibrate': [True, False],
    }
    defaults = ConcAorta.defaults | WaterExchangeFX.defaults | Relax.defaults | Magnetization.defaults | {
        'magnitude': Signal.defaults['magnitude'],
        'calibrate': False,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        
        self._conc = ConcAorta(**self.config)
        self._wex = WaterExchangeFX(
            imap=iomap, 
            omap=iomap, 
            **self.config,
        )
        self._signal = ConcToSignal(
            imap=iomap | {'tR':'t'},
            omap=iomap,
            **self.config,
        ) 
        self.map_io(imap, omap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent'}
        inputs |= self._conc.mapped_inputs()
        inputs |= self._wex.mapped_inputs() 
        inputs |= self._signal.mapped_inputs()

        inputs -= {'r1', 'r2', 'r2s', 'r1i'}
        inputs -= {'R1ib_a'} 
        inputs -= self._conc.mapped_outputs()
        inputs -= self._wex.mapped_outputs()
        inputs -= (self._signal.mapped_outputs() - self._signal.mapped_inputs())
        return inputs  
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._signal.mapped_outputs() 
        # remove uninformative outputs
        outputs -= {'vi_a', 'v_a', 'c_a', 'ci_a'}
        return outputs
    
    def lexicon_data(self): 
        p = {'v_a': 1}
        p |= self._signal.lexicon_data(QVALUES, nc=1)
        return self.update_data(QVALUES | p)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Constants
        if 'R1b_a' in p:
            p['R1ib_a'] = p['R1b_a']

        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rel = {'r1': rb['r1'], 'r2': rb['r2'], 'r2s': rb['r2s'], 'r1i': rb['r1']}

        # Compute
        p |= self._conc(p)
        p |= self._wex(p) 
        p |= self._signal(p, **rel)

        return self.map_results(p)