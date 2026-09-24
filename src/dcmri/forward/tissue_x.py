import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcTissueX
from dcmri.relaxivity.modules_rois import RelaxivityTissueX
from dcmri.bloch.modules_rois import WaterExchangeTissueX
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels


configs = ConcToSignal.configs | WaterExchangeTissueX.configs | RelaxivityTissueX.configs | ConcTissueX.configs
defaults = ConcToSignal.defaults | WaterExchangeTissueX.defaults | RelaxivityTissueX.defaults | ConcTissueX.defaults

configs['inflow'].discard('inlet')
configs.pop('tof_corr')
defaults.pop('tof_corr')

class ForwardTissueX(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {}
    _all_outputs = {}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._conc = ConcTissueX(**self.config)
        self._tissue_rel = RelaxivityTissueX(**self.config)
        self._tissue_wex = WaterExchangeTissueX(**self.config)
        self._conc_to_signal = ConcToSignal(**self.config)

        self.map_io(imap, omap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._conc(p)
        p |= self._tissue_rel(p) 
        p |= self._tissue_wex(p) 

        p['tacq'] = p['dt'] * (p['ci'].size - 1)
        p |= self._conc_to_signal(p)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        inputs |= self._tissue_rel.mapped_inputs() 
        inputs |= self._tissue_wex.mapped_inputs() 
        inputs |= self._conc_to_signal.mapped_inputs()

        inputs -= {'tacq'}
        inputs -= self._conc.new_mapped_outputs()
        inputs -= self._tissue_rel.new_mapped_outputs()
        inputs -= self._tissue_wex.new_mapped_outputs()
        inputs -= self._conc_to_signal.new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._conc_to_signal.mapped_outputs() 
        return outputs
    
    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            f'iScal': np.arange(n0, dtype=int),
            f'Scal': Scal, 
        }
        nt = 180
        ci = np.ones(nt)
        data['c_ar'] = ci
        return data