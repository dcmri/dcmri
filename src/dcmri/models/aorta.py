from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.relaxivity.modules_water_exchange import WaterExchangeFX
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal, CalibrateSignal


class AortaModel(Module):
    """Whole-body model for the aorta signal."""

    configs = ConcAorta.configs | Relax.configs | {
        't2s_relaxation': {None, 'lin', 'quad'},
        'sequence': Magnetization.configs['sequence'],
        'magnitude': Signal.configs['magnitude'],
        'calibrate': [True, False],
    }
    defaults = ConcAorta.defaults | Relax.defaults | {
        'sequence': Magnetization.defaults['sequence'],
        'magnitude': Signal.defaults['magnitude'],
        'calibrate': False,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Configure modules
        self._conc = ConcAorta(**config)
        self._wex = WaterExchangeFX(
                imap={'Fi':'Fi_a'}, 
                omap={'fx':'fx_a', 'Fw':'Fw_a'}, 
                **config,
            )
        self._relax_tissue = Relax( 
            sequence = self.config['sequence'],
            imap = {'v':'v_a', 'c':'c_a', 'fx':'fx_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'r1':f'r1_a', 'r2':f'r2_a', 'r2s':f'r2s_a'},
            omap = {'v':'v_a', 'R1':'R1_a', 'R2':'R2_a', 'R2s':'R2s_a'},
        )
        self._relax_inlets = Relax( 
            sequence = self.config['sequence'],
            imap = {'v':'Fi_a', 'c':'c_a', 'fx':'fx_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'r1':f'r1i_a'},
            omap = {'v':'Fi_a', 'R1':'R1i_a'},
        )
        self._magn = Magnetization(
            imap = {'tR':'t', 'R1':'R1_a', 'R2':'R2_a', 'R2s':'R2s_a', 'R1i':'R1i_a', 'B1corr':'B1corr_a', 'v':'v_a', 'Fw':'Fw_a', 'Fi':'Fi_a'}, 
            omap = {'tM':'tM_a', 'M':'M_a'},
            inflow=True, **config, # make this a config setting
        )
        if self.config['calibrate']:
            self._derive_s0 = CalibrateSignal(
                imap={'tSb':'tSb_a', 'Sb':'Sb_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'R1ib':'R1b_a', 'B1corr': 'B1corr_a', 'v':'v_a', 'Fw':'Fw_a', 'Fi':'Fi_a'}, 
                omap={'S0':'S0_a'},
                inflow=True, **config,
            ) 
        self._signal = Signal(
            imap = {'tM':'tM_a', 'M':'M_a', 'S0':'S0_a', 'B1corr':'B1corr_a'},
            omap = {'tS':'tS_a', 'S':'S_a'},
            **config
        )
        self.map_io(imap, omap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent', 'CO', 'vol_a'}
        inputs |= self._conc.mapped_inputs() 
        inputs |= self._wex.mapped_inputs() 
        inputs |= self._relax_tissue.mapped_inputs() 
        inputs |= self._relax_inlets.mapped_inputs() 
        inputs |= self._magn.mapped_inputs() 
        inputs |= self._signal.mapped_inputs() 
        if self.config['calibrate']:
            inputs |= self._derive_s0.mapped_inputs() 

        # Remove constants and derived parameters
        inputs -= {'r1_a', 'r2_a', 'r2s_a', 'r1i_a', 'R1ib_a'}
        inputs -= self._conc.mapped_outputs()
        inputs -= self._wex.mapped_outputs()
        inputs -= self._relax_tissue.mapped_outputs() 
        inputs -= {'R1i_a'}
        inputs -= self._magn.mapped_outputs()
        if self.config['calibrate']:
            inputs -= self._derive_s0.mapped_outputs() 
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._relax_tissue.mapped_outputs() 
        if 'R1_a' in outputs:
            outputs |= self._relax_inlets.mapped_outputs() 
        outputs |= self._magn.mapped_outputs()
        if self.config['calibrate']:
            outputs |= self._derive_s0.mapped_outputs()
        outputs |= self._signal.mapped_outputs()
        # remove uninformative outputs
        outputs -= {'vi_a', 'v_a', 'c_a', 'ci_a'}
        return outputs
    
    def map_lexicon(self, qvalues): 
        p = {'v_a': 1}
        if self.config['calibrate']:
            p |= self._derive_s0.map_lexicon(qvalues)
        return self.update_data(p)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Constants
        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        p |= {
            'r1_a': rb['r1'],
            'r2_a': rb['r2'],
            'r2s_a': rb['r2s'],
            'r1i_a': rb['r1'],
        }

        results = self._conc(p) 
        results |= self._wex(p, **results)
        results |= self._relax_tissue(p, **results) 
        if 'R1_a' in results:
            results |= self._relax_inlets(p, **results)
        results |= self._magn(p, **results) 
        if self.config['calibrate']:
            results |= self._derive_s0(p, **results)
        results |= self._signal(p, **results) 

        return self.map_results(results)