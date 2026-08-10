import numpy as np

from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.relaxivity.modules_water_exchange import WaterExchangeFX, WaterExchangeLiver
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal, CalibrateSignal


class AortaLiverModel(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = ConcAortaLiver.configs | WaterExchangeLiver.configs | Relax.configs  | {
        't2s_relaxation': {None, 'lin', 'quad'},
        'sequence': Magnetization.configs['sequence'],
        'magnitude': Signal.configs['magnitude'],
        'calibrate': [True, False],
    }
    defaults = ConcAortaLiver.defaults | WaterExchangeLiver.defaults | Relax.defaults | {
        'sequence': Magnetization.defaults['sequence'],
        'magnitude': Signal.defaults['magnitude'],
        'calibrate': False,
    }
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Configure modules
        self._conc = ConcAortaLiver(**config)
        self._wex = {
            'a': WaterExchangeFX(
                imap={'Fi':'Fi_a'}, 
                omap={'fx':'fx_a', 'Fw':'Fw_a'}, 
                **config,
            ),
            'l': WaterExchangeLiver(**config), 
        }

        # self._conc_to_signal[roi] = ConcToSignal(...)   
        self._relax_tissue = {}
        self._relax_inlets = {}
        if self.config['calibrate']:
            self._derive_s0 = {}
        self._magn = {}
        self._signal = {}
        for roi in ['a', 'l']:
            self._relax_tissue[roi] = Relax(
                sequence = self.config['sequence'],
                imap = {'v':f'v_{roi}', 'c':f'c_{roi}', 'fx':f'fx_{roi}', 'R1b':f'R1b_{roi}', 'R2b':f'R2b_{roi}', 'R2sb':f'R2sb_{roi}', 'r1':f'r1_{roi}', 'r2':f'r2_{roi}', 'r2s':f'r2s_{roi}'},
                omap = {'v':f'v_{roi}', 'R1':f'R1_{roi}', 'R2':f'R2_{roi}', 'R2s':f'R2s_{roi}'},
            )
            self._relax_inlets[roi] = Relax(
                sequence = self.config['sequence'],
                imap = {'v':f'Fi_{roi}', 'c':f'ci_{roi}', 'fx':f'fx_{roi}', 'R1b':f'R1ib_{roi}', 'r1':f'r1i_{roi}'},
                omap = {'v':f'Fi_{roi}', 'R1':f'R1i_{roi}'},
            ) 
            self._magn[roi] = Magnetization(
                imap = {'tR':'t', 'R1':f'R1_{roi}', 'R2':f'R2_{roi}', 'R2s':f'R2s_{roi}', 'R1i':f'R1i_{roi}', 'v':f'v_{roi}', 'Fw':f'Fw_{roi}', 'Fi':f'Fi_{roi}', 'B1corr':f'B1corr_{roi}'}, 
                omap = {'tM':f'tM_{roi}', 'M':f'M_{roi}'},
                inflow=True, # Make this configurable
                **config,
            ) 
            if self.config['calibrate']:
                self._derive_s0[roi] = CalibrateSignal(
                    imap={'tSb':f'tSb_{roi}', 'Sb':f'Sb_{roi}', 'R1b':f'R1b_{roi}', 'R2b':f'R2b_{roi}', 'R2sb':f'R2sb_{roi}', 'R1ib':f'R1ib_{roi}', 'B1corr': f'B1corr_{roi}', 'v':f'v_{roi}', 'Fw':f'Fw_{roi}', 'Fi':f'Fi_{roi}'}, 
                    omap={'S0':f'S0_{roi}'},
                    inflow=True, **config,
                )
            self._signal[roi] = Signal(
                    imap = {'tM':f'tM_{roi}', 'M':f'M_{roi}', 'S0':f'S0_{roi}', 'B1corr':f'B1corr_{roi}'},
                    omap = {'tS':f'tS_{roi}', 'S':f'S_{roi}'},
                    **config
            )
        self.map_io(imap, omap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent', 'CO', 'vol_a', 'fCO_l'}
        inputs |= self._conc.mapped_inputs()
        for roi in ['a', 'l']:
            inputs |= self._wex[roi].mapped_inputs() 
            inputs |= self._relax_tissue[roi].mapped_inputs()
            inputs |= self._relax_inlets[roi].mapped_inputs()
            inputs |= self._magn[roi].mapped_inputs()
            inputs |= self._signal[roi].mapped_inputs()
            if self.config['calibrate']:
                inputs |= self._derive_s0[roi].mapped_inputs() 

        inputs -= {'r1_a', 'r2_a', 'r2s_a', 'r1i_a', 'R1ib_a'} 
        inputs -= {'r1_l', 'r2_l', 'r2s_l', 'r1i_l', 'r2i_l', 'r2si_l', 'R1ib_l'} 
        inputs -= self._conc.mapped_outputs()
        for roi in ['a', 'l']:
            inputs -= self._wex[roi].mapped_outputs()
            inputs -= self._relax_tissue[roi].mapped_outputs() 
            inputs -= {f'R1i_{roi}'}
            inputs -= self._magn[roi].mapped_outputs()
            if self.config['calibrate']:
                inputs -= self._derive_s0[roi].mapped_outputs() 
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in ['a', 'l']:
            outputs |= self._relax_tissue[roi].mapped_outputs() 
            if f'R1_{roi}' in outputs:
                outputs |= self._relax_inlets[roi].mapped_outputs() 
            outputs |= self._magn[roi].mapped_outputs()
            if self.config['calibrate']:
                outputs |= self._derive_s0[roi].mapped_outputs()
            outputs |= self._signal[roi].mapped_outputs()
        return outputs
    
    def map_lexicon(self, qvalues: dict=None): 
        p = {'v_a': 1, 'PSw': 1}
        for roi in ['a', 'l']:
            if self.config['calibrate']:
                p |= self._derive_s0[roi].map_lexicon(qvalues)

        p['R1b_l'] = [qvalues['R1b_l'], qvalues['R1b_l']]
        p['R2b_l'] = [qvalues['R2b_l'], qvalues['R2b_l']]
        return self.update_data(p)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Inlet R1
        if 'R1b_a' in p:
            p['R1ib_a'] = [p['R1b_a']]
            p['R1ib_l'] = [p['R1b_a'], np.nan]

        # Relaxivities
        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = const.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        p |= {
            'r1_a': rb['r1'],
            'r2_a': rb['r2'],
            'r2s_a': rb['r2s'],
            'r1_l': [rb['r1'], rh['r1']],
            'r2_l': [rb['r2'], rh['r2']], 
            'r2s_l': rb['r2s'],
            'r1i_a': rb['r1'],
            'r1i_l': [rb['r1'], np.nan],
        }

        # Compute
        results = self._conc(p)

        for roi in ['a', 'l']:
            results |= self._wex[roi](p, **results) 
            results |= self._relax_tissue[roi](p, **results)
            if f'R1_{roi}' in results:
                results |= self._relax_inlets[roi](p, **results)
            results |= self._magn[roi](p, **results)
            if self.config['calibrate']:
                results |= self._derive_s0[roi](p, **results)
            results |= self._signal[roi](p, **results)

        return self.map_results(results)