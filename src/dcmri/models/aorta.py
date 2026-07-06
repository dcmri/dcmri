from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal, CalibrateSignal


class AortaModel(Module):
    """Whole-body model for the aorta signal."""

    configs = {
        'bolus': ConcAorta.configs['bolus'],
        'heartlung': ConcAorta.configs['heartlung'],
        'organs': ConcAorta.configs['organs'],
        't2s_relaxation': {None, 'lin', 'quad'},
        'sequence': Magnetization.configs['sequence'],
        'magnitude': Signal.configs['magnitude'],
    }
    defaults = {
        'bolus': ConcAorta.defaults['bolus'],
        'heartlung': ConcAorta.defaults['heartlung'],
        'organs': ConcAorta.defaults['organs'],
        't2s_relaxation': 'lin',
        'sequence': Magnetization.defaults['sequence'],
        'magnitude': Signal.defaults['magnitude'],
    }

    def __init__(self, imap:dict=None, **config):
        self.set_config(config)

        # Check configuration
        props = set(SEQUENCES[self.config['sequence']]['parameters']['tissue'])
        if 'R2s' in props:
            if not self.config['t2s_relaxation']:
                raise InvalidConfiguration(f"The t2s_relaxation option can't be None for T2*-weighted sequences.")
            t2s_relaxation = self.config['t2s_relaxation']
        else:
            t2s_relaxation = None            

        # Configure modules
        self._scal = CalibrateSignal(
            imap={'Sb': 'Sb_a', 'R1b':'R1b_a', 'R2b':'R2b_a', 'R2sb':'R2sb_a', 'R1ib':'R1b_a', 'B1corr': 'B1corr_a'}, 
            inflow=True, **config,
        )
        self._conc = ConcAorta(**config)
        self._relax = Relax( # inflow and roi are both blood - just one needed
            imap = {'R1b': 'R1b_a', 'R2b': 'R2b_a', 'R2sb': 'R2sb_a'},
            t1_relaxation='lin' if 'R1' in props else None, 
            t2_relaxation='lin' if 'R2' in props else None, 
            t2s_relaxation=t2s_relaxation,
            fast_water_exchange=True, 
        )
        self._magn = Magnetization(
            imap={'B1corr': 'B1corr_a'}, inflow=True, **config,
        )
        self._signal = Signal(**config)
        self.set_output_map({'C': 'ca', 'R1': 'R1_a', 'R2': 'R2_a', 'R2s': 'R2s_a', 'M': 'Ma', 'S': 'Sa'})
        self.map_inputs(imap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent', 'CO', 'vol_a'}
        inputs |= self._scal.mapped_inputs()
        inputs |= self._conc.mapped_inputs()
        inputs |= self._relax.mapped_inputs() - {'C'}
        inputs |= self._magn.mapped_inputs() - {'R1', 'R2', 'R2s', 'R1i'}
        inputs |= self._signal.mapped_inputs() - {'M'}  

        # Take away constants and derived
        return inputs - {'r1', 'r2', 'r2s', 'v', 'Fw', 'me', 'Fi', 'S0'}
    
    def outputs(self):
        outputs = self._conc.outputs()
        outputs |= self._relax.outputs()
        outputs |= self._signal.outputs()
        outputs |= self._magn.outputs()
        return self.map_outputs(outputs)
    
    def map_lexicon(self, qvalues, data={}):
        p = self.map_data(qvalues, data)
        p_lex = self._scal.map_lexicon(qvalues, data)        
        return p | {'Sb_a': p_lex['Sb'], 'tacq': p_lex['tacq']}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        # Add constants and derived parameters
        p |= const.relaxivity(p['field_strength'], 'blood', p['agent'])
        p |= {'v': 1, 'Fw': p['CO'] / p['vol_a'], 'me': 1, 'Fi': p['CO'] / p['vol_a']} 
        p |= self._scal(p) # Add S0

        # Compute successive stages
        conc = self._conc(p) # returns ca and ci
        relax_roi = self._relax(p, C=conc['ca']) # returns R1, R2, R2s as needed
        if 'R1' in relax_roi:
            relax_inflow = self._relax(p, C=conc['ci'])
            relax_roi |= {'R1i': relax_inflow['R1']}
        magn = self._magn(p, **relax_roi) # returns M
        signal = self._signal(p, **magn) # returns S

        # Map results to outputs
        results = signal | magn | relax_roi | conc
        return self.map_results(results)