from dcmri.core.module import Module
from dcmri.core.sequences import SEQUENCES
from dcmri.core.exceptions import InvalidConfiguration
from dcmri.utils import const
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.signal.modules_tissue import Signal


class AortaLiverModel(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = {
        'bolus': ConcAortaLiver.configs['bolus'],
        'heartlung': ConcAortaLiver.configs['heartlung'],
        'organs': ConcAortaLiver.configs['organs'],
        'lagut': ConcAortaLiver.configs['lagut'],
        'liver': ConcAortaLiver.configs['liver'],
        'non_stationary': ConcAortaLiver.configs['non_stationary'],
        't2s_relaxation': {None, 'lin', 'quad'},
        'sequence': Magnetization.configs['sequence'],
        'magnitude': Signal.configs['magnitude'],
    }
    defaults = {
        'bolus': ConcAortaLiver.defaults['bolus'],
        'heartlung': ConcAortaLiver.defaults['heartlung'],
        'organs': ConcAortaLiver.defaults['organs'],
        'lagut': ConcAortaLiver.defaults['lagut'],
        'liver': ConcAortaLiver.defaults['liver'],
        'non_stationary': ConcAortaLiver.defaults['non_stationary'],
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
        self._conc = ConcAortaLiver(**config)
        self._relax = {
            roi: Relax(
                imap = {f"R1b": f"R1b_{roi}", f"R2b": f"R2b_{roi}", f"R2sb": f"R2sb_{roi}"},
                t1_relaxation='lin' if 'R1' in props else None, 
                t2_relaxation='lin' if 'R2' in props else None, 
                t2s_relaxation=t2s_relaxation,
                fast_water_exchange=True, 
            ) for roi in ['a', 'l']
        }
        self._magn = {
            roi: Magnetization(
                imap={'B1corr': f"B1corr_{roi}"}, 
                inflow=True if roi =='l' else False, 
                **config,
            ) for roi in ['a', 'l']
        }
        self._signal = {
            roi: Signal(**config) for roi in ['a', 'l']
        }

        # Configure I/O
        self.set_output_map({'C':'ca', 'R1':'R1_a', 'R2':'R2_a', 'R2s':'R2s_a', 'M':'Ma', 'S':'Sa'})
        self.map_inputs(imap)
        
    def inputs(self) -> set:
        inputs = {'field_strength', 'agent', 'CO', 'vol_a'}
        inputs |= self._conc.mapped_inputs()
        inputs |= self._relax.mapped_inputs() - {'C'}
        inputs |= self._magn.mapped_inputs() - {'tR', 'R1', 'R2', 'R2s', 'R1i'}
        inputs |= self._signal.mapped_inputs() - {'tM', 'M'} 
        inputs -= {'r1', 'r2', 'r2s', 'v', 'Fw', 'me', 'Fi'} # Constants and derived
        return inputs 
    
    def outputs(self):
        outputs = self._conc.outputs()
        outputs |= self._relax.outputs()
        outputs |= self._signal.outputs()
        outputs |= self._magn.outputs()
        return self.map_outputs(outputs)
    
    def map_lexicon(self, qvalues): 
        return qvalues

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = const.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])

        results = {}

        # Compute concentrations
        results |= self._conc(p) 

        # Aorta
        p |= rb
        relax = self._relax['a'](p, C=results['ca']) # This needs inflow too
        magn = self._magn['a'](p, tR=results['t'], **relax)
        signal = self._signal['a'](p, **magn)
        results |= relax | magn | signal

        # Liver
        p |= {'r1': [rb['r1'], rh['r1']], 'r2': [rb['r2'], rh['r2']], 'r2s': rb['r2s']}
        relax = self._relax['l'](p, C=results['Cl']) 
        if 'R1' in relax['l']:
            relax_inflow = self._relax['a'](p, C=results['ci'])
            relax |= {'R1i': relax_inflow['R1']}
        magn = self._magn['l'](p, tR=results['t'], **relax) 
        signal = self._signal['l'](p, **magn)
        results |= relax | magn | signal 

        # Map results to outputs
        return self.map_results(results)