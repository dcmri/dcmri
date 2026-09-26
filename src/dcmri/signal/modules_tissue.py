
import numpy as np
from scipy.special import i0, i1

from dcmri.core.module import Module
from dcmri.relaxivity.modules_tissue import ConcToRelax
from dcmri.bloch.modules_tissue import Magnetization
from dcmri.bloch.functions_sequences import channels


def signal_rice(nu, sigma)-> np.ndarray:
    if sigma==0:
        return nu
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        K = nu**2 / (2*sigma**2)
        arg = K/2
        pref = sigma * np.sqrt(np.pi/2)
        rice_mean = pref * np.exp(-K/2) * ((1+K)*i0(arg) + K*i1(arg))
    # Nan values are points where the distribution is indistinguisable from Gaussian
    return np.where(np.isnan(rice_mean) | np.isinf(rice_mean), nu, rice_mean)



class Signal(Module):
    configs = {
        'magnitude': {False, True},
        'trigger': {False, True},
        'calibrate': {False, True},
    }
    defaults = {
        'magnitude': True,
        'trigger': False,
        'calibrate': False,
    }
    _all_inputs = {'M', 'S0', 'NSR', 'iScal', 'Scal', 'tM', 'iStrig'}
    _all_outputs = {'S0', 'tS', 'S'}

    def inputs(self):
        inputs = {'tM', 'M'} 
        if self.config['magnitude']:
            inputs |= {'NSR'}
        if self.config['calibrate']:
            inputs |= {'iScal', 'Scal'} 
        else:
            inputs |= {'S0'}
        if self.config['trigger']:
            inputs |= {'iStrig'}
        return inputs
    
    def outputs(self):
        outputs = {'tS', 'S'} # (channels, components, times)
        if self.config['calibrate']:
            outputs |= {'S0'}
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)

        if p['M'].ndim not in [4]:
            raise ValueError("Magnetization M must be a 4D array with shape (channels, components, compartments, times)")
        # (channels, components, compartments, times)

        # Sum xy components over compartments
        Mxy = p['M'][:, :2, :, :].sum(axis=2) 
        # (channels, components, times)

        results = {}

        # Build signal
        tS = p['tM']
        if self.config['calibrate']:
            S = Mxy # normalized signal (S0=1)
        else:
            S = p['S0'] * Mxy
        
        if self.config['magnitude']:
            S = np.linalg.norm(S, axis=1, keepdims=True)
            Sb = np.mean(S[:, 0, 0])
            noise_sdev = p['NSR'] / Sb if Sb != 0 else 0
            S = signal_rice(S, noise_sdev)
            # (channels, 1, times)

        if self.config['trigger']:
            if p['iStrig'] is not None:
                accept = p['iStrig']
                tS = tS[accept]
                S = S[:, :, accept]
                # (channels, components, times)

        if self.config['calibrate']:
            s_cal_norm = S[:, :, p['iScal']]
            s_cal = p['Scal']
            nozero = np.where(s_cal_norm != 0)
            results['S0'] = np.mean(s_cal[nozero] / s_cal_norm[nozero])
            S *= results['S0']

        results |= {'tS': tS, 'S': S}  # (channels, components, times)
        return self.map_results(results)
    
    def dummy_data(self, nt=5, nch=1):
        data = self.init_data()
        n_channels = nch
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'tM': np.zeros(nt),
            'M': np.ones((n_channels, 3, 1, nt)), # (channels, components, compartments, times)
            'iScal': np.zeros(n0, dtype=int),
            'Scal': Scal, 
            'iStrig': np.zeros(n0, dtype=int),
        }
        return self.input_data(data)


class RelaxToSignal(Module): 
    configs = Magnetization.configs | Signal.configs
    defaults = Magnetization.defaults | Signal.defaults

    _all_inputs = {'TR', 'TE2', 'Nz', 'Scal', 'TE', 'iStrig', 'FA', 'vw', 'Fwi', 'PA', 'tacq', 'tMi', 'TF', 'TA', 'TD', 'R2', 'iScal', 'TE1', 'SA', 'R1', 'inlets', 'NSR', 'me', 'Kw', 'Mzi', 'Nk0', 'R2s', 'iz', 'B1corr', 'S0', 'R1i', 'TP', 'tstart', 'Nph', 'tR'}
    _all_outputs = {'S', 'tS', 'S0', 'M', 'tM'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)
        self._magn = Magnetization(**self.config) 
        self._signal = Signal(**self.config)
        self.map_io(imap, omap, iomap)  
        
    def inputs(self):
        inputs = self._magn.mapped_inputs()
        inputs |= self._signal.mapped_inputs()
        inputs -= self._magn.new_mapped_outputs()
        return inputs 
   
    def outputs(self):
        outputs = self._magn.mapped_outputs()
        outputs |= self._signal.mapped_outputs()
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  
        p |= self._magn(p)
        p |= self._signal(p)
        return self.map_results(p)

    def dummy_data(self, nc=2, nt=5):
        n_channels = channels(self.config['sequence'])

        p = self.init_data()

        p |= self._magn.dummy_data(nc, nt)
        p |= self._signal.dummy_data(nt, n_channels)

        return self.input_data(p)



class ConcToSignal(Module): 
    configs = ConcToRelax.configs | RelaxToSignal.configs 
    defaults = ConcToRelax.defaults | RelaxToSignal.defaults

    _all_inputs = {'TR', 'Scal', 'R1ib', 'r2se', 'FA', 'r2', 'TD', 'iScal', 'TE1', 'inlets', 'NSR', 'me', 'Kw', 'Mzi', 'iz', 'B1corr', 'S0', 'TP', 'r2sq', 'C', 'TE2', 'Nz', 'R1b', 'v', 'TE', 'RM', 'iStrig', 'r2s', 'vw', 'Fwi', 'tC', 'PA', 'tacq', 'tMi', 'ci', 'TF', 'r1i', 'TA', 'SA', 'r1', 'Nk0', 'r2sv', 'R2b', 'R2sb', 'Nph', 'tstart'}
    _all_outputs = {'S', 'R2', 'R1', 'M', 'R2s', 'tS', 'S0', 'R1i', 'tM', 'tR'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap: dict=None, cmap: dict=None, **config):
        self.set_config(config, cmap)
        self._conc_to_relax = ConcToRelax(**self.config)
        self._relax_to_signal = RelaxToSignal(**self.config)
        self.map_io(imap, omap, iomap)  
        
    def inputs(self):
        inputs = {'tC'}
        inputs |= self._conc_to_relax.mapped_inputs()
        inputs |= self._relax_to_signal.mapped_inputs()
        inputs -= self._conc_to_relax.new_mapped_outputs()
        inputs -= self._relax_to_signal.new_mapped_outputs()
        inputs -= {'tR'}
        return inputs 
   
    def outputs(self):
        outputs = {'tR'}
        outputs |= self._conc_to_relax.mapped_outputs() 
        outputs |= self._relax_to_signal.mapped_outputs()
        return outputs

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs) 
        p['tR'] = p['tC']
        p |= self._conc_to_relax(p)
        p |= self._relax_to_signal(p)
        return self.map_results(p)

    def dummy_data(self, nc=2, nt=5):
        p = self.init_data()

        p |= self._conc_to_relax.dummy_data(nc, nt)
        p |= self._relax_to_signal.dummy_data(nc, nt)

        return self.input_data(p)
