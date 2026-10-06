from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train
from dcmri.forward.liver import ForwardLiver as Forward
from dcmri.kinetics.functions_liver import dpars_liver

configs = deepcopy(Forward.configs) 
defaults = deepcopy(Forward.defaults)

class InverseLiver(Module):

    configs = configs
    defaults = defaults

    _all_inputs = None
    _all_outputs = None

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        nt = len(time)
        return pred['S_li'][:, :, :nt].reshape(-1)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  

        # Reshape signal if needed
        if p['S_li'].ndim == 1:
            p['S_li'] = p['S_li'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal_li'] = p['S_li'][..., :p['nb']]
            p['iScal_li'] = np.arange(p['nb'])

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self.pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = p['tS_li']
        signal = p['S_li']
        p |= train(self._predict, time, signal, p, p['pfree'], **kwargs)

        p['pder'] = self.pder(p | p['popt'])

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'S_li', 'nb'}
            inputs -= {'Scal_li', 'iScal_li'}
        inputs |= {'tS_li', 'S_li', 'pfree'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'pder', 'loss'}
        return outputs
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'v_e_li': (0, 1), 'F_p_li': (0, 1)}),
            'tS_li': pred['tS_li'],
            'S_li': pred['S_li'],
        }
        return self.input_data(p, data)

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {'S0_li'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['phys', 'body']}
        p = dpars_liver(p, kinetics=self.config['kinetics'])
        return p

    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        p = self.map_data(data)

        # Reshape signal if needed
        if p['S_li'].ndim == 1:
            p['S_li'] = p['S_li'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            p['Scal_li'] = p['S_li'][..., :p['nb']]
            p['iScal_li'] = np.arange(p['nb'])

        pred = self.forward(p)

        xlim = xlim or [np.amin(pred['tR_li']), np.amax(pred['tR_li'])]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        nt = len(p['tS_li'])
        ax0.set_title('MRI Signal Prediction')
        for i in range(p['S_li'].shape[0]):
            for j in range(p['S_li'].shape[1]):
                ax0.plot(pred['tS_li'][:nt] / 60, p['S_li'][i, j, :nt], 'o', color='cornflowerblue', label='Data')
                ax0.plot(pred['tS_li'][:nt] / 60, pred['S_li'][i, j, :nt], '-', linewidth=3, color='darkblue', label='Prediction')
                ax0.plot(pred['tS_li'][:nt] / 60, pred['S_li'][i, j, :nt], 'o', color='darkblue')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim) / 60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        if '1I' in self.config['kinetics']:
            ax1.plot(pred['tC_li'] / 60, 1000 * p['ci_li'], '-', linewidth=3, color='darkred', label='Input')
        if '2I' in self.config['kinetics']:
            ax1.plot(pred['tC_li'] / 60, 1000 * p['ci_li'][0], '-', linewidth=3, color='darkred', label='Arterial')
            ax1.plot(pred['tC_li'] / 60, 1000 * p['ci_li'][1], '-', linewidth=3, color='purple', label='Portal')
        ax1.plot(pred['tC_li'] / 60, 1000 * pred['C_li'][0,:], '-.', linewidth=3, color='darkblue', label='Extracellular')
        ax1.plot(pred['tC_li'] / 60, 1000 * pred['C_li'][1,:], '-', linewidth=3, color='green', label='Hepatocytes]')

        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim) / 60)
        ax1.legend()

        if fname: 
            plt.savefig(fname)
        if show: 
            plt.show()
        else: 
            plt.close()