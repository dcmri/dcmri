from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.liver import InverseLiver as Inverse


class Liver():
    @classmethod
    def all_configs(cls, sample: int = None, seed: int = None, valid=False):
        return Inverse.all_configs(sample, seed, valid)
    
    def __init__(self, state: dict=None, **config):
        self._inverse = Inverse(**config)
        self._forward = self._inverse.forward
        self._state = self._forward.dummy_data(state)

    def state(self):
        return deepcopy(self._state)

    def predict(self) -> np.ndarray:
        return self._forward(self._state)

    def train(self, data: dict, pfree:dict=None, bounds: dict=None, nb=5, **kwargs):
        # Get free parameters
        default_pfree = self._inverse.pfree()     
        pfree = get_bounds(pfree, bounds, free_pars=default_pfree)

        # Apply inverse model
        inputs = self._state | data | {'pfree': pfree, 'nb': nb}
        result = self._inverse(inputs, **kwargs)

        # Update state
        self._state |= result['popt']

        return result

    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        pred = self._forward(self._state)

        signal_data = data['S_li']
        signal_pred = pred['S_li']

        return loss(signal_pred, signal_data, metric, nfree)

    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        pred = self._forward(self._state)

        xlim = xlim or [np.amin(pred['tR_li']), np.amax(pred['tR_li'])]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        ax0.set_title('MRI Signal Prediction')
        for i in range(data['S_li'].shape[0]):
            for j in range(data['S_li'].shape[1]):
                ax0.plot(data['tS_li'] / 60, data['S_li'][i, j, :], 'o', color='cornflowerblue', label='Data')
                ax0.plot(pred['tS_li'] / 60, pred['S_li'][i, j, :], '-', linewidth=3, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim) / 60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        if '1I' in self._inverse.config['kinetics']:
            ax1.plot(pred['tC_li'] / 60, 1000 * self._state['ci_li'], '-', linewidth=3, color='darkred', label='Input')
        if '2I' in self._inverse.config['kinetics']:
            ax1.plot(pred['tC_li'] / 60, 1000 * self._state['ci_li'][0], '-', linewidth=3, color='darkred', label='Arterial')
            ax1.plot(pred['tC_li'] / 60, 1000 * self._state['ci_li'][1], '-', linewidth=3, color='purple', label='Portal')
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



    


    # def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    # def print_params(self, *args, round_to=None, group=None, 
    #                  fixed_only=False, free_only=False, deriv=False):
    #     """Pretty print model parameters"""
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     if args != ():
    #         pars = {k: v for k, v in self._pars.items() if k in args}
    #     if fixed_only:
    #         pars = {k: v for k, v in pars.items() if k not in self._params('free')}
    #     if free_only:
    #         pars = {k: v for k, v in pars.items() if k in self._params('free')}
    #     print_params(pars, round_to=round_to, group=group)

