from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.kidney import InverseKidney as Inverse


class Kidney():
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

        signal_data = data['S_ki']
        signal_pred = pred['S_ki']

        return loss(signal_pred, signal_data, metric, nfree)


    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        prediction = self._forward(self._state)

        if xlim is None:
            xlim = [prediction['tR_ki'][0], prediction['tR_ki'][-1]]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        for i in range(data['S_ki'].shape[0]):
            for j in range(data['S_ki'].shape[1]):
                ax0.plot(data['tS_ki'] / 60, data['S_ki'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax0.plot(prediction['tS_ki'] / 60, prediction['S_ki'][i, j, :], linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(prediction['tC_ki'] / 60, 1000 * self._state['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(prediction['tC_ki'] / 60, 1000 * prediction['C_ki'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
        ax1.plot(prediction['tC_ki'] / 60, 1000 * prediction['C_ki'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Tubuli')
           
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


