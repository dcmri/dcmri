from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.cort_med import InverseCortMed as Inverse


class CortMed():
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

        signal_data = (data['S_kc'], data['S_km'])
        signal_pred = (pred['S_kc'], pred['S_km'])

        signal_data = np.concatenate([s.reshape(-1) for s in signal_data])
        signal_pred = np.concatenate([s.reshape(-1) for s in signal_pred])

        return loss(signal_pred, signal_data, metric, nfree)


    def plot(self, data: dict, xlim=None, fname=None, show=True):
        prediction = self._forward(self._state)

        if xlim is None: 
            xlim = [prediction['tR'][0], prediction['tR'][-1]]
        xlim = np.array(xlim) / 60

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot signals
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(prediction['tS_kc'], prediction['S_kc'], data['tS_kc'], data['S_kc'], ax0, ['lightcoral', 'darkred'])
        plot_data(prediction['tS_km'], prediction['S_km'], data['tS_km'], data['S_km'], ax0, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax1.set_title('Reconstruction of concentrations.')
        ax1.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax1.plot(prediction['tC'] / 60, 1000 * self._state['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C_kc'].sum(axis=0), linestyle='-', linewidth=3.0, color='darkred', label='Cortex')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C_km'].sum(axis=0), linestyle='-', linewidth=3.0, color='darkcyan', label='Medulla')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()