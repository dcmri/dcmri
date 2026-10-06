from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_bounds, get_quantity, print_quantities
from dcmri.utils.fit import loss
from dcmri.inverse.aorta import InverseAorta

class Aorta():
    """Whole-body model for the aorta.
    """
    def print_state(self, decimals=None):
        q = {k: get_quantity(k) for k in self._state.keys()}
        for k, v in self._state.items():
            q[k]['init'] = v
        title = f"{self.__class__.__name__} instance - state (n = {len(q)})"
        print_quantities(q, title, decimals=decimals)

    def set_state(self, state: dict):
        self._state |= self._forward.map_data(state)

    def state(self):
        return deepcopy(self._state)

    @classmethod
    def all_configs(cls, sample: int = None, seed: int = None, valid=False):
        return InverseAorta.all_configs(sample, seed, valid)

    @classmethod
    def print_configs(cls):
        InverseAorta.print_configs()

    def __init__(self, state: dict=None, **config):
        self._inverse = InverseAorta(**config)
        self._forward = self._inverse.forward
        self._state = self._forward.dummy_data(state)

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

        signal_pred = pred['S'].reshape(-1)
        signal_data = data['S'].reshape(-1)

        return loss(signal_pred, signal_data, metric, nfree)
    
    def plot(self, data: dict, xlim=None, fname:str=None, show=True, c_ref=None):
        pred = self._forward(self._state)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(pred['tS'], pred['S'], data['tS'], data['S'], ax0, ['lightcoral', 'darkred'])

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        if c_ref is not None:
            t, c = c_ref[0], c_ref[1].reshape(1, -1)
            ax1.plot(t / 60, 1000 * c[0], linestyle='-', color='lightcoral', linewidth=5, label='Reference')
        ax1.plot(pred['tC'] / 60, 1000 * pred['C'][0], linestyle='-', color='darkred', linewidth=3, label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: 
            plt.savefig(fname)
        if show: 
            plt.show()
        else: 
            plt.close()   