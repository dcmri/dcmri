from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.aorta_portal_liver import InverseAortaPortalLiver as Inverse


class AortaPortalLiver():
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

        signal_data = (data['S_ao'], data['S_pv'], data['S_li'])
        signal_pred = (pred['S_ao'], pred['S_pv'], pred['S_li'])

        signal_data = np.concatenate([s.reshape(-1) for s in signal_data])
        signal_pred = np.concatenate([s.reshape(-1) for s in signal_pred])

        return loss(signal_pred, signal_data, metric, nfree)

    def plot(self, data: dict, xlim=None, fname=None, show=True):
        pred = self._forward(self._state)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        xlim = np.array(xlim) / 60
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        
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

        plot_data(pred['tS_ao'], pred['S_ao'], data['tS_ao'], data['S_ao'], ax1, ['lightcoral', 'darkred'])
        plot_data(pred['tS_li'], pred['S_li'], data['tS_li'], data['S_li'], ax5, ['cornflowerblue', 'darkblue'])
        plot_data(pred['tS_pv'], pred['S_pv'], data['tS_pv'], data['S_pv'], ax3, ['orchid', 'purple'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax2.plot(pred['tC'] / 60, 1000 * pred['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(ylabel='Concentration (mM)', xlim=xlim)
        ax4.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax4.plot(pred['tC'] / 60, 1000 * pred['C_pv'][0], linestyle='-', color='purple', linewidth=2.0, label='Portal vein')
        ax4.legend()

        ax6.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax6.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax6.plot(pred['tC'] / 60, 1000 * pred['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
        ax6.plot(pred['tC'] / 60, 1000 * pred['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
        ax6.plot(pred['tC'] / 60, 1000 * pred['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax6.legend()

        if fname: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()

