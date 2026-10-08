from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.aorta_liver_split import InverseAortaLiverSplit as Inverse


class AortaLiverSplit():
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

        signal_data = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])
        signal_pred = (pred['S_1_ao'], pred['S_2_ao'], pred['S_1_li'], pred['S_2_li'])

        signal_data = np.concatenate([s.reshape(-1) for s in signal_data])
        signal_pred = np.concatenate([s.reshape(-1) for s in signal_pred])

        return loss(signal_pred, signal_data, metric, nfree)


    def plot(self, data: dict, xlim: list = None, fname: str = None, show=True):
        pred = self._forward(self._state)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)

        # Plot signals
        def _plot_data2scan(roi, ts, s, ax, color):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            for i in range(s[0].shape[0]):
                for j in range(s[0].shape[1]):
                    ax.plot(ts[0] / 60, s[0][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(ts[1] / 60, s[1][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(pred[f'tS_1_{roi}'] / 60, pred[f'S_1_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
                    ax.plot(pred[f'tS_2_{roi}'] / 60, pred[f'S_2_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan('ao',(data['tS_1_ao'], data['tS_2_ao']), (data['S_1_ao'], data['S_2_ao']), ax1, ['lightcoral', 'darkred'])
        _plot_data2scan('li',(data['tS_1_li'], data['tS_2_li']), (data['S_1_li'], data['S_2_li']), ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax2.plot(pred['tC'] / 60, 1000 * pred['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        if pred['C_li'].shape[0]==2:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        else:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()
    

    # WIP below here


    # def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     return export_params(pars, lexicon=QUANTITIES, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

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
    #     print_params(pars, round_to=round_to, group=group, lexicon=QUANTITIES)


