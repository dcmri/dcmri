import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_quantity, get_bounds
from dcmri.inverse.sig2conc import SignalToConc
from dcmri.core.types import Input
from dcmri.utils.fit import train_bat, loss
from dcmri.forward.tissue_x import ForwardTissueX



class TissueX():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = ForwardTissueX(**config)

        # Initialise model parameters
        pars = self._model.dummy_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if get_quantity(p)['group']=='phys'} 
            return params_free
        return params

    def _predict(self, time: tuple):
        pred = self._model(self._pars)
        return pred['S'][:, :, :len(time)].reshape(-1)

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> np.ndarray:
        """Predicts the data."""
        return self._model(self._pars)

    def train(
        self, data: dict, aif:dict=None, 
        free: dict=None, bounds: dict=None, n0=1, **kwargs):

        p = self._pars
        
        if aif is not None:
            input = Input(aif)
            ca = SignalToConc(**self._model.config)(
                p, S=input.signal, R1b=input.R1b, nb=n0, 
                B1corr=input.B1corr, 
            )
            t = np.arange(0, np.amax(data['tS']) + p['dt'], p['dt'])
            p['c_ar'] = np.interp(t, input.time, ca['C'])

        if self._model.config['calibrate']:
            p['Scal'] = data['S'][..., :n0]
            p['iScal'] = np.arange(n0)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)
        time = data['tS']
        signal = data['S']
        return train_bat(self._predict, time, signal, p, free, **kwargs)
        
    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        prediction = self._model(self._pars)

        if xlim is None:
            xlim = [prediction['tR'][0], prediction['tR'][-1]]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        for i in range(data['S'].shape[0]):
            for j in range(data['S'].shape[1]):
                ax0.plot(data['tS'] / 60, data['S'][i, j, :], marker='o', linestyle='None', color='cornflowerblue', label='Data')
                ax0.plot(prediction['tS'] / 60, prediction['S'][i, j, :], linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(prediction['tC'] / 60, 1000 * self._pars['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        if prediction['C'].shape[0] == 2:
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Interstitium')
        else:
            ax1.plot(prediction['tC'] / 60, 1000 * prediction['C'][0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Tissue')
           
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


    def cost(self, data: dict, metric: str = 'NRMS', nfree=None) -> float:
        time = data['tS']
        signal = data['S'].reshape(-1)

        signal_pred = self._predict(time)
        return loss(signal_pred, signal, metric, nfree)

    # Original plot function showing magnetization etc
    # Needs some adaptation

    # def plot(self, data: dict, xlim:list=None, fname:str=None, show=True, 
    #          sdev=None, round_to=None):
    #     clr = {
    #         'Plasma': 'darkred',
    #         'Interstitium': 'steelblue',
    #         'Extracellular': 'dimgrey',
    #         'Tissue': 'darkgrey',
    #         'Blood': 'darkred',
    #         'Extravascular': 'blue',
    #         'Tissue cells': 'lightblue',
    #         'Blood + Interstitium': 'purple',
    #     }
    #     plot_labels_kin = {
    #         '2CX': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         '2CU': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'HF': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'HFU': (['vb', 'vi'], ['Blood', 'Interstitium']),
    #         'FX': (['ve'], ['Extracellular']),
    #         'NX': (['vb'], ['Blood']), 
    #         'NXP': (['vb'], ['Blood']),
    #         'U': (['vb'], ['Blood']),
    #         'WV': (['vi'], ['Interstitium']),
    #     }
    #     def plot_labels_relax(kin, wex) -> list:

    #         if wex == 'FF':
    #             return ['Tissue']

    #         if wex in ['RR', 'NN', 'NR', 'RN']:
    #             if kin == 'WV':
    #                 return ['Interstitium', 'Tissue cells']
    #             else:
    #                 return ['Blood', 'Interstitium', 'Tissue cells']

    #         if wex in ['RF', 'NF']:
    #             if kin == 'WV':
    #                 return ['Extravascular']
    #             else:
    #                 return ['Blood', 'Extravascular']

    #         if wex in ['FR', 'FN']:
    #             if kin == 'WV':
    #                 return ['Interstitium', 'Tissue cells']
    #             else:
    #                 return ['Blood + Interstitium', 'Tissue cells']
                
    #     prediction = self._model(self._pars)
    #     if xlim is None:
    #         xlim = [np.amin(t), np.amax(t)]
    #     xlim = np.array(xlim) / 60

    #     if self._model.config['water_exchange'] != 'FF':
    #         fig, ax = plt.subplots(2, 2, figsize=(10, 12))
    #         fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #         ax00 = ax[0, 0]
    #         ax01 = ax[0, 1]
    #         ax10 = ax[1, 0]
    #         ax11 = ax[1, 1]
    #         ax_text = None
    #     else:
    #         fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #         fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #         ax00 = ax[0]
    #         ax01 = ax[1]
    #         ax_text = ax[2]

    #     ax00.set_title('MRI signals')
    #     for ci in range(self._shape[1]):
    #         if time is not None:
    #             ax00.plot(time / 60, self._predict_all(time)[0, ci, :], marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
    #             ax00.plot(time / 60, signal[0, ci, :], marker='x', linestyle='None', color='darkblue', label='Data')
    #         ax00.plot(t / 60, S[0, ci, :], linestyle='-', linewidth=3.0, color='darkblue', label='Model')
    #     ax00.set(ylabel='MRI signal (a.u.)', xlabel='Time (min)', xlim=xlim)
    #     ax00.legend()

    #     conc_comp, conc_label = plot_labels_kin[self._cnfg['kinetics']]
    #     relax_comp = plot_labels_relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])
    
    #     ax01.set_title('Tissue concentration in indicator compartments')
    #     ax01.plot(t / 60, 1000 * self._pars['c_a'], linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial blood')
    #     for k, vk in enumerate(conc_comp):
    #         # ck = C[k, ...] / p[vk] if p[vk] > 0 else 0 * C[k, ...]
    #         ax01.plot(t / 60, 1000 * C[0, k, :], linestyle='-', linewidth=3.0, label=conc_label[k], color=clr[conc_label[k]])
    #     ax01.set(ylabel='Concentration (mM)', xlabel='Time (min)', xlim=xlim)
    #     ax01.legend()

    #     if self._cnfg['water_exchange'] != 'FF':
    #         ax11.set_title('Concentration in water compartments')
    #         for i in range(c.shape[1]):
    #             ax11.plot(t / 60, 1000 * c[0, i, :], linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
    #         ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
    #         ax11.legend()

    #         ax10.set_title('Magnetization in water compartments')
    #         for i in range(Mz.shape[1]):
    #             mi = Mz[0, i, :] / v[i] if v[i] > 0 else 0 * Mz[0, i, :]
    #             ax10.plot(t / 60, mi, linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
    #         ax10.set(xlabel='Time (min)', ylabel='Magnetization (a.u.)', xlim=xlim)
    #         ax10.legend()

    #     if ax_text is not None:

    #         if sdev is None:
    #             pars = self._params('free')
    #         else:
    #             pars = list(sdev.keys())
    #             sdev = {k: float(v) for k, v in sdev.items()}

    #         vals = {k: p[k][0] for k in pars}
    #         msg = string_params(vals, sdev, round_to)
    #         msg = "\n".join(list(msg.values()))
    #         ax_text.set_title('Free parameters')
    #         ax_text.axis("off")  # hide axes
    #         ax_text.text(0, 0.9, f"Kinetics: {self._cnfg['kinetics']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.85, f"Water exchange: {self._cnfg['water_exchange']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.8, f"Sequence: {self._cnfg['sequence']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.75, f"R2* model: {self._cnfg['t2s_relaxation']}", fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
    #         ax_text.text(0, 0.6, msg, fontsize=10, transform=ax_text.transAxes, ha="left", va="top")

    #     if fname is not None: 
    #         plt.savefig(fname=fname)
    #     if show: 
    #         plt.show()
    #     else: 
    #         plt.close()



