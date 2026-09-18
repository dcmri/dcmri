import matplotlib.pyplot as plt
import numpy as np


from dcmri.core.types import Input
from dcmri.inverse.sig2conc import SignalToConc
from dcmri.core.tools import get_quantity, get_bounds
from dcmri.core.types import Input
from dcmri.utils.fit import train_bat, loss
from dcmri.models.cort_med import CortMedModel


class CortMed():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = CortMedModel(**config)

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
        return (
            pred['S_kc'][:, :, :len(time[0])].reshape(-1),
            pred['S_km'][:, :, :len(time[1])].reshape(-1),  
        )

    # ==========================================
    # User interface
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
            t = np.arange(0, np.amax(data['tS_kc']) + p['dt'], p['dt'])
            p['c_ar'] = np.interp(t, input.time, ca['C'])

        if self._model.config['calibrate']:
            for roi in ['kc', 'km']:
                p[f'Scal_{roi}'] = data[f'S_{roi}'][..., :n0]
                p[f'iScal_{roi}'] = np.arange(n0)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)

        time = (data['tS_kc'], data['tS_km'])
        signal = (data['S_kc'], data['S_km'])
        return train_bat(self._predict, time, signal, p, free, **kwargs)


    def plot(self, data: dict, xlim=None, fname=None, show=True):
        prediction = self._model(self._pars)

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
        ax1.plot(prediction['tC'] / 60, 1000 * self._pars['c_ar'], '-', linewidth=3, color='darkred', label='Arterial Pred')
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


    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        time = (data['tS_kc'], data['tS_km'])
        signal = (data['S_kc'], data['S_km'])

        pred = self._predict(time)
        signal = np.concatenate([s.reshape(-1) for s in signal])
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)