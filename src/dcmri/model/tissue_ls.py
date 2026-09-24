import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_quantity
from dcmri.kinetics.functions_tissue_ls import irf_ls
from dcmri.inverse.sig2conc import SignalToConc
from dcmri.core.types import Input
from dcmri.utils.fit import loss
from dcmri.forward.tissue_ls import ForwardTissueLS
from dcmri.core.module import InvalidConfig


class TissueLS():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = ForwardTissueLS(**config)

        if self._model.config['sequence'] not in SignalToConc.configs['sequence']:
            raise InvalidConfig(f"{self._model.config['sequence']} is not invertible.")

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
        self, data: dict, aif:dict=None, n0=1, **kwargs):

        p = self._pars
    
        if aif is not None:
            input = Input(aif)
            ca = SignalToConc(**self._model.config)(
                p, S=input.signal, R1b=input.R1b, nb=n0, 
                B1corr=input.B1corr, 
            )
            t = np.arange(0, np.amax(data['tS']) + p['dt'], p['dt'])
            p['ci'] = np.interp(t, input.time, ca['C'])
        else:
            t = p['dt'] * np.arange(p['ci'].size)

        if self._model.config['calibrate']:
            p['Scal'] = data['S'][..., :n0]
            p['iScal'] = np.arange(n0)

        C = SignalToConc(**self._model.config)(
            p, S=data['S'][0,0,:], R1b=data['R1'][0,0], nb=n0, 
            B1corr=p['B1corr'], 
        )
        C = np.interp(t, data['tS'], C['C'])

        irf = irf_ls(p['ci'], C, p['dt'])

        return {'irf': irf}


    def plot(self, data: dict, xlim:list=None, fname:str=None, show=True):
        prediction = self._model(self._pars)
        p = self._pars

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
        ax1.plot(prediction['tC'] / 60, 1000 * self._pars['ci'], '-', linewidth=3, color='darkred', label='Arterial Pred')
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

