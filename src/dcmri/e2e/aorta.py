

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.tools import get_quantity, get_bounds
from dcmri.models.aorta import AortaModel
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat

class Aorta():
    """Whole-body model for the aorta.
    """
    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = AortaModel(**config)

        # Initialise model parameters
        pars = self._model.dummy_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if get_quantity(p)['group']=='phys'}
            params_free |= {p for p in ['BAT'] if p in params}
            return params_free
        return params

    def _predict(self, time):
        pred = self._model(self._pars)
        return pred['S'][:, :, :len(time)]
    
    def _train(
        self, time: np.ndarray, signal: np.ndarray, free:dict=None,
        bounds:dict=None, n0=10, n_bat=1, **kwargs,
    ):
        p = self._pars
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)

        # Estimate parameters
        bat = estimate_bat(time, signal, n0)
        p['BAT'] = max(bat - p['T_hl'], 0)

        if self._model.config['calibrate']:
            p['Scal'] = signal[..., :n0]
            p['iScal'] = np.arange(n0)

        # Perform training
        return train_bat(self._predict, time, signal, p, free, n_bat=n_bat, **kwargs)
    
    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        prediction = self._model(self._pars)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                ax0.plot(time / 60, signal[i, j, :], marker='o', color='lightcoral', alpha=0.5, label='Data')
                ax0.plot(prediction['tS'] / 60, prediction['S'][i, j, :], linestyle='-', color='darkred', linewidth=3, label='Prediction')                
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(prediction['tC'] / 60, 1000 * prediction['C_ao'][0], linestyle='-', color='darkred', linewidth=3, label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> np.ndarray:
        """Predicts the aorta data."""
        return self._model(self._pars)
    
    def train(
            self, data: dict, free: dict=None, 
            bounds: dict=None, n0=10, n_bat=1, **kwargs
        ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            bounds (dict, optional): Override default bounds for specific parameters.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        time = data['tS']
        signal = data['S']

        if signal.ndim==1:
            signal = signal.reshape(1, 1, -1)  
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + (time[-1] - time[-2])
        return self._train(time, signal, free, bounds, n0, n_bat, **kwargs)
    
    def plot(self, data: dict, fname:str=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of signals
            signal (tuple): Liver signals            
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        time = data['tS']
        signal = data['S']

        if signal.ndim==1:
            signal = signal.reshape(1, 1, -1)  
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + (time[-1] - time[-2])
        self._plot(time, signal, fname, show)

    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        """Return the goodness-of-fit

        Args:
            time (np.ndarray): array with time points
            signal (array-like): array with signal data for all pixels.
            metric (str, optional): Which metric to use (see notes for 
                possible values). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.

        Notes:
            Available options are: 
            
            - 'RMS': Root-mean-square.
            - 'NRMS': Normalized root-mean-square. 
            - 'AIC': Akaike information criterion. 
            - 'cAIC': Corrected Akaike information criterion for small 
                models.
            - 'BIC': Bayesian information criterion.
        """
        time = data['tS']
        signal = data['S']

        if signal.ndim==1:
            signal = signal.reshape(1, 1, -1)  
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + (time[-1] - time[-2])
        signal_pred = self._predict(time)
        return loss(signal_pred, signal, metric, nfree)