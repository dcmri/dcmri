"""Whole-body model for the aorta signal.

This model uses a whole-body pharmacokinetic architecture to predict the 
MRI signal in the aorta by modeling the injection, heart-lung transit, 
and systemic circulation (see :ref:`whole-body-tissues`). 

Args:
    organs (str, optional): Model for the systemic organs. 
        Options: 'comp' (1-compartment), '2cxm' (2-compartment exchange). 
        Defaults to 'comp'.
    heartlung (str, optional): Model for the heart-lung system. 
        Options: 'pfcomp' (plug-flow), 'chain'. Defaults to 'pfcomp'.
    sequence (str, optional): Imaging sequence model. 
        Options: 'SS' (steady-state), 'SR' (saturation-recovery), 
        'SSI' (steady state with inflow), 'lin' (linear). 
        Defaults to 'SS'.
    **params: Variable list of model parameters (e.g., CO=100, BAT=20). 
        Defaults are used for any that are not provided. See table 
        :ref:`Aorta-defaults` for a list of parameters and 
        their default values.

Raises:
    ValueError: If invalid configurations or unknown parameters are provided.

Example:

    Use the model to fit minipig aorta data with inflow 
    correction:

.. plot::
    :include-source:
    :context: close-figs

    >>> import numpy as np
    >>> import pydmr
    >>> import dcmri as dc

    Read the dataset:

    >>> datafile = dc.fetch('minipig_renal_fibrosis')
    >>> data = pydmr.read(datafile, 'nest')
    >>> rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']

    Initialize the tissue:

    >>> aorta = dc.Aorta(
    ...     sequence='SSI',
    ...     heartlung='chain',
    ...     organs='comp',
    ...     field_strength=pars['B0'],
    ...     t0=15, 
    ...     agent="gadoterate",
    ...     weight=pars['weight'],
    ...     dose=pars['dose'],
    ...     rate=pars['rate'],
    ...     TR=pars['TR'],
    ...     FA=pars['FA'],
    ...     TS=pars['TS'],
    ...     CO=60, 
    ...     R1b=1/dc.const.T1(pars['B0'], 'blood'),
    ... )

    Create an array of time points:

    >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

    Train the system to the data:ConcAorta

    >>> aorta.train(time, rois['Aorta'])

    Plot the reconstructed signals and concentrations:

    >>> aorta.plot(time, rois['Aorta'])

    Print the model parameters:

    >>> aorta.print_params(round_to=4)
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QVALUES, QUANTITIES
from dcmri.models.aorta import AortaModel
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat

class Aorta(SuperRoiModel):
    """Whole-body model for the aorta.
    """
    def __init__(self, data: dict=None, **config):
        if data is None:
            data = {}

        self._version = '1.0'
        self._model = AortaModel(**config)

        # Initialise model parameters
        pars = QVALUES | self._model.map_lexicon(QVALUES) | data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if p in QUANTITIES and QUANTITIES[p]['group']=='phys'}
            params_free |= {p for p in ['BAT'] if p in params}
            return params_free
        return params

    def _predict(self, time):
        pred = self._model(self._pars)
        return pred['S_a'][:, :, :len(time)]
    
    def _train(
        self, time: np.ndarray, signal: np.ndarray, free:dict=None,
        bounds:dict=None, n0=10, n_bat=1, **kwargs,
    ):
        p = self._pars

        # Estimate parameters
        bat = estimate_bat(time, signal, n0)
        p['BAT'] = max(bat - p['Thl'], 0)

        if self._model.config['calibrate']:
            p['Sb_a'] = signal[..., :n0]
            p['tSb_a'] = time[:n0]

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train_bat(self._predict, time, signal, p, free, n_bat=n_bat, **kwargs)
    
    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        prediction = self._model(self._pars)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                ax0.plot(time / 60, signal[i, j, :], marker='o', color='lightcoral', alpha=0.5, label='Data')
                ax0.plot(prediction['tS_a'] / 60, prediction['S_a'][i, j, :], linestyle='-', color='darkred', linewidth=3, label='Prediction')                
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(prediction['t'] / 60, 1000 * prediction['C_a'][0], linestyle='-', color='darkred', linewidth=3, label='Reconstruction')
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
        time = data['tS_a']
        signal = data['S_a']

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
        time = data['tS_a']
        signal = data['S_a']

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
        time = data['tS_a']
        signal = data['S_a']

        if signal.ndim==1:
            signal = signal.reshape(1, 1, -1)  
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + (time[-1] - time[-2])
        signal_pred = self._predict(time)
        return loss(signal_pred, signal, metric, nfree)