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

from dcmri.core.module import Module
from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QVALUES
from dcmri.core.sequences import SEQUENCES
from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.signal.modules_tissue import Signal



class SignalAorta(Module):
    """Whole-body model for the aorta signal."""
    configs = {
        'heartlung': ConcAorta.configs['heartlung'],
        'organs': ConcAorta.configs['organs'],
        'sequence': Signal.configs['sequence'],
    }
    defaults = {
        'heartlung': 'pfcomp', 
        'organs': 'comp', 
        'sequence': '3D-SPGR-SS', 
    }
    def __init__(self, config:dict=None, imap:dict=None):
        self.set_config(config)
        props = set(SEQUENCES[self.config['sequence']]['parameters']['tissue'])

        self._conc_aorta = ConcAorta(config)
        self._relax_rate = Relax(
            config = self.config | {'tissue_props': props}, 
            imap = {'R1b': 'R1b_a', 'R2b': 'R2b_a', 'R2sb': 'R2sb_a', 'c': 'C'},
        )
        self._signal = Signal(
            config = self.config | {'calibrate': True},
            imap = {'Sb': 'Sb_a', 'B1corr': 'B1corr_a'},
        )
        self.map_inputs(imap)

    def inputs(self, group=None) -> set:
        if group == 'phys':
            inputs = self._conc_aorta.inputs()

        inputs = {'field_strength', 'agent'}
        inputs |= self._conc_aorta.inputs()
        inputs |= self._relax_rate.inputs()
        inputs |= self._signal.inputs()   
        return inputs
    
    def outputs(self):
        outputs = self._conc_aorta.outputs()
        outputs |= self._relax_rate.outputs()
        outputs |= self._signal.outputs()
        return outputs

    def __call__(self, data: dict) -> dict:
        p = self.map_data(data)
        p |= const.relaxivity(p['field_strength'], 'blood', p['agent'])

        conc = self._conc_aorta(p)
        relax = self._relax_rate(p | conc)
        signal = self._signal(p | relax)
        return signal | relax | conc


class Aorta(SuperRoiModel):
    """Whole-body model for the aorta signal.
    """

    def __init__(self, config: dict=None, data: dict=None):
        self._version = '1.0'
        self._signal_model = SignalAorta(config)
        self._set_params(QVALUES | data)

        # Set multi-channel baseline if not done by the user
        if self._signal_model.config['sequence'] in ['Eq-DE-EPI', 'DE-EPI']:
            if 'Sb_a' not in data:
                self._pars['Sb_a'] = np.full(2, self._pars['Sb_a'])

    def _time(self): # Measure Module?
        p = self._pars
        return np.arange(0, p['tmax'], p['dt'])

    def _predict(self, time):
        p = self._pars
        t = self._time()
        signal = self._signal_model(p)
        return sample(time, t, signal['S'], self._pars['TS'])
    
        # TODO: Build Sample Module, so _predict(time) becomes
        # return self._signal_model(p | {'time': time})['signal']

    def _train(
        self, time: np.ndarray, signal: np.ndarray, free:dict=None,
        bounds:dict=None, n0=1, **kwargs,
    ):
        # Estimate parameters
        p = self._pars
        bat = estimate_bat(time, signal)
        p['BAT'] = max(bat - p['Thl'], 0)
        p['Sb_a'] = signal[..., :n0]

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train(self._predict, time, signal, p, free, **kwargs)
    
    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        # self._compute_signal()
        signal = self._signal_model(self._pars)
        t = self._time()
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        if signal.ndim==1:
            ax0.plot(time/60, signal, marker='o', color='lightcoral', alpha=0.5, label='Data')
            ax0.plot(t/60, self._S, linestyle='-', color='darkred', linewidth=3, label='Prediction')
        else:
            for i in range(signal.shape[0]):
                ax0.plot(time/60, signal[i,:], marker='o', color='lightcoral', alpha=0.5, label='Data')
                ax0.plot(t/60, self._S[i,:], linestyle='-', color='darkred', linewidth=3, label='Prediction')                
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(t/60, 1000*self._C, linestyle='-', color='darkred', linewidth=3, label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, select=None) -> list:
        """Return a list of model parameters"""
        return self._signal_model.inputs(select)

    def time(self) -> np.ndarray:
        """Internal time array"""
        return self._time()

    def conc(self) -> np.ndarray:
        """Returns the predicted aorta blood concentration."""
        signal = self._signal_model(self._pars)
        return signal['C']

    def relax(self) -> tuple:
        """Returns the predicted relaxation rates."""
        signal = self._signal_model(self._pars)
        return signal['R']
    
    def signal(self) -> np.ndarray:
        """Returns time points and predicted liver signal."""
        signal = self._signal_model(self._pars)
        return signal['S']

    def predict(self, time:np.ndarray) -> np.ndarray:
        """Predicts the aorta signal at specified time points."""
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        return self._predict(time)
    
    def train(
            self, time: np.ndarray, signal: np.ndarray, free: dict=None, 
            bounds: dict=None, n0=10, **kwargs
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
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)        
        return self._train(time, signal, free, bounds, n0, **kwargs)
    
    def plot(self, time: np.ndarray, signal:np.ndarray, 
             fname:str=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of signals
            signal (tuple): Liver signals            
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._plot(time, signal, fname, show)

    def cost(self, time: dict, signal: dict, metric: str='NRMS', nfree=None) -> float:
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
            - 'BIC': Baysian information criterion.
        """
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        signal_pred = self._predict(time)
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]