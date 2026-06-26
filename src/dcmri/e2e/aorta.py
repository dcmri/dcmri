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
from dcmri.core.quantities import QVALUES
from dcmri.core.sequences import SEQUENCES
from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.tissue import Relax
from dcmri.bloch.tissue import Signal


class Aorta(SuperRoiModel):
    """Whole-body model for the aorta signal.

    This model uses a whole-body pharmacokinetic architecture to predict the 
    MRI signal in the aorta by modeling the injection, heart-lung transit, 
    and systemic circulation.

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        sequence (str, optional): Imaging sequence.
        **params: override parameter defaults
    """
    configs = {
        'heartlung': ConcAorta.configs['heartlung'],
        'organs': ConcAorta.configs['organs'],
        'sequence': Signal.configs['sequence'],
    }
    def __init__(
        self, 
        heartlung='pfcomp',
        organs='comp',
        sequence='3D-SPGR-SS',
        **params,
    ):
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
            'sequence': sequence, 
        }
        self._version = '1.0'
        self._set_config(cnfg)
        self._set_params(QVALUES | params)

        # Set multi-channel baseline if not done by the user
        if sequence in ['Eq-DE-EPI', 'DE-EPI']:
            if 'Sb_a' not in params:
                self._pars['Sb_a'] = np.full(2, self._pars['Sb_a'])

    # ==========================================
    # Backend
    # ==========================================

    # Helper function
    def _tissue_props(self):
        return set(SEQUENCES[self._cnfg['sequence']]['parameters']['tissue'])

    # ==========================================
    # Model Parameters
    # ==========================================

    def _params(self, select=None):
        props = self._tissue_props()
        pars = []
        if select is None:
            # Explicit parameters
            pars = [
                'field_strength', 'agent', 'tmax', 'dt', 'TS',
                'R1b_a', 'R2b_a', 'R2sb_a', 
                'Sb_a', 'B1corr_a', 
            ]
            # Implicit parameters
            pars += ConcAorta(**self._cnfg).params() 
            pars += Relax(tissue_props=props, **self._cnfg).params()
            pars += Signal(calibrate=True, **self._cnfg).params()
            
        if select == 'free':
            pars += ConcAorta(**self._cnfg).params('free')  

        derived = [
            'c', 
            'r1', 'r2', 'r2s', 'R1b', 'R2b', 'R2sb', 
            'R1', 'R2', 'R2s', 'Sb', 'B1corr',
            ]
        pars = {p for p in pars if p not in derived}
        return list(pars) 

    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_conc(self) -> np.ndarray:
        p = self._pars
        self._C = ConcAorta(**self._cnfg, defaults=p)()

    def _compute_relax(self):
        self._compute_conc()
        p = self._pars
        props = self._tissue_props()
        relaxivity = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        baseline_relaxation_rate = {f"{relax_rate}b": p[f"{relax_rate}b_a"] for relax_rate in props} 

        inputs = self._pars | relaxivity | baseline_relaxation_rate | {'c': self._C}
        config = self._cnfg | {'tissue_props': props}
        self._R = Relax(defaults=inputs, **config)()

    def _compute_signal(self):
        self._compute_relax()
        p = self._pars

        inputs = self._pars | self._R | {'Sb': p['Sb_a'], 'B1corr': p['B1corr_a']}
        config = self._cnfg | {'calibrate': True}
        self._S = Signal(defaults=inputs, **config)()

    def _time(self):
        p = self._pars
        return np.arange(0, p['tmax'], p['dt'])

    def _predict(self, time):
        self._compute_signal()
        t = self._time()
        return sample(time, t, self._S, self._pars['TS'])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

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
    
    # ==========================================
    # Plot
    # ==========================================

    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        self._compute_signal()
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
        return self._params(select)

    def time(self) -> np.ndarray:
        """Internal time array"""
        return self._time()

    def conc(self) -> np.ndarray:
        """Returns the predicted aorta blood concentration."""
        self._compute_conc()
        return self._C

    def relax(self) -> tuple:
        """Returns the predicted relaxation rates."""
        self._compute_relax()
        return self._R
    
    def signal(self) -> np.ndarray:
        """Returns time points and predicted liver signal."""
        self._compute_signal()
        return self._S

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