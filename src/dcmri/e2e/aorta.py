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
    ...     R10=1/dc.const.T1(pars['B0'], 'blood'),
    ... )

    Create an array of time points:

    >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

    Train the system to the data:

    >>> aorta.train(time, rois['Aorta'])

    Plot the reconstructed signals and concentrations:

    >>> aorta.plot(time, rois['Aorta'])

    Print the model parameters:

    >>> aorta.print_params(round_to=4)
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import const
from dcmri.kinetics import ConcAorta
from dcmri.lexicon import SEQUENCES
from dcmri.core import SuperModel
from dcmri.bloch import Signal
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss



class Aorta(SuperModel):
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
        'heartlung': ['comp', 'pfcomp', 'chain'],
        'organs': ['comp','2cxm'],
        'sequence': ['3D-SPGR-SS', '3D-SR-SPGR-SS', '3D-SPGR-SSI'],
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
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def _params(self, select=None):
        # Sequence parameters
        seq = self._cnfg['sequence']
        sequence = SEQUENCES[seq]['parameters']['prep']
        sequence += SEQUENCES[seq]['parameters']['read']
        replace = {'S0': 'S0_a', 'B1corr': 'B1corr_a'}
        sequence = [replace.get(x, x) for x in sequence]
    
        free_inflow = ['TF', 'S0_a'] if seq == '3D-SPGR-SSI' else []

        c = ConcAorta(**self._cnfg)

        if select in [None, 'all']:
            return c._params() + sequence + ['TS', 'R10_a', 'R20s_a']
        if select in ['free']:
            return c._params('body') + free_inflow     

    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])

    def _compute_conc(self) -> np.ndarray:
        self._C = ConcAorta(**self._cnfg)(**self._pars)

    def _compute_relax(self):
        self._compute_conc()
        p = self._pars
        r1 = const.r1(p['field_strength'], 'blood', p['agent'])
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        self._R1 = p['R10_a'] + r1 * self._C
        self._R2s = p['R20s_a'] + r2s * self._C

    def _compute_signal(self):
        self._compute_relax()
        p = self._pars
        seq = self._cnfg['sequence']
        self._S = Signal(seq, **p)(
            R1=self._R1, 
            R2s=self._R2s,
            S0=p['S0_a'], 
            B1corr=p['B1corr_a'],
        )

    def _predict(self, time):
        self._compute_time()
        self._compute_signal()
        return sample(time, self._t, self._S, self._pars['TS'])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: tuple, signal: tuple, n0: int):
        p = self._pars

        # Estimate BAT based on peak signal
        if self._cnfg['heartlung']=='comp':
            offset = p['Thl']
        else:
            offset = (1 - p['Dhl']) * p['Thl']
        bat = time[np.argmax(signal)] - offset
        p['BAT'] = max(bat, 0)

        # Scaling Factor (S0) aorta
        seq = self._cnfg['sequence']
        s_ref = Signal(seq, **p)(R1=p['R10_a'], R2s=p['R20s_a'], S0=1, B1corr=p['B1corr_a'])
        p['S0_a'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0


    def _train(
        self, time: tuple, signal: tuple, free:dict=None, 
        bounds:dict=None, n0=1, **kwargs,
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")
        
        # Optimization
        return train(self._predict, time, signal, self._pars, free, **kwargs)

    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        self._compute_signal()
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(time/60, signal, 'ko', alpha=0.5, label='Data')
        ax0.plot(self._t/60, self._S, 'r-', linewidth=2, label='Prediction')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(self._t/60, 1000*self._C, 'r-', label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()


    # ---- Public API ----

    def time(self) -> np.ndarray:
        """Internal time array"""
        self._compute_time()
        return self._t

    def conc(self) -> np.ndarray:
        """Returns the predicted aorta blood concentration."""
        self._compute_conc()
        return self._C

    def relax(self) -> tuple:
        """Returns the predicted relaxation rates."""
        self._compute_relax()
        return self._R1, self._R2s
    
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
            self, time: tuple, signal: tuple, free: dict=None, 
            bounds: dict=None, n0=1, **kwargs
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


    def cost(self, time: dict, signal: dict, metric: str = 'NRMS', nfree=None) -> float:
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