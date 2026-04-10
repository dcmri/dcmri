from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.signal_to_conc import SignalToConc
from dcmri import sig, utils, cort_med
from dcmri.ui import SuperModel, Input
from dcmri.lexicon import SEQUENCES
from dcmri.utils import lib


class CortMed(SuperModel):
    """
    General model for renal cortico-medullary data.

    **warning**: This model is functional but under active 
    development. Future versions may change without warning.

    See Also:
        `Kidney`, `Liver`

    Example:

        Derive model parameters from simulated data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use `fake_kidney` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_kidney(CNR=100)

        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.KidneyCortMed(
        ...     aif = aif,
        ...     dt = time[1],
        ...     agent = 'gadoterate',
        ...     TR = 0.005,
        ...     FA = 15,
        ...     TC = 0.2,
        ...     n0 = 10,
        ... )

        Train the model on the ROI data and predict signals and concentrations:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> model.plot(time, roi, ref=gt)
    """

    configs = {
        'kinetics': ['7C'],
        'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']
      }
    
    def __init__(self, kinetics='7C', sequence='3D-SPGR-SS', **params):
        self._version = '1.0'
        self._cnfg = self._set_config(kinetics=kinetics, sequence=sequence)
        self._pars = self._set_pars(**params)

    def _params(self, select=None):
        kin, seq = self._cnfg['kinetics'], self._cnfg['sequence']
        pars_kin = cort_med.Conc(kin)._params()
        pars_seq = SEQUENCES[seq]['parameters']['prep']
        pars_seq += SEQUENCES[seq]['parameters']['read']

        if select is None:
            pars_list = [
                'c_a', 'dt', 'field_strength', 'agent',
                'H', 'S0_c', 'S0_m', 'R10_c', 'R10_m', 'TS',
            ]
            pars_list += pars_kin + pars_seq
        elif select=='free':
            pars_list = pars_kin
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================   

    def _compute_concentration(self):
        p = self._pars
        kin = self._cnfg['kinetics']
        ca = p['c_a'] / (1 - p['H'])
        self._Cc, self._Cm = cort_med.Conc(kin, **p)(ca, dt=p['dt'])

    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1c = p['R10_c'] + rp * self._Cc.sum(axis=0)
        self._R1m = p['R10_m'] + rp * self._Cm.sum(axis=0)

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars
        seq = self._cnfg['sequence']
        self._Sc = sig.Signal(seq, **p)(R1=self._R1c, TE=0)
        self._Sm = sig.Signal(seq, **p)(R1=self._R1m, TE=0)

    def _set_time(self):
        p = self._pars
        self._t = p['dt'] * np.arange(p['c_a'].size)
        
    def _predict(self, time) -> Tuple[np.ndarray, np.ndarray]:
        self._set_time()
        self._compute_signal()
        return (
            utils.sample(time[0], self._t, self._Sc, self._pars['TS']),
            utils.sample(time[1], self._t, self._Sm, self._pars['TS']),
        )
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================
    
    def _estimate_parameters(self, signal: np.ndarray, n0: int, aif: Input):
        p = self._pars
        seq = self._cnfg['sequence']

        # Estimate S0
        s_ref_c = sig.Signal(seq, **p)(R1=p['R10_c'], S0=1, TE=0)
        s_ref_m = sig.Signal(seq, **p)(R1=p['R10_m'], S0=1, TE=0)
        p['S0_c'] = np.mean(signal[0][:n0]) / s_ref_c if s_ref_c > 0 else 0
        p['S0_m'] = np.mean(signal[1][:n0]) / s_ref_m if s_ref_m > 0 else 0

        if aif is not None:
            rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
            ca = SignalToConc(seq, **p)(
                aif.signal, S0=None, R1=aif.R10, n0=n0, 
                B1corr=aif.B1corr, r1=rp,
            )
            p['c_a'] = np.interp(self._t, aif.time, ca)

    def _train(
        self, time, signal, free, bounds, n0, aif, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        
        self._estimate_parameters(signal, n0, aif)
        free = self._set_free_pars(free, bounds)
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)
    
    def _plot(self, time, signal, xlim, fname, show):
        self._set_time()
        self._compute_signal()
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(time[0]/60, signal[0], marker='o', linestyle='None', color='cornflowerblue', label='Cortex data')
        ax0.plot(time[1]/60, signal[1], marker='x', linestyle='None', color='cornflowerblue', label='Medulla data')
        ax0.plot(self._t/60, self._Sc, linestyle='-', linewidth=3.0, color='darkblue', label='Cortex prediction')
        ax0.plot(self._t/60, self._Sm, linestyle='--', linewidth=3.0, color='darkblue', label='Medulla prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations.')

        ax1.plot(self._t/60, 1000*self._Cc.sum(axis=0), linestyle='-', linewidth=3.0, color='darkblue', label='Cortex prediction')
        ax1.plot(self._t/60, 1000*self._Cm.sum(axis=0), linestyle='--', linewidth=3.0, color='darkblue', label='Medulla prediction')
        ax1.plot(self._t/60, 1000*self._pars['c_a'], linestyle='-', linewidth=3.0, color='darkred', label='Arterial prediction')
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    # ==========================================
    # Public API
    # ==========================================

    def time(self) -> dict:
        """Cortex and medulla signal times"""
        self._set_time()
        return {
            'cort': self._t, 
            'med': self._t,
        }

    def conc(self) -> dict:
        """Returns cortex and medulla concentrations."""
        self._compute_concentration()
        return {
            'cort': self._Cc, 
            'med': self._Cm,
        }

    def relax(self) -> dict:
        """Returns cortex and medulla relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return {
            'cort': self._R1c, 
            'med': self._R1m,
        }

    def signal(self) -> dict:
        """Returns cortex and medulla signal."""
        self._compute_signal()
        return {
            'cort': self._Sc, 
            'med': self._Sm,
        }

    def predict(self, time: dict) -> dict:
        """Predicts cortex and medulla signal at specific time points."""
        time = (time['cort'], time['med'])
        self._set_time()
        if max(self._t) < np.max(np.concatenate(time)) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        signal = self._predict(time)
        return {
            'cort': signal[0],
            'med': signal[1],
        }
    
    def train(
        self, time: tuple, signal: tuple, aif:Input=None, 
        free: dict=None, bounds:dict=None, n0=1, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

        Args:
            time (tuple): Time points of cortex and medulla signals
            signal (tuple): Cortex and medulla signals
            free (dict, optional): Free parameters and their
                bounds. If not provided, a default set of free parameters is used.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            aif (dict, optional): AIF signal, time and baseline R1.
            kwargs: any keyword parameters accepted by 
                `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        time = (time['cort'], time['med'])
        signal = (signal['cort'], signal['med'])
        self._set_time()
        if max(self._t) < np.max(np.concatenate(time)) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._train(time, signal, free, bounds, n0, aif, **kwargs)


    def plot(self, time: tuple, signal: tuple, xlim=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of cortex and medulla signals
            signal (tuple): Cortex and medulla signals            
            xlim (list, optional): Lower and upper boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        time = (time['cort'], time['med'])
        signal = (signal['cort'], signal['med'])
        self._plot(time, signal, xlim, fname, show)

    def cost(self, time: np.ndarray, signal: np.ndarray, metric: str = 'NRMS', nfree=None) -> float:
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
        time = (time['cort'], time['med'])
        signal = np.concatenate((signal['cort'], signal['med']))
        signal_pred = np.concatenate(self._predict(time))
        cost = utils.loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
