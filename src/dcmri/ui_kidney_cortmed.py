from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import dcmri.pk as pk
import dcmri.kidney as pkk
import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
from dcmri.ui import SuperModel
from dcmri.lexicon import LEXICON


class KidneyCortMed(SuperModel):
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

    def __init__(self, sequence='SR', **params):
        # Check inputs
        if sequence not in ['SS', 'SR', 'lin']:
            raise ValueError(f"Sequence {sequence} is not available.")
        
        # Initialize parameters
        self._version = '1.0'
        self._cnfg = {'sequence': sequence}
        self._pars = {p: deepcopy(LEXICON[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    def _pars_list(self, select=None):
        pars_kin = ['Fp', 'Eg', 'fc', 'Tglom', 'Tv', 'Tpt', 'Tlh', 'Tdt', 'Tcd']
        pars_seq = {
            'SR': ['B1corr', 'FA', 'TR', 'TC', 'TP', 'TS'],
            'SS': ['B1corr', 'FA', 'TR', 'TS'],
            'lin': ['TS'],
        }[self._cnfg['sequence']]

        if select is None:
            pars_list = [
                'c_a', 'dt', 'field_strength', 'agent',
                'H', 'T_a', 'S0_c', 'S0_m', 'R10_c', 'R10_m',
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
        ca = pk.flux(p['c_a'], p['T_a'], dt=p['dt'], model='plug')
        self._Cc, self._Cm = pkk.conc_kidney_cm(
            ca, p['Fp'], p['Eg'], p['fc'], p['Tglom'], p['Tv'], 
            p['Tpt'], p['Tlh'], p['Tdt'], p['Tcd'], 
            dt=p['dt'], sum=False, kinetics='7C'
        )

    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1c = p['R10_c'] + rp * self._Cc.sum(axis=0)
        self._R1m = p['R10_m'] + rp * self._Cm.sum(axis=0)

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars
        if self._cnfg['sequence'] == 'SR':
            self._Sc = sig.signal_spgr(
                p['S0_c'], self._R1c, p['TC'], p['TR'], 
                p['B1corr'] * p['FA'], p['TP']
            )
            self._Sm = sig.signal_spgr(
                p['S0_m'], self._R1m, p['TC'], p['TR'], 
                p['B1corr'] * p['FA'], p['TP']
            )
        elif self._cnfg['sequence'] == 'SS':
            self._Sc = sig.signal_ss(p['S0_c'], self._R1c, p['TR'], p['B1corr'] * p['FA'])
            self._Sm = sig.signal_ss(p['S0_m'], self._R1m, p['TR'], p['B1corr'] * p['FA'])
        elif self._cnfg['sequence'] == 'lin':
            self._Sc = sig.signal_lin(p['S0_c'], self._R1c)
            self._Sm = sig.signal_lin(p['S0_m'], self._R1m)

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
    
    def _estimate_parameters(self, signal: np.ndarray, n0: int, aif: dict):
        p = self._pars
        if self._cnfg['sequence'] == 'SR':
            fa_t = p['B1corr'] * p['FA']
            Scref = sig.signal_spgr(1, p['R10_c'], p['TC'], p['TR'], fa_t, p['TP'])
            Smref = sig.signal_spgr(1, p['R10_m'], p['TC'], p['TR'], fa_t, p['TP'])
        elif self._cnfg['sequence'] == 'SS':
            fa_t = p['B1corr'] * p['FA']
            Scref = sig.signal_ss(1, p['R10_c'], p['TR'], fa_t)
            Smref = sig.signal_ss(1, p['R10_m'], p['TR'], fa_t)
        elif self._cnfg['sequence'] == 'lin':
            Scref = sig.signal_lin(1, p['R10_c'])
            Smref = sig.signal_lin(1, p['R10_m'])

        p['S0_c'] = np.mean(signal[0][:n0]) / Scref if Scref > 0 else 0
        p['S0_m'] = np.mean(signal[1][:n0]) / Smref if Smref > 0 else 0

        if aif is not None:
            r1 = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
            if self._cnfg['sequence'] == 'SR':
                fa_a = aif['B1corr'] * p['FA']
                ca = sig.conc_spgr(aif['signal'], p['TC'], p['TR'], fa_a, p['TP'], 1/aif['R10'], r1)
            elif self._cnfg['sequence'] == 'SS':
                fa_a = aif['B1corr'] * p['FA']
                ca = sig.conc_ss(aif['signal'], p['TR'], fa_a, 1/aif['R10'], r1, n0)
            elif self._cnfg['sequence'] == 'lin':
                ca = sig.conc_lin(aif['signal'], 1/aif['R10'], r1, n0)
            uniform_time = np.arange(0, np.max(aif['time']) + p['TS'] + p['dt'], p['dt'])
            p['c_a'] = np.interp(uniform_time, aif['time'], ca)

    def _train(self, time, signal, free, bounds, n0, aif, **kwargs) -> Tuple[dict, dict, np.ndarray]:
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

    def time(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cortex and medulla signal times"""
        self._set_time()
        return self._t, self._t

    def conc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns cortex and medulla concentrations."""
        self._compute_concentration()
        return self._Cc, self._Cm

    def relax(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns cortex and medulla relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return self._R1c, self._R1m

    def signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns cortex and medulla signal."""
        self._compute_signal()
        return self._Sc, self._Sm

    def predict(self, time: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts cortex and medulla signal at specific time points."""
        self._set_time()
        if max(self._t) < np.max(np.concatenate(time)) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._predict(time)
    
    def train(
        self, time: tuple, signal: tuple, free: dict=None, 
        bounds:dict=None, n0=1, aif:dict=None, **kwargs
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
        self._plot(time, signal, xlim, fname, show)
