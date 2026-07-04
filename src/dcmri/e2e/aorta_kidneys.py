"""Joint model for signals from aorta and both kidneys.

This model uses a whole body model to simultaneously predict 
signals in aorta and kidneys (see :ref:`whole-body-tissues`). 

See Also:
    `Aorta`, `Kidney`

Args:
    organs (str, optional): Model for the organs in the whole-body 
        model. The options are 'comp' (one compartment) and '2cxm' 
        (two-compartment exchange). Defaults to 'comp'.
    heartlung (str, optional): Model for the heart-lung system in 
        the whole-body model. Options are 'pfcomp' (plug-flow 
        compartment) or 'chain'. Defaults to 'pfcomp'.
    kidneys (str, optional): Model for the kidneys. Options are 
        '2CF' (Two-compartment filtration) and 'HF' (High-flow). 
        Defaults to '2CF'. 
    sequence (str, optional): imaging sequence model. Possible 
        values are 'SS' (steady-state), 'SR' (saturation-recovery), 
        'SSI' (steady state with inflow correction) and 'lin' 
        (linear). Defaults to 'SS'.
    agent (str, optional): Generic name of the contrast agent 
        injected. Defaults to 'gadoterate'.
    params (dict, optional): values for the model parameters,
        specified as keyword parameters. Defaults are used for any 
        that are not provided. See table 
        :ref:`AortaKidneys-defaults` for a list of parameters and 
        their default values.


Example:

    Use the model to fit minipig data with inflow correction:

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

    Create an array of time points:

    >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

    Initialize the tissue:

    >>> aorta_kidneys = dc.AortaKidneys(
    ...     sequence='SSI',
    ...     heartlung='chain',
    ...     organs='comp',
    ...     agent="gadoterate",
    ...     dt=0.25,
    ...     field_strength=pars['B0'],
    ...     weight=pars['weight'],
    ...     dose=pars['dose'],
    ...     rate=pars['rate'],
    ...     R1ba=1/dc.const.T1(pars['B0'], 'blood'),
    ...     R1b_lk=1/dc.const.T1(pars['B0'], 'kidney'),
    ...     R1b_rk=1/dc.const.T1(pars['B0'], 'kidney'),
    ...     vol_lk=85,
    ...     vol_rk=85,
    ...     TR=pars['TR'],
    ...     FA=pars['FA'],
    ...     TS=pars['TS'],
    ...     CO=60,   
    ...     t0=15, 
    ... )

    Define time and signal data

    >>> t = (time, time, time)
    >>> signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])

    Train the system to the data:

    >>> aorta_kidneys.train(t, signal)

    Plot the reconstructed signals and concentrations:

    >>> aorta_kidneys.plot(t, signal)

    Print the model parameters:

    >>> aorta_kidneys.print_params(round_to=4)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    Bolus arrival time (BAT): 16.7422 (0.2853) sec
    Inflow time (TF): 0.2801 (0.0133) sec
    Cardiac output (CO): 72.762 (12.4426) mL/sec
    Heart-lung mean transit time (Thl): 16.2249 (0.3069) sec
    Organs blood mean transit time (To): 14.3793 (1.2492) sec
    Body extraction fraction (Eb): 0.0751 (0.0071)
    Heart-lung dispersion (Dhl): 0.0795 (0.0041)
    Renal plasma flow (RPF): 3.3489 (0.7204) mL/sec
    Differential renal function (DRF): 0.9085 (0.0212)
    Differential renal plasma flow (DRPF): 0.812 (0.0169)
    Left kidney arterial mean transit time (Ta_lk): 0.6509 (0.2228) sec
    Left kidney plasma volume (vp_lk): 0.099 (0.0186) mL/cm3
    Left kidney tubular mean transit time (Tt_lk): 46.9705 (3.3684) sec
    Right kidney arterial mean transit time (Ta_rk): 1.4206 (0.2023) sec
    Right kidney plasma volume (vp_rk): 0.1294 (0.0175) mL/cm3
    Right kidney tubular mean transit time (Tt_rk): 4497.8301 (39890.3818) sec
    Aorta signal scaling factor (S0a): 4912.776 (254.2363) a.u.
    ----------------------------
    Fixed and derived parameters
    ----------------------------
    Filtration fraction (FF): 0.0812
    Glomerular Filtration Rate (GFR): 0.2719 mL/sec
    Left kidney plasma flow (RPF_lk): 2.7194 mL/sec
    Right kidney plasma flow (RPF_rk): 0.6295 mL/sec
    Left kidney glomerular filtration rate (GFR_lk): 0.247 mL/sec
    Right kidney glomerular filtration rate (GFR_rk): 0.0249 mL/sec
    Left kidney plasma flow (Fp_lk): 0.032 mL/sec/cm3
    Left kidney plasma mean transit time (Tp_lk): 2.838 sec
    Left kidney vascular mean transit time (Tv_lk): 3.0958 sec
    Left kidney tubular flow (Ft_lk): 0.0029 mL/sec/cm3
    Left kidney filtration fraction (FF_lk): 0.0908
    Left kidney extraction fraction (E_lk): 0.0833
    Right kidney plasma flow (Fp_rk): 0.0074 mL/sec/cm3
    Right kidney plasma mean transit time (Tp_rk): 16.8121 sec
    Right kidney vascular mean transit time (Tv_rk): 17.4762 sec
    Right kidney tubular flow (Ft_rk): 0.0003 mL/sec/cm3
    Right kidney filtration fraction (FF_rk): 0.0395
    Right kidney extraction fraction (E_rk): 0.038
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
from dcmri.kinetics.modules_conc import ConcAortaKidneys
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.signal.modules_tissue import Signal

class AortaKidneys(SuperRoiModel):
    """Joint model for signals from aorta and both kidneys.

    A whole body model to simultaneously predict 
    signals in aorta and both kidneys. 

    See Also:
        `Aorta`, `Kidney`

    Args:
        heartlung (str, optional): Model for the heart-lung system. 
        organs (str, optional): Model for the systemic organs. 
        kidneys (str, optional): Kidney tracer-kinetic model.
        sequence (str, optional): Imaging sequence.
        liver_clearance (bool, optional): Contrast agent with liver clearance.
        **params: override parameter defaults

    """
    configs = {
        'heartlung': ConcAortaKidneys.configs['heartlung'],
        'organs': ConcAortaKidneys.configs['organs'],
        'kidneys': ConcAortaKidneys.configs['kidneys'],
        # 'sequence': Signal.configs['sequence'],
    }
    def __init__(
        self, 
        heartlung='pfcomp', 
        organs='comp', 
        kidneys='2CF', 
        sequence='3D-SPGR-SS',
        **params,
    ):
        self._version = '1.0'
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
            'kidneys': kidneys, 
            'sequence': sequence, 
        }
        self._set_config(cnfg)
        self._set_params(QVALUES | params)

        # Set multi-channel baseline if not done by the user
        if sequence in ['Eq-DE-EPI', 'DE-EPI']:
            for roi in ['a', 'lk', 'rk']:
                Sb = f"Sb_{roi}"
                if Sb not in params:
                    self._pars[Sb] = np.full(2, self._pars[Sb])

    # ==========================================
    # Backend
    # ==========================================

    # Helper function
    def _sequence(self, roi):
        seq = self._cnfg['sequence']
        if roi in ['lk', 'rk'] and seq == '3D-SPGR-SSI':
            return '3D-SPGR-SS'
        return seq
    
    # Helper function
    def _tissue_props(self, roi):
        return set(SEQUENCES[self._sequence(roi)]['parameters']['tissue'])
    
    # ==========================================
    # Model Parameters
    # ==========================================
    
    def _params(self, select=None):
        pars = []
        if select is None:
            # Explicit parameters
            pars += ['field_strength', 'agent', 'dt', 'tmax', 'TS']
            for roi in ['a', 'lk', 'rk']:
                pars += [f"{relax_rate}b_{roi}" for relax_rate in self._tissue_props(roi)]
                pars += [f"Sb_{roi}", f"B1corr_{roi}"]

            # Implicit parameters
            pars += ConcAortaKidneys(**self._cnfg).params()
            for roi in ['a', 'lk', 'rk']:
                pars += Relax(tissue_props=self._tissue_props(roi), **self._cnfg).params()
                pars += Signal(calibrate=True, sequence=self._sequence(roi)).params()
        
        if select=='free':
            pars += ConcAortaKidneys(**self._cnfg).params('free')

        # Derived parameters
        derived = [
            'C',
            'r1', 'r2', 'r2s', 'R1b', 'R2b', 'R2sb', 
            'R1', 'R2', 'R2s', 'Sb', 'B1corr',
        ]
        pars = {p for p in pars if p not in derived}
        return list(pars)
    
    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_conc(self):
        p = self._pars
        self._C = ConcAortaKidneys(**self._cnfg, defaults=p)()
            
    def _compute_relax(self):  
        self._compute_conc()
        p = self._pars
        self._R = {}
        relaxivity = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        for roi in ['a', 'lk', 'rk']:
            props = self._tissue_props(roi)
            baseline_relaxation_rate = {f"{relax_rate}b": p[f"{relax_rate}b_{roi}"] for relax_rate in props}

            inputs = self._pars | relaxivity | baseline_relaxation_rate | {'C': self._C[roi]}
            config = self._cnfg | {'tissue_props': props, 'fast_water_exchange': True}
            self._R[roi] = Relax(defaults=inputs, **config)()

    def _compute_signal(self):
        self._compute_relax()
        p = self._pars
        self._S = {}
        for roi in ['a', 'lk', 'rk']:
            inputs = self._pars | self._R[roi] | {'Sb': p[f'Sb_{roi}'], 'B1corr': p[f'B1corr_{roi}']}
            config = self._cnfg | {'calibrate': True, 'sequence': self._sequence(roi)}
            self._S[roi] = Signal(defaults=inputs, **config)()

    def _time(self):
        p = self._pars
        return np.arange(0, p['tmax'], p['dt'])
     
    def _predict(self, time: dict) -> dict:
        self._compute_signal()
        t = self._time()
        p = self._pars
        signal = [
            sample(time[i], t, self._S[roi], p['TS']) 
            for i, roi in enumerate(['a', 'lk', 'rk'])
        ]
        return tuple(signal)
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _train(
        self, time: dict, signal: dict, free: dict, 
        bounds: dict, n0: int, **kwargs
    ):
        p = self._pars

        # Estimate BAT 
        bat = estimate_bat(time[0], signal[0], n0)
        p['BAT'] = max(bat - p['Thl'], 0)

        # Estimate baseline
        for i, roi in enumerate(['a', 'lk', 'rk']):
            p[f"Sb_{roi}"] = signal[i][..., :n0]
            # p[f"Sb_{roi}"] = np.mean(signal[i][..., :n0], axis=-1)

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train(self._predict, time, signal, self._pars, free, **kwargs)

    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _plot(self, time, signal, xlim, fname, show):
        self._compute_signal()
        t = self._time()
        p = self._pars

        if xlim is None: xlim = [t[0], t[-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(wspace=0.3)

        def plot_data(sig, ts, s, roi, ax, color):
            ax.set(xlabel='Time (min)', ylabel=f'{roi} signal (a.u.)', xlim=xlim)
            if s.ndim==1:
                ax.plot(ts / 60, s, marker='o', color=color[0], label='Data', linestyle='None')
                ax.plot(t / 60, sig, linestyle='-', color=color[1], linewidth=3.0, label='Prediction')
            else:
                for i in range(s.shape[0]):
                    ax.plot(ts / 60, s[i,:], marker='o', color=color[0], label='Data', linestyle='None')
                    ax.plot(t / 60, sig[i,:], linestyle='-', color=color[1], linewidth=3.0, label='Prediction')
            ax.legend()

        plot_data(self._S['a'], time[0], signal[0], 'Aorta', ax1, ['lightcoral', 'darkred'])
        plot_data(self._S['lk'], time[1], signal[1], 'Left kidney', ax3, ['cornflowerblue', 'darkblue'])
        plot_data(self._S['rk'], time[2], signal[2], 'Right kidney', ax5, ['cornflowerblue', 'darkblue'])

        # Plot aorta
        cb_lk = self._C['lk'][0,:] / (p['vp_lk'] / (1 - p['H']))
        cb_rk = self._C['rk'][0,:] / (p['vp_rk'] / (1 - p['H']))

        ax2.set(xlabel='Time (min)', ylabel='Blood conc (mM)', xlim=xlim)
        ax2.plot(t / 60, 0 * t, color='gray')
        ax2.plot(t / 60, 1000 * self._C['a'], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.plot(t / 60, 1000 * cb_lk, linestyle='--', color='lightcoral', linewidth=2.0, label='Left kidney')
        ax2.plot(t / 60, 1000 * cb_rk, linestyle='-.', color='lightcoral', linewidth=2.0, label='Right kidney')
        ax2.legend()

        def plot_conc_kidney(C, kid, ax):
            ax.set(xlabel='Time (min)', ylabel=f'{kid} conc (mM)', xlim=xlim)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * C[0, :], linestyle='-', color='darkred', linewidth=2.0, label='Blood')
            ax.plot(t / 60, 1000 * C[1, :], linestyle='-', color='darkcyan', linewidth=2.0, label='Tubuli')
            ax.plot(t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
            ax.legend()

        plot_conc_kidney(self._C['lk'], 'Left kidney', ax4)
        plot_conc_kidney(self._C['rk'], 'Right kidney', ax6)

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()


    # ==========================================
    # User Interface
    # ==========================================

    def params(self, select=None) -> list:
        """Return a list of model parameters"""
        return self._params(select)

    def time(self) -> dict:
        """Internal time array

        Returns:
            Tuple: (aorta_time, portal_time, liver_time).
        """
        t = self._time()
        return {
            'aorta': t, 
            'kidney_left': t, 
            'kidney_right': t,
        }

    def conc(self) -> dict:
        """Concentrations in aorta and kidney.

        Args:
            sum (bool, optional): If set to true, the kidney 
              concentrations are the sum over all compartments. If 
              set to false, the compartmental concentrations are 
              returned individually. Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, left 
              kidney concentrations, right kidney concentrations.
        """
        self._compute_conc()
        return {
            'aorta': self._C['a'], 
            'kidney_left': self._C['lk'], 
            'kidney_right': self._C['rk'],
        }

    def relax(self) -> dict:
        """Relaxation rates in aorta and kidney.

        Returns:
            dict: aorta relaxation rates, left kidney 
              relaxation rates, right kidney relaxation rates.
        """
        self._compute_relax()
        return {
            'aorta': self._R['a'], 
            'kidney_left': self._R['lk'], 
            'kidney_right': self._R['rk'],
        }
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal()
        return {
            'aorta': self._S['a'], 
            'kidney_left': self._S['lk'], 
            'kidney_right': self._S['rk'],
        }

    def predict(self, time) -> dict:
        """Predict the data at given time

        Args:
            time (tuple): Tuple of 3 arrays with time points for 
              aorta, left kidney and right kidney, in that order. 
              The three arrays can all be different in length and value.

        Returns:
            tuple: Tuple of 3 arrays with signals for aorta, left 
              kidney and right kidney, in that order. The three 
              arrays can all be different in length and value but 
              each has to have the same length as its corresponding 
              array of time points.
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])

        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        signal = self._predict(time)
        return {
            'aorta': signal[0],
            'kidney_left': signal[1],
            'kidney_right': signal[2],
        }

    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=10, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

       Args:
            time (tuple): (time_aorta, time_portal, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_portal, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific params.
            n0 (int, optional): Baseline points for S0 estimation. Defaults to 1.
            **kwargs: Passed to scipy.optimize.curve_fit via utils.train.

        Returns:
            vals, sdev, pcov: Values, standard deviations and 
              covariance matrix of free parameters

        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['kidney_left'], 
                signal['kidney_right'], 
            )

        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))

        return self._train(time, signal, free, bounds, n0, **kwargs)

    def plot(
        self, time: dict, signal: dict, xlim: list = None, 
        fname: str = None, show = True,
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The two arrays can be 
              different in length and value.
            signal (array-like): tuple of 3 arrays with signals for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value but each has to have the same 
              length as its corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['kidney_left'], 
                signal['kidney_right'], 
            )

        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._plot(time, signal, xlim, fname, show)

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
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'], 
                time['kidney_right'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['kidney_left'], 
                signal['kidney_right'], 
            )
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]