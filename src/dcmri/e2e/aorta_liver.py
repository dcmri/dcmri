"""Joint model for aorta and liver signals.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver.  

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    liver (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only single-inlet models 
        are allowed. Defaults to '1I-IC-HFD'.
    non_stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to None.
    sequence (str, optional): imaging sequence. Possible values are '3D-SPGR-SS'
        and 'SR'. Defaults to '3D-SPGR-SS'.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`AortaLiver-parameters` and
        :ref:`AortaLiver-defaults` for a list of parameters and their
        default values.

See Also:
    `AortaLiver2scan`

Example:

    Use the model to reconstruct concentrations from experimentally 
    derived signals.

.. plot::
    :include-source:
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import dcmri as dc

    Use `fake.tissue` to generate synthetic test data from 
    experimentally-derived concentrations:

    Use `fake.liver` to generate synthetic test data:

    >>> time, aif, vif, roi, gt = dc.fake.liver()

    Since this model generates two time curves, the x- and y-data are 
    tuples:

    >>> time, signal = (time,time), (aif,roi)

    Build an aorta-liver model and parameters to match the 
    conditions of the fake liver data:

    >>> model = dc.AortaLiver(
    ...     dt = 0.5,
    ...     tmax = 180,
    ...     weight = 70,
    ...     agent = 'gadoxetate',
    ...     field_strength = 3.0,
    ...     dose = 0.2,
    ...     rate = 3,
    ...     TR = 0.005,
    ...     FA = 15,
    ... )

    Train the model on the data:

    >>> model.train(time, signal, n0=10, xtol=1e-3)

    Plot the reconstructed signals and concentrations and compare 
    against the experimentally derived data:

    >>> model.plot(time, signal)

    We can also have a look at the model parameters after training:

    >>> model.print_params(round_to=3)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    First bolus arrival time (BAT): 13.231 (0.266) sec
    Cardiac output (CO): 102.893 (4.182) mL/sec
    Heart-lung mean transit time (T(hl)): 16.285 (0.409) sec
    Heart-lung dispersion (D(hl)): 0.324 (0.016)
    Organs blood mean transit time (To): 19.578 (5.583) sec
    Organs extraction fraction (Eo): 0.363 (0.075)
    Organs extravascular mean transit time (Toe): 46.775 (53.05) sec
    Body extraction fraction (Eb): 0.029 (0.168)
    Apparent liver extracellular volume fraction (ve_app): 0.307 (0.564) mL/cm3
    Extracellular mean transit time (Te): 44.748 (79.644) sec
    Extracellular dispersion (De): 0.916 (0.148)
    Hepatic plasma clearance (Ktrans): 0.002 (0.003) mL/sec/cm3
    Hepatocellular mean transit time (Th): 712.855 (5899.078) sec
    ----------------------------
    Fixed and derived parameters
    ----------------------------
    Aorta first baseline R1 (R1ba): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.169 a.u.
    Liver first baseline R1 (R1bl): 1.33 Hz
    Liver first signal scale factor (S0l): 150.0 a.u.
    Liver volume (vol): 1000 cm3
    Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.tools import print_params, export_params
from dcmri.core.quantities import QVALUES
from dcmri.core.sequences import SEQUENCES
from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_tissue import Relax
from dcmri.signal.modules_tissue import Signal


class AortaLiver(SuperRoiModel):
    """Joint model or aorta and liver signals.

    A whole-body model to simultaneously predict signals in 
    aorta and liver.  

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `AortaLiver2scan`
    """
    configs = {
        'heartlung': ConcAortaLiver.configs['heartlung'],
        'organs': ConcAortaLiver.configs['organs'],
        'liver': ConcAortaLiver.configs['liver'],
        'non_stationary': ConcAortaLiver.configs['non_stationary'],
        'sequence': Signal.configs['sequence'],
    }

    def __init__(
            self, 
            heartlung='pfcomp', 
            organs='comp', 
            liver='1I-EC', 
            non_stationary=None, 
            sequence='3D-SPGR-SS', 
            **params,
        ):
        self._version = '1.0'
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
            'liver': liver, 
            'non_stationary': non_stationary, 
            'sequence': sequence,
        }
        self._set_config(cnfg)

        # Setup modules
        self._conc = ConcAortaLiver(**cnfg)
        self._relax = {
            roi: Relax(tissue_props=self._tissue_props(roi), fast_water_exchange=True)
            for roi in ['a', 'l']
        }
        self._signal = {
            roi: Signal(calibrate=True, sequence=self._sequence(roi))
            for roi in ['a', 'l']
        }

        
        self._set_params(QVALUES | params)

        # Set multi-channel baseline if not done by the user
        if sequence in ['Eq-DE-EPI', 'DE-EPI']:
            for roi in ['a', 'l']:
                Sb = f"Sb_{roi}"
                if Sb not in params:
                    self._pars[Sb] = np.full(2, self._pars[Sb])

    # ==========================================
    # Backend
    # ==========================================

    # Helper function
    def _sequence(self, roi):
        seq = self._cnfg['sequence']
        if roi in ['l'] and seq == '3D-SPGR-SSI':
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
            for roi in ['a', 'l']:
                pars += [f"{relax_rate}b_{roi}" for relax_rate in self._tissue_props(roi)]
                pars += [f"Sb_{roi}", f"B1corr_{roi}"]

            # Implicit parameters
            pars += self._conc.params()
            for roi in ['a', 'l']:
                pars += self._relax[roi].params()
                pars += self._signal[roi].params()
        
        if select=='free':
            pars += self._conc.params('free')

        # Derived parameters
        derived = [
            'C',
            'r1', 'r2', 'r2s','R1b', 'R2b', 'R2sb', 
            'R1', 'R2', 'R2s', 'Sb', 'B1corr',
        ]
        pars = {p for p in pars if p not in derived}
        return list(pars)
    
    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_conc(self):
        params = self._pars
        self._C = self._conc(**params)

    def _compute_relax(self):  
        self._compute_conc()
        p = self._pars
        
        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = const.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        relaxivity = {
            'a': rb,
            'l': {'r1': [rb['r1'], rh['r1']], 'r2': [rb['r2'], rh['r2']], 'r2s': rb['r2s']}
        }

        self._R = {}
        for roi in ['a', 'l']:
            props = self._tissue_props(roi)
            baseline_relaxation_rate = {f"{relax_rate}b": p[f"{relax_rate}b_{roi}"] for relax_rate in props}
            params = self._pars | relaxivity[roi] | baseline_relaxation_rate | {'C': self._C[roi]}
            self._R[roi] = self._relax[roi](**params)

    def _compute_signal(self):
        self._compute_relax()
        p = self._pars
        self._S = {}
        for roi in ['a', 'l']:
            params = self._pars | self._R[roi] | {'Sb': p[f'Sb_{roi}'], 'B1corr': p[f'B1corr_{roi}']}
            self._S[roi] = self._signal[roi](**params)

    def _time(self):
        p = self._pars
        return np.arange(0, p['tmax'], p['dt'])

    def _predict(self, time: dict) -> dict:
        self._compute_signal()
        t = self._time()
        p = self._pars
        signal = [
            sample(time[i], t, self._S[roi], p['TS']) 
            for i, roi in enumerate(['a', 'l'])
        ]
        return tuple(signal)
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _train(
        self, time: tuple, signal: tuple, free: dict, 
        bounds: dict, n0: int, **kwargs
    ):
        p = self._pars

        # Estimate BAT 
        bat = estimate_bat(time[0], signal[0], n0)
        p['BAT'] = max(bat - p['Thl'], 0)

        # Estimate baseline
        for i, roi in enumerate(['a', 'l']):
            p[f"Sb_{roi}"] = signal[i][..., :n0]

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train(self._predict, time, signal, self._pars, free, **kwargs)


    def _plot(
        self, time: tuple, signal: tuple, xlim: list, fname: str, 
        show: bool
    ):
        self._compute_signal()
        t = self._time()
        p = self._pars

        if xlim is None: xlim = [t[0], t[-1]]
        xlim = np.array(xlim)/60
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot signals
        def _plot_data(sig, ts, s, ax, clr):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            if s.ndim==1:
                ax.plot(ts / 60, s, marker='o', color=clr[0], label='Data', linestyle='None')
                ax.plot(t / 60, sig, linestyle='-', color=clr[1], linewidth=3.0, label='Prediction')
            else:
                for i in range(s.shape[0]):
                    ax.plot(ts / 60, s[i,:], marker='o', color=clr[0], label='Data', linestyle='None')
                    ax.plot(t / 60, sig[i,:], linestyle='-', color=clr[1], linewidth=3.0, label='Prediction')
            ax.legend()

        _plot_data(self._S['a'], time[0], signal[0], ax1, ['lightcoral', 'darkred'])
        _plot_data(self._S['l'], time[1], signal[1], ax3, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(t / 60, 0 * t, color='gray')
        ax2.plot(t / 60, 1000 * self._C['a'], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(t / 60, 0 * t, color='gray')
        if self._C['l'].shape[0]==2:
            ax4.plot(t/60, 1000 * self._C['l'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(t/60, 1000 * self._C['l'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(t/60, 1000 * self._C['l'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        # else:
        #     ax4.plot(self._t/60, 1000*self._Cl[0,:], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname: plt.savefig(fname=fname)
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
            tuple: (aorta_time, liver_time)        
        """
        t = self._time()
        return {
            'aorta': t, 
            'liver': t,
        }

    def conc(self) -> dict:
        """Return concentrations in aorta and liver.

        Returns:
            tuple: (aorta_blood_conc, liver_tissue_conc)
        """
        self._compute_conc()
        return {
            'aorta': self._C['a'], 
            'liver': self._C['l'],
        }

    def relax(self) -> dict:
        """Return relaxation rates in aorta and liver.

        Returns:
            tuple: (aorta_R1, liver_R1)
        """
        self._compute_relax()
        return {
            'aorta': self._R['a'], 
            'liver': self._R['l'], 
        }
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: (time, aorta_signal, liver_signal)
        """
        self._compute_signal()
        return {
            'aorta': self._S['a'], 
            'liver': self._S['l'], 
        }

    def predict(self, time: dict) -> dict:
        """Predict the signals at given time time points.

        Args:
            time (tuple): Tuple of (time_aorta, time_liver) arrays.

        Returns:
            tuple: Tuple of (signal_aorta, signal_liver) arrays.
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(2 * [time])

        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))

        signal = self._predict(time)
        return {
            'aorta': signal[0],
            'liver': signal[1],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=10, **kwargs
    ) -> tuple:
        """Train the model free parameters.

        Args:
            time (tuple): (time_aorta, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points for S0 estimation.
            **kwargs: Arguments passed to scipy.optimize.curve_fit.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(2 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['liver'], 
            )
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        return self._train(time, signal, free, bounds, n0, **kwargs)

    def plot(
        self, time: dict, signal: dict, xlim=None, fname=None, 
        show=True,
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            signal (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
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
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(2 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['liver'], 
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
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(2 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['liver'], 
            )
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
        

    def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['liver'])
        return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    def print_params(self, *args, round_to=None, group=None, 
                     fixed_only=False, free_only=False, deriv=False):
        """Pretty print model parameters"""
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['liver'])
        if args != ():
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self._params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self._params('free')}
        print_params(pars, round_to=round_to, group=group)

