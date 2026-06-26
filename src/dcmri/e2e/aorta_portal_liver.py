"""Joint model for aorta, portal vein and liver signals.

This model uses a whole-body model to simultaneously predict signals in 
aorta, portal vein and liver.  

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    liver (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only dual-inlet models 
        are allowed. Defaults to '2I-EC'.
    non_stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to None.
    sequence (str, optional): imaging sequence. Possible values are 'SS'
        and 'SSI' (steady-state with aortic inflow correction). Defaults 
        to 'SS'.
    free (dict, optional): Dictionary with free parameters and their
        bounds. If not provided, a default set of free parameters is used.
        Defaults to None.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`AortaLiver-parameters` and
        :ref:`AortaLiver-defaults` for a list of parameters and their
        default values.

See Also:
    `AortaLiver`

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

    >>> time, aif, vif, roi, _ = dc.fake.liver(sequence='SSI')

    Since this model generates 3 time curves, the x- and y-data are 
    tuples:

    >>> xdata, ydata = (time, time, time), (aif, vif, roi)

    Build an aorta-portal-liver model and parameters to match the 
    conditions of the fake liver data:

    >>> model = dc.AortaPortalLiver(
    ...     liver = '2I-IC',
    ...     sequence = 'SSI',
    ...     dt = 0.5,
    ...     tmax = 180,
    ...     weight = 70,
    ...     agent = 'gadoxetate',
    ...     dose = 0.2,
    ...     rate = 3,
    ...     field_strength = 3.0,
    ...     TR = 0.005,
    ...     FA = 15,
    ...     TS = 0.5,
    ... )

    Train the model on the data:

    >>> model.train(xdata, ydata, n0=10, xtol=1e-3)

    Plot the reconstructed signals and concentrations and compare 
    against the experimentally derived data:

    >>> model.plot(xdata, ydata)

    We can also have a look at the model parameters after training:


"""
import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QVALUES
from dcmri.core.sequences import SEQUENCES
from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.tissue import Relax
from dcmri.bloch.tissue import Signal


class AortaPortalLiver(SuperRoiModel):
    """Joint prediction of aorta, portal vein and liver signals.

    A whole-body model to simultaneously predict signals in 
    aorta, portal vein and liver.  

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `AortaLiver`
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
        self._set_params(QVALUES | params)

        # Set multi-channel baseline if not done by the user
        if sequence in ['Eq-DE-EPI', 'DE-EPI']:
            for roi in ['a', 'pv', 'l']:
                Sb = f"Sb_{roi}"
                if Sb not in params:
                    self._pars[Sb] = np.full(2, self._pars[Sb])

    # ==========================================
    # Backend
    # ==========================================

    # Helper function
    def _sequence(self, roi):
        seq = self._cnfg['sequence']
        if roi in ['pv', 'l'] and seq == '3D-SPGR-SSI':
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
            pars += ['field_strength', 'agent', 'uv', 'dt', 'tmax', 'TS']
            for roi in ['a', 'pv', 'l']:
                pars += [f"{relax_rate}b_{roi}" for relax_rate in self._tissue_props(roi)]
                pars += [f"Sb_{roi}", f"B1corr_{roi}"]

            # Implicit parameters
            pars += ConcAortaLiver(**self._cnfg).params()
            for roi in ['a', 'pv', 'l']:
                pars += Relax(tissue_props=self._tissue_props(roi), **self._cnfg).params()
                pars += Signal(calibrate=True, sequence=self._sequence(roi)).params()
        
        if select=='free':
            pars += ConcAortaLiver(**self._cnfg).params('free')

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
        self._C = ConcAortaLiver(**self._cnfg, defaults=p)()

    def _compute_relax(self):  
        self._compute_conc()
        p = self._pars
        
        rb = const.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = const.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        relaxivity = {
            'a': rb,
            'pv': {k: v * p['uv'] for k, v in rb.items()},
            'l': {'r1': [rb['r1'], rh['r1']], 'r2': [rb['r2'], rh['r2']], 'r2s': rb['r2s']}
        }

        self._R = {}
        for roi in ['a', 'pv', 'l']:
            props = self._tissue_props(roi)
            baseline_relaxation_rate = {f"{relax_rate}b": p[f"{relax_rate}b_{roi}"] for relax_rate in props}

            inputs = self._pars | relaxivity[roi] | baseline_relaxation_rate | {'C': self._C[roi]}
            config = self._cnfg | {'tissue_props': props, 'fast_water_exchange': True}
            self._R[roi] = Relax(defaults=inputs, **config)()
    
    def _compute_signal(self):
        self._compute_relax()
        p = self._pars

        self._S = {}
        for roi in ['a', 'pv', 'l']:
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
            for i, roi in enumerate(['a', 'pv', 'l'])
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
        for i, roi in enumerate(['a', 'pv', 'l']):
            p[f"Sb_{roi}"] = signal[i][..., :n0]

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train(self._predict, time, signal, self._pars, free, **kwargs)


    # ==========================================
    # I/O and Reporting
    # ==========================================


    def _plot(
        self, time: dict, signal: dict, xlim: list, fname: str, 
        show: bool,
    ):
        self._compute_signal()
        t = self._time()
        p = self._pars

        if xlim is None: xlim = [t[0], t[-1]]
        xlim = np.array(xlim)/60
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        
        # Plot signals
        def plot_data(sig, ts, s, ax, clr):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            if s.ndim==1:
                ax.plot(ts / 60, s, marker='o', color=clr[0], label='Data', linestyle='None')
                ax.plot(t / 60, sig, linestyle='-', color=clr[1], linewidth=3.0, label='Prediction')
            else:
                for i in range(s.shape[0]):
                    ax.plot(ts / 60, s[i,:], marker='o', color=clr[0], label='Data', linestyle='None')
                    ax.plot(t / 60, sig[i,:], linestyle='-', color=clr[1], linewidth=3.0, label='Prediction')
            ax.legend()

        plot_data(self._S['a'], time[0], signal[0], ax1, ['lightcoral', 'darkred'])
        plot_data(self._S['pv'], time[1], signal[1], ax3, ['orchid', 'purple'])
        plot_data(self._S['l'], time[2], signal[2], ax5, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(t / 60, 0 * t, color='gray')
        ax2.plot(t / 60, 1000 * self._C['a'], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(ylabel='Concentration (mM)', xlim=xlim)
        ax4.plot(t / 60, 0 * t, color='gray')
        ax4.plot(t / 60, 1000 * self._C['pv'], linestyle='-', color='purple', linewidth=2.0, label='Portal vein')
        ax4.legend()

        ax6.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax6.plot(t / 60, 0 * t, color='gray')
        if self._C['l'].shape[0]==2:
            ax6.plot(t / 60, 1000 * self._C['l'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax6.plot(t / 60, 1000 * self._C['l'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax6.plot(t / 60, 1000 * self._C['l'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        # else:
        #     ax6.plot(self._t/60, 1000*self._Cl, linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax6.legend()

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
            tuple: (aorta_time, portal_time, liver_time).
        """
        t = self._time()
        return {
            'aorta': t, 
            'portal': t, 
            'liver': t,
        }

    def conc(self) -> dict:
        """Get concentrations in aorta, portal vein, and liver.

        Returns:
            tuple: (aorta_conc, portal_conc, liver_conc).
        """
        self._compute_conc()
        return {
            'aorta': self._C['a'], 
            'portal': self._C['pv'], 
            'liver': self._C['l'],
        }

    def relax(self) -> dict:
        """Get relaxation rates in aorta, portal vein, and liver.

        Returns:
            tuple: (R1_aorta, R1_portal, R1_liver).
        """
        self._compute_relax()
        return {
            'aorta': self._R['a'], 
            'portal': self._R['pv'], 
            'liver': self._R['l'],
        }
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal()
        return {
            'aorta': self._S['a'], 
            'portal': self._S['pv'], 
            'liver': self._S['l'], 
        }

    def predict(self, time: dict) -> dict:
        """Predict the signals at given time points.

        Args:
            time: Time points for (aorta, portal, liver).

        Returns:
            tuple: Predicted signals for (aorta, portal, liver).
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['portal'],
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])

        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))

        signal = self._predict(time)

        return {
            'aorta': signal[0],
            'portal': signal[1],
            'liver': signal[2],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=10, **kwargs
    ) -> tuple:
        """Train the free parameters against provided data.

        Args:
            time (tuple): (time_aorta, time_portal, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_portal, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific params.
            n0 (int, optional): Baseline points for S0 estimation. Defaults to 10.
            **kwargs: Passed to scipy.optimize.curve_fit via utils.train.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
 
        """
        if isinstance(time, dict):
            time = (
                time['aorta'],
                time['portal'],  
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['portal'], 
                signal['liver'], 
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
                time['portal'],  
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['portal'], 
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
                time['portal'], 
                time['liver'], 
            )
        elif isinstance(time, np.ndarray):
            time = tuple(3 * [time])
        if isinstance(signal, dict):
            signal = (
                signal['aorta'], 
                signal['portal'], 
                signal['liver'], 
            )
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]