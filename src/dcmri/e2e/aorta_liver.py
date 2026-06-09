"""Joint model for aorta and liver signals.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver.  

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    kinetics (str, optional): Tracer-kinetic liver model. See table 
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
    Aorta first baseline R1 (R10a): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.169 a.u.
    Liver first baseline R1 (R10l): 1.33 Hz
    Liver first signal scale factor (S0l): 150.0 a.u.
    Liver volume (vol): 1000 cm3
    Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.utils import const
from dcmri.kinetics.lib.input import ca_injection
from dcmri.kinetics.lib.aorta import flux_aorta
from dcmri.kinetics.conc import ConcLiver
from dcmri.lexicon.tools import print_params, export_params
from dcmri.lexicon.dicts import SEQUENCES
from dcmri.bloch.tissue import Signal
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.core.model import SuperModel
from dcmri.kinetics.lib.liver import dpars_liver

class AortaLiver(SuperModel):
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

    # ==========================================
    # User interface
    # ==========================================

    configs = {
        'kinetics': ['1I-EC', '1I-EC-HF', '1I-IC', '1I-IC-HF'],
        'non_stationary': [None, 'U', 'E', 'UE'],
        'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']
    }

    def __init__(
        self, 
        kinetics='1I-IC-HF', 
        non_stationary=None, 
        sequence='3D-SPGR-SS', 
        **params,
    ):
        self._version = '1.0'
        cnfg = {
            'kinetics': kinetics, 
            'non_stationary': non_stationary, 
            'sequence': sequence,
        }
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['kinetics'])
        return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    def print_params(self, *args, round_to=None, group=None, 
                     fixed_only=False, free_only=False, deriv=False):
        """Pretty print model parameters"""
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['kinetics'])
        if args != ():
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self._params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self._params('free')}
        print_params(pars, round_to=round_to, group=group)

    def time(self) -> dict:
        """Internal time array

        Returns:
            tuple: (aorta_time, liver_time)        
        """
        self._set_time()
        return {
            'aorta': self._t, 
            'liver': self._t,
        }

    def conc(self) -> dict:
        """Return concentrations in aorta and liver.

        Returns:
            tuple: (aorta_blood_conc, liver_tissue_conc)
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        return {
            'aorta': self._ca, 
            'liver': self._Cl,
        }

    def relax(self) -> dict:
        """Return relaxation rates in aorta and liver.

        Returns:
            tuple: (aorta_R1, liver_R1)
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        R1 = {
            'aorta': self._R1a, 
            'liver': self._R1l,
        }
        R2s = {
            'aorta': self._R2sa, 
            'liver': self._R2sl,
        }
        return R1, R2s
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: (time, aorta_signal, liver_signal)
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        return {
            'aorta': self._Sa, 
            'liver': self._Sl,
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
        signal = self._predict(time)
        return {
            'aorta': signal[0],
            'liver': signal[1],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=1, staged=False, **kwargs
    ) -> tuple:
        """Train the model free parameters.

        Args:
            time (tuple): (time_aorta, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points for S0 estimation.
            staged (bool, optional): If True, the training is performed in stages
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
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)

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
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
        
    def _params(self, select=None):
        if select is None:
            select = 'all'
        kin, ns, seq = self._cnfg['kinetics'], self._cnfg['non_stationary'], self._cnfg['sequence']

        aorta_kinetics = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eo', 'To_e', 'Eb']
        liver_kinetics = ConcLiver(kin, ns)._params()
        liver_kinetics = [k for k in liver_kinetics if k != 'T_a']
        liver_sequence = SEQUENCES[seq]['parameters']['prep']
        liver_sequence += SEQUENCES[seq]['parameters']['read']
        free_inflow = ['TF', 'S0_a'] if seq == '3D-SPGR-SSI' else []

        pars_list = {
            'all': aorta_kinetics + liver_kinetics + liver_sequence + [
                'dt', 'tmax', 'dose_tolerance', 'field_strength', 
                'weight', 'agent', 'dose', 'rate',
                'TS', 'H',
                'R10_a', 'R10_l',
                'R20s_a', 'R20s_l', 
                'S0_a', 'S0_l', 
                'B1corr', 'B1corr_a',   
                'vol_l', # to derive CL - not a primary parameter
            ],
            'free': aorta_kinetics + free_inflow + liver_kinetics,
            'free_liver': liver_kinetics,
            'free_aorta': aorta_kinetics,
        }
        return pars_list[select]
    
    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])

    def _compute_conc_aorta(self):
        self._set_time()
        p = self._pars
        
        conc = const.ca_conc(p['agent'])
        Ji = ca_injection(
            self._t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = flux_aorta(
            Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=['pfcomp', (p['Thl'], p['Dhl'])], 
            organs=['2cxm', ([p['To'], p['To_e']], p['Eo'])],
        )
        self._ca = Jb / p['CO']

    def _compute_relax_aorta(self):  
        self._compute_conc_aorta()
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1a = p['R10_a'] + rb * self._ca
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        self._R2sa = p['R20s_a'] + r2s * self._ca

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars
        self._Sa = Signal(self._cnfg['sequence'], **p)(
            R1=self._R1a, 
            R2s=self._R2sa,
            S0=p['S0_a'], 
            B1corr=p['B1corr_a'],
        )

    def _predict_aorta(self, time: np.ndarray):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_aorta()
        return sample(time, self._t, self._Sa, p['TS'])
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        p = self._pars
        cp = self._ca / (1 - p['H'])
        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        self._Cl = ConcLiver(kin, ns, **p)(cp, dt=p['dt'], T_a=0)
        
    def _compute_relax_liver(self):
        self._compute_conc_liver()
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        if self._Cl.shape[0] == 2:
            self._R1l = p['R10_l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
            self._R2sl = p['R20s_l'] + r2s * self._Cl.sum(axis=0) 
        # else:
        #     self._R1l = p['R10_l'] + rp * self._Cl.sum(axis=0) 
        #     self._R2sl = p['R20s_l'] + r2s * self._Cl.sum(axis=0) 

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        p = self._pars
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        self._Sl = Signal(seq, **p)(R1=self._R1l, R2s=self._R2sl, S0=p['S0_l'], B1corr=p['B1corr'])

    def _predict_liver(self, time: np.ndarray):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_liver()
        return sample(time, self._t, self._Sl, self._pars['TS'])
    
    # ==========================================
    # Forward Model: Liver and Aorta
    # ==========================================
    
    def _predict(self, time: tuple) -> tuple:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        return (
            self._predict_aorta(time[0]),
            self._predict_liver(time[1]),
        )
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: tuple, signal: tuple, n0: int):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        # 1. Estimate BAT
        t_hl, d_hl = p['Thl'], p['Dhl']
        bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        self._pars['BAT'] = max(bat, 0)
        
        # 2. Scaling Factor (S0) aorta
        seq = self._cnfg['sequence']
        s_ref = Signal(seq, **p)(R1=p['R10_a'], R2s=p['R20s_a'], S0=1, B1corr=p['B1corr_a'])
        p['S0_a'] = np.mean(signal[0][:n0]) / s_ref if s_ref > 0 else 0

        # 3. Scaling Factor (S0) liver
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        s_ref = Signal(seq, **p)(R1=p['R10_l'], R2s=p['R20s_l'])
        p['S0_l'] = np.mean(signal[1][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: tuple, signal: tuple, free: dict, 
        bounds: dict, n0: int, staged: bool, **kwargs
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")

        if staged:
            
            # Optimize Aorta parameters
            free_aorta = {k: v for k, v in free.items() if k in self._params('free_aorta')}
            train(self._predict_aorta, time[0], signal[0], self._pars, free_aorta, **kwargs)

            # Optimize Liver parameters
            free_liver = {k: v for k, v in free.items() if k in self._params('free_liver')}
            train(self._predict_liver, time[1], signal[1], self._pars, free_liver, **kwargs)

        # Joint Optimization
        return train(self._predict, time, signal, self._pars, free, **kwargs)


    def _plot(
        self, time: tuple, signal: tuple, xlim: list, fname: str, 
        show: bool
    ):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_aorta()
        self._compute_signal_liver()

        if xlim is None: xlim = [self._t[0], self._t[-1]]
        xlim = np.array(xlim)/60
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot signals
        def _plot_data(sig, t, s, ax, clr):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            ax.plot(t / 60, s, marker='o', color=clr[0], label='data', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=clr[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data(self._Sa, time[0], signal[0], ax1, ['lightcoral', 'darkred'])
        _plot_data(self._Sl, time[1], signal[1], ax3, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(self._t/60, 0*self._t, color='gray')
        ax2.plot(self._t/60, 1000*self._ca, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(self._t/60, 0*self._t, color='gray')
        if self._Cl.shape[0]==2:
            ax4.plot(self._t/60, 1000*self._Cl[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(self._t/60, 1000*self._Cl[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(self._t/60, 1000*self._Cl.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        # else:
        #     ax4.plot(self._t/60, 1000*self._Cl[0,:], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

