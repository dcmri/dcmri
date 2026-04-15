import matplotlib.pyplot as plt
import numpy as np

from dcmri import magnetization, pk, const
from dcmri.kinetics import ConcLiver
from dcmri.lexicon import SEQUENCES
from dcmri.pk import flux_aorta
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.core import SuperModel


class AortaPortalLiver(SuperModel):
    """Joint model for aorta, portal vein and liver signals.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta, portal vein and liver.  

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
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
        ...     kinetics = '2I-IC',
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

    configs = {
        'kinetics': ['2I-EC-HF', '2I-EC', '2I-IC-HF', '2I-IC', '2I-IC-U'],
        'non_stationary': [None, 'U', 'E', 'UE'],
        'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']
      }
    
    def __init__(
        self, 
        kinetics='2I-EC', 
        non_stationary=None, 
        sequence='3D-SPGR-SS', 
        **params,
    ):
        cnfg = {
            'kinetics': kinetics, 
            'non_stationary': non_stationary, 
            'sequence': sequence,
        }
        self._version = '1.0'
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

        if not kinetics.startswith('2'):
            raise ValueError("Only dual-inlet models are allowed.")

    def _params(self, select=None):
        if select is None:
            select = 'all'
        kin, ns, seq = self._cnfg['kinetics'], self._cnfg['non_stationary'], self._cnfg['sequence']

        aorta_kinetics = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eo', 'To_e', 'Eb']
        liver_kinetics = ConcLiver(kin, ns)._params()
        portal_kinetics = ['Tg', 'Dg', 'uv']
        kinetics = aorta_kinetics + liver_kinetics + portal_kinetics
        liver_sequence = SEQUENCES[seq]['parameters']['prep']
        liver_sequence += SEQUENCES[seq]['parameters']['read']
        inflow = ['TF'] if seq == '3D-SPGR-SSI' else []
        free_inflow = ['TF', 'S0_a'] if seq == '3D-SPGR-SSI' else []

        pars_list = {
            'all': kinetics + inflow + liver_sequence + [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate',
                'TS', 'H', 
                'R10_a', 'R10_v', 'R10_l', 'S0_a', 'S0_v', 'S0_l', 
                'B1corr', 'B1corr_a', 'B1corr_v', 
            ],
            'free': kinetics + free_inflow,
            'free_liver': liver_kinetics,
            'free_aorta': aorta_kinetics + free_inflow,
            'free_portal': portal_kinetics,
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
        Ji = pk.ca_injection(
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

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars
        self._Sa = magnetization.Signal(self._cnfg['sequence'], **p)(
            R1=self._R1a, 
            S0=p['S0_a'], 
            B1corr=p['B1corr_a'],
            TE=0, PA=0,
        )

    def _predict_aorta(self, time: np.ndarray):
        self._set_time()
        self._compute_signal_aorta()
        p = self._pars
        return sample(time, self._t, self._Sa, p['TS'])

    # ==========================================
    # Forward Model: Portal
    # ==========================================
    
    def _compute_conc_portal(self):
        p = self._pars
        self._cv = pk.flux_chain(self._ca, p['Tg'], p['Dg'], dt=p['dt'])
    
    def _compute_relax_portal(self):
        self._compute_conc_portal()
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1v = p['R10_v'] + rb * p['uv'] * self._cv
    
    def _compute_signal_portal(self):
        self._compute_relax_portal()
        p = self._pars
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        self._Sv = magnetization.Signal(seq, **p)(
            R1=self._R1v, 
            S0=p['S0_v'], 
            B1corr=p['B1corr_v'],
            TE=0,
        )

    def _predict_portal(self, time: np.ndarray) -> np.ndarray:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_portal()
        return sample(time, self._t, self._Sv, p['TS'])

    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        p = self._pars
  
        ca_plasma = self._ca / (1 - p['H'])
        cv_plasma = self._cv / (1 - p['H'])
        cp = (ca_plasma, cv_plasma)
        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        self._Cl = ConcLiver(kin, ns, **p)(cp, dt=p['dt'])
        
    def _compute_relax_liver(self):
        self._compute_conc_liver()
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])

        if self._Cl.shape[0] == 2:
            self._R1l = p['R10_l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = p['R10_l'] + rp * self._Cl[0,:]

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        p = self._pars

        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        self._Sl = magnetization.Signal(seq, **p)(R1=self._R1l, S0=p['S0_l'], TE=0)

    def _predict_liver(self, time: np.ndarray) -> np.ndarray:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_liver()
        return sample(time, self._t, self._Sl, p['TS'])
    
    # ===========================================
    # Forward Model: Liver, Portal Vein and Aorta
    # ===========================================
    
    def _predict(self, time: dict) -> dict:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        return (
            self._predict_aorta(time[0]),
            self._predict_portal(time[1]),
            self._predict_liver(time[2]),
        )

    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: dict, signal: dict, n0: int):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        # Estimate BAT based on peak signal
        t_hl, d_hl = p['Thl'], p['Dhl']
        bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        p['BAT'] = max(bat, 0)

        # 2. Scaling Factor (S0) aorta
        seq = self._cnfg['sequence']
        s_ref = magnetization.Signal(seq, **p)(R1=p['R10_a'], S0=1, B1corr=p['B1corr_a'], TE=0, PA=0)
        p['S0_a'] = np.mean(signal[0][:n0]) / s_ref if s_ref > 0 else 0

        # 3. Scaling Factor (S0) portal vein
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        s_ref = magnetization.Signal(seq, **p)(R1=p['R10_v'], S0=1, B1corr=p['B1corr_v'], TE=0)
        p['S0_v'] = np.mean(signal[1][:n0]) / s_ref if s_ref > 0 else 0

        # 4. Scaling Factor (S0) liver
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']
        s_ref = magnetization.Signal(seq, **p)(R1=p['R10_l'], S0=1, B1corr=p['B1corr'], TE=0)
        p['S0_l'] = np.mean(signal[2][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: dict, signal: dict, free: dict, 
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

            # Optimize Portal parameters
            free_portal = {k: v for k, v in free.items() if k in self._params('free_portal')}
            train(self._predict_portal, time[1], signal[1], self._pars, free_portal, **kwargs)

            # Optimize Liver parameters
            free_liver = {k: v for k, v in free.items() if k in self._params('free_liver')}
            train(self._predict_liver, time[2], signal[2], self._pars, free_liver, **kwargs)

        # Joint Optimization
        return train(self._predict, time, signal, self._pars, free, **kwargs)


    # ==========================================
    # I/O and Reporting
    # ==========================================


    def _plot(
        self, time: dict, signal: dict, xlim: list, fname: str, 
        show: bool,
    ):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_aorta()
        self._compute_signal_portal()
        self._compute_signal_liver()

        if xlim is None: xlim = [self._t[0], self._t[-1]]
        xlim = np.array(xlim)/60
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        
        # Plot signals
        def plot_data(sig, t, s, ax, clr):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            ax.plot(t / 60, s, marker='o', color=clr[0], label='data', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=clr[1], linewidth=3.0, label='fit')
            ax.legend()

        plot_data(self._Sa, time[0], signal[0], ax1, ['lightcoral', 'darkred'])
        plot_data(self._Sv, time[1], signal[1], ax3, ['orchid', 'purple'])
        plot_data(self._Sl, time[2], signal[2], ax5, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(self._t/60, 0*self._t, color='gray')
        ax2.plot(self._t/60, 1000*self._ca, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(ylabel='Concentration (mM)', xlim=xlim)
        ax4.plot(self._t/60, 0*self._t, color='gray')
        ax4.plot(self._t/60, 1000*self._cv, linestyle='-', color='purple', linewidth=2.0, label='Portal vein')
        ax4.legend()

        ax6.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax6.plot(self._t/60, 0*self._t, color='gray')
        if self._Cl.ndim==2:
            ax6.plot(self._t/60, 1000*self._Cl[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax6.plot(self._t/60, 1000*self._Cl[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax6.plot(self._t/60, 1000*self._Cl.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        else:
            ax6.plot(self._t/60, 1000*self._Cl, linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax6.legend()

        if fname: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()


    # ==========================================
    # Public API: dict Extraction
    # ==========================================

    

    def time(self) -> dict:
        """Internal time array

        Returns:
            tuple: (aorta_time, portal_time, liver_time).
        """
        self._set_time()
        return {
            'aorta': self._t, 
            'portal': self._t, 
            'liver': self._t,
        }

    def conc(self) -> dict:
        """Get concentrations in aorta, portal vein, and liver.

        Returns:
            tuple: (aorta_conc, portal_conc, liver_conc).
        """
        self._compute_conc_aorta()
        self._compute_conc_portal()
        self._compute_conc_liver()
        return {
            'aorta': self._ca, 
            'portal': self._cv, 
            'liver': self._Cl,
        }

    def relax(self) -> dict:
        """Get relaxation rates in aorta, portal vein, and liver.

        Returns:
            tuple: (R1_aorta, R1_portal, R1_liver).
        """
        self._compute_relax_aorta()
        self._compute_relax_portal()
        self._compute_relax_liver()
        return {
            'aorta': self._R1a, 
            'portal': self._R1v, 
            'liver': self._R1l,
        }
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal_aorta()
        self._compute_signal_portal()
        self._compute_signal_liver()
        return {
            'aorta': self._Sa, 
            'portal': self._Sv, 
            'liver': self._Sl,
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
        else:
            time = tuple(3 * [time])

        signal = self._predict(time)

        return {
            'aorta': signal[0],
            'portal': signal[1],
            'liver': signal[2],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=1, staged=False, **kwargs
    ) -> tuple:
        """Train the free parameters against provided data.

        Args:
            time (tuple): (time_aorta, time_portal, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_portal, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific params.
            n0 (int, optional): Baseline points for S0 estimation. Defaults to 1.
            staged (bool, optional): If True, the training is performed in stages
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
        else:
            time = tuple(3 * [time])
        signal = (
            signal['aorta'], 
            signal['portal'], 
            signal['liver'], 
        )
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)

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
        else:
            time = tuple(3 * [time])
        signal = (
            signal['aorta'], 
            signal['portal'], 
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
                time['portal'], 
                time['liver'], 
            )
        else:
            time = tuple(3 * [time])
        signal = np.concatenate((
            signal['aorta'], 
            signal['portal'], 
            signal['liver'], 
        ))
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]