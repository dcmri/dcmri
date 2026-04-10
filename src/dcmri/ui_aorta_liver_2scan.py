import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, sig, utils, ui, liver, pk_lib
from dcmri.lexicon import SEQUENCES
from dcmri.lexicon import LEXICON

LEXICON = LEXICON | {
    'S02_a': {'init': 1, 'bounds': [0, 2], 'name': 'Aorta second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},
    'S02_l': {'init': 1, 'bounds': [0, 2], 'name': 'Liver second signal scale factor', 'unit': 'a.u.', 'bounds_type': 'mult'},

}

class AortaLiver2scan(ui.SuperModel):
    """Joint model for aorta and liver signals measured over two scans.

    This model uses a whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two separate scans.

    For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
    For more detail on the liver model, see :ref:`liver-tissues`. 

    Args:
        kinetics (str, optional): Tracer-kinetic liver model. See table 
          :ref:`table-liver-models` for options - only single-inlet models 
          are allowed. Defaults to '1I-IC-HFD'.
        stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to 'UE'.
        stationary (str, optional): Stationarity regime of the hepatocytes. 
          The options are 'UE', 'E', 'U' or None. For more detail 
          see :ref:`liver-tissues`. Defaults to 'UE'.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`AortaLiver2scan-parameters` and
          :ref:`AortaLiver2scan-defaults` for a list of parameters and their
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

        Use `fake_tissue` to generate synthetic test data from 
        experimentally-derived concentrations:

        >>> time, aif, roi, gt = dc.fake_tissue2scan(R10=1/dc.T1(3.0,'liver'))

        Since this model generates four time curves, the x- and y-data are 
        tuples:

        >>> time = (time[0], time[1], time[0], time[1])
        >>> signal = (aif[0], aif[1], roi[0], roi[1])

        Build an aorta-liver model and parameters to match the conditions of 
        the fake tissue data:

        >>> model = dc.AortaLiver2scan(
        ...     dt = 0.5,
        ...     tmax = 420,
        ...     weight = 70,
        ...     agent = 'gadodiamide',
        ...     dose = 0.2,
        ...     dose2 = 0.2,
        ...     rate = 3,
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     FA2 = 15,
        ...     TS = 0.5,
        ...     Th_i = 120,
        ...     Th_f = 120,
        ... )

        In this case we have defined different initial values for Th as 
        the defaults are optimized for the slow passage through hepatocytes. 
        We also need to reset the parameter bounds:

        >>> model.free['Th_i'] = [0, np.inf]
        >>> model.free['Th_f'] = [0, np.inf]

        Train the model on the data:

        >>> model.train(time, signal, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare against 
        the experimentally derived data:

        >>> model.plot(time, signal)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Aorta second signal scale factor (S02a): 195.824 (2.025) a.u.
        Liver second signal scale factor (S02l): 297.854 (4.9) a.u.
        Second bolus arrival time (BAT2): 254.512 (0.137) sec
        First bolus arrival time (BAT): 14.288 (0.132) sec
        Cardiac output (CO): 203.199 (5.406) mL/sec
        Heart-lung mean transit time (Thl): 15.236 (0.263) sec
        Heart-lung dispersion (Dhl): 0.381 (0.009)
        Organs blood mean transit time (To): 23.761 (3.052) sec
        Organs extraction fraction (Eo): 0.287 (0.053)
        Organs extravascular mean transit time (Toe): 50.274 (17.44) sec
        Body extraction fraction (Eb): 0.078 (0.015)
        Apparent liver extracellular volume fraction (ve_app): 0.053 (0.008) mL/cm3
        Extracellular mean transit time (Te): 1.298 (0.552) sec
        Extracellular dispersion (De): 1.0 (0.7)
        Initial hepatic plasma clearance (Ktrans_i): 0.005 (0.001) mL/sec/cm3
        Final hepatic plasma clearance (Ktrans_f): 0.005 (0.001) mL/sec/cm3
        Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
        Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Aorta first baseline R1 (R10a): 0.614 Hz
        Aorta first signal scale factor (S0a): 100.117 a.u.
        Liver first baseline R1 (R10l): 1.33 Hz
        Liver first signal scale factor (S0(l)): 150.003 a.u.
        Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
        Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec


    Notes:

        Table :ref:`AortaLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaLiver-defaults` list all 
        possible parameters and their default settings. 

        .. _AortaLiver2scan-parameters:
        .. list-table:: **Aorta-Liver 2-scan parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - dt, tmax
              - Always
              - Time axis for forward model
            * - dose_tolerance
              - Always
              - Stopping criterion for whole-body model
            * - field_strength, weight, agent, dose, dose2, rate
              - Always
              - Injection protocol
            * - R10a, R102a, R10l, R102l, S0a, S02a, S0(l), S02l
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`) for aorta and liver 
            * - FA, TR, TS, FA2
              - Always
              - :ref:`params-per-sequence`
            * - TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - BAT, BAT2, CO, Thl, Dhl, To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - H, ve, De
              - Always
              - :ref:`table-liver-models`
            * - khe, khe_i, kh_f, Th, Th_i, Th_f
              - Depends on **stationary**
              - :ref:`table-liver-models`

        .. _AortaLiver2scan-defaults:
        .. list-table:: **Aorta-Liver 2-scan parameter defaults**
            :widths: 5 10 10 10 10
            :header-rows: 1

            * - Parameter
              - Type
              - Value
              - Bounds
              - Free/Fixed
            * - 
              - **Simulation**
              -
              - 
              - 
            * - dt
              - Simulation
              - 0.5
              - [0, inf]
              - Fixed
            * - tmax
              - Simulation
              - 120
              - [0, inf]
              - Fixed
            * - dose_tolerance
              - Simulation
              - 0.1
              - [0, 1]
              - Fixed
            * - 
              - **Injection**
              -
              - 
              - 
            * - field_strength
              - Injection
              - 3
              - [0, inf]
              - Fixed
            * - weight
              - Injection
              - 70
              - [0, inf]
              - Fixed
            * - agent
              - Injection
              - 'gadoxetate'
              - None
              - Fixed
            * - dose
              - Injection
              - 0.0125
              - [0, inf]
              - Fixed
            * - rate
              - Injection
              - 1
              - [0, inf]
              - Fixed
            * - 
              - **Signal**
              -
              - 
              - 
            * - R10a
              - Signal
              - 0.7
              - [0, inf]
              - Fixed
            * - R10l
              - Signal
              - 0.7
              - [0, inf]
              - Fixed
            * - S0a
              - Signal
              - 1
              - [0, inf]
              - Free
            * - S0(l)
              - Signal
              - 1
              - [0, inf]
              - Free
            * - 
              - **Sequence**
              -
              - 
              - 
            * - FA
              - Sequence
              - 15
              - [0, inf]
              - Fixed
            * - FA2
              - Sequence
              - 15
              - [0, inf]
              - Fixed
            * - S0
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - TC
              - Sequence
              - 0.1
              - [0, inf]
              - Fixed
            * - TR
              - Sequence
              - 0.005
              - [0, inf]
              - Fixed
            * - TS
              - Sequence
              - 0
              - [0, inf]
              - Fixed
            * - 
              - **Whole body**
              -
              - 
              - 
            * - BAT
              - Whole body
              - 1200
              - [0, inf]
              - Free
            * - CO
              - Whole body
              - 100
              - [0, inf]
              - Free
            * - Thl
              - Whole body
              - 10
              - [0, 30]
              - Free
            * - Dhl
              - Whole body
              - 0.2
              - [0.05, 0.95]
              - Free
            * - To
              - Whole body
              - 20
              - [0, 60]
              - Free
            * - Eo
              - Whole body
              - 0.15
              - [0, 0.5]
              - Free
            * - Toe
              - Whole body
              - 120
              - [0, 800]
              - Free
            * - Eb
              - Whole body
              - 0.05
              - [0.01, 0.15]
              - Free
            * - 
              - **Liver**
              -
              - 
              - 
            * - H
              - Kinetic
              - 0.45
              - [0, 1]
              - Fixed
            * - Te
              - Kinetic
              - 30
              - [0.1, 60]
              - Free
            * - De
              - Kinetic
              - 0.85
              - [0, 1]
              - Free
            * - ve
              - Kinetic
              - 0.3
              - [0.01, 0.6]
              - Free
            * - khe
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - khe_i
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - khe_f
              - Kinetic
              - 0.003
              - [0, 0.1]
              - Free
            * - Th
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - Th_i
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - Th_f
              - Kinetic
              - 1800
              - [600, 36000]
              - Free
            * - vol
              - Kinetic
              - 1000
              - [0, 10000]
              - Free
    """

    configs = {
        'kinetics': ['1I-EC-D', '1I-EC', '1I-IC', '1I-IC-HF', '1I-IC-HFD', '1I-IC-HFDU'],
        'non_stationary': [None, 'U', 'E', 'UE'],
        'sequence': ['3D-SPGR-SS', '3D-SPGR-SSI']
    }

    def __init__(
        self, 
        kinetics = '1I-IC-HFD', 
        non_stationary=None, 
        sequence='3D-SPGR-SS', 
        **params,
      ):
        self._version = '1.0'

        cnfg = {'kinetics': kinetics, 'non_stationary': non_stationary, 'sequence': sequence}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(LEXICON, **params)

        if not kinetics.startswith('1'):
            raise ValueError('Only single-inlet models are allowed.')
                    
    def _params(self, select=None):
        if select is None:
            select = 'all'
        kin, ns, seq = self._cnfg['kinetics'], self._cnfg['non_stationary'], self._cnfg['sequence']

        aorta_kinetics = ['BAT', 'BAT2', 'CO', 'Thl', 'Dhl', 'To', 'Eo', 'To_e', 'Eb']
        liver_kinetics = liver.Conc(kin, ns)._params()
        kinetics = aorta_kinetics + liver_kinetics
        liver_sequence = SEQUENCES[seq]['parameters']['prep']
        liver_sequence += SEQUENCES[seq]['parameters']['read']
        if 'FA' in liver_sequence:
            liver_sequence += ['FA2']
    
        inflow = ['TF'] if seq == '3D-SPGR-SSI' else []
        free_inflow = ['TF', 'S0_a'] if seq == '3D-SPGR-SSI' else []

        pars_list = {
            'all': kinetics + inflow + liver_sequence + [
                'dt', 'tmax', 't_scan2', 'dose_tolerance', 'field_strength', 
                'agent', 'weight', 'dose', 'dose2', 'rate', 
                'TS', 'H', 
                'R10_a', 'R10_l', 'S0_a', 'S0_l', 'S02_a', 'S02_l',
                'B1corr', 'B1corr_a', 'B1corr_2', 'B1corr_2_a',
            ],
            'all free': kinetics + free_inflow + ['S02_a', 'S02_l'],
            'free_aorta': aorta_kinetics + free_inflow + ['S02_a'],
            'free_liver': liver_kinetics + ['S02_l'],
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

        conc = lib.ca_conc(p['agent'])
        J1 = lib.ca_injection(
            self._t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        J2 = lib.ca_injection(
            self._t, p['weight'], conc, p['dose2'], p['rate'], p['BAT2']
        )
        Jb = pk_lib.aorta_flux(
            J1 + J2, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung = ['pfcomp', (p['Thl'], p['Dhl'])],
            organs = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]
        )
        self._ca = Jb / p['CO']

    def _compute_relax_aorta(self):
        self._compute_conc_aorta()
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1a = p['R10_a'] + rb * self._ca

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars

        self._Sa = np.zeros_like(self._t)
        seq = self._cnfg['sequence']

        # First scan signal
        t = self._t < p['t_scan2']
        self._Sa[t] = sig.Signal(seq, **p)(R1=self._R1a[t], S0=p['S0_a'], B1corr=p['B1corr_a'], TE=0, PA=0)

        # Second scan signal
        t = self._t >= p['t_scan2']
        self._Sa[t] = sig.Signal(seq, **p)(R1=self._R1a[t], S0=p['S02_a'], B1corr=p['B1corr_2_a'], FA=p['FA2'], TE=0, PA=0)

    def _predict_aorta(self, time: tuple):
        self._compute_signal_aorta()
        p = self._pars
        return (
            utils.sample(time[0], self._t, self._Sa, p['TS']),
            utils.sample(time[1], self._t, self._Sa, p['TS']),
        )
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        p = self._pars
        cp = self._ca / (1 - p['H'])
        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        self._Cl = liver.Conc(kin, ns, **p)(cp, dt=p['dt'])

    def _compute_relax_liver(self):
        self._compute_conc_liver()
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'plasma', p['agent'])
        rh = lib.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])

        if self._Cl.ndim==2:
            self._R1l = p['R10_l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = p['R10_l'] + rp * self._Cl

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        p = self._pars

        self._Sl = np.zeros_like(self._t)
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']

        # First scan signal
        t = self._t < p['t_scan2']
        self._Sl[t] = sig.Signal(seq, **p)(R1=self._R1l[t], S0=p['S0_l'], TE=0)
        
        # Second scan signal
        t = self._t >= p['t_scan2']
        self._Sl[t] = sig.Signal(seq, **p)(R1=self._R1l[t], S0=p['S02_l'], B1corr=p['B1corr_2'], FA=p['FA2'], TE=0)

    def _predict_liver(self, time: tuple):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_liver()
        return (
            utils.sample(time[0], self._t, self._Sl, p['TS']),
            utils.sample(time[1], self._t, self._Sl, p['TS']),
        )
    
    # ===========================================
    # Forward Model: Liver and Aorta
    # ===========================================
    
    def _predict(self, time: dict) -> dict:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        s_a = self._predict_aorta(time[:2])
        s_l = self._predict_liver(time[2:])
        return s_a + s_l

    # ==========================================
    # Inverse Model: Training
    # ==========================================   

    def _estimate_parameters(
        self, time: dict, signal: dict, n0: int, R102a: float, 
        R102l: float
    ):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        seq_aorta = self._cnfg['sequence']
        seq_liver = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']

        # Estimate BAT and BAT2 and ajust their bounds
        t_hl, d_hl = p['Thl'], p['Dhl']
        bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        bat2 = time[1][np.argmax(signal[1])] - (1 - d_hl) * t_hl
        p['BAT'] = max(bat, 0)
        p['BAT2'] = max(bat2, 0)

        # Scaling Factor (S0) aorta
        s_ref = sig.Signal(seq_aorta, **p)(R1=p['R10_a'], S0=1, B1corr=p['B1corr_a'], TE=0, PA=0)
        p['S0_a'] = np.mean(signal[0][:n0]) / s_ref if s_ref > 0 else 0

        # Scaling Factor (S0) liver
        s_ref = sig.Signal(seq_liver, **p)(R1=p['R10_l'], S0=1, TE=0)
        p['S0_l'] = np.mean(signal[2][:n0]) / s_ref if s_ref > 0 else 0

        # Second Scaling Factor (S02) aorta
        if R102a is None:
            p['S02_a'] = p['S0_a']
        else:
            s_ref = sig.Signal(seq_aorta, **p)(R1=R102a, S0=1, B1corr=p['B1corr_2_a'], FA=p['FA2'], TE=0, PA=0)
            p['S02_a'] = np.mean(signal[1][:n0]) / s_ref if s_ref > 0 else 0

        # Second Scaling Factor (S02) liver
        if R102l is None:
            p['S02_l'] = p['S0_l']
        else:
            s_ref = sig.Signal(seq_liver, **p)(R1=R102l, S0=1, B1corr=p['B1corr_2'], FA=p['FA2'], TE=0, PA=0)
            p['S02_l'] = np.mean(signal[3][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: dict, signal: dict, free: dict, 
        bounds: dict, n0: int, R102a: float, R102l: float, 
        staged: bool, **kwargs,
    ):
        self._estimate_parameters(time, signal, n0, R102a, R102l)
        free = self._set_free_pars(free, bounds, lexicon=LEXICON)
    
        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")

        if staged:
            # Train free aorta parameters on aorta data
            free_aorta = {k: v for k, v in free.items() if k in self._params('free_aorta')}
            utils.train(self._predict_aorta, time[:2], signal[:2], self._pars, free_aorta, **kwargs)

            # Train free liver parameters on liver data
            free_liver = {k: v for k, v in free.items() if k in self._params('free_liver')}
            utils.train(self._predict_liver, time[2:], signal[2:], self._pars, free_liver, **kwargs)

        # Joint Optimization
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)
    

    # ==========================================
    # I/O and Reporting
    # ==========================================


    def _plot(
        self, time: dict, signal: dict, xlim=None, fname=None, 
        show=True
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
        def _plot_data2scan(sig, t, s, ax, color):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            ax.plot(np.concatenate(t)/60, np.concatenate(s), marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan(self._Sa, time[:2], signal[:2], ax1, ['lightcoral', 'darkred'])
        _plot_data2scan(self._Sl, time[2:], signal[2:], ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(self._t / 60, 0 * self._t, color='gray')
        ax2.plot(self._t / 60, 1000 * self._ca, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(self._t / 60, 0 * self._t, color='gray')
        if self._Cl.ndim==2:
            ax4.plot(self._t / 60, 1000 * self._Cl[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(self._t / 60, 1000 * self._Cl[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(self._t / 60, 1000 * self._Cl.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        else:
            ax4.plot(self._t / 60, 1000 * self._Cl, linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')        
        ax4.legend()

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()


    # ==========================================
    # Public API: dict Extraction
    # ==========================================

    def time(self) -> dict:
        """Internal time array
        
        Returns:
            tuple: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.      
        """
        self._set_time()
        p = self._pars
        t, t2 = self._t, p['t_scan2']
        tacq1, tacq2 = t[t < t2], t[t >= t2]
        return {
            ('aorta', 1): tacq1, 
            ('aorta', 2): tacq2, 
            ('liver', 1): tacq1, 
            ('liver', 2): tacq2, 
        }
    
    def conc(self) -> dict:
        """Concentrations in aorta and liver.

        Returns:
            tuple: aorta conc scan 1, aorta conc scan 2, 
              liver conc scan 1, liver conc scan 2.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        t, t2 = self._t, self._pars['t_scan2']
        return {
            ('aorta', 1): self._ca[t < t2], 
            ('aorta', 2): self._ca[t >= t2], 
            ('liver', 1): self._Cl[:, t < t2], 
            ('liver', 2): self._Cl[:, t >= t2], 
        }
    
    def relax(self) -> dict:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: aorta R1 scan 1, aorta R1 scan 2, 
              liver R1 scan 1, liver R1 scan 2.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        t, t2 = self._t, self._pars['t_scan2']
        return {
            ('aorta', 1): self._R1a[t < t2], 
            ('aorta', 2): self._R1a[t >= t2], 
            ('liver', 1): self._R1l[t < t2], 
            ('liver', 2): self._R1l[t >= t2], 
        }
    
    def signal(self) -> dict:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta signal scan 1, aorta signal scan 2, 
              liver signal scan 1, liver signal scan 2.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        t, t2 = self._t, self._pars['t_scan2']
        return {
            ('aorta', 1): self._Sa[t < t2], 
            ('aorta', 2): self._Sa[t >= t2], 
            ('liver', 1): self._Sl[t < t2], 
            ('liver', 2): self._Sl[t >= t2], 
        }
    
    def predict(self, time: dict) -> dict:
        """Predict the data at given time points

        Args:
            time: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.

        Returns:
            tuple: aorta data scan 1, aorta data scan 2, 
              liver data scan 1, liver data scan 2.
        """
        if isinstance(time, dict):
            time = (
                time[('aorta', 1)], 
                time[('aorta', 2)], 
                time[('liver', 1)], 
                time[('liver', 2)], 
            )
        else:
            time = tuple(4 * [time])

        signal = self._predict(time)

        return {
            ('aorta', 1): signal[0],
            ('aorta', 2): signal[1],
            ('liver', 1): signal[2],
            ('liver', 2): signal[3],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=1, R102a: float = None, 
        R102l: float = None, staged=False, **kwargs,
    ) -> tuple:
        """Train the free parameters

        Args:
            time (tuple): (time_1_aorta, time_2_aorta, time_1_liver, time_2_liver)
            signal (tuple): (signal_1_aorta, signal_2_aorta, signal_1_liver, signal_2_liver).
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R102a (float, optional): R1 value in arterial blood before the second injection. 
            R102l (float, optional): R1 value in liver before the second injection. 
            staged (bool, optional): If True, the training is performed in stages
            kwargs: any other keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """

        if isinstance(time, dict):
            time = (
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        else:
            time = tuple(4 * [time])

        signal = (
            signal['aorta', 1], 
            signal['aorta', 2], 
            signal['liver', 1], 
            signal['liver', 2], 
        )

        return self._train(time, signal, free, bounds, n0, R102a, R102l, staged, **kwargs)

    def plot(
        self, time: dict, signal: dict, xlim: list = None, 
        fname: str = None, show=True
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """

        if isinstance(time, dict):
            time = (
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        else:
            time = tuple(4 * [time])

        signal = (
            signal['aorta', 1], 
            signal['aorta', 2], 
            signal['liver', 1], 
            signal['liver', 2], 
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
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        else:
            time = tuple(4 * [time])

        signal = np.concatenate((
            signal['aorta', 1], 
            signal['aorta', 2], 
            signal['liver', 1], 
            signal['liver', 2], 
        ))
        signal_pred = np.concatenate(self._predict(time))
        cost = utils.loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
