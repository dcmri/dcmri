from copy import deepcopy
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, sig, utils, pk, pk_aorta, ui, liver
from dcmri.lexicon import LEXICON
import dcmri.lexicon_utils as lexicon

# Shorthand notation for data type hint
Data = Tuple[np.ndarray, np.ndarray, np.ndarray]


class AortaPortalLiver(ui.SuperModel):
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

        Use `fake_tissue` to generate synthetic test data from 
        experimentally-derived concentrations:

        Use `fake_liver` to generate synthetic test data:

        >>> time, aif, vif, roi, _ = dc.fake_liver(sequence='SSI')

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

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        First bolus arrival time (BAT): 14.616 (1.1) sec
        Cardiac output (CO): 100.09 (2.736) mL/sec
        Heart-lung mean transit time (Thl): 14.402 (1.375) sec
        Heart-lung dispersion (Dhl): 0.391 (0.013) 
        Organs blood mean transit time (To): 27.811 (5.291) sec
        Organs extraction fraction (Eo): 0.29 (0.105)
        Organs extravascular mean transit time (Toe): 70.621 (102.614) sec
        Body extraction fraction (Eb): 0.013 (0.23)
        Aorta inflow time (TF): 0.409 (0.014) sec
        Liver extracellular volume fraction (ve): 0.479 (0.112) mL/cm3
        Liver plasma flow (Fp): 0.018 (0.001) mL/sec/cm3
        Arterial flow fraction (fa): 0.087 (0.074)
        Arterial transit time (Ta): 2.398 (1.356) sec
        Hepatocellular uptake rate (khe): 0.006 (0.003) mL/sec/cm3
        Hepatocellular mean transit time (Th): 683.604 (2554.75) sec
        Gut mean transit time (Tg): 10.782 (0.614) sec
        Gut dispersion (Dg): 0.893 (0.07)
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Hematocrit (H): 0.45
        Arterial venous blood flow (Fa): 0.002 mL/sec/cm3
        Portal venous blood flow (Fv): 0.016 mL/sec/cm3
        Extracellular mean transit time (Te): 27.216 sec
        Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
        Hepatocellular tissue uptake rate (Khe): 0.012 mL/sec/cm3
        Biliary excretion rate (kbh): 0.001 mL/sec/cm3

    Notes:

        Table :ref:`AortaPortalLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaPortalLiver-defaults` list 
        all possible parameters and their default settings. 

        .. _AortaPortalLiver-parameters:
        .. list-table:: **Aorta-Liver parameters**
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
              - For estimating baseline signal
            * - field_strength, weight, agent, dose, rate
              - Always
              - Injection protocol
            * - R10a, R10l, S0a, S0v, S0l 
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`)for aorta and liver 
            * - FA, TR, TS
              - Always
              - :ref:`params-per-sequence`
            * - TF
              - If **sequence** is 'SSI'
              - To model aorta inflow effects
            * - BAT, CO, Thl, Dhl, To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - Tg , Dg
              - Always
              - Gut dispersion
            * - H, ve, Fp, fa, Ta, Tg, khe, khe_i, kh_f, Th, Th_i, Th_f.
              - Depends on **kinetics** and **stationary**
              - :ref:`table-liver-models`

        .. _AortaPortalLiver-defaults:
        .. list-table:: **Aorta-Liver parameter defaults**
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
            * - S0v
              - Signal
              - 1
              - [0, inf]
              - Free
            * - S0l
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
            * - S0
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - TF
              - Sequence
              - 0.5
              - [0, 10]
              - Free
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
              - **Portal vein**
              -
              - 
              - 
            * - Tg
              - Kinetic
              - 15
              - [0.1, 60]
              - Free
            * - Dg
              - Kinetic
              - 0.85
              - [0, 1]
              - Free
            * - uv
              - Kinetic
              - 1
              - [0, 1]
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
            * - ve
              - Kinetic
              - 0.3
              - [0.01, 0.6]
              - Free
            * - Fp
              - Kinetic
              - 0.01
              - [0, 0.1]
              - Free
            * - fa
              - Kinetic
              - 0.2
              - [0, 0.1]
              - Free
            * - Ta
              - Kinetic
              - 0.5
              - [0, 3]
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
    def __init__(
        self, 
        kinetics='2I-EC', 
        non_stationary=None, 
        sequence='SS', 
        **params,
    ):
        
        # --- 1. Configuration Validation ---
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics: {e}") from e
        if sequence not in ['SR', 'SS', 'SSI', 'lin']:
            raise ValueError(f"Sequence '{sequence}' is not available.")
        if not kinetics.startswith('2'):
            raise ValueError("Only dual-inlet models are allowed.")

        # --- 2. State Initialization ---
        self._version = '1.0'
        self._cnfg = {'kinetics': kinetics, 'non_stationary': non_stationary, 'sequence': sequence}
        self._pars = lexicon.init(self._pars_list(), LEXICON)

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    def _pars_list(self, select='all'):
        kin, ns, seq = self._cnfg['kinetics'], self._cnfg['non_stationary'], self._cnfg['sequence']

        aorta_kinetics = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eo', 'To_e', 'Eb']
        liver_kinetics = list(liver.params_liver(kin, ns).keys())
        portal_kinetics = ['Tg', 'Dg', 'uv']
        kinetics = aorta_kinetics + liver_kinetics + portal_kinetics
        liver_sequence = {
            'SR': ['FA', 'TR', 'TC', 'TP'],
            'SS': ['FA', 'TR'], 
            'lin': [],
            'SSI': ['FA', 'TR'],
        }[seq]
        inflow = ['TF'] if seq=='SSI' else []

        pars_list = {
            'all': kinetics + inflow + liver_sequence + [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate',
                'TS', 'H', 
                'R10_a', 'R10_v', 'R10_l', 'S0_a', 'S0_v', 'S0_l', 
                'B1corr', 'B1corr_a', 'B1corr_v', 
            ],
            'free': kinetics + inflow + ['S0_a', 'S0_l', 'S0_v'],
            'sequence': liver_sequence + inflow, 
            'liver_sequence': liver_sequence,
            'liver_kinetics': liver_kinetics,
            'liver_fit': liver_kinetics,
            'aorta_fit': aorta_kinetics + inflow,
            'portal_fit': portal_kinetics,
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
        Ji = lib.ca_injection(
            self._t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=['pfcomp', (p['Thl'], p['Dhl'])], 
            organs=['2cxm', ([p['To'], p['To_e']], p['Eo'])],
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
        pars = {k: p[k] for k in self._pars_list(select='sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr_a']
        self._Sa = sig.signal(self._cnfg['sequence'], self._R1a, p['S0_a'], **pars)

    def _predict_aorta(self, time: np.ndarray):
        self._set_time()
        self._compute_signal_aorta()
        p = self._pars
        return utils.sample(time, self._t, self._Sa, p['TS'])

    # ==========================================
    # Forward Model: Portal
    # ==========================================
    
    def _compute_conc_portal(self):
        p = self._pars
        self._cv = pk.flux_chain(self._ca, p['Tg'], p['Dg'], dt=p['dt'])
    
    def _compute_relax_portal(self):
        self._compute_conc_portal()
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1v = p['R10_v'] + rb * p['uv'] * self._cv
    
    def _compute_signal_portal(self):
        self._compute_relax_portal()
        p = self._pars
        pars = {k: p[k] for k in self._pars_list(select='liver_sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr_v']
        seq = 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence']
        self._Sv = sig.signal(seq, self._R1v, p['S0_v'], **pars)

    def _predict_portal(self, time: np.ndarray) -> np.ndarray:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_portal()
        return utils.sample(time, self._t, self._Sv, p['TS'])

    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        p = self._pars
        pars = {k: p[k] for k in self._pars_list(select='liver_kinetics')}

        ca_plasma = self._ca / (1 - p['H'])
        cv_plasma = self._cv / (1 - p['H'])
        cp = (ca_plasma, cv_plasma)
        self._Cl = liver.conc_liver(
            cp, dt=p['dt'], sum=False, kinetics=self._cnfg['kinetics'], 
            non_stationary=self._cnfg['non_stationary'], **pars
        )
        
    def _compute_relax_liver(self):
        self._compute_conc_liver()
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'plasma', p['agent'])
        rh = lib.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])

        if self._Cl.ndim == 2:
            self._R1l = p['R10_l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = p['R10_l'] + rp * self._Cl

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        p = self._pars

        pars = {k: p[k] for k in self._pars_list(select='liver_sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr']
        seq = 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence']
        self._Sl = sig.signal(seq, self._R1l, p['S0_l'], **pars)

    def _predict_liver(self, time: np.ndarray) -> np.ndarray:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        self._compute_signal_liver()
        return utils.sample(time, self._t, self._Sl, p['TS'])
    
    # ===========================================
    # Forward Model: Liver, Portal Vein and Aorta
    # ===========================================
    
    def _predict(self, time: Data) -> Data:
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

    def _estimate_parameters(self, time: Data, signal: Data, n0: int):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        # Estimate BAT based on peak signal
        t_hl, d_hl = p['Thl'], p['Dhl']
        bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        p['BAT'] = max(bat, 0)

        # 2. Scaling Factor (S0) aorta
        pars = {k: p[k] for k in self._pars_list(select='sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr_a']
        s_ref = sig.signal(self._cnfg['sequence'], p['R10_a'], 1, **pars)
        p['S0_a'] = np.mean(signal[0][:n0]) / s_ref if s_ref > 0 else 0

        # 3. Scaling Factor (S0) portal vein
        pars = {k: p[k] for k in self._pars_list(select='liver_sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr_v']
        seq = 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence']
        s_ref = sig.signal(seq, p['R10_v'], 1, **pars)
        p['S0_v'] = np.mean(signal[1][:n0]) / s_ref if s_ref > 0 else 0

        # 4. Scaling Factor (S0) liver
        pars = {k: p[k] for k in self._pars_list(select='liver_sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr']
        seq = 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence']
        s_ref = sig.signal(seq, p['R10_l'], 1, **pars)
        p['S0_l'] = np.mean(signal[2][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: Data, signal: Data, free: dict, 
        bounds: dict, n0: int, staged: bool, **kwargs
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == 'SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")
        
        if staged:
            # Optimize Aorta parameters
            free_aorta = {k: v for k, v in free.items() if k in self._pars_list('aorta_fit')}
            utils.train(self._predict_aorta, time[0], signal[0], self._pars, free_aorta, **kwargs)

            # Optimize Portal parameters
            free_portal = {k: v for k, v in free.items() if k in self._pars_list('portal_fit')}
            utils.train(self._predict_portal, time[1], signal[1], self._pars, free_portal, **kwargs)

            # Optimize Liver parameters
            free_liver = {k: v for k, v in free.items() if k in self._pars_list('liver_fit')}
            utils.train(self._predict_liver, time[2], signal[2], self._pars, free_liver, **kwargs)

        # Joint Optimization
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)


    # ==========================================
    # I/O and Reporting
    # ==========================================


    def _plot(
        self, time: Data, signal: Data, xlim: list, fname: str, 
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
    # Public API: Data Extraction
    # ==========================================

    

    def time(self) -> Data:
        """Internal time array

        Returns:
            Tuple: (aorta_time, portal_time, liver_time).
        """
        self._set_time()
        return self._t, self._t, self._t

    def conc(self) -> Data:
        """Get concentrations in aorta, portal vein, and liver.

        Returns:
            Tuple: (aorta_conc, portal_conc, liver_conc).
        """
        self._compute_conc_aorta()
        self._compute_conc_portal()
        self._compute_conc_liver()
        return self._ca, self._cv, self._Cl

    def relax(self) -> Data:
        """Get relaxation rates in aorta, portal vein, and liver.

        Returns:
            Tuple: (R1_aorta, R1_portal, R1_liver).
        """
        self._compute_relax_aorta()
        self._compute_relax_portal()
        self._compute_relax_liver()
        return self._R1a, self._R1v, self._R1l
    
    def signal(self) -> Data:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal_aorta()
        self._compute_signal_portal()
        self._compute_signal_liver()
        return self._Sa, self._Sv, self._Sl

    def predict(self, time: Data) -> Data:
        """Predict the signals at given time points.

        Args:
            time: Time points for (aorta, portal, liver).

        Returns:
            Tuple: Predicted signals for (aorta, portal, liver).
        """
        return self._predict(time)
    
    def train(
        self, time: Data, signal: Data, free: dict = None, 
        bounds: dict = None, n0=1, staged=False, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
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
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)

    def plot(
        self, time: Data, signal: Data, xlim: list = None, 
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
        self._plot(time, signal, xlim, fname, show)