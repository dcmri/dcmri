import json
from copy import deepcopy
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk
import dcmri.liver as liver



PARAMS = liver.PARAMS_LIVER | {
    # --- Experimental Setup ---
    'field_strength': {'init': 3.0, 'name': 'Field strength', 'unit': 'T'},

    # --- Simulation Constants ---
    'dt': {'init': 0.5, 'name': 'Time step', 'unit': 's'},
    'tmax': {'init': 180, 'name': 'Max time', 'unit': 's'},
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},

    # --- Injection & Contrast Agent ---
    'agent': {'init': 'gadoterate', 'name': 'Contrast agent', 'unit': None},
    'weight': {'init': 70, 'name': 'Weight', 'unit': 'kg'},
    'dose': {'init': 0.1, 'name': 'Dose', 'unit': 'mL/kg'},
    'rate': {'init': 1, 'name': 'Injection rate', 'unit': 'mL/s'},
    'BAT': {'init': 60, 'bounds': [-30, 30], 'name': 'Bolus arrival time', 'unit': 's'},

    # --- Physiological & Pharmacokinetic (aorta) ---
    'H': {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},
    'CO': {'init': 100.0, 'bounds': [0.0, 300.0], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'T(hl)': {'init': 10.0,  'bounds': [0.0, 30.0], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'D(hl)': {'init': 0.2,   'bounds': [0.05, 0.95], 'name': 'Heart-lung dispersion', 'unit': ''},
    'T(o)': {'init': 20.0,  'bounds': [0.0, 60.0], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'E(o)': {'init': 0.15,  'bounds': [0.0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'T(o,e)': {'init': 120.0, 'bounds': [0.0, 800.0], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'E(b)': {'init': 0.05,  'bounds': [0.01, 0.15], 'name': 'Body extraction fraction', 'unit': ''},

    # --- Physiological & Pharmacokinetic (portal) ---
    'T(g)': {'init': 15.0, 'bounds': [0.1, 60.0], 'name': 'Gut mean transit time', 'unit': 'sec'},
    'D(g)': {'init': 0.85, 'bounds': [0.0, 1.0], 'name': 'Gut dispersion', 'unit': ''},
    'u(v)': {'init': 1.0,  'bounds': [0.0, 1.0], 'name': 'Portal vein volume fraction', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'FA': {'init': 15.0, 'bounds': [0.0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'TS': {'init': None, 'name': 'Sampling time', 'unit': 'sec'},
    'TF': {'init': 0.5, 'bounds': [0.0, 5.0], 'name': 'Aorta inflow time', 'unit': 'sec'},

    # --- Baseline Relaxation & Scaling ---
    'R10(a)': {'init': 1/1.5, 'bounds': [0.0, 5], 'name': 'Aorta baseline R1', 'unit': 'Hz'},
    'R10(v)': {'init': 1/0.8, 'bounds': [0.0, 5], 'name': 'Portal baseline R1', 'unit': 'Hz'},
    'R10(l)': {'init': 1/0.8, 'bounds': [0.0, 5], 'name': 'Liver baseline R1', 'unit': 'Hz'},
    'S0(a)': {'init': 1.0, 'bounds': [0.0, 5], 'name': 'Aorta signal scale factor', 'unit': 'a.u.'},
    'S0(l)': {'init': 1.0, 'bounds': [0.0, 5], 'name': 'Liver signal scale factor', 'unit': 'a.u.'},
    'S0(v)': {'init': 1.0, 'bounds': [0.0, 5], 'name': 'Portal venous signal scale factor', 'unit': 'a.u.'},
}


class AortaPortalLiver:
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
        self._version = '1.0'
        
        # --- 1. Configuration Validation ---
        if sequence not in ['SS', 'SSI']:
            raise ValueError(f"Sequence '{sequence}' is not available.")
        if not kinetics.startswith('2'):
            raise ValueError("Only dual-inlet models are allowed.")

        # --- 2. State Initialization ---
        self._kinetics = kinetics 
        self._non_stationary = non_stationary
        self._sequence = sequence 

       # Initialize parameters
        self._pars = {p: deepcopy(PARAMS[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    def _pars_list(self, select=None):
        pars_sequence = {'SSI': ['FA', 'TR', 'TF'], 'SS': ['FA', 'TR']}
        aorta_kinetics = ['BAT', 'CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)', 'E(b)']
        portal_kinetics = ['T(g)', 'D(g)', 'u(v)']
        liver_kinetics = list(liver.params_liver(self._kinetics, self._non_stationary).keys())
        if select is None:
            pars_list = [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate',
                'R10(a)', 'R10(v)', 'R10(l)', 'S0(a)', 'S0(v)', 'S0(l)'
            ]
            pars_list += ['H', 'vol'] + aorta_kinetics + portal_kinetics + liver_kinetics
            pars_list += ['TS'] + pars_sequence[self._sequence]
        elif select=='free':
            pars_list = aorta_kinetics + portal_kinetics + liver_kinetics
        elif select=='export': # exclude liver
            pars_list = ['S0(a)', 'S0(v)', 'S0(l)'] + aorta_kinetics + portal_kinetics
        elif select=='liver':
            pars_list = liver_kinetics
        elif select=='aorta':
            pars_list = aorta_kinetics
            pars_list += ['TF'] if self._sequence=='SSI' else []
        elif select=='portal':
            pars_list = portal_kinetics

        return pars_list

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        """Build time axis"""
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])

    def _compute_conc_aorta(self):
        """Internal: Calculate blood concentration in the aorta."""
        self._set_time()
        organs = ['2cxm', ([self._pars['T(o)'], self._pars['T(o,e)']], self._pars['E(o)'])]

        conc = lib.ca_conc(self._pars['agent'])
        injection_flux = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        aorta_flux = pk_aorta.flux_aorta(
            injection_flux, 
            E=self._pars['E(b)'], 
            dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
            heartlung=['pfcomp', (self._pars['T(hl)'], self._pars['D(hl)'])],
            organs=organs
        )
        self._ca = aorta_flux / self._pars['CO']

    def _compute_relax_aorta(self):
        """Internal: Calculate longitudinal relaxation rate in the aorta."""
        self._compute_conc_aorta()
        rb = lib.relaxivity(
            self._pars['field_strength'], 'blood', self._pars['agent']
        )
        self._R1a = self._pars['R10(a)'] + rb * self._ca

    def _compute_signal_aorta(self):
        """Internal: Calculate MRI signal in the aorta."""
        self._compute_relax_aorta()
        if self._sequence == 'SSI':
            self._Sa = sig.signal_spgr(
                self._pars['S0(a)'], self._R1a, self._pars['TF'], 
                self._pars['TR'], self._pars['FA'], n0=1
            )
        else:
            self._Sa = sig.signal_ss(
                self._pars['S0(a)'], self._R1a, self._pars['TR'], 
                self._pars['FA']
            )

    def _predict_aorta(self, time: np.ndarray) -> np.ndarray:
        """Internal: Sample aorta signal at specific time points."""
        self._compute_signal_aorta()
        return utils.sample(time, self._t, self._Sa, self._pars['TS'])

    # ==========================================
    # Forward Model: Portal
    # ==========================================
    
    def _compute_conc_portal(self):
        """Internal: Compute portal-venous concentration."""
        # Note: Depends on self._ca from aorta computation
        self._cv = pk.flux_chain(
            self._ca, self._pars['T(g)'], self._pars['D(g)'], dt=self._pars['dt']
        )
    
    def _compute_relax_portal(self):
        """Internal: Calculate longitudinal relaxation rate in the portal vein."""
        self._compute_conc_portal()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1v = self._pars['R10(v)'] + rb * self._pars['u(v)'] * self._cv
    
    def _compute_signal_portal(self):
        """Internal: Calculate MRI signal in the portal vein."""
        self._compute_relax_portal()
        self._Sv = sig.signal_ss(self._pars['S0(v)'], self._R1v, self._pars['TR'], self._pars['FA']
        )
    
    def _predict_portal(self, time: np.ndarray) -> np.ndarray:
        """Internal: Sample portal signal at specific time points."""
        self._compute_signal_portal()
        return utils.sample(time, self._t, self._Sv, self._pars['TS'])

    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        """Calculate tissue concentration in the liver."""
        pars = self._pars_dict(select='liver')

        hct = self._pars['H']
        ca_plasma = self._ca / (1 - hct)
        cv_plasma = self._cv / (1 - hct)
        c_plasma = (ca_plasma, cv_plasma)
        
        self._Cl = liver.conc_liver(
            c_plasma, 
            dt=self._pars['dt'], 
            kinetics=self._kinetics, 
            non_stationary=self._non_stationary, 
            sum=False, 
            **pars
        )
        
    def _compute_relax_liver(self):
        """Calculates the liver longitudinal relaxation rate R1."""
        self._compute_conc_liver()
        rp = lib.relaxivity(self._pars['field_strength'], 'plasma', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])

        if self._Cl.ndim == 2:
            # Multi-compartment model
            self._R1l = self._pars['R10(l)'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10(l)'] + rp * self._Cl

    def _compute_signal_liver(self):
        """Internal: Calculates the liver MRI signal."""
        self._compute_relax_liver()
        self._Sl = sig.signal_ss(self._pars['S0(l)'], self._R1l, self._pars['TR'], self._pars['FA'])

    def _predict_liver(self, time: np.ndarray) -> np.ndarray:
        """Internal: Predict liver signal at specific time points."""
        self._compute_signal_liver()
        return utils.sample(time, self._t, self._Sl, self._pars['TS'])

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> np.ndarray:
        """Internal time array

        Returns:
            Tuple: (aorta_time, portal_time, liver_time).
        """
        self._set_time()
        return self._t, self._t, self._t

    def conc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get concentrations in aorta, portal vein, and liver.

        Returns:
            Tuple: (aorta_conc, portal_conc, liver_conc).
        """
        self._compute_conc_aorta()
        self._compute_conc_portal()
        self._compute_conc_liver()
        return self._ca, self._cv, self._Cl

    def relax(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get relaxation rates in aorta, portal vein, and liver.

        Returns:
            Tuple: (R1_aorta, R1_portal, R1_liver).
        """
        self._compute_relax_aorta()
        self._compute_relax_portal()
        self._compute_relax_liver()
        return self._R1a, self._R1v, self._R1l
    
    def signal(self) -> tuple:
        """Return signals in aorta and liver.

        Returns:
            tuple: (time, aorta_signal, liver_signal)
        """
        self._compute_signal_aorta()
        self._compute_signal_portal()
        self._compute_signal_liver()
        return self._Sa, self._Sv, self._Sl

    def predict(self, time: tuple=None) -> tuple:
        """Predict the signals at given time points.

        Args:
            time (Tuple[np.ndarray, ...]): Time points for (aorta, portal, liver).

        Returns:
            Tuple[np.ndarray, ...]: Predicted signals for (aorta, portal, liver).
        """
        if time is None:
            time = self.time()
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        sa = self._predict_aorta(time[0])
        sv = self._predict_portal(time[1])
        sl = self._predict_liver(time[2])
        return sa, sv, sl

    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def train(
        self, time: tuple, signal: tuple, free: dict=None, 
        bounds: dict=None, n0=1, **kwargs
    ):
        """Train the free parameters against provided data.

        Args:
            time (tuple): (time_aorta, time_portal, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_portal, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific params.
            n0 (int, optional): Baseline points for S0 estimation. Defaults to 1.
            **kwargs: Passed to scipy.optimize.curve_fit via utils.train.

        Returns:
            AortaPortalLiver: The trained model instance.
        """
        # Heuristic initialization
        self._estimate_parameters(time, signal, n0)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds)

        # Aorta optimization
        free_aorta = {p:v for p, v in free.items() if p in self._pars_list('aorta')}
        utils.train(self._predict_aorta, time[0], signal[0], self._pars, free_aorta, **kwargs)

        # Portal optimization
        free_portal = {p:v for p, v in free.items() if p in self._pars_list('portal')}
        utils.train(self._predict_portal, time[1], signal[1], self._pars, free_portal, **kwargs)

        # Liver optimization
        free_liver = {p:v for p, v in free.items() if p in self._pars_list('liver')}
        utils.train(self._predict_liver, time[2], signal[2], self._pars, free_liver, **kwargs)

        # Joint optimization
        pcov, sdev = utils.train(self.predict, time, signal, self._pars, free, **kwargs)
        pars = {
            p: {
                'name': deepcopy(PARAMS[p]['name']), 
                'unit': deepcopy(PARAMS[p]['unit']), 
                'value': self._pars[p], 
                'sdev': sdev[p] if sdev is not None else None
            } 
            for p in free
        }
        return pars, pcov
    
    def _estimate_parameters(self, time: tuple, signal: tuple, n0: int):
        """Internal: Heuristic estimation of BAT and signal scaling (S0)."""

        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        # Estimate BAT based on peak signal
        t_hl, d_hl = self._pars['T(hl)'], self._pars['D(hl)']
        heuristic_bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        self._pars['BAT'] = max(heuristic_bat, 0)

        # S0 Estimation for Aorta
        if self._sequence == 'SSI':
            s_ref_a = sig.signal_spgr(
                1, self._pars['R10(a)'], self._pars['TF'], 
                self._pars['TR'], self._pars['FA'], n0=1,
            )
        else:
            s_ref_a = sig.signal_ss(
                1, self._pars['R10(a)'], self._pars['TR'], 
                self._pars['FA']
            )
            
        # S0 Estimation for Portal and Liver
        s_ref_v = sig.signal_ss(1, self._pars['R10(v)'], self._pars['TR'], self._pars['FA'])
        s_ref_l = sig.signal_ss(1, self._pars['R10(l)'], self._pars['TR'], self._pars['FA'])
        
        self._pars['S0(a)'] = np.mean(signal[0][:n0]) / s_ref_a
        self._pars['S0(v)'] = np.mean(signal[1][:n0]) / s_ref_v
        self._pars['S0(l)'] = np.mean(signal[2][:n0]) / s_ref_l

    def _set_free_pars(self, free: dict=None, bounds: dict=None):
        # --- 0. Set Defaults ---
        if free is None:
            free = {p: deepcopy(PARAMS[p]['bounds']) for p in self._pars_list('free')}
        
        # --- 1. Update Bounds ---
        if bounds is not None:
            for p, b in bounds.items():
                if b is None:
                    free.pop(p, None)
                else:
                    free[p] = b

        # --- 2. Boundary Validation ---
        for p, bnds in free.items():
            if p not in self._pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            elif p == 'BAT':
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError("Bounds on BAT must be (negative, positive).")
            elif p in ['S0(a)', 'S0(v)', 'S0(l)']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(
                    f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}."
                )
            
        # --- 3. Relative to Absolute Bounds
        # Additive
        for par in ['BAT']:
            if par in free:
                free[par] = [  
                    self._pars[par] + free[par][0],
                    self._pars[par] + free[par][1],
                ]
        # Multiplicative
        for par in ['S0(a)', 'S0(v)', 'S0(l)']:
            if par in free:
                free[par] = [
                    self._pars[par] * free[par][0],
                    self._pars[par] * free[par][1],
                ]

        return free

    # ==========================================
    # I/O and Reporting
    # ==========================================

    def save(self, file: str):
        """Save model state to a JSON file."""
        if not file.endswith('.json'):
            file += '.json'

        data = {
            'model': self.__class__.__name__,
            'version': self._version,
            'kinetics': self._kinetics,
            'non_stationary': self._non_stationary,
            'sequence': self._sequence,
            'pars': self._pars,
        }

        with open(file, "w") as f:
            json.dump(data, f, indent=4)
        return self

    def load(self, file: str):
        """Load model state from a JSON file."""
        with open(file, "r") as f:
            data = json.load(f) 

        if data['model'] != self.__class__.__name__:
            raise ValueError(f"File belongs to {data['model']}, not {self.__class__.__name__}.")
        if data['version'] != self._version:
            raise ValueError(f"Version mismatch: {data['version']} vs {self._version}.")

        self._kinetics = data['kinetics']
        self._non_stationary = data['non_stationary']
        self._sequence = data['sequence']
        self._pars = data['pars']
        return self

    def export_params(self) -> Dict[str, List]:
        """Return model parameters with descriptions and standard deviations.

        Returns:
            dict: {short_name: [long_name, value, unit, sdev]}
        """
        pars_liver = {p: self._pars[p] for p in self._pars_list('liver')}
        pars_liver = liver.derived_params_liver(pars_liver, self._kinetics, self._pars['H'])

        value = self._pars | pars_liver
        export_pars = self._pars_list('export') + list(pars_liver.keys())
        pars = {
            p: {
                'name': deepcopy(PARAMS[p]['name']),
                'unit': deepcopy(PARAMS[p]['unit']),  
                'value': value[p], 
            } for p in export_pars
        }
        return pars  
    
    def print_params(self, round_to=None):
        """Print parameters and uncertainties to console."""
        pars = self.export_params()
        for p, v in pars.items():
            val = v['value']
            if round_to is not None:
                val = round(val, round_to)
            print(f"{v['name']} ({p}) = {val} {v['unit']}")

    def _pars_dict(self, *args, select=None):
        """Return the parameter values"""
        if len(args) == 0:
            pars = deepcopy(self._pars)
        else:
            pars = {k: v for k, v in self._pars.items() if k in list(args)}
        if select is not None:
            pars = {k: v for k, v in pars.items() if k in self._pars_list(select)}
        return pars

    def params(self, *args, as_dict=False):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            tuple or dict: values of parameters
        """
        pars = self._pars_dict(*args)
        if as_dict:
            return pars
        elif len(pars)==1:
            return list(pars.values())[0]
        else:
            return tuple(pars.values())
    
    def cost(self, time: tuple, signal: tuple, metric: str = 'NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            signal (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
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
        ypred = self.predict(time)
        if isinstance(signal, tuple):
            ypred = np.concatenate(ypred)
            signal = np.concatenate(signal)
        return utils.loss(ypred, signal, metric)

    def plot(
        self,
        time: tuple,
        signal: tuple,
        xlim: list = None, 
        fname: str = None, 
        show: bool = True,
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
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        self.signal()
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        fig.subplots_adjust(wspace=0.3)

        # Plot Aorta
        _plot_data(
            self._t, self._Sa, time[0], signal[0], ax1, xlim,
            color=['lightcoral', 'darkred']
        )
        # Plot Portal
        _plot_data(
            self._t, self._Sv, time[1], signal[1], ax3, xlim,
            color=['orchid', 'purple'],
        )
        # Plot Liver
        _plot_data(
            self._t, self._Sl, time[2], signal[2], ax5, xlim,
            color=['cornflowerblue', 'darkblue'],
            xlabel='Time (min)'
        )
        
        _plot_conc_aorta(self._t, self._ca, ax2, xlim)
        _plot_conc_portal(self._t, self._cv, ax4, xlim)
        _plot_conc_liver(self._t, self._Cl, ax6, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()



# ==========================================
# Private Helper Functions
# ==========================================


def _plot_conc_aorta(t, cb, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-',
            color='darkred', linewidth=2.0, label='Aorta')
    ax.legend()


def _plot_conc_portal(t, cv, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cv, linestyle='-',
            color='purple', linewidth=2.0, label='Portal vein')
    ax.legend()


def _plot_conc_liver(t, C, ax, xlim=None):
    color = 'darkblue'
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    if C.ndim==2:
        ax.plot(t/60, 1000*C[0, :], linestyle='-.',
                color=color, linewidth=2.0, label='Extracellular')
        ax.plot(t/60, 1000*C[1, :], linestyle='--',
                color=color, linewidth=2.0, label='Hepatocytes')
        ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
                color=color, linewidth=2.0, label='Liver')
    else:
        ax.plot(t/60, 1000*C, linestyle='-',
                color=color, linewidth=2.0, label='Liver')
    ax.legend()


def _plot_data(t: np.ndarray, sig: np.ndarray,
                    xdata: np.ndarray, ydata: np.ndarray,
                    ax, xlim, color=['black', 'black'],
                    xlabel=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel=xlabel, ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(xdata/60, ydata, marker='o',
            color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig, linestyle='-',
            color=color[1], linewidth=3.0, label='fit')
    ax.legend()