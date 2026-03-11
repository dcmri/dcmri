import json
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver


PARAMS = liver.PARAMS_LIVER | {
    # --- Experimental Setup ---
    'field_strength': {'init': 3.0, 'name': 'Magnetic field strength', 'unit': 'T'},
    't_scan2': {'init': 2 * 60 * 60, 'name': 'Start of second scan', 'unit': 'sec'},

    # --- Simulation Constants ---
    'dt': {'init': 0.5, 'name': 'Forward model time step', 'unit': 'sec'},
    'tmax': {'init': 4 * 60 * 60, 'name': 'Maximum acquisition time', 'unit': 'sec'},
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},
    
    # --- Injection & Contrast Agent ---
    'weight': {'init': 70.0, 'name': 'Subject weight', 'unit': 'kg'},
    'agent': {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},
    'dose': {'init': 0.05, 'name': 'First contrast agent dose', 'unit': 'mL/kg'},
    'rate': {'init': 1.0, 'name': 'Contrast agent injection rate', 'unit': 'mL/sec'},
    'dose2': {'init': 0.05, 'name': 'Second contrast agent dose', 'unit': 'mL/kg'},
    'BAT': {'init': 120.0, 'bounds': [-60.0, 60.0], 'name': 'First bolus arrival time', 'unit': 'sec'},
    'BAT2': {'init': 7200 + 900, 'bounds': [-60.0, 60.0], 'name': 'Second bolus arrival time', 'unit': 'sec'},

    # --- Physiological & Pharmacokinetic ---
    'H': {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},
    'CO': {'init': 100.0, 'bounds': [0.0, 300.0], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'T(hl)': {'init': 10.0, 'bounds': [0.0, 30.0], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'D(hl)': {'init': 0.2, 'bounds': [0.05, 0.95], 'name': 'Heart-lung logic dispersion', 'unit': ''},
    'T(o)': {'init': 20.0, 'bounds': [0.0, 60.0], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'E(o)': {'init': 0.15, 'bounds': [0.0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'T(o,e)': {'init': 120.0, 'bounds': [0.0, 800.0], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'E(b)': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Body extraction fraction', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'FA': {'init': 15.0, 'bounds': [0.0, 180.0], 'name': 'Flip angle', 'unit': 'deg'},
    'FA2': {'init': 15.0, 'bounds': [0.0, 180], 'name': 'Second flip angle', 'unit': 'deg'},
    'TC': {'init': 0.18, 'name': 'Time to center', 'unit': 'sec'},
    'TS': {'init': 2.0, 'name': 'Sampling time', 'unit': 'sec'},

    # --- Baseline Relaxation & Scaling ---
    'R10(a)': {'init': 1.0/1.5, 'bounds': [0.0, 5], 'name': 'Aorta first baseline R1', 'unit': 'Hz'},
    'R10(l)': {'init': 1.0/0.8, 'bounds': [0.0, 5], 'name': 'Liver first baseline R1', 'unit': 'Hz'},
    'S0(a)': {'init': 1.0, 'bounds': [0.0, 5.0], 'name': 'Aorta first signal scale factor', 'unit': 'a.u.'},
    'S0(l)': {'init': 2.0, 'bounds': [0.0, 5.0], 'name': 'Liver first signal scale factor', 'unit': 'a.u.'},
    'S02(a)': {'init': 2.0, 'bounds': [0.0, 5.0], 'name': 'Aorta second signal scale factor', 'unit': 'a.u.'},
    'S02(l)': {'init': 4.0, 'bounds': [0.0, 5.0], 'name': 'Liver second signal scale factor', 'unit': 'a.u.'},
}


class AortaLiver2scan:
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

    def __init__(
        self, 
        kinetics = '1I-IC-HFD', 
        non_stationary=None, 
        sequence='SS', 
        **params,
      ):
        self._version = '1.0'
        
        # Set configuration
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics/stationarity: {e}") from e
        if sequence not in ['SS', 'SR']:
            raise ValueError('Sequence ' + str(sequence) + ' is not available.')
        if not kinetics.startswith('1'):
            raise ValueError('Only single-inlet models are allowed.')
        
        self._kinetics = kinetics
        self._sequence = sequence 
        self._non_stationary = non_stationary

        # Initialize parameters
        self._pars = {p: deepcopy(PARAMS[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
    def _pars_list(self, select=None):
        aorta_kinetics = ['BAT', 'BAT2', 'CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)', 'E(b)']
        liver_kinetics = list(liver.params_liver(self._kinetics, self._non_stationary).keys())
        if select is None:
            pars_list = ['dt', 'tmax', 't_scan2', 'dose_tolerance', 'field_strength']
            pars_list += ['weight', 'agent', 'dose', 'dose2', 'rate']
            pars_list += ['R10(a)', 'R10(l)', 'S0(a)', 'S0(l)', 'S02(a)', 'S02(l)']
            pars_list += ['TS', 'FA2'] + {'SR': ['FA', 'TR', 'TC'], 'SS': ['FA', 'TR']}[self._sequence]
            pars_list += ['H'] + aorta_kinetics
            pars_list += ['vol'] + liver_kinetics 
        elif select=='free':
            pars_list = ['S02(a)', 'S02(l)'] + aorta_kinetics + liver_kinetics
        elif select=='export':
            pars_list = ['S0(a)', 'S0(l)', 'S02(a)', 'S02(l)'] + aorta_kinetics
        elif select=='first_scan':
            pars_list = {'SR': ['FA', 'TR', 'TC'], 'SS': ['FA', 'TR'], 'SRC': ['TC']}[self._sequence]
        elif select=='second_scan':
            pars_list = {'SR': ['FA2', 'TR', 'TC'], 'SS': ['FA2', 'TR'], 'SRC': ['TC']}[self._sequence]
        elif select=='liver':
            pars_list = liver_kinetics
        elif select=='aorta_fit':
            pars_list = ['S02(a)'] + aorta_kinetics
        elif select=='liver_fit':
            pars_list = ['S02(l)'] + liver_kinetics
        return pars_list

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        """Build time axis"""
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])
        
    def _compute_conc_aorta(self):
        """Calculate blood concentration in the aorta."""
        self._set_time()
        organs = ['2cxm', ([self._pars['T(o)'], self._pars['T(o,e)']], self._pars['E(o)'])]
        conc = lib.ca_conc(self._pars['agent'])
        J1 = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        J2 = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose2'], 
            self._pars['rate'], self._pars['BAT2']
        )
        Jb = pk_aorta.flux_aorta(
            J1 + J2, E=self._pars['E(b)'], dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
            heartlung = ['pfcomp', (self._pars['T(hl)'], self._pars['D(hl)'])],
            organs = organs
        )
        self._ca = Jb / self._pars['CO']

    def _compute_relax_aorta(self):
        """Calculate longitudinal relaxation rate in the aorta."""
        self._compute_conc_aorta()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1a = self._pars['R10(a)'] + rb * self._ca

    def _compute_signal_aorta(self):
        """Calculate MRI signal in the aorta."""
        self._compute_relax_aorta()
        pars1 = self._pars_dict(select='first_scan')
        pars2 = self._pars_dict(select='second_scan')
        pars2['FA'] = pars2.pop('FA2', None)

        self._Sa = np.zeros(self._t.size)
        t1 = self._t < self._pars['t_scan2']
        t2 = self._t >= self._pars['t_scan2']
        self._Sa[t1] = sig.signal(self._sequence, self._R1a[t1], self._pars['S0(a)'], **pars1)
        self._Sa[t2] = sig.signal(self._sequence, self._R1a[t2], self._pars['S02(a)'], **pars2)

    def _predict_aorta(self, time):
        """Sample aorta signal at specific time points."""
        self._compute_signal_aorta()
        return (
            utils.sample(time[0], self._t, self._Sa, self._pars['TS']),
            utils.sample(time[1], self._t, self._Sa, self._pars['TS']),
        )
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        """Calculate tissue concentration in the liver."""
        pars = self._pars_dict(select='liver')

        cp = self._ca / (1 - self._pars['H'])
        self._Cl = liver.conc_liver(
            cp, dt=self._pars['dt'], sum=False, kinetics=self._kinetics, 
            non_stationary=self._non_stationary, **pars,
        )

    def _compute_relax_liver(self):
        """Calculate tissue concentration in the liver."""
        self._compute_conc_liver()
        rp = lib.relaxivity(self._pars['field_strength'], 'plasma', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])

        if self._Cl.ndim==2:
            self._R1l = self._pars['R10(l)'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10(l)'] + rp * self._Cl

    def _compute_signal_liver(self):
        """Calculate MRI signal in the liver."""
        pars1 = self._pars_dict(select='first_scan')
        pars2 = self._pars_dict(select='second_scan')
        pars2['FA'] = pars2.pop('FA2', None)
        
        self._compute_relax_liver()
        self._Sl = np.zeros(self._t.size)
        t1 = self._t <= self._pars['t_scan2']
        t2 = self._t > self._pars['t_scan2']
        self._Sl[t1] = sig.signal(self._sequence, self._R1l[t1], self._pars['S0(l)'], **pars1)
        self._Sl[t2] = sig.signal(self._sequence, self._R1l[t2], self._pars['S02(l)'], **pars2)

    def _predict_liver(self, time):
        self._compute_signal_liver()
        return (
            utils.sample(time[0], self._t, self._Sl, self._pars['TS']),
            utils.sample(time[1], self._t, self._Sl, self._pars['TS']),
        )
    
    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> np.ndarray:
        """Internal time array
        
        Returns:
            tuple: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.      
        """
        self._set_time()
        t, t2 = self._t, self._pars['t_scan2']
        tacq1, tacq2 = t[t < t2], t[t >= t2]
        return tacq1, tacq2, tacq1, tacq2
    
    def conc(self) -> tuple:
        """Concentrations in aorta and liver.

        Returns:
            tuple: aorta conc scan 1, aorta conc scan 2, 
              liver conc scan 1, liver conc scan 2.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        t, t2 = self._t, self._pars['t_scan2']
        ca = self._ca[t < t2], self._ca[t >= t2]
        Cl = self._Cl[t < t2], self._Cl[t >= t2]
        return ca + Cl
    
    def relax(self) -> tuple:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: aorta R1 scan 1, aorta R1 scan 2, 
              liver R1 scan 1, liver R1 scan 2.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        t, t2 = self._t, self._pars['t_scan2']
        R1a = self._R1a[t < t2], self._R1a[t >= t2]
        R1l = self._R1l[t < t2], self._R1l[t >= t2]
        return R1a + R1l
    
    def signal(self) -> tuple:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta signal scan 1, aorta signal scan 2, 
              liver signal scan 1, liver signal scan 2.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        t, t2 = self._t, self._pars['t_scan2']
        Sa = self._Sa[t < t2], self._Sa[t >= t2]
        Sl = self._Sl[t < t2], self._Sl[t >= t2]
        return Sa + Sl
    
    def predict(self, time: tuple=None) -> tuple:
        """Predict the data at given time points

        Args:
            time: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.

        Returns:
            tuple: aorta data scan 1, aorta data scan 2, 
              liver data scan 1, liver data scan 2.
        """
        if time is None:
            time = self.time()
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        Sa = self._predict_aorta(time[:2])
        Sl = self._predict_liver(time[2:])
        return Sa + Sl

    # ==========================================
    # Inverse Model: Training
    # ==========================================    

    def train(
        self, time: tuple, signal: tuple, free: dict=None, 
        bounds:dict=None, R102a=None, R102l=None, n0=1, **kwargs,
    ):
        """Train the free parameters

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in 
                the first scan, aorta in the second stand, liver in the first 
                scan, and liver in the second scan, in that order. The four 
                arrays can be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
                first scan, aorta in the second stand, liver in the first scan, 
                and liver in the second scan, in that order. The arrays can be 
                different in length but each has to have the same length as its 
                corresponding array of time points.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R102a (float, optional): R1 value in arterial blood before the 
                second injection. If provided this is used to estimate the 
                baseline S0a in the artery. Else this is initialized to S0a. 
                Defaults to None.
            R102l (float, optional): R1 value in liver before the 
                second injection. If provided this is used to estimate the 
                baseline S0(l) in the liver. Else this is initialized to S0(l). 
                Defaults to None.
            kwargs: any other keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        # Initial heuristics for BAT and S0
        self._estimate_parameters(time, signal, n0, R102a, R102l)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds)       

        # Train free aorta parameters on aorta data
        free_aorta = {p:v for p, v in free.items() if p in self._pars_list('aorta_fit')}
        utils.train(self._predict_aorta, time[:2], signal[:2], self._pars, free_aorta, **kwargs)

        # Train free liver parameters on liver data
        free_liver = {p:v for p, v in free.items() if p in self._pars_list('liver_fit')}
        utils.train(self._predict_liver, time[2:], signal[2:], self._pars, free_liver, **kwargs)

        # Train all parameters on all data
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
    
    def _estimate_parameters(self, time, signal, n0, R102a, R102l):
        """Heuristic estimation of BAT and signal scaling (S0)."""
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        # Estimate BAT and BAT2 and ajust their bounds
        T, D = self._pars['T(hl)'], self._pars['D(hl)']
        self._pars['BAT'] = time[0][np.argmax(signal[0])] - (1-D)*T
        self._pars['BAT2'] = time[1][np.argmax(signal[1])] - (1-D)*T

        # Estimate S0
        pars = self._pars_dict(select='first_scan')
        Srefb = sig.signal(self._sequence, self._pars['R10(a)'], 1, **pars)
        Srefl = sig.signal(self._sequence, self._pars['R10(l)'], 1, **pars)
        self._pars['S0(a)'] = np.mean(signal[0][:n0]) / Srefb
        self._pars['S0(l)'] = np.mean(signal[2][:n0]) / Srefl

        # Estimate S02
        pars = self._pars_dict(select='second_scan')
        pars['FA'] = pars.pop('FA2', None)
        if R102a is None:
            self._pars['S02(a)'] = self._pars['S0(a)']
        else:
            Sref2b = sig.signal(self._sequence, R102a, 1, **pars)
            self._pars['S02(a)'] = np.mean(signal[1][:n0]) / Sref2b   
        if R102l is None:
            self._pars['S02(l)'] = self._pars['S0(l)']
        else:
            Sref2l = sig.signal(self._sequence, R102l, 1, **pars)
            self._pars['S02(l)'] = np.mean(signal[3][:n0]) / Sref2l    


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
            
            # Parameters with relative bounds
            if p in ['BAT', 'BAT2']:
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Bounds on BAT must be (negative, positive).")
            elif p in ['S0(a)', 'S0(l)', 'S02(a)', 'S02(l)']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")
            
            # Absolute bounds
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds

        # Additive
        for par in ['BAT', 'BAT2']:
            if par in free:
                free[par] = [  
                    self._pars[par] + free[par][0],
                    self._pars[par] + free[par][1],
                ]

        # Multiplicative
        for par in ['S0(a)', 'S0(l)', 'S02(a)', 'S02(l)']:
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
        """Save the current state of the model as a json file.

        Args:
            file (str): complete path of the json file. 
        """
        if file.split('.')[-1] != 'json':
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


    def load(self, file):
        """Load the saved state of the model from a json file

        Args:
            file (str): complete path of the json file. 
        """
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

    def export_params(self) -> dict:
        """Return model parameters with their descriptions

        Args:
            type (str, optional): Type of output. If 'dict', a dictionary is 
              returned. If 'list', a list is returned. Defaults to 'dict'.

        Returns:
            dict: Dictionary with one item for each model parameter. The key 
            is the short parameter name, and the value is a 
            4-element list with [long parameter name, value, unit, sdev].

        """
        # Add derived parameters
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
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
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

    def cost(self, time: tuple, signal: tuple, metric='NRMS') -> float:
        """Return the goodness-of-fit

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
        y = self.predict(time)
        if isinstance(signal, tuple):
            y = np.concatenate(y)
            signal = np.concatenate(signal)
        return utils.loss(y, signal, metric)

    def plot(self, time: tuple, signal: tuple,
             xlim=None, fname=None, show=True):
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
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        self.signal()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        _plot_data2scan(self._t, self._Sa, time[:2], signal[:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_data2scan(self._t, self._Sl, time[2:], signal[2:],
                        ax3, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_conc_aorta(self._t, self._ca, ax2, xlim)
        _plot_conc_liver(self._t, self._Cl, ax4, xlim)
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


# ==========================================
# Private Plotting Helpers
# ==========================================


def _plot_conc_aorta(t, cb, ax, xlim=None):
    if xlim is None:
        xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Concentration (mM)',
           xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-',
            color='darkred', linewidth=2.0, label='Aorta')
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
                color=color, linewidth=2.0, label='Tissue')
    else:
        ax.plot(t/60, 1000*C, linestyle='-',
                color=color, linewidth=2.0, label='Tissue')        
    ax.legend()

def _plot_data2scan(t, 
                    sig,
                    time, 
                    signal,
                    ax, xlim, color=['black', 'black']):
    if xlim is None:
        xlim = [0, t[-1]]
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', 
           xlim=np.array(xlim)/60)
    ax.plot(np.concatenate(time)/60, np.concatenate(signal),
            marker='o', color=color[0], label='fitted data', linestyle='None')
    ax.plot(t/60, sig,
            linestyle='-', color=color[1], linewidth=3.0, label='fit')
    ax.legend()



