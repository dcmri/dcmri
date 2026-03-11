import os
import warnings
from copy import deepcopy
import json

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk

INF_BOUND = 1e6

PARAMS = {

    # --- Experimental Setup ---
    'field_strength': {'init': 3.0, 'name': 'Magnetic field strength', 'unit': 'T'},
    't2':             {'init': 2 * [2 * 60 * 60], 'name': 'Start of second scan', 'unit': 'sec'},

    # --- Simulation Constants ---
    'dt':             {'init': 0.5, 'name': 'Forward model time step', 'unit': 'sec'},
    'tmax':           {'init': 2 * [4 * 60 * 60], 'name': 'Maximum acquisition time', 'unit': 'sec'},
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},
    
    # --- Injection & Contrast Agent ---
    'weight':         {'init': 70.0, 'name': 'Subject weight', 'unit': 'kg'},
    'agent':          {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},
    'dose':           {'init': 2 * [0.05], 'name': 'First contrast agent dose', 'unit': 'mL/kg'},
    'rate':           {'init': 1, 'name': 'Contrast agent injection rate', 'unit': 'mL/sec'},
    'dose2':          {'init': 2 * [0.05], 'name': 'Second contrast agent dose', 'unit': 'mL/kg'},
    'BAT':            {'init': 2 * [120], 'bounds': 2 * [[-60, 60]], 'name': 'First bolus arrival time', 'unit': 'sec'},
    'BAT2':           {'init': 2 * [7200 + 900], 'bounds': 2 * [[-60, 60]], 'name': 'Second bolus arrival time', 'unit': 'sec'},

    # Body
    'H':              {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},
    'CO':             {'init': 100, 'bounds': [0, 300], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'Thl':            {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'Dhl':            {'init': 0.2, 'bounds': [0.05, 0.95], 'name': 'Heart-lung dispersion', 'unit': ''},
    'To':             {'init': 20, 'bounds': [0, 60], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'Eo':             {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'Toe':            {'init': 120, 'bounds': [0, 800], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'GFR':            {'init': 2, 'bounds': [0, 4], 'name': 'Glomerular filtration rate', 'unit': 'mL/sec'},

    # Liver
    've':             {'init': 0.3, 'bounds': [0.01, 0.6], 'name': 'Liver extracellular volume fraction', 'unit': 'mL/cm3'},
    'Tg':             {'init': 30, 'bounds': [0.1, 60], 'name': 'Gut mean transit time', 'unit': 'sec'},
    'Dg':             {'init': 0.85, 'bounds': [0, 1], 'name': 'Gut dispersion', 'unit': ''},
    'khe_i':          {'init': 2 * [0.002], 'bounds': 2 * [[0.0, 0.1]], 'name': 'Initial hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'khe_f':          {'init': 2 * [0.002], 'bounds': 2 * [[0.0, 0.1]], 'name': 'Final hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'Th_i':           {'init': 2 * [30 * 60], 'bounds': 2 * [[10*60, 10*60*60]], 'name': 'Initial hepatocellular mean transit time', 'unit': 'sec'},
    'Th_f':           {'init': 2 * [30 * 60], 'bounds': 2 * [[10*60, 10*60*60]], 'name': 'Final hepatocellular mean transit time', 'unit': 'sec'},
    'Kbh':            {'init': 2 * [1e-9], 'bounds': 2 * [[1e-9, 1e-3]], 'name': 'Biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_i':          {'init': 2 * [1e-9], 'bounds': 2 * [[1e-9, 1e-3]], 'name': 'Initial biliary tissue excretion rate', 'unit': '/sec'},
    'Kbh_f':          {'init': 2 * [1e-9], 'bounds': 2 * [[1e-9, 1e-3]], 'name': 'Final biliary tissue excretion rate', 'unit': '/sec'},
    'vol':            {'init': 2 * [1000], 'bounds': 2 * [[0, 10000]], 'name': 'Liver volume', 'unit': 'cm3'},
    'vh':             {'name': 'Hepatocellular volume fraction', 'unit': 'mL/cm3'},
    'khe':            {'name': 'Hepatocellular uptake rate', 'unit': 'mL/sec/cm3'},
    'kbh':            {'name': 'Biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_i':          {'name': 'Initial biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'kbh_f':          {'name': 'Final biliary excretion rate', 'unit': 'mL/sec/cm3'},
    'Th':             {'name': 'Hepatocellular mean transit time', 'unit': 'sec'},
    'Khe':            {'name': 'Hepatocellular tissue uptake rate', 'unit': '/sec'},
    'CL':             {'name': 'Liver plasma clearance', 'unit': 'mL/sec'},
    'CL_i':           {'name': 'Initial liver plasma clearance', 'unit': 'mL/sec'},
    'CL_f':           {'name': 'Final liver plasma clearance', 'unit': 'mL/sec'},
    'Eb':             {'name': 'Body extraction fraction', 'unit': ''},
    'Eb_i':           {'name': 'Initial body extraction fraction', 'unit': ''},
    'Eb_f':           {'name': 'Final body extraction fraction', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'TR':             {'init': 0.005, 'bounds': [0, INF_BOUND], 'name': 'Repetition time', 'unit': 'sec'},
    'FA':             {'init': 15.0, 'bounds': [0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'FA2':            {'init': 15.0, 'bounds': [0, INF_BOUND], 'name': 'Second flip angle', 'unit': 'deg'},
    'TC':             {'init': 0.180, 'bounds': [0, INF_BOUND], 'name': 'Time to center', 'unit': 'sec'},
    'TS':             {'init': None, 'bounds': [0, INF_BOUND], 'name': 'Sampling time', 'unit': 'sec'},

    # --- Baseline Relaxation & Scaling ---
    'R10a':           {'init': 2 * [1/lib.T1(3.0, 'blood')], 'bounds': 2 * [[0, INF_BOUND]], 'name': 'Aorta first baseline R1', 'unit': 'Hz'},
    'R10l':           {'init': 2 * [1/lib.T1(3.0, 'liver')], 'bounds': 2 * [[0, INF_BOUND]], 'name': 'Liver first baseline R1', 'unit': 'Hz'},
    'S0a':            {'init': 2 * [1], 'bounds': 2 * [[0, INF_BOUND]], 'name': 'Aorta first signal scale factor', 'unit': 'a.u.'},
    'S0l':            {'init': 2 * [1], 'bounds': 2 * [[0, INF_BOUND]], 'name': 'Liver first signal scale factor', 'unit': 'a.u.'},
    'S02a':           {'init': 2 * [1], 'bounds': 2 * [[0, 2]], 'name': 'Aorta second signal scale factor', 'unit': 'a.u.'},
    'S02l':           {'init': 2 * [1], 'bounds': 2 * [[0, 2]], 'name': 'Liver second signal scale factor', 'unit': 'a.u.'},
}


class Liver2scanDrugEffect():
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

        >>> model.free['Th_i'] = [0, INF_BOUND]
        >>> model.free['Th_f'] = [0, INF_BOUND]

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
        Liver first signal scale factor (S0l): 150.003 a.u.
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
            * - R10a, R102a, R10l, R102l, S0a, S02a, S0l, S02l
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

    def __init__(self, **params):
        self._version = '1.0'

        self._pcov = None
        self._free = None

        # Initialize parameters
        self._pars = {p: deepcopy(PARAMS[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    def _pars_list(self, select=None):
        if select is None:
            pars_list = ['dt', 'tmax', 'dose_tolerance', 'field_strength', 'weight', 'agent', 'dose', 'rate']
            pars_list += ['H', 'R10a', 'R10l', 'S0a', 'S0l', 'S02a', 'S02l', 'BAT', 'BAT2']
            pars_list += ['FA', 'TR', 'TS', 't2', 'dose2', 'FA2']
            pars_list += ['CO', 'GFR', 'Thl', 'Dhl', 'To', 'Eo', 'Toe']
            pars_list += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'vol']
            return pars_list
        if select == 'default_free':
            pars_list = ['CO', 'Thl', 'Dhl', 'To', 'Eo', 'Toe', 'BAT', 'BAT2', 'S02a']  # aorta
            pars_list += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'S02l']  # liver
            return pars_list
        elif select=='first_scan':
            return ['FA', 'TR']
        elif select=='second_scan':
            return ['FA2', 'TR']
        elif select=='aorta_fit':
            return ['CO', 'Thl', 'Dhl', 'To', 'Eo', 'Toe', 'BAT', 'BAT2', 'S02a', 'khe_i', 'khe_f']
        elif select=='liver_fit':
            return ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'S02l']
        elif select=='export':
            pars_list = ['CO', 'Thl', 'Dhl', 'To', 'Eo', 'Toe', 'BAT', 'BAT2', 'S02a', 'S0a']  # aorta
            pars_list += ['Tg', 'Dg', 've', 'khe_i', 'khe_f', 'Th_i', 'Th_f', 'S02l', 'S0l']  # liver
            return pars_list

    
    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        self._t_control = _time(self._pars, visit=0)
        self._t_drug = _time(self._pars, visit=1)

    def _compute_conc_aorta(self):
        self._t_control, self._ca_control = _conc_aorta(self._pars, visit=0)
        self._t_drug, self._ca_drug = _conc_aorta(self._pars, visit=1)

    def _compute_relax_aorta(self):
        self._compute_conc_aorta()
        self._R1a_control = _relax_aorta(self._ca_control, self._pars, visit=0)
        self._R1a_drug = _relax_aorta(self._ca_drug, self._pars, visit=1)

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        self._Sa_control = _signal(self._R1a_control, 'aorta', self._pars, visit=0)
        self._Sa_drug = _signal(self._R1a_drug, 'aorta', self._pars, visit=1)

    def _predict_aorta(self, time):
        self._compute_signal_aorta()
        return _sample_signal(self._Sa_control, self._Sa_drug, self._pars, time)
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        self._Cl_control = _conc_liver(self._ca_control, self._pars, visit=0)
        self._Cl_drug = _conc_liver(self._ca_drug, self._pars, visit=1)

    def _compute_relax_liver(self):
        self._compute_conc_liver()
        self._R1l_control = _relax_liver(self._Cl_control, self._pars, visit=0)
        self._R1l_drug = _relax_liver(self._Cl_drug, self._pars, visit=1)

    def _compute_signal_liver(self):
        self._compute_relax_liver()
        self._Sl_control = _signal(self._R1l_control, 'liver', self._pars, visit=0)
        self._Sl_drug = _signal(self._R1l_drug, 'liver', self._pars, visit=1)

    def _predict_liver(self, time: tuple) -> tuple:
        self._compute_signal_liver()
        return _sample_signal(self._Sl_control, self._Sl_drug, self._pars, time)

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> tuple:
        """Time points in aorta and liver for the two visits"""
        self._set_time()
        return self._t_control, self._t_drug
  
    def conc(self) -> tuple:
        """Concentrations in aorta and liver.

        Returns:
            dict: time points, aorta blood concentrations, liver 
              concentrations.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        return (
            self._t_control, self._ca_control, self._Cl_control, 
            self._t_drug, self._ca_drug, self._Cl_drug
        )
    
    def relax(self) -> tuple:
        """Relaxation rates in aorta and liver.

        Returns:
            dict: time points, aorta blood R1, liver 
              R1.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        return (
            self._t_control, self._R1a_control, self._R1l_control, 
            self._t_drug, self._R1a_drug, self._R1l_drug
        )
    
    def signal(self) -> tuple:
        """Signal in aorta and liver.

        Returns:
            dict: time points, aorta blood signal, liver 
              signal.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        return (
            self._t_control, self._Sa_control, self._Sl_control, 
            self._t_drug, self._Sa_drug, self._Sl_drug
        )
    
    def predict(self, time: tuple) -> tuple:
        """Predict the data at given time points

        Args:
            time (tuple): tuple of 8 arrays with time points. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second stand, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.

        Returns:
            tuple: tuple of 8 arrays with signals corresponding to time.
        """
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for visit in [0, 1]:
            self._pars['tmax'][visit] = max([
                self._pars['tmax'][visit], 
                self._pars['dt'] + np.max(time[4 * visit: 4 * visit + 4]) + ts
            ])
        Sa = self._predict_aorta((time[0], time[1], time[4], time[5]))
        Sl = self._predict_liver((time[2], time[3], time[6], time[7]))
        return Sa[0], Sa[1], Sl[0], Sl[1], Sa[2], Sa[3], Sl[2], Sl[3]

    # ==========================================
    # Inverse Model: Training
    # ==========================================  

    def train(
            self, time: tuple, signal: tuple, free=None, 
            bounds:dict=None, R102a=None, R102l=None, n0=[1, 1], 
            **kwargs,
        ):
        # x,y: 
        # (aorta scan 1 visit 1, aorta scan 2 visit 1, liver scan 1 visit 1, liver scan 2 visit 1)
        # (aorta scan 1 visit 2, aorta scan 2 visit 2, liver scan 1 visit 2, liver scan 2 visit 2)
        """Train the free parameters

        Args:
            time (tuple): tuple of 8 arrays with time points. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second scan, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.
            signal (tuple): tuple of 8 arrays with signals. The first 
              four are from the control visit: aorta in 
              the first scan, aorta in the second scan, liver in the first 
              scan, and liver in the second scan, in that order. 
              The second group of 4 is the same data for the treatment visit.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R102a (list, optional): R1 value in arterial blood before the 
                second injection. If provided this is used to estimate the 
                baseline S0a in the artery. Else this is initialized to S0a. 
                Defaults to None.
            R102l (list, optional): R1 value in liver before the 
                second injection. If provided this is used to estimate the 
                baseline S0l in the liver. Else this is initialized to S0l. 
                Defaults to None.

            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            AortaLiver2scan: A reference to the model instance.
        """
        
        if free is None:
            free_pars = self._pars_list('default_free')
            free = {p: deepcopy(PARAMS[p]['bounds']) for p in free_pars}
            
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
            if p=='BAT': 
                for visit in [0,1]:
                    if (bnds[visit][0] > 0) or (bnds[visit][1] < 0):
                        raise ValueError(f"Bounds on BAT must be (negative, positive).")
            elif p=='BAT2':
                for visit in [0,1]:
                    if (bnds[visit][0] > 0) or (bnds[visit][1] < 0):
                        raise ValueError(f"Bounds on BAT2 must be (negative, positive).")
            elif not isinstance(PARAMS[p]['init'], list):
                if not (bnds[0] <= self._pars[p] <= bnds[1]):
                    raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")
            else:
                for visit in [0,1]:
                    if not (bnds[visit][0] <= self._pars[p][visit] <= bnds[visit][1]):
                        raise ValueError(f"Initial {p} ({self._pars[p][visit]}) for visit {visit} is out of bounds {bnds[visit]}.")

        self._free = free

        # --- 3. Step-wise Training ---
        # 3.1 Initial heuristics for BAT and S0
        self._estimate_parameters(time, signal, n0, R102a, R102l)  
 
        # Train free aorta parameters on aorta data (DISABLE!!)
        pars_aorta = self._pars_list('aorta_fit')
        free_aorta = {p:v for p, v in self._free.items() if p in pars_aorta}
        time_aorta = (time[0], time[1], time[4], time[5])
        signal_aorta = (signal[0], signal[1], signal[4], signal[5])
        _train(self._predict_aorta, time_aorta, signal_aorta, self._pars, free_aorta, **kwargs)

        # Train free liver parameters on liver data (DISABLE!!)
        pars_liver = self._pars_list('liver_fit')
        free_liver = {p:v for p, v in self._free.items() if p in pars_liver}
        time_liver = (time[2], time[3], time[6], time[7])
        signal_liver = (signal[2], signal[3], signal[6], signal[7])
        _train(self._predict_liver, time_liver, signal_liver, self._pars, free_liver, **kwargs)

        # Train all parameters on all data
        self._pcov = _train(self.predict, time, signal, self._pars, self._free, **kwargs)

        return self
        # return {p: v for p, v in self.export_params().items() if p in self._free}
    
    def _estimate_parameters(self, time, signal, n0, R102a, R102l):
        """Heuristic estimation of BAT and signal scaling (S0)."""

        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for visit in [0, 1]:
            self._pars['tmax'][visit] = max([
                self._pars['tmax'][visit], 
                self._pars['dt'] + np.max(time[4 * visit: 4 * visit + 4]) + ts
            ])

            # Estimate BAT and BAT2
            T, D = self._pars['Thl'], self._pars['Dhl']
            self._pars['BAT'][visit] = time[4 * visit][np.argmax(signal[4 * visit])] - (1-D)*T
            if 'BAT' in self._free:
                self._free['BAT'][visit] = [  
                    self._pars['BAT'][visit] + self._free['BAT'][visit][0],
                    self._pars['BAT'][visit] + self._free['BAT'][visit][1],
                ]
            self._pars['BAT2'][visit] = time[1 + 4 * visit][np.argmax(signal[1 + 4 * visit])] - (1-D)*T
            if 'BAT2' in self._free:
                self._free['BAT2'][visit] = [
                    self._pars['BAT2'][visit] + self._free['BAT2'][visit][0],
                    self._pars['BAT2'][visit] + self._free['BAT2'][visit][1],
                ]

            # Estimate S0
            pars = self._pars_dict(select='first_scan')
            Srefb = sig.signal('SS', self._pars['R10a'][visit], 1, **pars)
            Srefl = sig.signal('SS', self._pars['R10l'][visit], 1, **pars)
            self._pars['S0a'][visit] = np.mean(signal[0 + 4 * visit][:n0[visit]]) / Srefb
            if 'S0a' in self._free:
                self._free['S0a'][visit] = [
                    self._pars['S0a'][visit] * self._free['S0a'][visit][0],
                    self._pars['S0a'][visit] * self._free['S0a'][visit][1],
                ]
            self._pars['S0l'][visit] = np.mean(signal[2 + 4 * visit][:n0[visit]]) / Srefl
            if 'S0l' in self._free:
                self._free['S0l'][visit] = [
                    self._pars['S0l'][visit] * self._free['S0l'][visit][0],
                    self._pars['S0l'][visit] * self._free['S0l'][visit][1],
                ]

            # Estimate S02
            pars = self._pars_dict(select='second_scan')
            pars['FA'] = pars.pop('FA2', None)
            if R102a is None:
                self._pars['S02a'][visit] = self._pars['S0a'][visit]
            else:
                Sref2b = sig.signal('SS', R102a[visit], 1, **pars)
                self._pars['S02a'][visit] = np.mean(signal[1 + 4 * visit][:n0[visit]]) / Sref2b
            if 'S02a' in self._free:
                self._free['S02a'][visit] = [
                    self._pars['S02a'][visit] * self._free['S02a'][visit][0],
                    self._pars['S02a'][visit] * self._free['S02a'][visit][1],
                ]
            if R102l is None:
                self._pars['S02l'][visit] = self._pars['S0l'][visit]
            else:
                Sref2l = sig.signal('SS', R102l[visit], 1, **pars)
                self._pars['S02l'][visit] = np.mean(signal[3 + 4 * visit][:n0[visit]]) / Sref2l 
            if 'S02l' in self._free:
                self._free['S02l'][visit] = [
                    self._pars['S02l'][visit] * self._free['S02l'][visit][0],
                    self._pars['S02l'][visit] * self._free['S02l'][visit][1],
                ]

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
            'pars': self._pars,
            'free': self._free,
            'pcov': self._pcov,
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
            
        self._pars = data['pars']
        self._free = data['free']
        self._pcov = data['pcov']
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
        def sdev_effect(v, dv):
            # z = (x-y)/y = x/y-1 
            # dz = sqrt((dy * dz/dy)**2 + (dx * dz/dx)**2)
            # dz/dx = 1/y
            # dz/dy = -x/y**2
            x, y = v[0], v[1]
            dx, dy = dv[0], dv[1]
            dz_dx = 1 / y if y != 0 else 0
            dz_dy = - x / y**2 if y != 0 else 0
            return np.sqrt((dy * dz_dy)**2 + (dx * dz_dx)**2)
                    
        # List parameters for export
        pars_deriv = _derived_params(self._pars)
        export_pars = self._pars_list('export') + list(pars_deriv.keys())

        # Add sdev.
        sdev = None
        if self._pcov is not None:
            sdev = _sdev(self._pcov, self._free)

        all_pars = self._pars | pars_deriv
        pars = {}
        for p in export_pars:

            # Initialize sdev_p
            sdev_p = [0, 0] if isinstance(all_pars[p], list) else 0
            if sdev is not None:
                if p in sdev:
                    sdev_p = sdev[p]

            # Format output for p
            pars[p] = [
                deepcopy(PARAMS[p]['name']), 
                all_pars[p], 
                deepcopy(PARAMS[p]['unit']), 
                sdev_p,
            ]

            if isinstance(all_pars[p], list):
                # Add effect size

                pars[f'{p}_effect'] = [
                    deepcopy(PARAMS[p]['name']), 
                    100 * (all_pars[p][1] - all_pars[p][0]) / all_pars[p][0], 
                    '%', 
                    sdev_effect(all_pars[p], sdev_p)
                ]

        return pars
    
    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        for p, v in pars.items():
            name, unit = v[0], v[2]
            if round_to is None:
                val = v[1]
                err = v[3]
            elif np.isscalar(v[1]):
                val = round(v[1], round_to)
                err = round(v[3], round_to)
                print(f"{name} ({p}): {val} ({err}) {unit}")
            else:
                val = [round(v[1][i], round_to) for i in [0,1]]
                err = [round(v[3][i], round_to) for i in [0,1]] 
                print(f"{name} ({p}): [{val[0]}, {val[1]}] ({err[0]}, {err[1]}) {unit}")                  

    def _pars_dict(self, *args, select=None):
        """Return the parameter values"""
        if len(args) == 0:
            pars = deepcopy(self._pars)
        else:
            pars = {k: v for k, v in self._pars.items() if k in list(args)}
        if select is not None:
            pars = {k: v for k, v in pars.items() if k in self._pars_list(select)}
        return pars
            
    def pars(self, *args, as_dict=False):
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
            ref (tuple, optional): Tuple of optional test data in the form 
              (x,y), where x is an array with x-values and y is an array with 
              y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        for visit in [0, 1]:
            self._pars['tmax'][visit] = max([
                self._pars['tmax'][visit], 
                self._pars['dt'] + np.max(time[4 * visit: 4 * visit + 4]) + ts
            ])
        
        self.signal()
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('First visit')
        ax2.set_title('Second visit')
        ax3.set_title('First visit')
        ax4.set_title('Second visit')

        _plot_data2scan(self._t_control, self._Sa_control, time[0:2], signal[0:2],
                        ax1, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_data2scan(self._t_drug, self._Sa_drug, time[4:6], signal[4:6],
                        ax2, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_conc_aorta(self._t_control, self._ca_control, ax3, xlim)
        _plot_conc_aorta(self._t_drug, self._ca_drug, ax4, xlim)

        _plot_data2scan(self._t_control, self._Sl_control, time[2:4], signal[2:4],
                        ax5, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_data2scan(self._t_drug, self._Sl_drug, time[6:8], signal[6:8],
                        ax6, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_conc_liver(self._t_control, self._Cl_control, ax7, xlim)
        _plot_conc_liver(self._t_drug, self._Cl_drug, ax8, xlim)

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    # def to_dmr(self, file=None, subject='Subject', study='drug_effect'):
    #     dmr = {'data': {}, 'pars': {}, 'sdev':{}}
    #     pars = self.export_params()
    #     for p in pars:
    #         if np.isscalar(pars[p][1]):
    #             dmr['data'][p] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1], str) else 'float']
    #             dmr['pars'][(subject, study, p)] = pars[p][1]
    #             dmr['sdev'][(subject, study, p)] = pars[p][3]
    #         else:
    #             dmr['data'][f'{p}_control'] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1][0], str) else 'float']
    #             dmr['pars'][(subject, study, p)] = pars[p][1][0]
    #             dmr['sdev'][(subject, study, p)] = pars[p][3][0]   
    #             dmr['data'][f'{p}_drug'] = [pars[p][0], pars[p][2], 'str' if isinstance(pars[p][1][1], str) else 'float']
    #             dmr['pars'][(subject, study, p)] = pars[p][1][1]
    #             dmr['sdev'][(subject, study, p)] = pars[p][3][1]  
    #     if file is not None:             
    #         pydmr.write(file, dmr)
    #     return dmr
    


def _derived_params(pars):
    # Compute derived parameters
    vh = 1 - pars['ve'] / (1 - pars['H'])
    khe = [np.mean([pars['khe_i'][i], pars['khe_f'][i]]) for i in [0,1]]
    Th = [np.mean([pars['Th_i'][i], pars['Th_f'][i]]) for i in [0,1]]
    pars_deriv = {
        'khe': khe,
        'Th': Th,
        'vh': vh,
        'kbh_i': [_div(vh, pars['Th_i'][i]) for i in [0,1]],
        'kbh_f': [_div(vh, pars['Th_f'][i]) for i in [0,1]],
        'kbh': [_div(vh, Th[i]) for i in [0,1]],
        'Khe': [_div(khe[i], pars['ve']) for i in [0,1]],
        'Kbh': [_div(1, Th[i]) for i in [0,1]], # Needs integration over t
        'Kbh_i': [_div(1, pars['Th_i'][i]) for i in [0,1]],
        'Kbh_f': [_div(1, pars['Th_f'][i]) for i in [0,1]],
        'CL_i': [pars['khe_i'][i] * pars['vol'][i] for i in [0,1]],
        'CL_f': [pars['khe_f'][i] * pars['vol'][i] for i in [0,1]],
    }
    pars_deriv['CL'] = [np.mean([pars_deriv['CL_i'][i], pars_deriv['CL_f'][i]]) for i in [0,1]]
    pars_deriv['Eb_i'] = [_div(pars_deriv['CL_i'][i], pars_deriv['CL_i'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    pars_deriv['Eb_f'] = [_div(pars_deriv['CL_f'][i], pars_deriv['CL_f'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    # Needs integration
    pars_deriv['Eb'] = [_div(pars_deriv['CL'][i], pars_deriv['CL'][i] + pars['CO'] * (1 - pars['H'])) for i in [0,1]]
    return pars_deriv


def _interp(ki, kf, t):
    #slope = (kf - ki) / (6 * 60 * 60)
    slope = (kf - ki) / t.max()
    return ki + t * slope


def _time(pars, visit=None):
    t = np.arange(0, pars['tmax'][visit], pars['dt'])
    return t

def _conc_aorta(pars, visit=None):

    t = _time(pars, visit)
    conc = lib.ca_conc(pars['agent'])

    # Derive Eb
    khe = _interp(pars['khe_i'][visit], pars['khe_f'][visit], t)
    CL = khe * pars['vol'][visit] + pars['GFR']
    Eb = CL / (CL + pars['CO'] * (1 - pars['H']))
    
    # Compute flux
    J1 = lib.ca_injection(
        t, pars['weight'], conc, pars['dose'][visit], 
        pars['rate'], pars['BAT'][visit],
    )
    J2 = lib.ca_injection(
        t, pars['weight'], conc, pars['dose2'][visit], 
        pars['rate'], pars['BAT2'][visit],
    )
    Jb = pk_aorta.flux_aorta(
        J1 + J2, E=Eb, dt=pars['dt'], 
        tol=pars['dose_tolerance'],
        heartlung=['pfcomp', (pars['Thl'], pars['Dhl'])],
        organs=['2cxm', ([pars['To'], pars['Toe']], pars['Eo'])],
    )
    return t, Jb/pars['CO']


def _conc_liver(cb, pars, visit=None):

    t = _time(pars, visit)
    
    # Determine khe, Kbh over the visit duration
    khe = _interp(pars['khe_i'][visit], pars['khe_f'][visit], t)
    Th = _interp(pars['Th_i'][visit], pars['Th_f'][visit], t)
    
    # Compute concentrations
    cp = cb / (1 - pars['H'])
    cp = pk.flux_pfcomp(cp, pars['Tg'], pars['Dg'], dt=pars['dt'])  
    Ce = pars['ve'] * cp
    Ch = pk.conc(khe * cp, Th, dt=pars['dt'], model="nscomp")

    # Return results
    return np.stack((Ce, Ch))


def _relax_aorta(ca, pars, visit=None):
    rb = lib.relaxivity(pars['field_strength'], 'blood', pars['agent'])
    R1a = pars['R10a'][visit] + rb * ca
    return R1a

def _relax_liver(Cl, pars, visit=None):
    rp = lib.relaxivity(pars['field_strength'], 'plasma', pars['agent'])
    rh = lib.relaxivity(pars['field_strength'], 'hepatocytes', pars['agent'])
    R1l = pars['R10l'][visit] + rp * Cl[0, :] + rh * Cl[1, :]
    return R1l

def _signal(R1, roi, pars, visit=None):
    
    t = np.arange(0, pars['tmax'][visit], pars['dt'])
    t1 = t <= pars['t2'][visit]
    t2 = t > pars['t2'][visit]

    S = np.zeros(t.size)
    S[t1] = sig.signal('SS', R1[t1], pars[f'S0{roi[0]}'][visit], FA=pars['FA'], TR=pars['TR'])
    S[t2] = sig.signal('SS', R1[t2], pars[f'S02{roi[0]}'][visit], FA=pars['FA2'], TR=pars['TR'])

    return S


def _sample_signal(S_control, S_drug, pars, time) -> tuple:

    t_control = np.arange(0, pars['tmax'][0], pars['dt'])
    t_drug = np.arange(0, pars['tmax'][1], pars['dt'])
    return (
        utils.sample(time[0], t_control, S_control, pars['TS']),
        utils.sample(time[1], t_control, S_control, pars['TS']),
        utils.sample(time[2], t_drug, S_drug, pars['TS']),
        utils.sample(time[3], t_drug, S_drug, pars['TS']),
    )


        

def _train(predict, time, signal, pars, free, **kwargs): 

    if free == {}:
        return 

    if isinstance(signal, tuple):
        signal = np.concatenate(signal)

    p0 = _compute_normalized_pars(pars, free)
 
    def predict_normalized(_, *normalized_pars):
        _update_original_pars(pars, normalized_pars, free)
        ypred = predict(time)
        return np.concatenate(ypred) if isinstance(ypred, tuple) else ypred

    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, signal, p0, bounds=(0, 1), **kwargs,
        )
        pcov = pcov.tolist()
    except RuntimeError as e:
        warnings.warn(f"Curve fit failed: {e}. Using initial values.")
        fitted_pars, pcov = p0, None

    _update_original_pars(pars, fitted_pars, free)
    return pcov

def _normalize(v, bounds):
    return (v - bounds[0]) / (bounds[1] - bounds[0])

def _renormalize(v, bounds):
    return v * (bounds[1] - bounds[0]) + bounds[0]

def _compute_normalized_pars(original_pars, free_pars):
    p0 = []
    for p in free_pars:
        if np.isscalar(original_pars[p]):
            p0.append(_normalize(original_pars[p], free_pars[p]))
        else:
            p0.append(_normalize(original_pars[p][0], free_pars[p][0]))
            p0.append(_normalize(original_pars[p][1], free_pars[p][1]))
    return p0

def _update_original_pars(original_pars, normalized_pars, free):
    i = 0
    for p in free:
        if np.isscalar(original_pars[p]):
            original_pars[p] = _renormalize(normalized_pars[i], free[p])
            i += 1
        else:
            original_pars[p][0] = _renormalize(normalized_pars[i], free[p][0])
            original_pars[p][1] = _renormalize(normalized_pars[i+1], free[p][1])
            i += 2

def _sdev(pcov: np.ndarray, free: dict):
    sdev = {}
    pcov = np.array(pcov)
    i = 0
    for p in free.keys():
        if np.isscalar(free[p][0]):
            sdev[p] = _renormalize(np.sqrt(pcov[i, i]), free[p])
            i += 1
        else:
            sdev[p] = [
                _renormalize(np.sqrt(pcov[i, i]), free[p][0]),
                _renormalize(np.sqrt(pcov[i+1, i+1]), free[p][1]),
            ]
            i += 2

    return sdev



# Helper functions for plotting

def _plot_conc_aorta(t: np.ndarray, cb: np.ndarray, ax, xlim=None):
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
    ax.plot(t/60, 1000*C[0, :], linestyle='-.',
            color=color, linewidth=2.0, label='Extracellular')
    ax.plot(t/60, 1000*C[1, :], linestyle='--',
            color=color, linewidth=2.0, label='Hepatocytes')
    ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-',
            color=color, linewidth=2.0, label='Tissue')       
    ax.legend()

def _plot_data2scan(t: tuple[np.ndarray, np.ndarray], 
                    sig: tuple[np.ndarray, np.ndarray],
                    time: tuple[np.ndarray, np.ndarray], 
                    signal: tuple[np.ndarray, np.ndarray],
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



def _div(a, b):
    with np.errstate(divide='ignore'):
        return np.divide(a, b)

