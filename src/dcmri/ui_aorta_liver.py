import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver


# --- Global Parameter Definitions ---


PARAMS = liver.PARAMS_LIVER | {
    # --- Experimental Setup ---
    'field_strength': {'init': 3.0, 'name': 'Magnetic field strength', 'unit': 'T'},

    # --- Simulation Constants ---
    'dt': {'init': 0.5, 'name': 'Forward model time step', 'unit': 'sec'},
    'tmax': {'init': 4 * 60 * 60, 'name': 'Maximum acquisition time', 'unit': 'sec'},
    'dose_tolerance': {'init': 0.1, 'name': 'Dose tolerance', 'unit': ''},
    
    # --- Injection & Contrast Agent ---
    'weight': {'init': 70.0, 'name': 'Subject weight', 'unit': 'kg'},
    'agent': {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},
    'dose': {'init': 0.05, 'name': 'First contrast agent dose', 'unit': 'mL/kg'},
    'rate': {'init': 1, 'name': 'Contrast agent injection rate', 'unit': 'mL/sec'},
    'BAT': {'init': 120, 'bounds': [-60, 60], 'name': 'First bolus arrival time', 'unit': 'sec'},

    # --- Physiological & Pharmacokinetic ---
    'H': {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},
    'CO': {'init': 100, 'bounds': [0, 300], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'T(hl)': {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'D(hl)': {'init': 0.2, 'bounds': [0.05, 0.95], 'name': 'Heart-lung dispersion', 'unit': ''},
    'T(o)': {'init': 20, 'bounds': [0, 60], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'E(o)': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'T(o,e)': {'init': 120, 'bounds': [0, 800], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'E(b)': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Body extraction fraction', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'FA': {'init': 15.0, 'bounds': [0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'TC': {'init': 0.180, 'name': 'Time to center', 'unit': 'sec'},
    'TS': {'init': 2.0, 'name': 'Sampling time', 'unit': 'sec'},

    # --- Baseline Relaxation & Scaling ---
    'R10(a)': {'init': 1/1.5, 'bounds': [0, 5], 'name': 'Aorta first baseline R1', 'unit': 'Hz'},
    'R10(l)': {'init': 1/0.8, 'bounds': [0, 5], 'name': 'Liver first baseline R1', 'unit': 'Hz'},
    'S0(a)': {'init': 1, 'bounds': [0, 5], 'name': 'Aorta first signal scale factor', 'unit': 'a.u.'},
    'S0(l)': {'init': 2, 'bounds': [0, 5], 'name': 'Liver first signal scale factor', 'unit': 'a.u.'},
}

class AortaLiver:
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
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
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

        Use `fake_tissue` to generate synthetic test data from 
        experimentally-derived concentrations:

        Use `fake_liver` to generate synthetic test data:

        >>> time, aif, vif, roi, gt = dc.fake_liver()

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

    Notes:

        Table :ref:`AortaLiver-parameters` lists the parameters that are 
        relevant in each regime. Table :ref:`AortaLiver-defaults` list all 
        possible parameters and their default settings. 

        .. _AortaLiver-parameters:
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
              - Stopping criterion on whole-body model
            * - field_strength, weight, agent, dose, rate
              - Always
              - Injection protocol
            * - R10a, R10l, S0a, S0l 
              - Always
              - Precontrast R1 (:ref:`relaxation-params`) and 
                S0 (:ref:`params-per-sequence`)for aorta and liver 
            * - FA, TR, TS
              - Always
              - :ref:`params-per-sequence`
            * - TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - BAT, CO, T(hl), D(hl), To, Eo, Tie, Eb
              - Always
              - :ref:`whole-body-tissues`
            * - H, ve, De
              - Always
              - :ref:`table-liver-models`
            * - khe, khe_i, kh_f, Th, Th_i, Th_f
              - Depends on **stationary**
              - :ref:`table-liver-models`

        .. _AortaLiver-defaults:
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
            * - T(hl)
              - Whole body
              - 10
              - [0, 30]
              - Free
            * - D(hl)
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
        kinetics='1I-IC-HFD', 
        non_stationary=None, 
        sequence='SS', 
        **params,
    ):
        self._version = '1.0'
        
        # Set Configuration
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics/stationarity: {e}") from e
        if sequence not in ['SS', 'SR']:
            raise ValueError(f"Sequence '{sequence}' is not available.")
        if not kinetics.startswith('1'):
            raise ValueError('Only single-inlet models are allowed.')

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
        aorta_kinetics = ['BAT', 'CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(o)', 'T(o,e)', 'E(b)']
        liver_kinetics = list(liver.params_liver(self._kinetics, self._non_stationary).keys())
        if select is None:
            pars_list = ['dt', 'tmax', 'dose_tolerance', 'field_strength']
            pars_list += ['weight', 'agent', 'dose', 'rate']
            pars_list += ['R10(a)', 'R10(l)', 'S0(a)', 'S0(l)']
            pars_list += ['TS'] + {'SR': ['FA', 'TR', 'TC'], 'SS': ['FA', 'TR']}[self._sequence]
            pars_list += ['H', 'vol'] + aorta_kinetics + liver_kinetics
        elif select=='free':
            pars_list = aorta_kinetics + liver_kinetics
        elif select=='export':
            pars_list = ['S0(a)', 'S0(l)'] + aorta_kinetics
        elif select=='sequence':
            pars_list = {'SR': ['FA', 'TR', 'TC'], 'SS': ['FA', 'TR']}[self._sequence]
        elif select=='liver':
            pars_list = liver_kinetics
        elif select=='aorta':
            pars_list = aorta_kinetics
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
        Ji = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self._pars['E(b)'], dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
            heartlung=['pfcomp', (self._pars['T(hl)'], self._pars['D(hl)'])],
            organs=organs
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
        pars = self._pars_dict(select='sequence')
        self._Sa = sig.signal(self._sequence, self._R1a, self._pars['S0(a)'], **pars)

    def _predict_aorta(self, time):
        """Sample aorta signal at specific time points."""
        self._compute_signal_aorta()
        return utils.sample(time, self._t, self._Sa, self._pars['TS'])
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        """Calculate tissue concentration in the liver."""
        pars = self._pars_dict(select='liver')
        
        # Account for hematocrit to get plasma concentration
        cp = self._ca / (1 - self._pars['H'])
        self._Cl = liver.conc_liver(
            cp, dt=self._pars['dt'], sum=False, 
            kinetics=self._kinetics, 
            non_stationary=self._non_stationary, **pars
        )
        
    def _compute_relax_liver(self):
        """Calculate longitudinal relaxation rate in the liver."""
        self._compute_conc_liver()
        rp = lib.relaxivity(self._pars['field_strength'], 'plasma', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])

        if self._Cl.ndim == 2:
            self._R1l = self._pars['R10(l)'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10(l)'] + rp * self._Cl

    def _compute_signal_liver(self):
        """Calculate MRI signal in the liver."""
        self._compute_relax_liver()
        pars = self._pars_dict(select='sequence')
        self._Sl = sig.signal(self._sequence, self._R1l, self._pars['S0(l)'], **pars)

    def _predict_liver(self, time):
        """Sample liver signal at specific time points."""
        self._compute_signal_liver()
        return utils.sample(time, self._t, self._Sl, self._pars['TS'])
    
    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> tuple:
        """Internal time array

        Returns:
            tuple: (aorta_time, liver_time)        
        """
        self._set_time()
        return self._t, self._t

    def conc(self) -> tuple:
        """Return concentrations in aorta and liver.

        Returns:
            tuple: (aorta_blood_conc, liver_tissue_conc)
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        return self._ca, self._Cl

    def relax(self) -> tuple:
        """Return relaxation rates in aorta and liver.

        Returns:
            tuple: (aorta_R1, liver_R1)
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        return self._R1a, self._R1l
    
    def signal(self) -> tuple:
        """Return signals in aorta and liver.

        Returns:
            tuple: (time, aorta_signal, liver_signal)
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        return self._Sa, self._Sl

    def predict(self, time: tuple=None) -> tuple:
        """Predict the signals at given time time points.

        Args:
            time (tuple): Tuple of (time_aorta, time_liver) arrays.

        Returns:
            tuple: Tuple of (signal_aorta, signal_liver) arrays.
        """
        if time is None:
            time = self.time()
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts

        Sa = self._predict_aorta(time[0])
        Sl = self._predict_liver(time[1])
        return Sa, Sl
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def train(
        self, time: tuple, signal: tuple, free: dict=None, 
        bounds: dict=None, n0=1, **kwargs
    ):
        """Train the model free parameters.

        Args:
            time (tuple): (time_aorta, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points for S0 estimation.
            **kwargs: Arguments passed to scipy.optimize.curve_fit.

        Returns:
            AortaLiver: The trained model instance.
        """
        # Initial heuristics for BAT and S0
        self._estimate_parameters(time, signal, n0)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds) 

        # Optimize Aorta parameters
        free_aorta = {p:v for p, v in free.items() if p in self._pars_list('aorta')}
        utils.train(self._predict_aorta, time[0], signal[0], self._pars, free_aorta, **kwargs)

        # Optimize Liver parameters
        free_liver = {p:v for p, v in free.items() if p in self._pars_list('liver')}
        utils.train(self._predict_liver, time[1], signal[1], self._pars, free_liver, **kwargs)

        # Joint Optimization
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

        # 1. Bolus Arrival Time
        T, D = self._pars['T(hl)'], self._pars['D(hl)']
        self._pars['BAT'] = time[0][np.argmax(signal[0])] - (1-D)*T
        self._pars['BAT'] = max([self._pars['BAT'], 0])
        
        # 2. Scaling Factors (S0)
        pars = self._pars_dict(select='sequence')
        Srefb = sig.signal(self._sequence, self._pars['R10(a)'], 1, **pars)
        Srefl = sig.signal(self._sequence, self._pars['R10(l)'], 1, **pars)
        self._pars['S0(a)'] = np.mean(signal[0][:n0]) / Srefb
        self._pars['S0(l)'] = np.mean(signal[1][:n0]) / Srefl

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
            elif p=='BAT':
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Bounds on BAT must be (negative, positive).")
            elif p in ['S0(a)', 'S0(l)']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds
        # Additive
        for par in ['BAT']:
            if par in free:
                free[par] = [  
                    self._pars[par] + free[par][0],
                    self._pars[par] + free[par][1],
                ]
        # Multiplicative
        for par in ['S0(a)', 'S0(l)']:
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
    
    def export_params(self) -> dict:
        """Export model parameters with metadata (name, unit, sdev)."""
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
    
    def cost(self, time: tuple, signal: tuple, metric='NRMS') -> float:
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
    
    def plot(self, time: tuple, signal: tuple, xlim=None, 
             fname=None, show=True):
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
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(np.concatenate(time)) + ts
        
        self.signal()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot Aorta Data and Conc
        _plot_data1scan(self._t, self._Sa, time[0], signal[0], ax1, xlim,
                        color=['lightcoral', 'darkred'])
        _plot_conc_aorta(self._t, self._ca, ax2, xlim)
        
        # Plot Liver Data and Conc
        _plot_data1scan(self._t, self._Sl, time[1], signal[1], ax3, xlim,
                        color=['cornflowerblue', 'darkblue'])
        _plot_conc_liver(self._t, self._Cl, ax4, xlim)
        
        if fname:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


# ==========================================
# Private Plotting Helpers
# ==========================================

def _plot_conc_aorta(t, cb, ax, xlim=None):
    if xlim is None: xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    ax.plot(t/60, 1000*cb, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
    ax.legend()

def _plot_conc_liver(t, C, ax, xlim=None):
    color = 'darkblue'
    if xlim is None: xlim = [t[0], t[-1]]
    ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xlim)/60)
    ax.plot(t/60, 0*t, color='gray')
    if C.ndim == 2:
        ax.plot(t/60, 1000*C[0, :], linestyle='-.', color=color, linewidth=2.0, label='Extracellular')
        ax.plot(t/60, 1000*C[1, :], linestyle='--', color=color, linewidth=2.0, label='Hepatocytes')
        ax.plot(t/60, 1000*(C[0, :]+C[1, :]), linestyle='-', color=color, linewidth=2.0, label='Tissue')
    else:
        ax.plot(t/60, 1000*C, linestyle='-', color=color, linewidth=2.0, label='Tissue')        
    ax.legend()

def _plot_data1scan(t, sig, time, signal, ax, xlim, color=['black', 'black']):
    """Helper to plot fitted signal vs experimental data for a single scan"""
    if xlim is None:
        xlim = [t[0], t[-1]]
    
    # Configure axes (convert seconds to minutes for display)
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
    
    # Plot experimental data points
    ax.plot(time/60, signal, marker='o', color=color[0], 
            label='fitted data', linestyle='None')
    
    # Plot continuous fit line
    ax.plot(t/60, sig, linestyle='-', color=color[1], 
            linewidth=3.0, label='fit')
    
    ax.legend()