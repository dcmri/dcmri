import warnings
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.liver as liver

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

        >>> xdata, ydata = (time,time), (aif,roi)

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

        >>> model.train(xdata, ydata, n0=10, xtol=1e-3)

        Plot the reconstructed signals and concentrations and compare 
        against the experimentally derived data:

        >>> model.plot(xdata, ydata)

        We can also have a look at the model parameters after training:

        >>> model.print_params(round_to=3)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        First bolus arrival time (BAT): 13.231 (0.266) sec
        Cardiac output (CO): 102.893 (4.182) mL/sec
        Heart-lung mean transit time (Thl): 16.285 (0.409) sec
        Heart-lung dispersion (Dhl): 0.324 (0.016)
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
            * - BAT, CO, Thl, Dhl, To, Eo, Tie, Eb
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
        kinetics='1I-IC-HFD', 
        non_stationary=None, 
        sequence='SS', 
        **params,
    ):
        self._version = '1.0'
        
        # --- 1. Configuration Validation ---
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics/stationarity: {e}") from e
            
        if sequence not in ['SS', 'SR']:
            raise ValueError(f"Sequence '{sequence}' is not available.")
            
        if not kinetics.startswith('1'):
            raise ValueError('Only single-inlet models are allowed.')

        # --- 2. State Initialization ---
        self._kinetics = kinetics
        self._sequence = sequence 
        self._non_stationary = non_stationary
        self._free = None
        self._pcov = None

        # --- 3. Parameter List Construction ---
        # Combine aorta, signal, liver, and sequence-specific parameters
        pars_list = list((PARAMS_COMMON | PARAMS_SIGNAL | PARAMS_AORTA).keys())
        pars_list += _sequence_pars(self._sequence)
        pars_list += liver.params_liver(self._kinetics, self._non_stationary)
        pars_list += ['TS', 'vol'] 

        # --- 4. Parameter Value Initialization ---
        defaults = (PARAMS_COMMON | PARAMS_SIGNAL | PARAMS_AORTA | 
                    PARAMS_SEQUENCE | liver.PARAMS_LIVER)
        self._pars = {p: defaults[p]['init'] for p in pars_list}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in pars_list:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _compute_conc_aorta(self):
        """Internal: Calculate blood concentration in the aorta."""
        organs = ['2cxm', ([self._pars['To'], self._pars['Toe']], self._pars['Eo'])]
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])
        
        conc = lib.ca_conc(self._pars['agent'])
        Ji = lib.ca_injection(
            self._t, self._pars['weight'], conc, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self._pars['Eb'], dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
            heartlung=['pfcomp', (self._pars['Thl'], self._pars['Dhl'])],
            organs=organs
        )
        self._ca = Jb / self._pars['CO']

    def _compute_relax_aorta(self):  
        """Internal: Calculate longitudinal relaxation rate in the aorta."""
        self._compute_conc_aorta()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1a = self._pars['R10a'] + rb * self._ca

    def _compute_signal_aorta(self):
        """Internal: Calculate MRI signal in the aorta."""
        self._compute_relax_aorta()
        pars = {k: v for k, v in self._pars.items() if k in _sequence_pars(self._sequence)}
        self._Sa = sig.signal(self._sequence, self._R1a, self._pars['S0a'], **pars)

    def _predict_aorta(self, xdata):
        """Internal: Sample aorta signal at specific time points."""
        tmax = max(xdata)
        if self._pars['tmax'] < tmax:
            raise ValueError(f"xdata exceeds tmax ({self._pars['tmax']}). Increase tmax.")
        self._compute_signal_aorta()
        return utils.sample(xdata, self._t, self._Sa, self._pars['TS'])
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        """Internal: Calculate tissue concentration in the liver."""
        pars_keys = liver.params_liver(self._kinetics, self._non_stationary)
        pars = {p: self._pars[p] for p in pars_keys}
        
        # Account for hematocrit to get plasma concentration
        cp = self._ca / (1 - self._pars['H'])
        self._Cl = liver.conc_liver(
            cp, dt=self._pars['dt'], sum=False, 
            kinetics=self._kinetics, non_stationary=self._non_stationary, **pars
        )
        
    def _compute_relax_liver(self):
        """Internal: Calculate longitudinal relaxation rate in the liver."""
        self._compute_conc_liver()
        rp = lib.relaxivity(self._pars['field_strength'], 'plasma', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])

        if self._Cl.ndim == 2:
            # Multi-compartment liver model
            self._R1l = self._pars['R10l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10l'] + rp * self._Cl

    def _compute_signal_liver(self):
        """Internal: Calculate MRI signal in the liver."""
        self._compute_relax_liver()
        pars = {k: v for k, v in self._pars.items() if k in _sequence_pars(self._sequence)}
        self._Sl = sig.signal(self._sequence, self._R1l, self._pars['S0l'], **pars)

    def _predict_liver(self, xdata: np.ndarray) -> np.ndarray:
        """Internal: Sample liver signal at specific time points."""
        tmax = max(xdata)
        if self._pars['tmax'] < tmax:
            raise ValueError(f"xdata exceeds tmax ({self._pars['tmax']}). Increase tmax.")
        self._compute_signal_liver()
        return utils.sample(xdata, self._t, self._Sl, self._pars['TS'])
    
    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def conc(self) -> tuple:
        """Return concentrations in aorta and liver.

        Returns:
            tuple: (time, aorta_blood_conc, liver_tissue_conc)
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        return self._t, self._ca, self._Cl

    def relax(self) -> tuple:
        """Return relaxation rates in aorta and liver.

        Returns:
            tuple: (time, aorta_R1, liver_R1)
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        return self._t, self._R1a, self._R1l
    
    def signal(self) -> tuple:
        """Return signals in aorta and liver.

        Returns:
            tuple: (time, aorta_signal, liver_signal)
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        return self._t, self._Sa, self._Sl

    def predict(self, xdata: tuple) -> tuple:
        """Predict the signals at given xdata time points.

        Args:
            xdata (tuple): Tuple of (time_aorta, time_liver) arrays.

        Returns:
            tuple: Tuple of (signal_aorta, signal_liver) arrays.
        """
        Sa = self._predict_aorta(xdata[0])
        Sl = self._predict_liver(xdata[1])
        return Sa, Sl
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def train(self, xdata: tuple, ydata: tuple, free: dict=None, bounds:dict=None, n0=1, **kwargs):
        """Train the model free parameters.

        Args:
            xdata (tuple): (time_aorta, time_liver) arrays.
            ydata (tuple): (signal_aorta, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points for S0 estimation.
            **kwargs: Arguments passed to scipy.optimize.curve_fit.

        Returns:
            AortaLiver: The trained model instance.
        """
        # --- 1. Initialize Free Parameters ---
        if free is None:
            defaults = PARAMS_AORTA | liver.PARAMS_LIVER
            free_pars = list(PARAMS_AORTA.keys())
            free_pars += liver.params_liver(self._kinetics, self._non_stationary)
            free = {p: defaults[p]['bounds'] for p in free_pars}
            
            if bounds is not None:
                for p, b in bounds.items():
                    if p not in free:
                        raise ValueError(f"'{p}' is not a free parameter. Use 'free' to define it.")
                    free[p] = b

        # --- 2. Boundary Validation ---
        for p, bnds in free.items():
            if p not in self._pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            if p=='BAT':
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Bounds on BAT must be (negative, positive).")
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")
 
        self._free = free
        
        # --- 3. Step-wise Training ---
        # 3.1 Initial heuristics for BAT and S0
        self._estimate_parameters(xdata, ydata, n0)

        # 3.2 Optimize Aorta parameters
        pars_aorta = list(PARAMS_AORTA.keys())
        free_aorta = {p: v for p, v in self._free.items() if p in pars_aorta}
        if free_aorta != {}:
            _train(self._predict_aorta, xdata[0], ydata[0], self._pars, free_aorta, **kwargs)

        # 3.3 Optimize Liver parameters
        pars_liver = list(liver.PARAMS_LIVER.keys())
        free_liver = {p: v for p, v in self._free.items() if p in pars_liver}
        if free_liver != {}:
            _train(self._predict_liver, xdata[1], ydata[1], self._pars, free_liver, **kwargs)

        # 3.4 Joint Optimization
        self._pcov = _train(self.predict, xdata, ydata, self._pars, self._free, **kwargs)

        return self
    
    def _estimate_parameters(self, xdata: tuple, ydata: tuple, n0: int):
        """Internal: Heuristic estimation of BAT and signal scaling (S0)."""
        # 1. Bolus Arrival Time
        T, D = self._pars['Thl'], self._pars['Dhl']
        self._pars['BAT'] = xdata[0][np.argmax(ydata[0])] - (1-D)*T
        self._pars['BAT'] = max([self._pars['BAT'], 0])
        
        # Shift BAT bounds relative to heuristic
        self._free['BAT'] = [
            self._pars['BAT'] + self._free['BAT'][0],
            self._pars['BAT'] + self._free['BAT'][1],
        ]

        # 2. Scaling Factors (S0)
        s_pars = {k: v for k, v in self._pars.items() if k in _sequence_pars(self._sequence)} 
        Srefb = sig.signal(self._sequence, self._pars['R10a'], 1, **s_pars)
        Srefl = sig.signal(self._sequence, self._pars['R10l'], 1, **s_pars)
        self._pars['S0a'] = np.mean(ydata[0][:n0]) / Srefb
        self._pars['S0l'] = np.mean(ydata[1][:n0]) / Srefl

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
            'free': self._free,
            'pcov': self._pcov,
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
        self._free = data['free']
        self._pcov = data['pcov']
        return self
    
    def export_params(self) -> dict:
        """Export model parameters with metadata (name, unit, sdev)."""
        # Calculate derived kinetic parameters
        all_pars = liver.derived_params_liver(self._pars, self._kinetics, self._pars['H'])
        meta = PARAMS_AORTA | liver.PARAMS_LIVER | PARAMS_SIGNAL

        exported = {}
        for p, val in all_pars.items():
            if p in meta:
                exported[p] = [meta[p]['name'], val, meta[p]['unit'], 0.0]

        # Map standard deviations from covariance matrix
        if self._pcov is not None:
            for i, p in enumerate(self._free.keys()):
                sdev = _renormalize(np.sqrt(np.array(self._pcov)[i, i]), self._free[p])
                if p in exported:
                    exported[p][-1] = sdev
        return exported

    def print_params(self, round_to=None):
        """Print parameters and uncertainties to console."""
        pars = self.export_params()
        for p, (name, val, unit, sdev) in pars.items():
            if round_to is not None and not isinstance(val, str):
                val, sdev = round(val, round_to), round(sdev, round_to)
            print(f"{name} ({p}): {val} (+/- {sdev}) {unit}")

    def params(self, *args, round_to=None):
        """Get specific parameter values."""
        if len(args) == 1:
            val = self._pars[args[0]]
            return round(val, round_to) if round_to else val
        
        subset = {p: self._pars[p] for p in args if p in self._pars}
        if round_to:
            return {p: round(v, round_to) for p, v in subset.items()}
        return subset
    
    def cost(self, xdata: tuple, ydata: tuple, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
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
        ypred = self.predict(xdata)
        if isinstance(ydata, tuple):
            ypred = np.concatenate(ypred)
            ydata = np.concatenate(ydata)
        return utils.loss(ypred, ydata, metric)
    
    def plot(self, xdata: tuple, ydata: tuple, xlim=None, ref=None, fname=None, show=True):
        """Plot the model fit against data

        Args:
            xdata (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            ydata (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
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
        tmax = max([max(x) for x in xdata])
        if self._pars['tmax'] < tmax:
            raise ValueError(f"xdata exceeds tmax ({self._pars['tmax']}). Increase tmax.")
        
        self._compute_signal_aorta()
        self._compute_signal_liver()
        self.conc()

        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot Aorta Data and Conc
        _plot_data1scan(self._t, self._Sa, xdata[0], ydata[0], ax1, xlim,
                        color=['lightcoral', 'darkred'], test=None if ref is None else ref[0])
        _plot_conc_aorta(self._t, self._ca, ax2, xlim)
        
        # Plot Liver Data and Conc
        _plot_data1scan(self._t, self._Sl, xdata[1], ydata[1], ax3, xlim,
                        color=['cornflowerblue', 'darkblue'], test=None if ref is None else ref[1])
        _plot_conc_liver(self._t, self._Cl, ax4, xlim)
        
        if fname:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

# ==========================================
# Private Helper Functions
# ==========================================

def _sequence_pars(sequence):
    """Return parameters associated with specific sequences."""
    return {'SR': ['FA', 'TR', 'TC'], 'SS': ['FA', 'TR']}[sequence]

def _train(predict, xdata, ydata, pars, free, **kwargs):
    """Internal optimization logic using normalized parameter values."""
    if isinstance(ydata, tuple):
        ydata = np.concatenate(ydata)

    p0 = _compute_normalized_pars(pars, free)

    def predict_normalized(_, *normalized_pars):
        _update_original_pars(pars, normalized_pars, free)
        ypred = predict(xdata)
        return np.concatenate(ypred) if isinstance(ypred, tuple) else ypred

    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, None, ydata, p0, bounds=(0, 1), **kwargs
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
    return [_normalize(original_pars[p], free_pars[p]) for p in free_pars]

def _update_original_pars(original_pars, normalized_pars, free):
    for i, p in enumerate(free):
        original_pars[p] = _renormalize(normalized_pars[i], free[p])

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

def _plot_data1scan(t, sig, xdata, ydata, ax, xlim, color=['black', 'black'], test=None):
    """Helper to plot fitted signal vs experimental data for a single scan.

    Args:
        t (np.ndarray): High-resolution time array (seconds).
        sig (np.ndarray): Predicted signal array.
        xdata (np.ndarray): Observed time points (seconds).
        ydata (np.ndarray): Observed signal values.
        ax (matplotlib.axes.Axes): Axis to plot on.
        xlim (list): Plotting limits for the x-axis.
        color (list, optional): Colors for [data_points, fit_line]. Defaults to ['black', 'black'].
        test (tuple, optional): (x, y) test data for validation. Defaults to None.
    """
    if xlim is None:
        xlim = [t[0], t[-1]]
    
    # Configure axes (convert seconds to minutes for display)
    ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xlim)/60)
    
    # Plot experimental data points
    ax.plot(xdata/60, ydata, marker='o', color=color[0], 
            label='fitted data', linestyle='None')
    
    # Plot continuous fit line
    ax.plot(t/60, sig, linestyle='-', color=color[1], 
            linewidth=3.0, label='fit')
    
    # Plot test data if provided
    if test is not None:
        ax.plot(np.array(test[0])/60, test[1], color='black',
                marker='D', linestyle='None', label='Test data')
    
    ax.legend()



# --- Global Parameter Definitions ---

"""Common model execution and physiological parameters."""
PARAMS_COMMON = {
    'dt': {'init': 0.5, 'bounds': [0, 1e9], 'name': 'Forward model time step', 'unit': 'sec'},
    'tmax': {'init': 180, 'bounds': [0, 1e9], 'name': 'Maximum acquisition time', 'unit': 'sec'},
    'dose_tolerance': {'init': 0.1, 'bounds': [0, 1e9], 'name': 'Dose tolerance', 'unit': ''},
    'field_strength': {'init': 3.0, 'bounds': [0, 1e9], 'name': 'Magnetic field strength', 'unit': 'T'},
    'weight': {'init': 70.0, 'bounds': [0, 1e9], 'name': 'Subject weight', 'unit': 'kg'},
    'agent': {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},
    'dose': {'init': 0.025, 'bounds': [0, 1e9], 'name': 'First contrast agent dose', 'unit': 'mL/kg'},
    'rate': {'init': 1, 'bounds': [0, 1e9], 'name': 'Contrast agent injection rate', 'unit': 'mL/sec'},
    'H': {'init': 0.45, 'bounds': [0, 1], 'name': 'Hematocrit', 'unit': ''},
}

"""Parameters for signal scaling and baseline relaxation."""
PARAMS_SIGNAL = {
    'R10a': {'init': 1/1.5, 'bounds': [0, 1e9], 'name': 'Aorta first baseline R1', 'unit': 'Hz'},
    'S0a': {'init': 1, 'bounds': [0, 1e9], 'name': 'Aorta first signal scale factor', 'unit': 'a.u.'},
    'R10l': {'init': 1/0.8, 'bounds': [0, 1e9], 'name': 'Liver first baseline R1', 'unit': 'Hz'},
    'S0l': {'init': 1, 'bounds': [0, 1e9], 'name': 'Liver first signal scale factor', 'unit': 'a.u.'},
}

"""Aorta and systemic circulation kinetic parameters."""
PARAMS_AORTA = {
    'BAT': {'init': 60, 'bounds': [-60, 60], 'name': 'First bolus arrival time', 'unit': 'sec'},
    'CO': {'init': 100, 'bounds': [0, 300], 'name': 'Cardiac output', 'unit': 'mL/sec'},
    'Thl': {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung mean transit time', 'unit': 'sec'},
    'Dhl': {'init': 0.2, 'bounds': [0.05, 0.95], 'name': 'Heart-lung dispersion', 'unit': ''},
    'To': {'init': 20, 'bounds': [0, 60], 'name': 'Organs blood mean transit time', 'unit': 'sec'},
    'Eo': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organs extraction fraction', 'unit': ''},
    'Toe': {'init': 120, 'bounds': [0, 800], 'name': 'Organs extravascular mean transit time', 'unit': 'sec'},
    'Eb': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Body extraction fraction', 'unit': ''},
}

"""Imaging sequence specific parameters."""
PARAMS_SEQUENCE = {
    'TR': {'init': 0.005, 'bounds': [0, 1e9], 'name': 'Repetition time', 'unit': 'sec'},
    'FA': {'init': 15.0, 'bounds': [0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'TC': {'init': 0.180, 'bounds': [0, 1e9], 'name': 'Time to center', 'unit': 'sec'},
    'TS': {'init': None, 'bounds': [0, 1e9], 'name': 'Sampling time', 'unit': 'sec'},
}


