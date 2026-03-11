import json
from copy import deepcopy
from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.liver as liver
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk


# ---- Constants ----
dt_init, tmax_init = 0.5, 180
t_init = np.arange(0, tmax_init, dt_init, dtype=float)
ca_init = pk_aorta.aif_tristan(t_init, agent='gadoxetate', BAT=20)
cv_init = pk.flux_pfcomp(ca_init, 10, 0.5)

PARAMS = liver.PARAMS_LIVER | {
    # --- Input Functions ---
    'c(a)': {'init': ca_init, 'name': 'Arterial blood concentration', 'unit': 'M'},
    'c(v)': {'init': cv_init, 'name': 'Portal venous blood concentration', 'unit': 'M'},

    # --- Simulation Constants ---
    'dt': {'init': dt_init, 'name': 'Forward model time step', 'unit': 'sec'},
    'tmax': {'init': tmax_init, 'name': 'Maximum acquisition time', 'unit': 'sec'},

    # --- Experimental Setup ---
    'field_strength': {'init': 3.0, 'name': 'Field strength', 'unit': 'T'},

    # --- Injection & Contrast Agent ---
    'agent': {'init': 'gadoxetate', 'name': 'Contrast agent', 'unit': None},

    # --- Physiological & Pharmacokinetic ---
    'H': {'init': 0.45, 'name': 'Hematocrit', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'B1corr(v)': {'init': 1, 'bounds': [0, 5], 'name': 'Venous B1-corr', 'unit': ''},
    'B1corr(a)': {'init': 1, 'bounds': [0, 5], 'name': 'Arterial B1-corr', 'unit': ''},
    'B1corr': {'init': 1, 'bounds': [0, 5], 'name': 'Tissue B1-corr', 'unit': ''},
    'FA': {'init': 15, 'bounds': [0, 180], 'name': 'Flip angle', 'unit': 'deg'},
    'TR': {'init': 0.005, 'name': 'Repetition time', 'unit': 'sec'},
    'TC': {'init': 0.2, 'name': 'Time to center', 'unit': 'sec'},
    'TP': {'init': 0.05, 'name': 'Preparation delay', 'unit': 'sec'},
    'TS': {'init': 0, 'name': 'Sampling time', 'unit': 'sec'},

    # --- Baseline Relaxation & Scaling ---
    'R10(a)': {'init': 0.7, 'bounds': [0, 5], 'name': 'Arterial R10', 'unit': 'Hz'},
    'R10(v)': {'init': 0.7, 'bounds': [0, 5], 'name': 'Venous R10', 'unit': 'Hz'},
    'R10': {'init': 0.7, 'bounds': [0, 5], 'name': 'Tissue R10', 'unit': 'Hz'},
    'S0': {'init': 1.0, 'bounds': [0, 5], 'name': 'Signal scaling', 'unit': 'a.u.'},
}


class Liver:
    """General model for liver tissue.

    This is the standard interface for liver tissues with known input 
    function(s). For more detail see :ref:`liver-tissues`.

    Args:
        kinetics (str, optional): Tracer-kinetic model. See table 
          :ref:`table-liver-models` for options. Defaults to '2I-EC'.
        non_stationary (str, optional): For intracellular tracers - stationarity 
          regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
          For more detail see :ref:`liver-tissues`. Defaults to None.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`Liver-parameters` and
          :ref:`Liver-defaults` for a list of parameters and their
          default values.

    See Also:
        `Tissue`

    Example:

        Fit a dual-inlet liver model:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import dcmri as dc

        Use `fake_liver` to generate synthetic test data:

        >>> time, aif, vif, roi, gt = dc.fake_liver()

        Build a tissue model and set the constants to match the experimental 
        conditions of the synthetic test data. Note the default model is the 
        dual-inlet model for extracellular agents (2I-EC). Since the 
        synthetic data are generated with an intracellular agent, the default 
        for the kinetic model needs to be overwritten:

        >>> model = dc.Liver(
        ...     kinetics = '2I-IC',
        ...     t = time,
        ...     agent = 'gadoxetate',
        ...     field_strength = 3.0,
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10 = 1/dc.T1(3.0,'liver'),
        ...     R10a = 1/dc.T1(3.0, 'blood'), 
        ...     R10v = 1/dc.T1(3.0, 'blood'), 
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi, aif, vif, n0=10)

        Plot the reconstructed signals (left) and concentrations (right) and 
        compare the concentrations against the noise-free ground truth. Since 
        the data are analysed with an exact model, and there are no other data 
        errors present, this should fior the data exactly.

        >>> model.plot(time, roi, ref=gt)

    Notes:

        Table :ref:`Liver-parameters` lists the parameters that are relevant 
        in each regime. Table :ref:`Liver-defaults` list all possible 
        parameters and their default settings. 

        .. _Liver-parameters:
        .. list-table:: **Liver parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - field_strength, agent, R10
              - Always
              - :ref:`relaxation-params`
            * - R10a, B1corr_a
              - When aif is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - R10v, B1corr_v
              - When vif is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - S0, FA, TR, TS, B1corr
              - Always
              - :ref:`params-per-sequence`
            * - TP, TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - ve, Fp, fa, Ta, Tg, khe, khe_i, kh_f, Th, Th_i, Th_f.
              - Depends on **kinetics** and **stationary**
              - :ref:`table-liver-models`

        .. _Liver-defaults:
        .. list-table:: **Liver parameter defaults**
            :widths: 5 10 10 10 10
            :header-rows: 1

            * - Parameter
              - Type
              - Value
              - Bounds
              - Free/Fixed
            * - field_strength
              - Injection
              - 3
              - [0, inf]
              - Fixed
            * - agent
              - Injection
              - 'gadoxetate'
              - None
              - Fixed
            * - R10
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - R10a
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - R10v
              - Relaxation
              - 0.7
              - [0, inf]
              - Fixed
            * - B1corr
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - B1corr_a
              - Sequence
              - 1
              - [0, inf]
              - Fixed
            * - B1corr_v
              - Sequence
              - 1
              - [0, inf]
              - Fixed
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
            * - TP
              - Sequence
              - 0
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
            * - Ta
              - Kinetic
              - 2
              - [0, inf]
              - Free
            * - Tg
              - Kinetic
              - 10
              - [0, inf]
              - Free
            * - Fp
              - Kinetic
              - 0.008
              - [0, inf]
              - Free
            * - fa
              - Kinetic
              - 0.2
              - [0, inf]
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
        kinetics = '2I-EC',
        non_stationary = None,
        sequence = 'SS',
        **params,
    ):
        """Initializes the Liver model with configuration and parameters."""
        self._version = '1.0'

        # Validate configuration
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics/stationarity: {e}") from e
        if sequence not in ['SS', 'SR']:
            raise ValueError(f'Sequence {sequence} is not available in Liver().')

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
        # Define parameters based on sequence and kinetics
        liver_kinetics = list(liver.params_liver(self._kinetics, self._non_stationary).keys())
        pars_sequence = {
            'SR': ['S0', 'B1corr', 'FA', 'TR', 'TS', 'TC', 'TP'],
            'SS': ['S0', 'B1corr', 'FA', 'TR', 'TS'],
        }
        if select is None:
            pars_list = ['dt', 'tmax', 'field_strength', 'agent']
            pars_list += ['c(a)', 'R10(a)', 'B1corr(a)', 'R10']
            pars_list += ['H', 'vol'] + liver_kinetics
            pars_list += pars_sequence[self._sequence]
            if self._kinetics.startswith('2'):
                pars_list += ['R10(v)', 'B1corr(v)', 'c(v)']
        elif select=='liver':
            pars_list = liver_kinetics
        elif select=='free':
            pars_list = liver_kinetics
        elif select=='export':
            pars_list = ['S0', 'B1corr'] + liver_kinetics
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================

    def _set_time(self):
        """Build time axis"""
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])

    def _compute_concentration(self):
        """Calculates internal liver concentrations."""
        self._set_time()
    
        hct = self._pars['H']
        ca_plasma = self._pars['c(a)'] / (1 - hct)
        if 'c(v)' in self._pars:
            ca_plasma = (ca_plasma, self._pars['c(v)'] / (1 - hct))

        self._Cl = liver.conc_liver(
            ca_plasma, dt=self._pars['dt'], kinetics=self._kinetics,
            non_stationary=self._non_stationary, sum=False, 
            **self._pars_dict(select='liver'),
        )

    def _compute_relaxation_rate(self):
        """Calculates the longitudinal relaxation rate R1."""
        self._compute_concentration()
        rp = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])
        
        if self._Cl.ndim == 2:
            self._R1l = self._pars['R10'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10'] + rp * self._Cl

    def _compute_signal(self):
        """Calculates the MRI signal based on the sequence."""
        self._compute_relaxation_rate()
        fa_corr = self._pars['B1corr'] * self._pars['FA']
        
        if self._sequence == 'SR':
            self._Sl = sig.signal_spgr(
                self._pars['S0'], self._R1l, self._pars['TC'], 
                self._pars['TR'], fa_corr
            )
        else:
            self._Sl = sig.signal_ss(
                self._pars['S0'], self._R1l, self._pars['TR'], fa_corr
            )

    def _predict(self, time):
        """Predict data at specific time points."""
        self._compute_signal()
        return utils.sample(time, self._t, self._Sl, self._pars['TS'])

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> np.ndarray:
        """Internal time array

        Returns:
            tuple: (aorta_time, liver_time)        
        """
        self._set_time()
        return self._t
       
    def conc(self) -> np.ndarray:
        """Returns time points and liver concentrations."""
        self._compute_concentration()
        return self._Cl

    def relax(self) -> np.ndarray:
        """Returns time points and liver relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return self._R1l

    def signal(self) -> np.ndarray:
        """Returns time points and predicted liver signal."""
        self._compute_signal()
        return self._Sl

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predicts liver signal at specific time points."""
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        if self._pars['tmax'] < self._pars['dt'] + np.max(time) + ts:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {self._pars['tmax']/60} mins.')

        return self._predict(time)
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: np.ndarray=None, vif: np.ndarray=None, 
        free: dict=None, bounds:dict=None, n0=1, **kwargs
    ):
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            aif (array): arterial signal
            vif (array): portal-venous signal
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            bounds (dict, optional): Override default bounds for specific parameters.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            Liver: A reference to the model instance.
        """
        # Initial heuristics for BAT and S0
        self._estimate_parameters(time, signal, aif, vif, n0)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds) 

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
    
    def _estimate_parameters(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: np.ndarray=None, vif: np.ndarray=None, n0=1
    ):
        """Estimates initial baseline parameters from data."""
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        r1 = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])

        # Arterial concentration estimation
        if aif is not None:
            if self._sequence == 'SR':
                ca = sig.conc_src(aif, self._pars['TC'], 1/self._pars['R10(a)'], r1, n0)
            elif self._sequence == 'SS':
                fa_a = self._pars['B1corr(a)'] * self._pars['FA']
                ca = sig.conc_ss(aif, self._pars['TR'], fa_a, 1/self._pars['R10(a)'], r1, n0)
            self._pars['tmax'] = self._pars['dt'] + np.max(time) + ts
            self._set_time()
            self._pars['c(a)'] = np.interp(self._t, time, ca)

        # Venous concentration estimation
        if vif is not None:
            if self._sequence == 'SR':
                cv = sig.conc_src(vif, self._pars['TC'], 1/self._pars['R10(v)'], r1, n0)
            elif self._sequence == 'SS':
                fa_v = self._pars['B1corr(v)'] * self._pars['FA']
                cv = sig.conc_ss(vif, self._pars['TR'], fa_v, 1/self._pars['R10(v)'], r1, n0)
            self._pars['tmax'] = self._pars['dt'] + np.max(time) + ts
            self._set_time()
            self._pars['c(v)'] = np.interp(self._t, time, cv)

        if self._pars['tmax'] < self._pars['dt'] + np.max(time) + ts:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {self._pars['tmax']/60} mins.')

        # Estimate liver S0
        fa_t = self._pars['B1corr'] * self._pars['FA']
        if self._sequence == 'SR':
            s_ref = sig.signal_spgr(1, self._pars['R10'], self._pars['TC'], self._pars['TR'], fa_t)
        else:
            s_ref = sig.signal_ss(1, self._pars['R10'], self._pars['TR'], fa_t)
            
        self._pars['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0


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
            elif p in ['S0']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial {p} ({self._pars[p]}) is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds
        for par in ['S0']:
            if par in free:
                free[par] = [
                    self._pars[par] * free[par][0],
                    self._pars[par] * free[par][1],
                ]

        return free

    # ---- I/O and Reporting ----

    def save(self, file: str):
        """Saves model state to a JSON file."""
        if not file.endswith('.json'):
            file += '.json'

        # Convert arrays to lists
        export_pars = deepcopy(self._pars)
        export_pars['c(a)'] = self._pars['c(a)'].tolist()
        if 'c(v)' in self._pars:
            export_pars['c(v)'] = self._pars['c(v)'].tolist()

        state = {
            'model': self.__class__.__name__,
            'version': self._version,
            'kinetics': self._kinetics,
            'non_stationary': self._non_stationary,
            'sequence': self._sequence,
            'pars': export_pars,
        }

        with open(file, "w") as f:
            json.dump(state, f, indent=4)
        return self

    def load(self, file: str):
        """Loads model state from a JSON file."""
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

        # Convert lists to arrays
        self._pars['c(a)'] = np.array(self._pars['c(a)'])
        if 'c(v)' in self._pars:
            self._pars['c(v)'] = np.array(self._pars['c(v)'])

        return self

    def export_params(self) -> Dict[str, list]:
        """Returns model parameters with descriptions and uncertainties."""
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

    def cost(self, time: np.ndarray, signal: np.ndarray, metric: str = 'NRMS') -> float:
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
        y_pred = self.predict(time)
        return utils.loss(y_pred, signal, metric)

    def plot(self, time, signal, xlim=None, fname=None, show=True):
        """Visualizes predictions vs data."""
        self._compute_signal()
        xlim = xlim or [np.amin(time), np.amax(time)]
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(time/60, signal, 'o', color='cornflowerblue', label='Data')
        ax0.plot(self._t/60, self._Sl, '-', linewidth=3, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        
        cl_total = self._Cl if self._Cl.ndim == 1 else self._Cl.sum(axis=0)
        ax1.plot(self._t/60, 1000*cl_total, '-', linewidth=3, color='darkblue', label='Tissue Pred')
        ax1.plot(self._t/60, 1000*self._pars['c(a)'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        
        if 'c(v)' in self._pars:
            ax1.plot(self._t/60, 1000*self._pars['c(v)'], '-', linewidth=3, color='purple', label='Venous Pred')

        ax1.set(xlabel='Time (min)', ylabel='Conc (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.close()