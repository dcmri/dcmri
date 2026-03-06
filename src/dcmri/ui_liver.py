import warnings
import json
from copy import deepcopy
from typing import Optional, Union, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import dcmri.lib as lib
import dcmri.liver as liver
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta
import dcmri.pk as pk


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
        kinetics: str = '2I-EC',
        non_stationary: Optional[str] = None,
        sequence: str = 'SS',
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

        # Define parameters based on sequence and kinetics
        pars_sequence = {
            'SR': ['S0', 'B1corr', 'FA', 'TR', 'TS', 'TC', 'TP'],
            'SS': ['S0', 'B1corr', 'FA', 'TR', 'TS'],
        }
        pars_list = ['H', 'field_strength', 'agent', 'R10a', 'B1corr_a', 'R10', 'vol']
        pars_list += liver.params_liver(self._kinetics, self._non_stationary)
        pars_list += pars_sequence[self._sequence]
        pars_list += ['t', 'ca']
        if self._kinetics.startswith('2'):
            pars_list += ['R10v', 'B1corr_v', 'cv']

        # Initialize parameters
        defaults = PARAMS | liver.PARAMS_LIVER
        self._pars = {p: defaults[p]['init'] for p in pars_list}

        # Override with user-defined values
        for p, val in params.items():
            if p in pars_list:
                self._pars[p] = val
            else:
                raise ValueError(f"{p} is not a valid parameter for this config.")

        self._free = None
        self._pcov = None

    # ---- Internal forward model methods ----

    def _compute_concentration(self):
        """Calculates internal liver concentrations."""
        hct = self._pars['H']
        ca_plasma = self._pars['ca'] / (1 - hct)
        c_plasma = (ca_plasma, self._pars['cv'] / (1 - hct)) if 'cv' in self._pars else ca_plasma
        
        pars_keys = liver.params_liver(self._kinetics, self._non_stationary)
        pars = {p: self._pars[p] for p in pars_keys}
        
        self._Cl = liver.conc_liver(
            c_plasma, t=self._pars['t'], kinetics=self._kinetics,
            non_stationary=self._non_stationary, sum=False, **pars,
        )

    def _compute_relaxation_rate(self):
        """Calculates the longitudinal relaxation rate R1."""
        self._compute_concentration()
        r1 = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        rh = lib.relaxivity(self._pars['field_strength'], 'hepatocytes', self._pars['agent'])
        
        if self._Cl.ndim == 2:
            self._R1l = self._pars['R10'] + r1 * self._Cl[0, :] + rh * self._Cl[1, :]
        else:
            self._R1l = self._pars['R10'] + r1 * self._Cl

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
        return utils.sample(time, self._pars['t'], self._Sl, self._pars['TS'])

    # ---- Public API ----

    def input(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Returns the input functions (time, arterial, venous)."""
        if 'cv' in self._pars:
            return self._pars['t'], self._pars['ca'], self._pars['cv']
        else:
            return self._pars['t'], self._pars['ca'], None
        
    def conc(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns time points and liver concentrations."""
        self._compute_concentration()
        return self._pars['t'], self._Cl

    def relax(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns time points and liver relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return self._pars['t'], self._R1l

    def signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns time points and predicted liver signal."""
        self._compute_signal()
        return self._pars['t'], self._Sl

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predicts liver signal at specific time points."""
        if np.amax(time) > np.amax(self._pars['t']):
            raise ValueError(
                f"Acquisition window exceeds AIF duration. Max: {np.amax(self._pars['t'])/60:.2f} min."
            )
        return self._predict(time)
    
    # ---- Inverse model API ----

    def train(self, time, signal, aif=None, vif=None, n0=1, free=None, bounds:dict=None, **kwargs):
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
        # Initialize free parameters
        if free is None:
            defaults = PARAMS | liver.PARAMS_LIVER
            free_pars = liver.params_liver(self._kinetics, self._non_stationary)
            free = {p: defaults[p]['bounds'] for p in free_pars}

            if bounds is not None:
                for p, b in bounds.items():
                    if p not in free:
                        raise ValueError(f"'{p}' is not a free parameter. Use 'free' to define it.")
                    free[p] = b

        # Parameter boundary validation
        for p, bnds in free.items():
            if p not in self._pars:
                raise ValueError(f"{p} is not a valid free parameter.")
            if not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial value for {p} ({self._pars[p]}) is out of bounds {bnds}.")

        self._free = free
        self._estimate_parameters(time, signal, aif, vif, n0)
        self._pcov = utils.train(self.predict, time, signal, self._pars, self._free, **kwargs)

        return self
    
    def _estimate_parameters(self, time, signal, aif=None, vif=None, n0=1):
        """Estimates initial baseline parameters from data."""
        r1 = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])

        # Arterial concentration estimation
        if aif is not None:
            self._pars['t'] = time
            if self._sequence == 'SR':
                self._pars['ca'] = sig.conc_src(aif, self._pars['TC'], 1/self._pars['R10a'], r1, n0)
            elif self._sequence == 'SS':
                fa_a = self._pars['B1corr_a'] * self._pars['FA']
                self._pars['ca'] = sig.conc_ss(aif, self._pars['TR'], fa_a, 1/self._pars['R10a'], r1, n0)

        # Venous concentration estimation
        if vif is not None:
            self._pars['t'] = time
            if self._sequence == 'SR':
                self._pars['cv'] = sig.conc_src(vif, self._pars['TC'], 1/self._pars['R10v'], r1, n0)
            elif self._sequence == 'SS':
                fa_v = self._pars['B1corr_v'] * self._pars['FA']
                self._pars['cv'] = sig.conc_ss(vif, self._pars['TR'], fa_v, 1/self._pars['R10v'], r1, n0)

        # Estimate liver S0
        fa_t = self._pars['B1corr'] * self._pars['FA']
        if self._sequence == 'SR':
            s_ref = sig.signal_spgr(1, self._pars['R10'], self._pars['TC'], self._pars['TR'], fa_t)
        else:
            s_ref = sig.signal_ss(1, self._pars['R10'], self._pars['TR'], fa_t)
            
        self._pars['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0

    # ---- I/O and Reporting ----

    def save(self, file: str):
        """Saves model state to a JSON file."""
        if not file.endswith('.json'):
            file += '.json'

        # Convert arrays to lists
        export_pars = deepcopy(self._pars)
        export_pars['t'] = self._pars['t'].tolist()
        export_pars['ca'] = self._pars['ca'].tolist()
        if 'cv' in self._pars:
            export_pars['cv'] = self._pars['cv'].tolist()

        state = {
            'model': self.__class__.__name__,
            'version': self._version,
            'kinetics': self._kinetics,
            'non_stationary': self._non_stationary,
            'sequence': self._sequence,
            'pars': export_pars,
            'free': self._free,
            'pcov': self._pcov,
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
        self._free = data['free']
        self._pcov = data['pcov']

        # Convert lists to arrays
        self._pars['t'] = np.array(self._pars['t'])
        self._pars['ca'] = np.array(self._pars['ca'])
        if 'cv' in self._pars:
            self._pars['cv'] = np.array(self._pars['cv'])

        return self

    def export_params(self) -> Dict[str, list]:
        """Returns model parameters with descriptions and uncertainties."""
        defaults = liver.PARAMS_LIVER
        derived = liver.derived_params_liver(self._pars, self._kinetics, self._pars['H'])
        
        exported = {
            p: [defaults[p]['name'], derived[p], defaults[p]['unit'], 0]
            for p in derived if p in defaults
        }

        # Add standard deviation
        if self._pcov is not None:
            for i, p in enumerate(self._free.keys()):
                sdev = utils.renormalize(np.sqrt(np.array(self._pcov)[i, i]), self._free[p])
                if p in exported:
                    exported[p][-1] = sdev
        return exported

    def print_params(self, round_to: Optional[int] = None):
        """Prints parameters and uncertainties to the console."""
        params_dict = self.export_params()
        for p, v in params_dict.items():
            name, val, unit, err = v[0], v[1], v[2], v[3]
            if round_to is not None:
                val, err = round(val, round_to), round(err, round_to)
            print(f"{name} ({p}): {val} ({err}) {unit}")

    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        if len(args) == 1:
            if round_to is None:
                return self._pars[args[0]]
            else:
                return round(self._pars[args[0]], round_to)
        if round_to is None:
            return {p: v for p, v in self._pars.items() if p in list(args)}
        else:
            return {p: round(v, round_to) for p, v in self._pars.items() if p in list(args)}


    def cost(self, time: np.ndarray, signal: np.ndarray, metric: str = 'NRMS') -> float:
        """Returns goodness-of-fit metric."""
        y_pred = self.predict(time)
        return utils.loss(y_pred, signal, metric)

    def plot(self, time, signal, ref=None, xlim=None, fname=None, show=True):
        """Visualizes predictions vs data."""
        self._compute_signal()
        xlim = xlim or [np.amin(time), np.amax(time)]
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(time/60, signal, 'o', color='cornflowerblue', label='Data')
        ax0.plot(self._pars['t']/60, self._Sl, '-', linewidth=3, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['C'], 'o', color='cornflowerblue', label='Tissue GT')
            ax1.plot(ref['t']/60, 1000*ref['cb'], 'o', color='lightcoral', label='Arterial GT')
        
        cl_total = self._Cl if self._Cl.ndim == 1 else self._Cl.sum(axis=0)
        ax1.plot(self._pars['t']/60, 1000*cl_total, '-', linewidth=3, color='darkblue', label='Tissue Pred')
        ax1.plot(self._pars['t']/60, 1000*self._pars['ca'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        
        if 'cv' in self._pars:
            ax1.plot(self._pars['t']/60, 1000*self._pars['cv'], '-', linewidth=3, color='purple', label='Venous Pred')

        ax1.set(xlabel='Time (min)', ylabel='Conc (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.close()


# ---- Constants ----

t_init = np.arange(120, dtype=float)
ca_init = pk_aorta.aif_tristan(t_init, agent='gadoxetate', BAT=20)
cv_init = pk.flux_pfcomp(ca_init, 10, 0.5)

PARAMS = {
    't': {'init': t_init, 'bounds': None, 'name': 'Time', 'unit': 'sec'},
    'ca': {'init': ca_init, 'bounds': None, 'name': 'Arterial blood concentration', 'unit': 'M'},
    'cv': {'init': cv_init, 'bounds': None, 'name': 'Portal venous blood concentration', 'unit': 'M'},
    'field_strength': {'init': 3.0, 'bounds': [0, 1e9], 'name': 'Field strength', 'unit': 'T'},
    'agent': {'init': 'gadoxetate', 'bounds': None, 'name': 'Contrast agent', 'unit': None},
    'R10a': {'init': 0.7, 'bounds': [0, 1e9], 'name': 'Arterial R10', 'unit': 'Hz'},
    'B1corr_a': {'init': 1, 'bounds': [0, 1e9], 'name': 'Arterial B1-corr', 'unit': ''},
    'R10v': {'init': 0.7, 'bounds': [0, 1e9], 'name': 'Venous R10', 'unit': 'Hz'},
    'B1corr_v': {'init': 1, 'bounds': [0, 1e9], 'name': 'Venous B1-corr', 'unit': ''},
    'B1corr': {'init': 1, 'bounds': [0, 1e9], 'name': 'Tissue B1-corr', 'unit': ''},
    'FA': {'init': 15, 'bounds': [0, 1e9], 'name': 'Flip angle', 'unit': 'deg'},
    'TR': {'init': 0.005, 'bounds': [0, 1e9], 'name': 'Repetition time', 'unit': 'sec'},
    'TC': {'init': 0.2, 'bounds': [0, 1e9], 'name': 'Time to center', 'unit': 'sec'},
    'TP': {'init': 0.05, 'bounds': [0, 1e9], 'name': 'Preparation delay', 'unit': 'sec'},
    'TS': {'init': 0, 'bounds': [0, 1e9], 'name': 'Sampling time', 'unit': 'sec'},
    'R10': {'init': 0.7, 'bounds': [0, 1e9], 'name': 'Tissue R10', 'unit': 'Hz'},
    'S0': {'init': 1.0, 'bounds': [0, 1e9], 'name': 'Signal scaling', 'unit': 'a.u.'},
    'H': {'init': 0.45, 'bounds': [0, 1], 'name': 'Hematocrit', 'unit': ''},
}