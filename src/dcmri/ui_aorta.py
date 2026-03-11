import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta

# ---- Global Parameter Definitions ----

PARAMS = {
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

    # --- Physiological & Pharmacokinetic ---
    'CO': {'init': 100, 'bounds': [0, 500], 'name': 'Cardiac output', 'unit': 'mL/s'},
    'T(hl)': {'init': 10, 'bounds': [0, 30], 'name': 'Heart-lung MTT', 'unit': 's'},
    'D(hl)': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Heart-lung dispersion', 'unit': ''},
    'T(o)': {'init': 20, 'bounds': [0, 60], 'name': 'Organ blood MTT', 'unit': 's'},
    'E(o)': {'init': 0.15, 'bounds': [0, 0.5], 'name': 'Organ extraction', 'unit': ''},
    'T(o,e)': {'init': 120, 'bounds': [0, 800], 'name': 'Organ EES MTT', 'unit': 's'},
    'E(b)': {'init': 0.05, 'bounds': [0.01, 0.15], 'name': 'Body extraction', 'unit': ''},

    # --- MRI Sequence & Signal Parameters ---
    'TR': {'init': 0.005, 'name': 'TR', 'unit': 's'},
    'FA': {'init': 15, 'name': 'Flip angle', 'unit': 'deg'},
    'TC': {'init': 0.2, 'name': 'Time to k-center', 'unit': 's'},
    'TS': {'init': None, 'name': 'Sampling time', 'unit': 's'},
    'TF': {'init': 0.5, 'bounds': [0, 10], 'name': 'Inflow time', 'unit': 's'},

    # --- Baseline Relaxation & Scaling ---
    'R10': {'init': 0.7, 'bounds': [0, 5], 'name': 'Precontrast R1', 'unit': 'Hz'},
    'S0': {'init': 1.0, 'bounds': [0, 5], 'name': 'Signal scaling', 'unit': 'a.u.'},
}

class Aorta:
    """Whole-body model for the aorta signal.

    This model uses a whole-body pharmacokinetic architecture to predict the 
    MRI signal in the aorta by modeling the injection, heart-lung transit, 
    and systemic circulation (see :ref:`whole-body-tissues`). 

    Args:
        organs (str, optional): Model for the systemic organs. 
            Options: 'comp' (1-compartment), '2cxm' (2-compartment exchange). 
            Defaults to 'comp'.
        heartlung (str, optional): Model for the heart-lung system. 
            Options: 'pfcomp' (plug-flow), 'chain'. Defaults to 'pfcomp'.
        sequence (str, optional): Imaging sequence model. 
            Options: 'SS' (steady-state), 'SR' (saturation-recovery), 
            'SSI' (steady state with inflow), 'lin' (linear). 
            Defaults to 'SS'.
        **params: Variable list of model parameters (e.g., CO=100, BAT=20). 
            Defaults are used for any that are not provided. See table 
            :ref:`Aorta-defaults` for a list of parameters and 
            their default values.

    Raises:
        ValueError: If invalid configurations or unknown parameters are provided.

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _Aorta-defaults:
        .. list-table:: Aorta parameters. 
            :widths: 15 10 10 10
            :header-rows: 1

            * - Parameter
              - Value
              - Bounds
              - Usage
            * - **General**
              - 
              - 
              - 
            * - dt
              - 0.25
              - None
              - Always
            * - tmax
              - 120
              - None
              - Always
            * - dose_tolerance
              - 0.1
              - None
              - Always
            * - t0
              - 0
              - None
              - Always
            * - field_strength
              - 3
              - None
              - Always
            * - **Injection**
              -
              - 
              - 
            * - agent
              - 'gadoxetate'
              - None
              - Always
            * - weight
              - 70
              - None
              - Always
            * - dose
              - 0.0125
              - None
              - Always
            * - rate
              - 1
              - None
              - Always
            * - BAT
              - 60
              - [0, inf]
              - Always
            * - **Sequence**
              -
              - 
              - 
            * - TS
              - 0
              - None
              - Always
            * - TR
              - 0.005
              - None
              - sequence in ['SS', 'SSI']
            * - FA
              - 15
              - None
              - sequence in ['SR', 'SS', 'SSI']
            * - TC
              - 0.1
              - None
              - sequence == 'SR'
            * - TF
              - 0
              - None
              - sequence == 'SSI'
            * - **Aorta**
              -
              - 
              - 

            * - CO
              - 100
              - [0, 300]
              - Always
            * - Thl
              - 10
              - [0, 30]
              - Always
            * - Dhl
              - 0.2
              - [0.05, 0.95]
              - heartlung in ['pfcomp', 'chain']
            * - To
              - 20
              - [0, 60]
              - Always
            * - Eo
              - 0.15
              - [0, 0.5]
              - organs == '2cxm'
            * - Toe
              - 120
              - [0, 800]
              - organs == '2cxm'
            * - Eb
              - 0.05
              - [0.01, 0.15]
              - Always
            * - R10
              - 0.7
              - None
              - Always
            * - S0
              - 1
              - None
              - Always


    Example:

        Use the model to fit minipig aorta data with inflow 
        correction:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import pydmr
        >>> import dcmri as dc

        Read the dataset:

        >>> datafile = dc.fetch('minipig_renal_fibrosis')
        >>> data = pydmr.read(datafile, 'nest')
        >>> rois, pars = data['rois']['Pig']['Test'], data['pars']['Pig']['Test']

        Initialize the tissue:

        >>> aorta = dc.Aorta(
        ...     sequence='SSI',
        ...     heartlung='chain',
        ...     organs='comp',
        ...     field_strength=pars['B0'],
        ...     t0=15, 
        ...     agent="gadoterate",
        ...     weight=pars['weight'],
        ...     dose=pars['dose'],
        ...     rate=pars['rate'],
        ...     TR=pars['TR'],
        ...     FA=pars['FA'],
        ...     TS=pars['TS'],
        ...     CO=60, 
        ...     R10=1/dc.T1(pars['B0'], 'blood'),
        ... )

        Create an array of time points:

        >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

        Train the system to the data:

        >>> aorta.train(time, rois['Aorta'])

        Plot the reconstructed signals and concentrations:

        >>> aorta.plot(time, rois['Aorta'])

        Print the model parameters:

        >>> aorta.print_params(round_to=4)
    """

    def __init__(
        self, 
        organs='comp', 
        heartlung='pfcomp', 
        sequence='SS', 
        **params,
    ):
        self._version = '1.0'
        
        # Set Configuration
        valid_organs = ['comp', '2cxm']
        valid_hl = ['pfcomp', 'chain']
        valid_seq = ['SR', 'SS', 'SSI', 'lin']

        if organs not in valid_organs:
            raise ValueError(f"Invalid organs model '{organs}'. Options: {valid_organs}")
        if heartlung not in valid_hl:
            raise ValueError(f"Invalid heart-lung model '{heartlung}'. Options: {valid_hl}")
        if sequence not in valid_seq:
            raise ValueError(f"Invalid sequence '{sequence}'. Options: {valid_seq}")

        self._organs = organs
        self._heartlung = heartlung
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
        # Build parameter list based on configuration
        pars_organs = {'comp': [], '2cxm': ['T(o,e)', 'E(o)']}
        pars_sequence = {
            'SR': ['TS', 'TC', 'FA'],
            'SS': ['TS', 'TR', 'FA'], 
            'SSI': ['TS', 'TF', 'TR', 'FA'],
            'lin': ['TS']
        } 
        if select is None:
            pars_list = [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate', 'BAT',
                'CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(b)', 'R10', 'S0',
            ]
            pars_list += pars_sequence[self._sequence]
            pars_list += pars_organs[self._organs]
        if select=='free':
            # Determine defaults based on config
            free_pars_sequence = {'SR': [], 'SS': [], 'SSI': ['TF'], 'lin': []}
            pars_list = ['BAT', 'CO', 'T(hl)', 'D(hl)', 'T(o)', 'E(b)', 'S0']
            pars_list += pars_organs[self._organs]
            pars_list += free_pars_sequence[self._sequence]       
        return pars_list

    # ---- Internal Forward Model ----

    def _set_time(self):
        """Build time axis"""
        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])

    def _compute_concentration(self):
        """Calculates blood concentration (cb) over time."""
        self._set_time()
        if self._organs=='comp':
            organs_cfg = ['comp', (self._pars['T(o)'],)]
        elif self._organs=='2cxm':
            organs_cfg = ['2cxm', ([self._pars['T(o)'], self._pars['T(o,e)']], self._pars['E(o)'])]

        if self._heartlung=='pfcomp':
            hl_cfg = ['pfcomp', (self._pars['T(hl)'], self._pars['D(hl)'])]
        elif self._heartlung=='chain':
            hl_cfg = ['chain', (self._pars['T(hl)'], self._pars['D(hl)'])]

        conc_mol = lib.ca_conc(self._pars['agent'])
        Ji = lib.ca_injection(
            self._t, self._pars['weight'], conc_mol, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=self._pars['E(b)'], 
            heartlung=hl_cfg, 
            organs=organs_cfg, 
            dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
        )
        self._ca = Jb / self._pars['CO']

    def _compute_relaxation_rate(self):
        """Calculates longitudinal relaxation rate (R1b)."""
        self._compute_concentration()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1a = self._pars['R10'] + rb * self._ca

    def _compute_signal(self):
        """Calculates MRI signal (Sb) based on the chosen sequence."""
        self._compute_relaxation_rate()
        s0, r1 = self._pars['S0'], self._R1a
        
        if self._sequence == 'SR':
            self._Sa = sig.signal_free(s0, r1, self._pars['TC'], self._pars['FA'])
        elif self._sequence == 'SS':
            self._Sa = sig.signal_ss(s0, r1, self._pars['TR'], self._pars['FA'])
        elif self._sequence == 'SSI':
            self._Sa = sig.signal_spgr(s0, r1, self._pars['TF'], self._pars['TR'], self._pars['FA'], n0=1)
        elif self._sequence == 'lin':
            self._Sa = sig.signal_lin(s0, r1)

    def _predict(self, time):
        self._compute_signal()
        return utils.sample(time, self._t, self._Sa, self._pars['TS'])

    # ---- Public API ----

    def time(self) -> np.ndarray:
        """Internal time array"""
        self._set_time()
        return self._t

    def conc(self):
        """Returns the predicted aorta blood concentration."""
        self._compute_concentration()
        return self._ca

    def relax(self):
        """Returns the predicted longitudinal relaxation rate."""
        self._compute_relaxation_rate()
        return self._R1a

    def predict(self, time:np.ndarray=None) -> np.ndarray:
        """Predicts the aorta signal at specified time points."""
        if time is None:
            time = self.time()
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + ts
        
        return self._predict(time)

    def train(self, time: tuple, signal: tuple, free:dict=None, 
              bounds:dict=None, n0=1, **kwargs):
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
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
        self._estimate_parameters(time, signal, n0)

        # Check and update free parameters
        free = self._set_free_pars(free, bounds) 

        # Optimization
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
        """Initial heuristic estimation for S0 and BAT."""
        ts = self._pars['TS'] if self._pars['TS'] is not None else 0
        self._pars['tmax'] = self._pars['dt'] + np.max(time) + ts

        # Calculate reference signal for S0 normalization
        r10 = self._pars['R10']
        if self._sequence == 'SR':
            s_ref = sig.signal_free(1, r10, self._pars['TC'], self._pars['FA'])
        elif self._sequence == 'SS':
            s_ref = sig.signal_ss(1, r10, self._pars['TR'], self._pars['FA'])
        elif self._sequence == 'SSI':
            s_ref = sig.signal_spgr(1, r10, self._pars['TF'], self._pars['TR'], self._pars['FA'], n0=1)
        else: # lin
            s_ref = sig.signal_lin(1, r10)

        self._pars['S0'] = np.mean(signal[:n0]) / s_ref
        self._pars['BAT'] = time[np.argmax(signal)] - self._pars['T(hl)']
        self._pars['BAT'] = max([self._pars['BAT'], 0])

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

        # Validate Free Parameters
        if self._sequence == 'SSI' and 'S0' not in free:
            raise ValueError("For SSI sequence, 'S0' must be a free parameter.")

        for p, bnds in free.items():
            if p not in self._pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            elif p=='BAT':
                if (bnds[0] > 0) or (bnds[1] < 0):
                    raise ValueError(f"Bounds on BAT must be (negative, positive).")
            elif p in ['S0']: 
                if not (0 <= bnds[0] < bnds[1]):
                    raise ValueError(f"Invalid bounds on {p}: Bounds on S0 are relative and must be positive.")
            elif not (bnds[0] <= self._pars[p] <= bnds[1]):
                raise ValueError(f"Initial value for '{p}' ({self._pars[p]}) is out of bounds {bnds}.")

        # --- 3. Relative to Absolute Bounds
        # Additive
        for par in ['BAT']:
            if par in free:
                free[par] = [  
                    self._pars[par] + free[par][0],
                    self._pars[par] + free[par][1],
                ]
        # Multiplicative
        for par in ['S0']:
            if par in free:
                free[par] = [
                    self._pars[par] * free[par][0],
                    self._pars[par] * free[par][1],
                ]
        return free

    # ---- I/O and Reporting ----

    def save(self, file: str):
        """Save the current state of the model as a json file."""
        if not file.endswith('.json'):
            file += '.json'
        
        state = {
            'model': self.__class__.__name__,
            'version': self._version,
            'organs': self._organs,
            'heartlung': self._heartlung,
            'sequence': self._sequence,
            'pars': self._pars,
        }
        with open(file, "w") as f:
            json.dump(state, f, indent=4)
        return self

    def load(self, file: str):
        """Load the saved state of the model from a json file"""
        with open(file, "r") as f:
            data = json.load(f)

        if data['model'] != self.__class__.__name__:
            raise ValueError(f"File belongs to {data['model']}, not {self.__class__.__name__}.")
        if data['version'] != self._version:
            raise ValueError(f"Version mismatch: {data['version']} vs {self._version}.")

        self._organs = data['organs']
        self._heartlung = data['heartlung']
        self._sequence = data['sequence']
        self._pars = data['pars']
        return self

    def export_params(self) -> dict:
        """Returns parameters as a dict: {short_name: [long_name, value, unit, sdev]}."""
        pars = {
            p: {
                'name': deepcopy(PARAMS[p]['name']),
                'unit': deepcopy(PARAMS[p]['unit']),  
                'value': self._pars[p], 
            } for p in self._pars_list('free')
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
        # if select is not None:
        #     pars = {k: v for k, v in pars.items() if k in self._pars_list(select)}
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

    def cost(self, time, signal, metric='NRMS') -> float:
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
        y = self.predict(time)
        return utils.loss(y, signal, metric)

    def plot(self, xdata, ydata, fname=None, show=True):
        
        signal = self.predict(xdata)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(xdata/60, ydata, 'ko', alpha=0.5, label='Data')
        ax0.plot(xdata/60, signal, 'r-', linewidth=2, label='Fit')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Prediction')
        ax1.plot(self._t/60, 1000*self._ca, 'r-', label='Prediction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()