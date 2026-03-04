import warnings
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import dcmri.lib as lib
import dcmri.sig as sig
import dcmri.utils as utils
import dcmri.pk_aorta as pk_aorta

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

    def __init__(self, organs='comp', heartlung='pfcomp', sequence='SS', **params):
        self._version = '1.0'
        
        # Configuration Validation
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

        # Build parameter list based on configuration
        pars_organs = {'comp': [], '2cxm': ['Toe', 'Eo']}
        pars_sequence = {
            'SR': ['TS', 'TC', 'FA'],
            'SS': ['TS', 'TR', 'FA'], 
            'SSI': ['TS', 'TF', 'TR', 'FA'],
            'lin': ['TS']
        } 
        
        pars_list = [
            'dt', 'tmax', 'dose_tolerance', 't0', 'field_strength',
            'agent', 'weight', 'dose', 'rate', 'BAT',
            'CO', 'Thl', 'Dhl', 'To', 'Eb', 'R10', 'S0',
        ]
        pars_list += pars_sequence[self._sequence]
        pars_list += pars_organs[self._organs]

        # Initialize Parameters
        self._pars = {p: PARAMS[p]['init'] for p in pars_list}

        # Override with user values
        for p, v in params.items():
            if p in pars_list:
                self._pars[p] = v
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

        self._free = None
        self._pcov = None

    # ---- Internal Forward Model ----

    def _compute_concentration(self):
        """Calculates blood concentration (cb) over time."""
        if self._organs=='comp':
            organs_cfg = ['comp', (self._pars['To'],)]
        elif self._organs=='2cxm':
            organs_cfg = ['2cxm', ([self._pars['To'], self._pars['Toe']], self._pars['Eo'])]

        if self._heartlung=='pfcomp':
            hl_cfg = ['pfcomp', (self._pars['Thl'], self._pars['Dhl'])]
        elif self._heartlung=='chain':
            hl_cfg = ['chain', (self._pars['Thl'], self._pars['Dhl'])]

        self._t = np.arange(0, self._pars['tmax'], self._pars['dt'])
        conc_mol = lib.ca_conc(self._pars['agent'])
        
        # Injection flux
        Ji = lib.ca_injection(
            self._t, self._pars['weight'], conc_mol, self._pars['dose'], 
            self._pars['rate'], self._pars['BAT']
        )
        
        # Aorta flux using whole-body PK model
        Jb = pk_aorta.flux_aorta(
            Ji, E=self._pars['Eb'], 
            heartlung=hl_cfg, 
            organs=organs_cfg, 
            dt=self._pars['dt'], 
            tol=self._pars['dose_tolerance'],
        )
        self._cb = Jb / self._pars['CO']

    def _compute_relaxation_rate(self):
        """Calculates longitudinal relaxation rate (R1b)."""
        self._compute_concentration()
        rb = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])
        self._R1b = self._pars['R10'] + rb * self._cb

    def _compute_signal(self):
        """Calculates MRI signal (Sb) based on the chosen sequence."""
        self._compute_relaxation_rate()
        s0, r1 = self._pars['S0'], self._R1b
        
        if self._sequence == 'SR':
            self._Sb = sig.signal_free(s0, r1, self._pars['TC'], self._pars['FA'])
        elif self._sequence == 'SS':
            self._Sb = sig.signal_ss(s0, r1, self._pars['TR'], self._pars['FA'])
        elif self._sequence == 'SSI':
            self._Sb = sig.signal_spgr(s0, r1, self._pars['TF'], self._pars['TR'], self._pars['FA'], n0=1)
        elif self._sequence == 'lin':
            self._Sb = sig.signal_lin(s0, r1)

    def _predict(self, time):
        self._compute_signal()
        return utils.sample(time, self._t, self._Sb, self._pars['TS'])

    # ---- Public API ----

    def conc(self):
        """Returns the predicted aorta blood concentration."""
        self._compute_concentration()
        return self._t, self._cb

    def relax(self):
        """Returns the predicted longitudinal relaxation rate."""
        self._compute_relaxation_rate()
        return self._t, self._R1b

    def predict(self, time) -> np.ndarray:
        """Predicts the aorta signal at specified time points."""
        tacq = time[1] - time[0] if len(time) > 1 else 0
        self._pars['tmax'] = max(time) + tacq + self._pars['dt']
        if self._pars.get('TS') is not None:
            self._pars['tmax'] += self._pars['TS']
        return self._predict(time)

    def train(self, time, signal, free=None, **kwargs):
        """Trains the model to fit provided signal data."""
        if free is None:
            # Determine defaults based on config
            free_pars_organs = {'comp': [], '2cxm': ['Toe', 'Eo']}
            free_pars_sequence = {'SR': [], 'SS': [], 'SSI': ['TF'], 'lin': []}
            
            base_free = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eb', 'S0']
            base_free += free_pars_organs[self._organs]
            base_free += free_pars_sequence[self._sequence]
            
            # Map names to their global PARAMS bounds
            free = {p: PARAMS[p]['bounds'] for p in base_free}

        # Validate Free Parameters
        if self._sequence == 'SSI' and 'S0' not in free:
            raise ValueError("For SSI sequence, 'S0' must be a free parameter.")

        for p, bounds in free.items():
            if p not in self._pars:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            if bounds[0] > self._pars[p] or bounds[1] < self._pars[p]:
                raise ValueError(f"Initial value for '{p}' ({self._pars[p]}) is out of bounds {bounds}.")

        self._free = free
        self._estimate_parameters(time, signal)
        self._pcov = _train(self.predict, time, signal, self._pars, self._free, **kwargs)

        return self

    def _estimate_parameters(self, time, signal):
        """Initial heuristic estimation for S0 and BAT."""
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

        n0 = max([np.sum(time < self._pars['t0']), 1])
        self._pars['S0'] = np.mean(signal[:n0]) / s_ref
        self._pars['BAT'] = time[np.argmax(signal)] - self._pars['Thl']
        self._pars['BAT'] = max([self._pars['BAT'], 0])

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
            'free': self._free,
            'pcov': self._pcov,
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

        self._organs, self._heartlung, self._sequence = data['organs'], data['heartlung'], data['sequence']
        self._pars, self._free = data['pars'], data['free']
        self._pcov = data['pcov']
        return self

    def export_params(self) -> dict:
        """Returns parameters as a dict: {short_name: [long_name, value, unit, sdev]}."""
        exported = {}
        for p, val in self._pars.items():
            if p in PARAMS:
                exported[p] = [PARAMS[p]['name'], val, PARAMS[p]['unit'], 0.0]
                
        # Add standard deviation
        if self._pcov is not None:
            for i, p in enumerate(self._free.keys()):
                sdev = _renormalize(np.sqrt(np.array(self._pcov)[i, i]), self._free[p])
                if p in exported:
                    exported[p][-1] = sdev
        return exported

    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is 
              not provided, the values are not rounded. Defaults to None.
        """
        pars = self.export_params()
        for p, (name, val, unit, sdev) in pars.items():
            if round_to is not None:
                if not isinstance(val, str):
                    val, sdev = round(val, round_to), round(sdev, round_to)
            print(f"{name} ({p}): {val} (+/- {sdev}) {unit}")

    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to return

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        if len(args) == 1:
            val = self._pars[args[0]]
            return round(val, round_to) if round_to else val
        
        subset = {p: self._pars[p] for p in args if p in self._pars}
        if round_to:
            return {p: round(v, round_to) for p, v in subset.items()}
        return subset

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

    def plot(self, xdata, ydata, ref=None, fname=None, show=True):
        
        pred_signal = self.predict(xdata)
        t_cont, cb_cont = self.conc()
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(xdata/60, ydata, 'ko', alpha=0.5, label='Data')
        ax0.plot(xdata/60, pred_signal, 'r-', linewidth=2, label='Fit')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Prediction')
        if ref is not None:
            ax1.plot(ref['t']/60, 1000*ref['cb'], 'ks', label='Ground Truth')
        ax1.plot(t_cont/60, 1000*cb_cont, 'r-', label='Prediction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()

# ---- Helper Functions ----

def _train(predict, xdata, ydata, pars, free, **kwargs):

    p0 = _compute_normalized_pars(pars, free)

    def predict_normalized(xdata, *normalized_pars):
        _update_original_pars(pars, normalized_pars, free)
        return predict(xdata)

    try:
        fitted_pars, pcov = curve_fit(
            predict_normalized, xdata, ydata, p0, bounds=(0, 1), **kwargs
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

# ---- Global Parameter Definitions ----

PARAMS = {
    'dt': {'init': 0.25, 'bounds': [1e-3, 10], 'name': 'Time step', 'unit': 's'},
    'tmax': {'init': 120, 'bounds': [10, 1000], 'name': 'Max time', 'unit': 's'},
    'dose_tolerance': {'init': 0.1, 'bounds': [0, 1], 'name': 'Dose tolerance', 'unit': ''},
    't0': {'init': 0, 'bounds': [0, 100], 'name': 'Baseline duration', 'unit': 's'},
    'field_strength': {'init': 3.0, 'bounds': [0, 10], 'name': 'Field strength', 'unit': 'T'},
    'agent': {'init': 'gadoterate', 'bounds': None, 'name': 'Contrast agent', 'unit': None},
    'weight': {'init': 70, 'bounds': [1, 500], 'name': 'Weight', 'unit': 'kg'},
    'dose': {'init': 0.1, 'bounds': [0, 1], 'name': 'Dose', 'unit': 'mL/kg'},
    'rate': {'init': 1, 'bounds': [0, 10], 'name': 'Injection rate', 'unit': 'mL/s'},
    'TR': {'init': 0.005, 'bounds': [0, 1], 'name': 'TR', 'unit': 's'},
    'FA': {'init': 15, 'bounds': [0, 90], 'name': 'Flip angle', 'unit': 'deg'},
    'TC': {'init': 0.2, 'bounds': [0, 10], 'name': 'Time to k-center', 'unit': 's'},
    'TS': {'init': 0, 'bounds': [0, 10], 'name': 'Sampling time', 'unit': 's'},
    'TF': {'init': 0.5, 'bounds': [0, 2], 'name': 'Inflow time', 'unit': 's'},
    'BAT': {'init': 60, 'bounds': [0, 200], 'name': 'Bolus arrival time', 'unit': 's'},
    'CO': {'init': 100, 'bounds': [0, 500], 'name': 'Cardiac output', 'unit': 'mL/s'},
    'Thl': {'init': 10, 'bounds': [0, 50], 'name': 'Heart-lung MTT', 'unit': 's'},
    'Dhl': {'init': 0.2, 'bounds': [0.01, 0.99], 'name': 'Heart-lung dispersion', 'unit': ''},
    'To': {'init': 20, 'bounds': [0, 200], 'name': 'Organ blood MTT', 'unit': 's'},
    'Eo': {'init': 0.15, 'bounds': [0, 1], 'name': 'Organ extraction', 'unit': ''},
    'Toe': {'init': 120, 'bounds': [0, 1000], 'name': 'Organ EES MTT', 'unit': 's'},
    'Eb': {'init': 0.05, 'bounds': [0, 1], 'name': 'Body extraction', 'unit': ''},
    'R10': {'init': 0.7, 'bounds': [0, 5], 'name': 'Precontrast R1', 'unit': 'Hz'},
    'S0': {'init': 1.0, 'bounds': [0, 1e6], 'name': 'Signal scaling', 'unit': 'a.u.'},
}