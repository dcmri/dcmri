from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, sig, utils, pk_aorta, ui
from dcmri.lexicon import LEXICON

class Aorta(ui.SuperModel):
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

        self._version = '1.0'
        self._cnfg = {'organs': organs, 'heartlung': heartlung, 'sequence': sequence}
        self._pars = {p: deepcopy(LEXICON[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    def _pars_list(self, select=None):
        pars_organs = {'comp': [], '2cxm': ['To_e', 'Eo']}
        pars_sequence = {
            'SR': ['B1corr', 'FA', 'TR', 'TC', 'TP', 'TS'],
            'SS': ['B1corr', 'FA', 'TR', 'TS'], 
            'lin': ['TS'],
            'SSI': ['B1corr', 'FA', 'TR', 'TF', 'TS'],
        }

        if select is None:
            pars_list = [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate', 'BAT',
                'CO', 'Thl', 'Dhl', 'To', 'Eb', 'R10', 'S0',
            ]
            pars_list += pars_organs[self._cnfg['organs']]
            pars_list += pars_sequence[self._cnfg['sequence']]
        if select=='free':
            # Determine defaults based on config
            pars_list = ['BAT', 'CO', 'Thl', 'Dhl', 'To', 'Eb', 'S0']
            pars_list += pars_organs[self._cnfg['organs']]
            if self._cnfg['sequence']=='SSI': pars_list += ['TF']      
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================

    def _set_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])

    def _compute_concentration(self):
        self._set_time()
        p = self._pars

        if self._cnfg['organs']=='comp':
            organs_cfg = ['comp', (p['To'],)]
        elif self._cnfg['organs']=='2cxm':
            organs_cfg = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]

        if self._cnfg['heartlung']=='pfcomp':
            hl_cfg = ['pfcomp', (p['Thl'], p['Dhl'])]
        elif self._cnfg['heartlung']=='chain':
            hl_cfg = ['chain', (p['Thl'], p['Dhl'])]

        conc = lib.ca_conc(p['agent'])
        Ji = lib.ca_injection(
            self._t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=p['Eb'], heartlung=hl_cfg, organs=organs_cfg, 
            dt=p['dt'], tol=p['dose_tolerance'],
        )
        self._c = Jb / p['CO']

    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1 = p['R10'] + rb * self._c

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars
        if self._cnfg['sequence'] == 'SR':
            self._S = sig.signal_spgr(p['S0'], self._R1, p['TC'], p['TR'], p['B1corr'] * p['FA'], p['TP'])
        elif self._cnfg['sequence'] == 'SS':
            self._S = sig.signal_ss(p['S0'], self._R1, p['TR'], p['B1corr'] * p['FA'])
        elif self._cnfg['sequence'] == 'SSI':
            self._S = sig.signal_spgr(p['S0'], self._R1, p['TF'], p['TR'], p['B1corr'] * p['FA'], n0=1)
        elif self._cnfg['sequence'] == 'lin':
            self._S = sig.signal_lin(p['S0'], self._R1)

    def _predict(self, time):
        self._compute_signal()
        return utils.sample(time, self._t, self._S, self._pars['TS'])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: tuple, signal: tuple, n0: int):
        p = self._pars
        p['tmax'] = np.max(time) + p['dt'] + p['TS']

        # Estimate BAT
        BAT = time[np.argmax(signal)] - p['Thl'] * (1 - p['Dhl'])
        p['BAT'] = max([BAT, 0])

        # Calculate reference signal for S0 normalization
        if self._cnfg['sequence'] == 'SR':
            s_ref = sig.signal_spgr(1, p['R10'], p['TC'], p['TR'], p['B1corr'] * p['FA'], p['TP'])
        elif self._cnfg['sequence'] == 'SS':
            s_ref = sig.signal_ss(1, p['R10'], p['TR'], p['B1corr'] * p['FA'])
        elif self._cnfg['sequence'] == 'SSI':
            s_ref = sig.signal_spgr(1, p['R10'], p['TF'], p['TR'], p['B1corr'] * p['FA'], n0=1)
        elif self._cnfg['sequence'] == 'lin':
            s_ref = sig.signal_lin(1, p['R10'])

        p['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0


    def _train(
        self, time: tuple, signal: tuple, free:dict=None, 
        bounds:dict=None, n0=1, **kwargs,
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == 'SSI' and 'S0' not in free:
            raise ValueError("For SSI sequence, 'S0' must be a free parameter.")

        # Optimization
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)

    def _plot(self, time: np.ndarray, signal: np.ndarray, fname: str, show: bool):
        self._compute_signal()
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signal Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(time/60, signal, 'ko', alpha=0.5, label='Data')
        ax0.plot(self._t/60, self._S, 'r-', linewidth=2, label='Prediction')
        ax0.set_xlabel('Time (min)')
        ax0.set_ylabel('Signal (a.u.)')
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        ax1.plot(self._t/60, 1000*self._c, 'r-', label='Reconstruction')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mM)')
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()


    # ---- Public API ----

    def time(self) -> np.ndarray:
        """Internal time array"""
        self._set_time()
        return self._t

    def conc(self) -> np.ndarray:
        """Returns the predicted aorta blood concentration."""
        self._compute_concentration()
        return self._c

    def relax(self) -> np.ndarray:
        """Returns the predicted longitudinal relaxation rate."""
        self._compute_relaxation_rate()
        return self._R1
    
    def signal(self) -> np.ndarray:
        """Returns time points and predicted liver signal."""
        self._compute_signal()
        return self._S

    def predict(self, time:np.ndarray) -> np.ndarray:
        """Predicts the aorta signal at specified time points."""
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)
        return self._predict(time)
    
    def train(
            self, time: tuple, signal: tuple, free: dict=None, 
            bounds: dict=None, n0=1, **kwargs
        ) -> Tuple[dict, dict, np.ndarray]:
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
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(time)        
        return self._train(time, signal, free, bounds, n0, **kwargs)
    
    def plot(self, time: np.ndarray, signal:np.ndarray, 
             fname:str=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of signals
            signal (tuple): Liver signals            
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        self._plot(time, signal, fname, show)