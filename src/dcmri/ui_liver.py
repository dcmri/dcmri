from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, liver, sig, utils
from dcmri.ui import SuperModel
from dcmri.lexicon import LEXICON


class Liver(SuperModel):
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
        self, kinetics='2I-EC', non_stationary=None,
        sequence='SS', **params,
    ):
        # Validate configuration
        if sequence not in ['SS', 'SR', 'lin']:
            raise ValueError(f'Sequence {sequence} is not available in Liver().')
        try:
            liver.params_liver(kinetics, non_stationary)
        except Exception as e:
            raise ValueError(f"Invalid kinetics/stationarity: {e}") from e
       
        self._version = '1.0'
        self._cnfg = {'kinetics': kinetics, 'sequence': sequence, 'non_stationary': non_stationary}
        self._pars = {p: deepcopy(LEXICON[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
        # Test validity of parameters
        if kinetics.startswith('2'):
          if self._pars['c_a'].size != self._pars['c_v'].size:
              raise ValueError("Arterial- and venous inputs have different lengths")

    def _pars_list(self, select=None):
        pars_kin = list(liver.params_liver(self._cnfg['kinetics'], self._cnfg['non_stationary']).keys())
        pars_seq = {
            'SR': ['FA', 'TR', 'TC', 'TP'],
            'SS': ['FA', 'TR'],
            'lin': [],
        }[self._cnfg['sequence']]

        if select is None:
            pars_list = [
                'c_a', 'dt', 'field_strength', 'agent',
                'H', 'T_a', 'S0', 'R10', 'TS'
            ]
            pars_list += pars_kin + pars_seq
            if self._cnfg['kinetics'].startswith('2'):
                pars_list += ['c_v']
            if self._cnfg['sequence'] != 'lin':
                pars_list += ['B1corr']
        elif select=='free':
            pars_list = pars_kin
        elif select=='liver':
            pars_list = pars_kin
        elif select=='sequence':
            pars_list = pars_seq
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_concentration(self):
        p = self._pars
        ca_plasma = p['c_a'] / (1 - p['H'])
        if 'c_v' in p:
            ca_plasma = (ca_plasma, p['c_v'] / (1 - p['H']))

        liver_pars = {k: p[k] for k in self._pars_list('liver')}
        self._C = liver.conc_liver(
            ca_plasma, dt=p['dt'], kinetics=self._cnfg['kinetics'],
            non_stationary=self._cnfg['non_stationary'], sum=False, 
            **liver_pars,
        )

    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars

        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        rh = lib.relaxivity(p['field_strength'], 'hepatocytes', p['agent'])
        if self._C.ndim == 2:
            self._R1 = p['R10'] + rp * self._C[0, :] + rh * self._C[1, :]
        else:
            self._R1 = p['R10'] + rp * self._C

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars

        pars = {k: p[k] for k in self._pars_list(select='sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr']
        self._S = sig.signal(self._cnfg['sequence'], self._R1, p['S0'], **pars)

    def _set_time(self):
        p = self._pars
        self._t = p['dt'] * np.arange(p['c_a'].size)

    def _predict(self, time):
        self._set_time()
        self._compute_signal()
        return utils.sample(time, self._t, self._S, self._pars['TS'])

    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, signal: np.ndarray, n0: int, aif: dict, vif: dict):
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        seq_name = self._cnfg['sequence']

        # Estimate S0
        pars = {k: p[k] for k in self._pars_list(select='sequence')}
        if 'FA' in pars: pars['FA'] *= p['B1corr']
        s_ref = sig.signal(seq_name, p['R10'], 1, **pars)
        p['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0

        # Input concentrations
        def conc(input):
            seq_pars = {k: p[k] for k in self._pars_list('sequence')}
            if 'FA' in seq_pars: seq_pars['FA'] *= input['B1corr']
            ci = sig.conc(seq_name, input['signal'], input['R10'], rp, n0=n0, **seq_pars)
            return np.interp(self._t, input['time'], ci)
      
        if aif is not None: p['c_a'] = conc(aif)
        if vif is not None: p['c_v'] = conc(vif)

    def _train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: dict, vif: dict, free: dict, 
        bounds: dict, n0: int, **kwargs
    ):
        self._estimate_parameters(signal, n0, aif, vif)
        free = self._set_free_pars(free, bounds)
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)
   
    def _plot(self, time: np.ndarray, signal: np.ndarray, xlim:list, 
              fname:str, show:bool):
        self._set_time()
        self._compute_signal()
        p = self._pars
        xlim = xlim or [np.amin(time), np.amax(time)]
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Signals Plot
        ax0.set_title('MRI Signal Prediction')
        ax0.plot(time/60, signal, 'o', color='cornflowerblue', label='Data')
        ax0.plot(self._t/60, self._S, '-', linewidth=3, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='Signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        # Concentration Plot
        ax1.set_title('Concentration Reconstruction')
        if self._C.ndim == 1:
            ax1.plot(self._t/60, 1000*self._C, '-', linewidth=3, color='cornflowerblue', label='Liver')
        else:
            ax1.plot(self._t/60, 1000*self._C[0,:], '-.', linewidth=3, color='darkblue', label='Extracellular')
            ax1.plot(self._t/60, 1000*self._C[1,:], '-', linewidth=3, color='green', label='Hepatocytes]')
        ax1.plot(self._t/60, 1000*p['c_a'], '-', linewidth=3, color='darkred', label='Artery')
        if 'c_v' in p:
            ax1.plot(self._t/60, 1000*p['c_v'], '-', linewidth=3, color='purple', label='Portal Vein')

        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # Public API
    # ==========================================

    def time(self) -> np.ndarray:
        """Internal time array"""
        self._set_time()
        return self._t
       
    def conc(self) -> np.ndarray:
        """Returns liver concentrations."""
        self._compute_concentration()
        return self._C

    def relax(self) -> np.ndarray:
        """Returns liver relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return self._R1

    def signal(self) -> np.ndarray:
        """Returns predicted liver signal."""
        self._compute_signal()
        return self._S

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predicts liver signal at specific time points."""
        self._set_time()
        if max(self._t) < np.max(time) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._predict(time)
    
    def train(
            self, time: np.ndarray, signal: np.ndarray, 
            aif: dict=None, vif: dict=None, free: dict=None, 
            bounds: dict=None, n0=1, **kwargs
        ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            aif (dict, optional): AIF signal, time and baseline R1.
            vif (dict, optional): VIF signal, time and baseline R1.
            free (dict, optional): Dictionary with free parameters and their
              bounds. If not provided, a default set of free parameters is used.
              Defaults to None.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            kwargs: any keyword parameters accepted by 
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        self._set_time()
        if max(self._t) < np.max(time) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._train(time, signal, aif, vif, free, bounds, n0, **kwargs)

    def plot(self, time: np.ndarray, signal:np.ndarray, 
             xlim:list=None, fname:str=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of signals
            signal (tuple): Liver signals            
            xlim (list, optional): Lower and upper boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        self._plot(time, signal, xlim, fname, show)