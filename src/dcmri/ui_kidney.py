from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, kidney, sig, utils, pk
from dcmri.ui import SuperModel
from dcmri.lexicon import LEXICON


class Kidney(SuperModel):
    """General model for whole-kidney signals.

    See Also:
        `Liver`, `Tissue`

    Args:
        kinetics (str, optional): Kinetic model for the kidneys. 
          Options are '2CF' (Two-compartment filtration) and 'HF' 
          (High-flow). Defaults to '2CF'. 
        sequence (str, optional): imaging sequence model. Possible 
          values are 'SS' (steady-state), 'SR' (saturation-recovery), 
          and 'lin' (linear). Defaults to 'SS'.
        params (dict, optional): values for the model parameters,
          specified as keyword parameters. Defaults are used for any 
          that are not provided. See table 
          :ref:`Kidney-defaults` for a list of parameters and 
          their default values.

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _Kidney-defaults:
        .. list-table:: Kidney parameters. 
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
            * - field_strength
              - 3
              - None
              - Always
            * - agent
              - 'gadoterate'
              - None
              - Always
            * - t0
              - 0
              - None
              - Always
            * - **Sequence**
              -
              - 
              - 
            * - TS
              - 0
              - None
              - Always
            * - B1corr
              - 1
              - None
              - sequence in ['SS']
            * - FA
              - 15
              - None
              - sequence in ['SR', 'SS']
            * - TR
              - 0.005
              - None
              - sequence in ['SS']
            * - TC
              - 0.1
              - None
              - sequence == 'SR'
            * - TP
              - 0.05
              - None
              - sequence == 'SR'
            * - **AIF**
              - 
              - 
              - 
            * - B1corr_a
              - 1
              - None
              - sequence in ['SS']
            * - R10a
              - 0.7
              - None
              - Always
            * - **Kidney**
              -
              - 
              - 
            * - H
              - 0.45
              - None
              - Always
            * - Ta
              - 0
              - [0, 3]
              - Always
            * - vol
              - 150
              - None
              - Always
            * - Fp
              - 0.02
              - [0, 0.05]
              - kinetics == '2CF'
            * - vp
              - 0.15
              - [0, 0.3]
              - Always
            * - FF
              - 0.1
              - [0, 0.3]
              - kinetics == '2CF'
            * - Tt
              - 120
              - [0, inf]
              - Always
            * - Ft
              - 0.005
              - [0, 0.05]
              - Always
            * - R10
              - 0.65
              - None
              - Always
            * - S0
              - 1.0
              - [0, inf]
              - Always

    Example:

        Use the model to fit minipig data. The AIF is corrupted by 
        inflow effects so for the purpose of this example we will 
        use a standard input function:

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
        >>> time = pars['TS'] * np.arange(len(rois['LeftKidney']))

        Generate an AIF at high temporal resolution (250 msec):

        >>> dt = 0.25
        >>> t = np.arange(0, np.amax(time) + dt, dt) 
        >>> ca = dc.aif_tristan(
        ...    t, 
        ...    agent="gadoterate",
        ...    dose=pars['dose'],
        ...    rate=pars['rate'],
        ...    weight=pars['weight'],
        ...    CO=60,
        ...    BAT=time[np.argmax(rois['Aorta'])] - 20,
        >>> )        

        Initialize the tissue:

        >>> kidney = dc.Kidney(
        ...    ca=ca,
        ...    dt=dt,
        ...    kinetics='HF',
        ...    field_strength=pars['B0'],
        ...    agent="gadoterate",
        ...    t0=pars['TS'] * pars['n0'],
        ...    TS=pars['TS'], 
        ...    TR=pars['TR'],
        ...    FA=pars['FA'],
        ...    R10a=1/dc.T1(pars['B0'], 'blood'),
        ...    R10=1/dc.T1(pars['B0'], 'kidney'),
        >>> )

        Train the kidney on the data:

        >>> kidney.set_free(Ta=[0,30])
        >>> kidney.train(time, rois['LeftKidney'])
        
        Plot the reconstructed signals and concentrations:

        >>> kidney.plot(time, rois['LeftKidney'])

        Print the model parameters:

        >>> kidney.print_params(round_to=4)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Arterial mean transit time (Ta): 13.8658 (0.1643) sec
        Plasma volume (vp): 0.0856 (0.003) mL/cm3
        Tubular flow (Ft): 0.0024 (0.0001) mL/sec/cm3
        Tubular mean transit time (Tt): 116.296 (7.6526) sec

    """

    def __init__(self, kinetics='2CF', sequence='SS', **params):
        
        # Check configuration
        if kinetics not in ['2CF', 'HF']:
            raise ValueError(f"Kinetic model {kinetics} is not available.")
        if sequence not in ['SS', 'SR', 'lin']:
            raise ValueError(f"Sequence {sequence} is not available.")
        
        # Config
        self._version = '1.0'
        self._cnfg = {'kinetics': kinetics, 'sequence': sequence}
        self._pars = {p: deepcopy(LEXICON[p]['init']) for p in self._pars_list()}

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

    
    def _pars_list(self, select=None):
        pars_kin = list(kidney.params_kidney(self._cnfg['kinetics']).keys())
        pars_seq = {
            'SR': ['B1corr', 'FA', 'TR', 'TC', 'TP', 'TS'],
            'SS': ['B1corr', 'FA', 'TR', 'TS'],
            'lin': ['TS'],
        }[self._cnfg['sequence']]

        if select is None:
            pars_list = [
                'c_a', 'dt', 'field_strength', 'agent',
                'H', 'T_a', 'S0', 'R10',
            ]
            pars_list += pars_kin + pars_seq
        elif select=='free':
            pars_list = pars_kin
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================   

    def _compute_concentration(self):
        p = self._pars
        ca = pk.flux(p['c_a'], p['T_a'], dt=p['dt'], model='plug')
        if self._cnfg['kinetics'] == '2CF':
            self._C = kidney.conc_kidney(
                ca / (1 - p['H']), p['Fp'], p['vp'], p['FF'] * p['Fp'], 
                p['Tt'], dt=p['dt'], sum=False, kinetics='2CF',
            )
        elif self._cnfg['kinetics'] == 'HF':
           self._C = kidney.conc_kidney(
                ca / (1 - p['H']), p['vp'], p['Ft'], p['Tt'],
                dt=p['dt'], sum=False, kinetics='HF',
            )
        
    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars
        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1 = p['R10'] + rp * self._C.sum(axis=0)

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars

        if self._cnfg['sequence'] == 'SR':
            self._S = sig.signal_spgr(p['S0'], self._R1, p['TC'], p['TR'], p['B1corr'] * p['FA'], p['TP'])
        elif self._cnfg['sequence'] == 'SS':
            self._S = sig.signal_ss(p['S0'], self._R1, p['TR'], p['B1corr'] * p['FA'])
        elif self._cnfg['sequence'] == 'lin':
            self._S = sig.signal_lin(p['S0'], self._R1)

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

    def _estimate_parameters(self, signal: np.ndarray, n0: int, aif: dict):
        p = self._pars

        # Estimate S0
        if self._cnfg['sequence'] == 'SR':
            Sref = sig.signal_spgr(1, p['R10'], p['TC'], p['TR'], p['B1corr'] * p['FA'], p['TP'])
        elif self._cnfg['sequence'] == 'SS':
            Sref = sig.signal_ss(1, p['R10'], p['TR'], p['B1corr'] * p['FA'])
        elif self._cnfg['sequence'] == 'lin':
            Sref = sig.signal_lin(1, p['R10'])
        p['S0'] = np.mean(signal[:n0]) / Sref if Sref > 0 else 0

        if aif is not None:
            r1 = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
            if self._cnfg['sequence'] == 'SR':
                ca = sig.conc_spgr(aif['signal'], p['TC'], p['TR'], aif['B1corr'] * p['FA'], p['TP'], 1/aif['R10'], r1)
            elif self._cnfg['sequence'] == 'SS':
                ca = sig.conc_ss(aif['signal'], p['TR'], aif['B1corr'] * p['FA'], 1/aif['R10'], r1, n0)
            elif self._cnfg['sequence'] == 'lin':
                ca = sig.conc_lin(aif['signal'], 1/aif['R10'], r1, n0)
            uniform_time = np.arange(0, np.max(aif['time']) + p['TS'] + p['dt'], p['dt'])
            p['c_a'] = np.interp(uniform_time, aif['time'], ca)

    def _train(self, time: np.ndarray, signal: np.ndarray, aif: dict, 
               free: dict, bounds: dict, n0: int, **kwargs):
        self._estimate_parameters(signal, n0, aif)
        free = self._set_free_pars(free, bounds)
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)

    def _plot(self, time:np.ndarray, signal:np.ndarray, xlim:list, 
              fname:str, show:bool):
        self._set_time()
        self._compute_signal()
        if xlim is None:
            xlim = [np.amin(time), np.amax(time)]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

        # Signals Plot
        ax0.set_title('Prediction of the MRI signals.')
        ax0.plot(time/60, signal, marker='o', linestyle='None', color='cornflowerblue', label='Data')
        ax0.plot(self._t/60, self._S, linestyle='-', linewidth=3.0, color='darkblue', label='Prediction')
        ax0.set(xlabel='Time (min)', ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax0.legend()

        ax1.set_title('Reconstruction of concentrations')

        ax1.plot(self._t/60, 1000*self._pars['c_a'], '-', linewidth=3, color='darkred', label='Arterial Pred')
        ax1.plot(self._t/60, 1000*self._C[0,:], linestyle='-', linewidth=3.0, color='darkred', label='Blood')
        ax1.plot(self._t/60, 1000*self._C[1,:], linestyle='-', linewidth=3.0, color='darkcyan', label='Tubuli')
           
        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    # ==========================================
    # Public API
    # ==========================================

    def time(self) -> np.ndarray:
        """Kidney signal time points"""
        self._set_time()
        return self._t

    def conc(self) -> np.ndarray:
        """Returns time points and kidney concentrations."""
        self._compute_concentration()
        return self._C

    def relax(self) -> np.ndarray:
        """Returns time points and kidney relaxation rates (R1)."""
        self._compute_relaxation_rate()
        return self._R1

    def signal(self) -> np.ndarray:
        """Returns time points and predicted signal."""
        self._compute_signal()
        return self._S

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predicts kidney signal at specific time points."""
        self._set_time()
        if max(self._t) < np.max(time) + self._pars['TS']:
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._predict(time)
    
    def train(
        self, time: np.ndarray, signal: np.ndarray, aif: dict=None, 
        free: dict=None, bounds: dict=None, n0=1, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            aif (dict, optional): AIF signal, time and baseline R1.
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
        return self._train(time, signal, aif, free, bounds, n0, **kwargs)


    def plot(self, time: np.ndarray, signal:np.ndarray, 
             xlim:list=None, fname:str=None, show=True):
        """Plot the model fit against data

        Args:
            time (tuple): Time points of signals
            signal (tuple): Kidney signals            
            xlim (list, optional): Lower and upper boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        self._plot(time, signal, xlim, fname, show)
