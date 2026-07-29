"""General model for whole-kidney signals.

See Also:
    `Liver`, `Tissue`

Args:
    kinetics (str, optional): Kinetic model for the kidneys. 
        Options are '2CF' (Two-compartment filtration) and 'HF' 
        (High-flow). Defaults to '2CF'. 
    sequence (str, optional): imaging sequence model. Possible 
        values are '3D-SPGR-SS' (steady-state), 'SR' (saturation-recovery), 
        and 'lin' (linear). Defaults to '3D-SPGR-SS'.
    params (dict, optional): values for the model parameters,
        specified as keyword parameters. Defaults are used for any 
        that are not provided. See table 
        :ref:`Kidney-defaults` for a list of parameters and 
        their default values.

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
    >>> ca = dc.aif.tristan(
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
    ...    R1ba=1/dc.const.T1(pars['B0'], 'blood'),
    ...    R1b=1/dc.const.T1(pars['B0'], 'kidney'),
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

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.inverse.sig2conc import SignalToConc
from dcmri.utils import const
from dcmri.core.model import SuperModel
from dcmri.core.types import Input
from dcmri.core.sequences import SEQUENCES
from dcmri.kinetics.modules_conc import ConcKidney
from dcmri.signal.modules_tissue import Signal
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss

CONSTANTS = {'Fw': 0, 'v': 1, 'me': 1, 'noise_sdev':0}

class Kidney(SuperModel):
    """Whole-kidney signals with a known input.

    See Also:
        `Liver`, `Tissue`

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.
    """

    # ==========================================
    # User interface
    # ==========================================

    configs = {
        'kinetics': ['2CF', '2PF', 'CPF', '2CFU', '2PFU', 'HF', 'HFU'],
        'sequence': ['3D-SPGR-SS', '2D-SR-SPGR'],
    }

    def __init__(
        self, kinetics='2CF', sequence='3D-SPGR-SS', **params
    ):
        self._version = '1.0'
        cnfg = {'kinetics': kinetics, 'sequence': sequence}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

    def params(self, *args) -> dict: 
        """Model parameters and their values"""
        pars = self._pars
        if args == ():
            return pars
        for k in args:
            if k not in pars:
                raise ValueError(f"{k} is not a valid model parameter. Use print_params() to get a list of valid parameters.")
        values = [pars[k] for k in args]
        if len(args) == 1:
            return values[0]
        else:
            return values

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
        return self._R1, self._R2s

    def signal(self) -> np.ndarray:
        """Returns time points and predicted signal."""
        self._compute_signal()
        return self._S

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predicts kidney signal at specific time points."""
        self._set_time()
        if max(self._t) + self._pars['TS'] < np.max(time) :
            raise ValueError(f'The largest time point that can be predicted with the current AIF is {max(self._t)/60} mins.')
        return self._predict(time)
    
    def train(
        self, time: np.ndarray, signal: np.ndarray, aif:dict=None, 
        free: dict=None, bounds: dict=None, n0=1, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

        Args:
            time (array-like): Array with time points
            signal (array-like): Array with signal values
            aif (dict, optional): Dictionary with required key 'signal' (AIF signal) and 'R1b' (baseline R1).
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
        if aif is not None:
            input = Input(aif)
            p = self._pars
            seq = self._cnfg['sequence']
            rp = const.r1(p['field_strength'], 'blood', p['agent']) 
            ca = SignalToConc(seq, defaults=p)(
                input.signal, R1b=input.R1b, n0=n0, 
                B1corr=input.B1corr, r1=rp,
            )
            t = np.arange(0, np.amax(time) + p['dt'], p['dt'])
            p['ca'] = np.interp(t, input.time, ca)

        return self._train(time, signal, free, bounds, n0, **kwargs)


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

    def cost(self, time: np.ndarray, signal: np.ndarray, metric: str = 'NRMS', nfree=None) -> float:
        """Return the goodness-of-fit

        Args:
            time (np.ndarray): array with time points
            signal (array-like): array with signal data for all pixels.
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
        signal_pred = self._predict(time)
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
    

    # ==========================================
    # Backend
    # ==========================================

    
    def _params(self, select=None):
        pars_kin = ConcKidney(**self._cnfg).params()
        seq = self._cnfg['sequence']
        pars_seq = SEQUENCES[seq]['parameters']['prep']
        pars_seq += SEQUENCES[seq]['parameters']['read']

        if select is None:
            pars_list = [
                'ca', 'field_strength', 'agent',
                'H', 'S0', 'R1b', 'R2sb', 'TS',
            ]
            pars_list += pars_kin + pars_seq
        elif select=='free':
            pars_list = [k for k in pars_kin if k != 'Ta']
        derived = ['ci']
        pars_list = list({p for p in pars_list if p not in derived})
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================   

    def _compute_concentration(self):
        p = self._pars
        p['ci'] = p['ca'] / (1 - p['H'])
        self._C = ConcKidney(**self._cnfg)(**p)
       
    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars
        rp = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1 = p['R1b'] + rp * self._C.sum(axis=0)
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        self._R2s = p['R2sb'] + r2s * self._C.sum(axis=0)

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars
        seq = self._cnfg['sequence']
        self._S = Signal(seq, defaults=p)(R1=self._R1, R2s=self._R2s, **CONSTANTS)

    def _set_time(self):
        p = self._pars
        self._t = p['dt'] * np.arange(p['ca'].size)

    def _predict(self, time):
        self._set_time()
        self._compute_signal()
        return sample(time, self._t, self._S, self._pars['TS'])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, signal: np.ndarray, n0: int):
        p = self._pars
        seq = self._cnfg['sequence']
        
        # Estimate S0
        s_ref = Signal(seq, defaults=p)(R1=p['R1b'], R2s=p['R2sb'], S0=1, **CONSTANTS)
        p['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: np.ndarray, signal: np.ndarray,  
        free: dict, bounds: dict, n0: int, **kwargs,
    ):
        self._estimate_parameters(signal, n0)
        free = self._set_free_pars(free, bounds)
        return train(self._predict, time, signal, self._pars, free, **kwargs)

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

        ax1.plot(self._t/60, 1000*self._pars['ca'], '-', linewidth=3, color='darkred', label='Arterial Pred')
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

