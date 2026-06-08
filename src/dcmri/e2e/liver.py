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

    Use `fake.liver` to generate synthetic test data:

    >>> time, aif, vif, roi, gt = dc.fake.liver()

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
    ...     R10 = 1/dc.const.T1(3.0,'liver'),
    ...     R10a = 1/dc.const.T1(3.0, 'blood'), 
    ...     R10v = 1/dc.const.T1(3.0, 'blood'), 
    ... )

    Train the model on the ROI data:

    >>> model.train(time, roi, aif, vif, n0=10)

    Plot the reconstructed signals (left) and concentrations (right) and 
    compare the concentrations against the noise-free ground truth. Since 
    the data are analysed with an exact model, and there are no other data 
    errors present, this should fior the data exactly.

    >>> model.plot(time, roi, ref=gt)

"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri.inverse.sig2conc import SignalToConc
from dcmri.utils import const
from dcmri.core.model import SuperModel
from dcmri.core.types import Input
from dcmri.lexicon.dicts import SEQUENCES
from dcmri.lexicon.tools import print_params, export_params
from dcmri.kinetics.conc import ConcLiver
from dcmri.bloch.tissue import Signal
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.kinetics.lib.liver import dpars_liver


class Liver(SuperModel):
    """Liver tissue with known inputs.

    This is the standard interface for liver tissues with known input 
    function(s).

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `Tissue`

    """

    # ==========================================
    # User interface: Frontend
    # ==========================================

    configs = {
        'kinetics': ['1I-EC-D', '1I-EC', '2I-EC-HF', '2I-EC', '1I-IC', '1I-IC-HF', '1I-IC-HFD', '1I-IC-HFDU', '2I-IC-HF', '2I-IC', '2I-IC-U'],
        'non_stationary': [None, 'U', 'E', 'UE'],
        'sequence': ['3D-SPGR-SS', '2D-SR-SPGR-SS'],
    }
    
    def __init__(
        self, kinetics='2I-EC', non_stationary=None,
        sequence='3D-SPGR-SS', **params,
    ):
        self._version = '1.0'
        cnfg = {'kinetics': kinetics, 'non_stationary': non_stationary, 'sequence': sequence}
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

        # Test validity of parameters
        if kinetics.startswith('2'):
            if self._pars['c_a'].size != self._pars['c_v'].size:
                raise ValueError("Arterial- and venous inputs have different lengths")

    def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['kinetics'])
        return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    def print_params(self, *args, round_to=None, group=None, 
                     fixed_only=False, free_only=False, deriv=False):
        """Pretty print model parameters"""
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['kinetics'])
        if args != ():
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self._params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self._params('free')}
        print_params(pars, round_to=round_to, group=group)

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
        return self._R1, self._R2s

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
        def conc(input: Input):
            p = self._pars
            seq = self._cnfg['sequence']
            rp = const.r1(p['field_strength'], 'blood', p['agent'])
            ci = SignalToConc(seq, **p)(
                input.signal, S0=None, R10=input.R10, n0=n0, 
                B1corr=input.B1corr, r1=rp,
            )
            t = np.arange(0, np.amax(time) + p['dt'], p['dt'])
            return np.interp(t, input.time, ci)
      
        if aif is not None: self._pars['c_a'] = conc(Input(aif))
        if vif is not None: self._pars['c_v'] = conc(Input(vif))

        return self._train(time, signal, free, bounds, n0, **kwargs)

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
    # Private API: Backend
    # ==========================================

    def _params(self, select=None):
        if select is None:
            select = 'all'
        pars_kin = ConcLiver(self._cnfg['kinetics'], self._cnfg['non_stationary'])._params()
        seq = self._cnfg['sequence']
        pars_seq = SEQUENCES[seq]['parameters']['prep']
        pars_seq += SEQUENCES[seq]['parameters']['read']

        if select == 'all':
            pars_list = [
                'c_a', 'dt', 'field_strength', 'agent',
                'H', 'T_a', 'S0', 'R10', 'R20s', 'TS'
            ]
            pars_list += pars_kin + pars_seq
            if self._cnfg['kinetics'].startswith('2'):
                pars_list += ['c_v']
        elif select=='free':
            pars_list = pars_kin
        return pars_list

    # ==========================================
    # Forward Model
    # ==========================================

    def _compute_concentration(self):
        p = self._pars
        ca_plasma = p['c_a'] / (1 - p['H'])
        if 'c_v' in p:
            ca_plasma = (ca_plasma, p['c_v'] / (1 - p['H']))

        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        self._C = ConcLiver(kin, ns, **p)(ca_plasma, dt=p['dt'])

    def _compute_relaxation_rate(self):
        self._compute_concentration()
        p = self._pars

        rp = const.r1(p['field_strength'], 'blood', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        if self._C.shape[0] == 2:
            self._R1 = p['R10'] + rp * self._C[0, :] + rh * self._C[1, :]
            self._R2s = p['R20s'] + r2s * self._C.sum(axis=0)
        # else:
        #     self._R1 = p['R10'] + rp * self._C[0,:]
        #     self._R2s = p['R20s'] + r2s * self._C[0,:]

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars
        seq = self._cnfg['sequence']
        self._S = Signal(seq, **p)(R1=self._R1, R2s=self._R2s)

    def _set_time(self):
        p = self._pars
        self._t = p['dt'] * np.arange(p['c_a'].size)

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
        s_ref = Signal(seq, **p)(R1=p['R10'], R2s=p['R20s'], S0=1)
        p['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: np.ndarray, signal: np.ndarray, 
        free: dict, bounds: dict, n0: int, **kwargs
    ):
        self._estimate_parameters(signal, n0)
        free = self._set_free_pars(free, bounds)
        return train(self._predict, time, signal, self._pars, free, **kwargs)
   
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
        if self._C.shape[0] == 2:
            ax1.plot(self._t/60, 1000*self._C[0,:], '-.', linewidth=3, color='darkblue', label='Extracellular')
            ax1.plot(self._t/60, 1000*self._C[1,:], '-', linewidth=3, color='green', label='Hepatocytes]')
        # else:
        #     ax1.plot(self._t/60, 1000*self._C[0,:], '-', linewidth=3, color='cornflowerblue', label='Liver')

        ax1.plot(self._t/60, 1000*p['c_a'], '-', linewidth=3, color='darkred', label='Artery')
        if 'c_v' in p:
            ax1.plot(self._t/60, 1000*p['c_v'], '-', linewidth=3, color='purple', label='Portal Vein')

        ax1.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax1.legend()

        if fname: plt.savefig(fname)
        if show: plt.show()
        else: plt.close()

