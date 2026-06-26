"""Joint model for aorta and liver signals measured over two scans.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver, measured over two separate scans.

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    kinetics (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only single-inlet models 
        are allowed. Defaults to '1I-IC-HFD'.
    stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to 'UE'.
    stationary (str, optional): Stationarity regime of the hepatocytes. 
        The options are 'UE', 'E', 'U' or None. For more detail 
        see :ref:`liver-tissues`. Defaults to 'UE'.
    sequence (str, optional): imaging sequence. Possible values are 'SS'
        and 'SR'. Defaults to 'SS'.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`AortaLiver2scan-parameters` and
        :ref:`AortaLiver2scan-defaults` for a list of parameters and their
        default values.

See Also:
    `AortaLiver`

Example:

    Use the model to reconstruct concentrations from experimentally 
    derived signals.

.. plot::
    :include-source:
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import dcmri as dc

    Use `fake.tissue` to generate synthetic test data from 
    experimentally-derived concentrations:

    >>> time, aif, roi, gt = dc.fake.tissue2scan(R1b=1/dc.const.T1(3.0,'liver'))

    Since this model generates four time curves, the x- and y-data are 
    tuples:

    >>> time = (time[0], time[1], time[0], time[1])
    >>> signal = (aif[0], aif[1], roi[0], roi[1])

    Build an aorta-liver model and parameters to match the conditions of 
    the fake tissue data:

    >>> model = dc.AortaLiver2scan(
    ...     dt = 0.5,
    ...     tmax = 420,
    ...     weight = 70,
    ...     agent = 'gadodiamide',
    ...     dose = 0.2,
    ...     dose2 = 0.2,
    ...     rate = 3,
    ...     field_strength = 3.0,
    ...     TR = 0.005,
    ...     FA = 15,
    ...     FA2 = 15,
    ...     TS = 0.5,
    ...     Th_i = 120,
    ...     Th_f = 120,
    ... )

    In this case we have defined different initial values for Th as 
    the defaults are optimized for the slow passage through hepatocytes. 
    We also need to reset the parameter bounds:

    >>> model.free['Th_i'] = [0, np.inf]
    >>> model.free['Th_f'] = [0, np.inf]

    Train the model on the data:

    >>> model.train(time, signal, n0=10, xtol=1e-3)

    Plot the reconstructed signals and concentrations and compare against 
    the experimentally derived data:

    >>> model.plot(time, signal)

    We can also have a look at the model parameters after training:

    >>> model.print_params(round_to=3)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    Aorta second signal scale factor (S02a): 195.824 (2.025) a.u.
    Liver second signal scale factor (S02l): 297.854 (4.9) a.u.
    Second bolus arrival time (BAT2): 254.512 (0.137) sec
    First bolus arrival time (BAT): 14.288 (0.132) sec
    Cardiac output (CO): 203.199 (5.406) mL/sec
    Heart-lung mean transit time (Thl): 15.236 (0.263) sec
    Heart-lung dispersion (Dhl): 0.381 (0.009)
    Organs blood mean transit time (To): 23.761 (3.052) sec
    Organs extraction fraction (Eo): 0.287 (0.053)
    Organs extravascular mean transit time (Toe): 50.274 (17.44) sec
    Body extraction fraction (Eb): 0.078 (0.015)
    Apparent liver extracellular volume fraction (ve_app): 0.053 (0.008) mL/cm3
    Extracellular mean transit time (Te): 1.298 (0.552) sec
    Extracellular dispersion (De): 1.0 (0.7)
    Initial hepatic plasma clearance (Ktrans_i): 0.005 (0.001) mL/sec/cm3
    Final hepatic plasma clearance (Ktrans_f): 0.005 (0.001) mL/sec/cm3
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
    ----------------------------
    Fixed and derived parameters
    ----------------------------
    Aorta first baseline R1 (R1ba): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.117 a.u.
    Liver first baseline R1 (R1bl): 1.33 Hz
    Liver first signal scale factor (S0(l)): 150.003 a.u.
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.utils import const
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.kinetics.functions_input import ca_injection
from dcmri.kinetics.modules_conc import ConcLiver
from dcmri.kinetics.functions_aorta import flux_aorta
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.core.tools import print_params, export_params
from dcmri.core.sequences import SEQUENCES
from dcmri.core.quantities import QUANTITIES
from dcmri.bloch.tissue import Signal
from dcmri.core.model import SuperModel

CONSTANTS = {'Fw': 0, 'v': 1, 'me': 1, 'noise_sdev':0}


QUANTITIES = QUANTITIES | {
    'tmax': {'init': 4 * 60 * 60, 'name': 'Maximum acquisition time', 'unit': 'sec', 'group': 'hyper'},
    't_scan2': {'init': 2 * 60 * 60, 'name': 'Start of second scan', 'unit': 'sec', 'group': 'signal'},
    'dose_1': {'init': 0.05, 'name': 'First contrast agent dose', 'unit': 'mL/kg', 'group': 'indicator'},
    'dose_2': {'init': 0.05, 'name': 'Second contrast agent dose', 'unit': 'mL/kg', 'group': 'indicator'},
    'BAT_1': {'init': 120, 'bounds': [-60, 60], 'name': 'First bolus arrival time', 'unit': 'sec', 'group': 'signal', 'bounds_type': 'add'},
    'BAT_2': {'init': 7200 + 900, 'bounds': [-60, 60], 'name': 'Second bolus arrival time', 'unit': 'sec', 'group': 'signal', 'bounds_type': 'add'},
    'S0_1_a': {'init': 1, 'bounds': [0, 2], 'name': 'Aorta first signal scale factor', 'unit': 'a.u.', 'group': 'signal', 'bounds_type': 'mult'},
    'S0_1_l': {'init': 1, 'bounds': [0, 2], 'name': 'Liver first signal scale factor', 'unit': 'a.u.', 'group': 'signal', 'bounds_type': 'mult'},
    'S0_2_a': {'init': 1, 'bounds': [0, 2], 'name': 'Aorta second signal scale factor', 'unit': 'a.u.', 'group': 'signal', 'bounds_type': 'mult'},
    'S0_2_l': {'init': 1, 'bounds': [0, 2], 'name': 'Liver second signal scale factor', 'unit': 'a.u.', 'group': 'signal', 'bounds_type': 'mult'},
    'B1corr_1_a': {'init': 1, 'bounds': [0, 5], 'name': 'Arterial B1-correction factor', 'unit': '', 'group': 'signal'},
    'B1corr_1_l': {'init': 1, 'bounds': [0, 5], 'name': 'Liver B1-correction factor', 'unit': '', 'group': 'signal'},
    'B1corr_2_a': {'init': 1, 'bounds': [0, 5], 'name': 'Arterial B1-correction factor of a second scan', 'unit': '', 'group': 'signal'},
    'B1corr_2_l': {'init': 1, 'bounds': [0, 5], 'name': 'Liver B1-correction factor of a second scan', 'unit': '', 'group': 'signal'},
}

class AortaLiverDynamic(SuperModel):
    """Aorta and liver signals measured over two scans.

    A whole-body model to simultaneously predict signals in 
    aorta and liver, measured over two separate scans.

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `AortaLiver`
    """

    # ==========================================
    # User interface
    # ==========================================

    configs = {
        'kinetics': ['1I-EC', '1I-EC-HF', '1I-IC', '1I-IC-HF'],
        'non_stationary': [None, 'U', 'E', 'UE'],
        'sequence': ['ZTE-3D-SPGR-SS', '3D-SPGR-SS', '3D-SPGR-SSI']
    }

    def __init__(
        self, 
        kinetics = '1I-IC-HF', 
        non_stationary=None, 
        sequence='ZTE-3D-SPGR-SS', 
        **params,
      ):
        
        cnfg = {
            'kinetics': kinetics, 
            'non_stationary': non_stationary, 
            'sequence': sequence,
        }
        self._version = '1.0'
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(QUANTITIES, **params)

    def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._cnfg['kinetics'])
        return export_params(pars, lexicon=QUANTITIES, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

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
        print_params(pars, round_to=round_to, group=group, lexicon=QUANTITIES)

    def time(self) -> dict:
        """Internal time array
        
        Returns:
            tuple: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.      
        """
        self._set_time()
        p = self._pars
        t, t2 = self._t, p['t_scan2']
        tacq1, tacq2 = t[t < t2], t[t >= t2]
        return {
            ('aorta', 1): tacq1, 
            ('aorta', 2): tacq2, 
            ('liver', 1): tacq1, 
            ('liver', 2): tacq2, 
        }
    
    def conc(self) -> dict:
        """Concentrations in aorta and liver.

        Returns:
            tuple: aorta conc scan 1, aorta conc scan 2, 
              liver conc scan 1, liver conc scan 2.
        """
        self._compute_conc_aorta()
        self._compute_conc_liver()
        t, t2 = self._t, self._pars['t_scan2']
        return {
            ('aorta', 1): self._ca[t < t2], 
            ('aorta', 2): self._ca[t >= t2], 
            ('liver', 1): self._Cl[:, t < t2], 
            ('liver', 2): self._Cl[:, t >= t2], 
        }
    
    def relax(self) -> dict:
        """Relaxation rates in aorta and liver.

        Returns:
            tuple: aorta R1 scan 1, aorta R1 scan 2, 
              liver R1 scan 1, liver R1 scan 2.
        """
        self._compute_relax_aorta()
        self._compute_relax_liver()
        t, t2 = self._t, self._pars['t_scan2']
        R1 = {
            ('aorta', 1): self._R1a[t < t2], 
            ('aorta', 2): self._R1a[t >= t2], 
            ('liver', 1): self._R1l[t < t2], 
            ('liver', 2): self._R1l[t >= t2], 
        }
        R2s = {
            ('aorta', 1): self._R2sa[t < t2], 
            ('aorta', 2): self._R2sa[t >= t2], 
            ('liver', 1): self._R2sl[t < t2], 
            ('liver', 2): self._R2sl[t >= t2], 
        }
        return R1, R2s
    
    def signal(self) -> dict:
        """Signal in aorta and liver.

        Returns:
            tuple: aorta signal scan 1, aorta signal scan 2, 
              liver signal scan 1, liver signal scan 2.
        """
        self._compute_signal_aorta()
        self._compute_signal_liver()
        t, t2 = self._t, self._pars['t_scan2']
        return {
            ('aorta', 1): self._Sa[t < t2], 
            ('aorta', 2): self._Sa[t >= t2], 
            ('liver', 1): self._Sl[t < t2], 
            ('liver', 2): self._Sl[t >= t2], 
        }
    
    def predict(self, time: dict) -> dict:
        """Predict the data at given time points

        Args:
            time: aorta time scan 1, aorta time scan 2, 
              liver time scan 1, liver time scan 2.

        Returns:
            tuple: aorta data scan 1, aorta data scan 2, 
              liver data scan 1, liver data scan 2.
        """
        if isinstance(time, dict):
            time = (
                time[('aorta', 1)], 
                time[('aorta', 2)], 
                time[('liver', 1)], 
                time[('liver', 2)], 
            )
        elif len(time) == 2:
            time = tuple([time[0], time[1], time[0], time[1]])

        signal = self._predict(time)

        return {
            ('aorta', 1): signal[0],
            ('aorta', 2): signal[1],
            ('liver', 1): signal[2],
            ('liver', 2): signal[3],
        }
    
    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=1, R1b2a: float = None, 
        R1b2l: float = None, staged=False, **kwargs,
    ) -> tuple:
        """Train the free parameters

        Args:
            time (tuple): (time_1_aorta, time_2_aorta, time_1_liver, time_2_liver)
            signal (tuple): (signal_1_aorta, signal_2_aorta, signal_1_liver, signal_2_liver).
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points. Defaults to 1.
            R1b2a (float, optional): R1 value in arterial blood before the second injection. 
            R1b2l (float, optional): R1 value in liver before the second injection. 
            staged (bool, optional): If True, the training is performed in stages
            kwargs: any other keyword parameters accepted by 
              `scipy.optimize.curve_fit`.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """

        if isinstance(time, dict):
            time = (
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        elif len(time) == 2:
            time = tuple([time[0], time[1], time[0], time[1]])

        if isinstance(signal, dict):
            signal = (
                signal['aorta', 1], 
                signal['aorta', 2], 
                signal['liver', 1], 
                signal['liver', 2], 
            )

        return self._train(time, signal, free, bounds, n0, R1b2a, R1b2l, staged, **kwargs)

    def plot(
        self, time: dict, signal: dict, xlim: list = None, 
        fname: str = None, show=True
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 4 arrays with time points for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The four arrays can 
              be different in length and value.
            signal (tuple): tuple of 4 arrays with signals for aorta in the 
              first scan, aorta in the second stand, liver in the first scan, 
              and liver in the second scan, in that order. The arrays can be 
              different in length but each has to have the same length as its 
              corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """

        if isinstance(time, dict):
            time = (
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        elif len(time) == 2:
            time = tuple([time[0], time[1], time[0], time[1]])
        if isinstance(signal, dict):
            signal = (
                signal['aorta', 1], 
                signal['aorta', 2], 
                signal['liver', 1], 
                signal['liver', 2], 
            )
        self._plot(time, signal, xlim, fname, show)

    def cost(self, time: dict, signal: dict, metric: str = 'NRMS', nfree=None) -> float:
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
        if isinstance(time, dict):
            time = (
                time['aorta', 1], 
                time['aorta', 2], 
                time['liver', 1], 
                time['liver', 2], 
            )
        elif len(time) == 2:
            time = tuple([time[0], time[1], time[0], time[1]])
        if isinstance(signal, dict):
            signal = (
                signal['aorta', 1], 
                signal['aorta', 2], 
                signal['liver', 1], 
                signal['liver', 2], 
            )
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]
    
    # ==========================================
    # Backend
    # ==========================================
   
    def _params(self, select=None):
        if select is None:
            select = 'all'
        kin, ns, seq = self._cnfg['kinetics'], self._cnfg['non_stationary'], self._cnfg['sequence']

        aorta_kinetics = ['BAT_1', 'BAT_2', 'CO', 'Thl', 'Dhl', 'To', 'Eo', 'To_e', 'Eb']
        liver_kinetics = ConcLiver(kin, ns).params()
        liver_kinetics = [k for k in liver_kinetics if k != 'Ta']
        kinetics = aorta_kinetics + liver_kinetics
        liver_sequence = SEQUENCES[seq]['parameters']['prep']
        liver_sequence += SEQUENCES[seq]['parameters']['read']
        if 'FA' in liver_sequence:
            liver_sequence += ['FA_2']
    
        free_inflow = ['TF', 'S0_1_a'] if seq == '3D-SPGR-SSI' else []

        pars_list = {
            'all': kinetics + liver_sequence + [
                'dt', 'tmax', 't_scan2', 'dose_tolerance', 'field_strength', 
                'agent', 'weight', 'dose_1', 'dose_2', 'rate', 
                'TS', 'H', 
                'R1b_a', 'R1b_l', 'R2sb_a', 'R2sb_l', 
                'S0_1_a', 'S0_1_l', 'S0_2_a', 'S0_2_l',
                'B1corr_1_l', 'B1corr_1_a', 'B1corr_2_l', 'B1corr_2_a',
                'vol_l', # needed to derive CL
            ],
            'free': kinetics + free_inflow + ['S0_2_a', 'S0_2_l'],
            'free_aorta': aorta_kinetics + free_inflow + ['S0_2_a'],
            'free_liver': liver_kinetics + ['S0_2_l'],
        }
        return pars_list[select]

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])
        
    def _compute_conc_aorta(self):
        self._set_time()
        p = self._pars

        conc = const.ca_conc(p['agent'])
        J1 = ca_injection(
            self._t, p['weight'], conc, p['dose_1'], p['rate'], p['BAT_1']
        )
        J2 = ca_injection(
            self._t, p['weight'], conc, p['dose_2'], p['rate'], p['BAT_2']
        )
        Jb = flux_aorta(
            J1 + J2, dt=p['dt'], tol=p['dose_tolerance'],
            heartlung={'model': 'pfcomp', 'params': {'T':p[f'Thl'], 'D':p[f'Dhl']}},
            organs=[{'vr': 1 - p['Eb'], 'model': '2cxm', 'params': {'T':[p[f'To'], p[f'To_e']], 'E':p[f'Eo']}}],
        )
        self._ca = Jb / p['CO']

    def _compute_relax_aorta(self):
        self._compute_conc_aorta()
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1a = p['R1b_a'] + rb * self._ca
        r2s = const.r2s(p['field_strength'], 'blood', p['agent'])
        self._R2sa = p['R2sb_a'] + r2s * self._ca

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars

        self._Sa = np.zeros_like(self._t)
        seq = self._cnfg['sequence']

        # First scan signal
        t = self._t < p['t_scan2']
        self._Sa[t] = Signal(seq, defaults=p)(R1=self._R1a[t], R2s=self._R2sa[t], S0=p['S0_1_a'], B1corr=p['B1corr_1_a'], **CONSTANTS)

        # Second scan signal
        t = self._t >= p['t_scan2']
        self._Sa[t] = Signal(seq, defaults=p)(R1=self._R1a[t], R2s=self._R2sa[t], S0=p['S0_2_a'], B1corr=p['B1corr_2_a'], FA=p['FA_2'], **CONSTANTS)

    def _predict_aorta(self, time: tuple):
        self._compute_signal_aorta()
        p = self._pars
        return (
            sample(time[0], self._t, self._Sa, p['TS']),
            sample(time[1], self._t, self._Sa, p['TS']),
        )
    
    # ==========================================
    # Forward Model: Liver
    # ==========================================

    def _compute_conc_liver(self):
        p = self._pars
        cp = self._ca / (1 - p['H'])
        kin, ns = self._cnfg['kinetics'], self._cnfg['non_stationary']
        self._Cl = ConcLiver(kin, ns, defaults=p)(ci=cp, Ta=0, dt=p['dt'])

    def _compute_relax_liver(self):
        self._compute_conc_liver()
        p = self._pars
        rp = const.r1(p['field_strength'], 'plasma', p['agent'])
        rh = const.r1(p['field_strength'], 'hepatocytes', p['agent'])
        r2s = const.r2s(p['field_strength'], 'tissue', p['agent'])
        if self._Cl.shape[0] == 2:
            self._R1l = p['R1b_l'] + rp * self._Cl[0, :] + rh * self._Cl[1, :]
            self._R2sl = p['R2sb_l'] + r2s * self._Cl.sum(axis=0) 
        # else:
        #     self._R1l = p['R1b_l'] + rp * self._Cl[0,:]
        #     self._R2sl = p['R2sb_l'] + r2s * self._Cl[0,:]
        
    def _compute_signal_liver(self):
        self._compute_relax_liver()
        p = self._pars

        self._Sl = np.zeros_like(self._t)
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']

        # First scan signal
        t = self._t < p['t_scan2']
        self._Sl[t] = Signal(seq, defaults=p)(R1=self._R1l[t], R2s=self._R2sl[t], S0=p['S0_1_l'], **CONSTANTS)
        
        # Second scan signal
        t = self._t >= p['t_scan2']
        self._Sl[t] = Signal(seq, defaults=p)(R1=self._R1l[t], R2s=self._R2sl[t], S0=p['S0_2_l'], B1corr=p['B1corr_2_l'], FA=p['FA_2'], **CONSTANTS)

    def _predict_liver(self, time: tuple):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_liver()
        return (
            sample(time[0], self._t, self._Sl, p['TS']),
            sample(time[1], self._t, self._Sl, p['TS']),
        )
    
    # ===========================================
    # Forward Model: Liver and Aorta
    # ===========================================
    
    def _predict(self, time: dict) -> dict:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        s_a = self._predict_aorta(time[:2])
        s_l = self._predict_liver(time[2:])
        return s_a + s_l

    # ==========================================
    # Inverse Model: Training
    # ==========================================   

    def _estimate_parameters(
        self, time: dict, signal: dict, n0: int, R1b2a: float, 
        R1b2l: float
    ):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        seq_aorta = self._cnfg['sequence']
        seq_liver = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']

        # Estimate BAT and BAT2 and ajust their bounds
        t_hl, d_hl = p['Thl'], p['Dhl']
        bat = time[0][np.argmax(signal[0])] - (1 - d_hl) * t_hl
        bat2 = time[1][np.argmax(signal[1])] - (1 - d_hl) * t_hl
        p['BAT_1'] = max(bat, 0)
        p['BAT_2'] = max(bat2, 0)

        # Scaling Factor (S0) aorta
        s_ref = Signal(seq_aorta, defaults=p)(R1=p['R1b_a'], R2s=p['R2sb_a'], S0=1, B1corr=p['B1corr_1_a'], **CONSTANTS)
        p['S0_1_a'] = np.mean(signal[0][:n0]) / s_ref if s_ref > 0 else 0

        # Scaling Factor (S0) liver
        s_ref = Signal(seq_liver, defaults=p)(R1=p['R1b_l'], R2s=p['R2sb_l'], S0=1, B1corr=p['B1corr_1_l'], **CONSTANTS)
        p['S0_1_l'] = np.mean(signal[2][:n0]) / s_ref if s_ref > 0 else 0

        # Second Scaling Factor (S02) aorta
        if R1b2a is None:
            p['S0_2_a'] = p['S0_1_a']
        else:
            s_ref = Signal(seq_aorta, defaults=p)(R1=R1b2a, R2s=p['R2sb_a'], S0=1, B1corr=p['B1corr_2_a'], FA=p['FA_2'], **CONSTANTS)
            p['S0_2_a'] = np.mean(signal[1][:n0]) / s_ref if s_ref > 0 else 0

        # Second Scaling Factor (S02) liver
        if R1b2l is None:
            p['S0_2_l'] = p['S0_1_l']
        else:
            s_ref = Signal(seq_liver, defaults=p)(R1=R1b2l, R2s=p['R2sb_l'], S0=1, B1corr=p['B1corr_2_l'], FA=p['FA_2'], **CONSTANTS)
            p['S0_2_l'] = np.mean(signal[3][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: dict, signal: dict, free: dict, 
        bounds: dict, n0: int, R1b2a: float, R1b2l: float, 
        staged: bool, **kwargs,
    ):
        self._estimate_parameters(time, signal, n0, R1b2a, R1b2l)
        free = self._set_free_pars(free, bounds, lexicon=QUANTITIES)
    
        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI':
            for par in ['S0_1_a', 'S0_2_a']:
                if par not in free:
                    raise ValueError(f"For SSI sequence, '{par}' must be a free parameter.")     

        if staged:
            # Train free aorta parameters on aorta data
            free_aorta = {k: v for k, v in free.items() if k in self._params('free_aorta')}
            train(self._predict_aorta, time[:2], signal[:2], self._pars, free_aorta, **kwargs)

            # Train free liver parameters on liver data
            free_liver = {k: v for k, v in free.items() if k in self._params('free_liver')}
            train(self._predict_liver, time[2:], signal[2:], self._pars, free_liver, **kwargs)

        # Joint Optimization
        return train(self._predict, time, signal, self._pars, free, **kwargs)
    

    # ==========================================
    # I/O and Reporting
    # ==========================================


    def _plot(
        self, time: dict, signal: dict, xlim=None, fname=None, 
        show=True
    ):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))

        self._compute_signal_aorta()
        self._compute_signal_liver()

        if xlim is None: xlim = [self._t[0], self._t[-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)

        # Plot signals
        def _plot_data2scan(sig, t, s, ax, color):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            ax.plot(np.concatenate(t)/60, np.concatenate(s), marker='o', color=color[0], label='fitted data', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan(self._Sa, time[:2], signal[:2], ax1, ['lightcoral', 'darkred'])
        _plot_data2scan(self._Sl, time[2:], signal[2:], ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(self._t / 60, 0 * self._t, color='gray')
        ax2.plot(self._t / 60, 1000 * self._ca, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(self._t / 60, 0 * self._t, color='gray')
        if self._Cl.shape[0]==2:
            ax4.plot(self._t / 60, 1000 * self._Cl[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(self._t / 60, 1000 * self._Cl[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(self._t / 60, 1000 * self._Cl.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        # else:
        #     ax4.plot(self._t / 60, 1000 * self._Cl[0,:], linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')        
        ax4.legend()

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()


