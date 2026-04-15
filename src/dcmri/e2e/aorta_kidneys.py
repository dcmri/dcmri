from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import magnetization, const
from dcmri.lexicon import SEQUENCES
from dcmri.kinetics import ConcAorta
from dcmri.utils.misc import sample
from dcmri.utils.fit import train, loss
from dcmri.core import SuperModel
from dcmri.kinetics import ConcKidney


class AortaKidneys(SuperModel):
    """Joint model for signals from aorta and both kidneys.

    This model uses a whole body model to simultaneously predict 
    signals in aorta and kidneys (see :ref:`whole-body-tissues`). 

    See Also:
        `Aorta`, `Kidney`

    Args:
        organs (str, optional): Model for the organs in the whole-body 
          model. The options are 'comp' (one compartment) and '2cxm' 
          (two-compartment exchange). Defaults to 'comp'.
        heartlung (str, optional): Model for the heart-lung system in 
          the whole-body model. Options are 'pfcomp' (plug-flow 
          compartment) or 'chain'. Defaults to 'pfcomp'.
        kidneys (str, optional): Model for the kidneys. Options are 
          '2CF' (Two-compartment filtration) and 'HF' (High-flow). 
          Defaults to '2CF'. 
        sequence (str, optional): imaging sequence model. Possible 
          values are 'SS' (steady-state), 'SR' (saturation-recovery), 
          'SSI' (steady state with inflow correction) and 'lin' 
          (linear). Defaults to 'SS'.
        agent (str, optional): Generic name of the contrast agent 
          injected. Defaults to 'gadoterate'.
        params (dict, optional): values for the model parameters,
          specified as keyword parameters. Defaults are used for any 
          that are not provided. See table 
          :ref:`AortaKidneys-defaults` for a list of parameters and 
          their default values.


    Example:

        Use the model to fit minipig data with inflow correction:

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

        Create an array of time points:

        >>> time = pars['TS'] * np.arange(len(rois['Aorta']))

        Initialize the tissue:

        >>> aorta_kidneys = dc.AortaKidneys(
        ...     sequence='SSI',
        ...     heartlung='chain',
        ...     organs='comp',
        ...     agent="gadoterate",
        ...     dt=0.25,
        ...     field_strength=pars['B0'],
        ...     weight=pars['weight'],
        ...     dose=pars['dose'],
        ...     rate=pars['rate'],
        ...     R10a=1/dc.const.T1(pars['B0'], 'blood'),
        ...     R10_lk=1/dc.const.T1(pars['B0'], 'kidney'),
        ...     R10_rk=1/dc.const.T1(pars['B0'], 'kidney'),
        ...     vol_lk=85,
        ...     vol_rk=85,
        ...     TR=pars['TR'],
        ...     FA=pars['FA'],
        ...     TS=pars['TS'],
        ...     CO=60,   
        ...     t0=15, 
        ... )

        Define time and signal data

        >>> t = (time, time, time)
        >>> signal = (rois['Aorta'], rois['LeftKidney'], rois['RightKidney'])

        Train the system to the data:

        >>> aorta_kidneys.train(t, signal)

        Plot the reconstructed signals and concentrations:

        >>> aorta_kidneys.plot(t, signal)

        Print the model parameters:

        >>> aorta_kidneys.print_params(round_to=4)
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        Bolus arrival time (BAT): 16.7422 (0.2853) sec
        Inflow time (TF): 0.2801 (0.0133) sec
        Cardiac output (CO): 72.762 (12.4426) mL/sec
        Heart-lung mean transit time (Thl): 16.2249 (0.3069) sec
        Organs blood mean transit time (To): 14.3793 (1.2492) sec
        Body extraction fraction (Eb): 0.0751 (0.0071)
        Heart-lung dispersion (Dhl): 0.0795 (0.0041)
        Renal plasma flow (RPF): 3.3489 (0.7204) mL/sec
        Differential renal function (DRF): 0.9085 (0.0212)
        Differential renal plasma flow (DRPF): 0.812 (0.0169)
        Left kidney arterial mean transit time (Ta_lk): 0.6509 (0.2228) sec
        Left kidney plasma volume (vp_lk): 0.099 (0.0186) mL/cm3
        Left kidney tubular mean transit time (Tt_lk): 46.9705 (3.3684) sec
        Right kidney arterial mean transit time (Ta_rk): 1.4206 (0.2023) sec
        Right kidney plasma volume (vp_rk): 0.1294 (0.0175) mL/cm3
        Right kidney tubular mean transit time (Tt_rk): 4497.8301 (39890.3818) sec
        Aorta signal scaling factor (S0a): 4912.776 (254.2363) a.u.
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        Filtration fraction (FF): 0.0812
        Glomerular Filtration Rate (GFR): 0.2719 mL/sec
        Left kidney plasma flow (RPF_lk): 2.7194 mL/sec
        Right kidney plasma flow (RPF_rk): 0.6295 mL/sec
        Left kidney glomerular filtration rate (GFR_lk): 0.247 mL/sec
        Right kidney glomerular filtration rate (GFR_rk): 0.0249 mL/sec
        Left kidney plasma flow (Fp_lk): 0.032 mL/sec/cm3
        Left kidney plasma mean transit time (Tp_lk): 2.838 sec
        Left kidney vascular mean transit time (Tv_lk): 3.0958 sec
        Left kidney tubular flow (Ft_lk): 0.0029 mL/sec/cm3
        Left kidney filtration fraction (FF_lk): 0.0908
        Left kidney extraction fraction (E_lk): 0.0833
        Right kidney plasma flow (Fp_rk): 0.0074 mL/sec/cm3
        Right kidney plasma mean transit time (Tp_rk): 16.8121 sec
        Right kidney vascular mean transit time (Tv_rk): 17.4762 sec
        Right kidney tubular flow (Ft_rk): 0.0003 mL/sec/cm3
        Right kidney filtration fraction (FF_rk): 0.0395
        Right kidney extraction fraction (E_rk): 0.038
    """

    configs = ConcAorta.configs | {
        'kidneys': ['2CF', 'HF'],
        'sequence': ['3D-SPGR-SS', '3D-SR-SPGR-SS', '3D-SPGR-SSI'],
        'liver_clearance': [True, False],
      }

    def __init__(
        self, 
        heartlung='pfcomp', 
        organs='comp', 
        kidneys='2CF', 
        sequence='3D-SPGR-SS',
        liver_clearance=False, 
        **params,
    ):
        cnfg = {
            'heartlung': heartlung, 
            'organs': organs, 
            'kidneys': kidneys, 
            'sequence': sequence, 
            'liver_clearance': liver_clearance,
        }
        self._version = '1.0'
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)
    
    def _params(self, select='all'):
        
        seq = self._cnfg['sequence']

        aorta_conc = ConcAorta(self._cnfg['heartlung'], self._cnfg['organs'])

        kidneys = {
            '2CF': ['DRPF'],
            'HF': [],
        }[self._cnfg['kidneys']]

        sequence = SEQUENCES[seq]['parameters']['prep']
        sequence += SEQUENCES[seq]['parameters']['read']
        sequence = [x for x in sequence if x not in ['S0', 'B1corr']]
        
        inflow = ['TF'] if seq == '3D-SPGR-SSI' else []
        free_inflow = ['TF', 'S0_a'] if seq == '3D-SPGR-SSI' else []

        agent = ['FF'] if self._cnfg['liver_clearance'] else []

        pars_list = {
            'all': aorta_conc._params() + inflow + kidneys + agent + sequence + [
                'TS',
                'H', 'RPF', 'DRF',
                'T_a_lk', 'vp_lk', 'Tt_lk', 'vol_lk', 
                'T_a_rk', 'vp_rk', 'Tt_rk', 'vol_rk', 
                'R10_a', 'R10_lk', 'R10_rk',
                'S0_a', 'S0_lk', 'S0_rk',
                'B1corr_a', 'B1corr_lk', 'B1corr_rk',
            ],
            'free': aorta_conc._params('body') + free_inflow + kidneys + agent + [
                'RPF', 'DRF',
                'T_a_lk', 'vp_lk', 'Tt_lk',
                'T_a_rk', 'vp_rk', 'Tt_rk',
            ],
            'free_aorta': aorta_conc._params('body') + free_inflow,
            'free_kidneys': kidneys + agent + [
                'RPF', 'DRF',
                'T_a_lk', 'vp_lk', 'Tt_lk',
                'T_a_rk', 'vp_rk', 'Tt_rk'
            ],
        }
        return pars_list[select]
    

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _compute_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])

    def _compute_conc_aorta(self) -> np.ndarray:
        hl, orgs = self._cnfg['heartlung'], self._cnfg['organs']
        self._ca = ConcAorta(hl, orgs)(**self._pars)

    def _compute_relax_aorta(self):  
        self._compute_conc_aorta()
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1a = p['R10_a'] + rb * self._ca

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars
        self._Sa = magnetization.Signal(self._cnfg['sequence'], **p)(
            R1=self._R1a, 
            S0=p['S0_a'], 
            B1corr=p['B1corr_a'],
            TE=0, PA=0,
        )

    def _predict_aorta(self, time):
        self._compute_time()
        self._compute_signal_aorta()
        p = self._pars
        return sample(time, self._t, self._Sa, p['TS'])
    
    # ==========================================
    # Forward Model: Kidneys
    # ==========================================

    def _compute_conc_kidneys(self):
        p = self._pars

        if self._cnfg['liver_clearance']:
            FF = p['FF']
        else:
            FF = p['Eb'] / (1-p['Eb'])
        GFR = FF * p['RPF']
        Ft = {
            'lk': p['DRF'] * GFR / p['vol_lk'],
            'rk': (1 - p['DRF']) * GFR / p['vol_rk']
        }

        ca = self._ca / (1 - p['H'])
        conc = ConcKidney(self._cnfg['kidneys'])
        pk = conc.params()
        self._Ck = {}

        for k in ['lk', 'rk']:
            pk['T_a'] = p[f'T_a_{k}']
            pk['vp'] = p[f'vp_{k}']
            pk['Tt'] = p[f'Tt_{k}']
            pk['Ft'] = Ft[k]
            if 'Fp' in pk:
                pk['Fp'] = {
                    'lk': p['DRPF'] * p['RPF'] / p[f'vol_lk'],
                    'rk': (1 - p['DRPF']) * p['RPF'] / p['vol_rk']
                }[k]   
            self._Ck[k] = conc(ca, dt=p['dt'], **pk)              

    def _compute_relax_kidneys(self):
        self._compute_conc_kidneys()
        p = self._pars
        rb = const.r1(p['field_strength'], 'blood', p['agent'])
        self._R1k = {}
        for k in ['lk', 'rk']:
            self._R1k[k] = p[f'R10_{k}'] + rb * self._Ck[k].sum(axis=0) 

    def _compute_signal_kidneys(self):
        self._compute_relax_kidneys()
        p = self._pars
        seq = '3D-SPGR-SS' if self._cnfg['sequence']=='3D-SPGR-SSI' else self._cnfg['sequence']

        self._Sk = {}
        for k in ['lk', 'rk']:
            self._Sk[k] = magnetization.Signal(seq, **p)(
                R1=self._R1k[k], 
                S0=p[f'S0_{k}'], 
                B1corr=p[f'B1corr_{k}'], 
                TE=0,
            )

    def _predict_kidneys(self, time):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_kidneys()
        return (
            sample(time[0], self._t, self._Sk['lk'], p['TS']),
            sample(time[1], self._t, self._Sk['rk'], p['TS']),
        )
    
    # ===========================================
    # Forward Model: Liver, Portal Vein and Aorta
    # ===========================================
    
    def _predict(self, time: dict) -> dict:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        aorta = self._predict_aorta(time[0])
        kidneys = self._predict_kidneys(time[1:])
        return (aorta, kidneys[0], kidneys[1])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: dict, signal: dict, n0: int):
        p = self._pars
        p['tmax'] = np.max(np.concatenate(time)) + p['dt'] + p['TS']

        # Estimate BAT based on peak signal
        if self._cnfg['heartlung']=='comp':
            offset = p['Thl']
        else:
            offset = (1 - p['Dhl']) * p['Thl']
        bat = time[0][np.argmax(signal[0])] - offset
        p['BAT'] = max(bat, 0)

        # Estimate scaling Factors (S0)
        seq = self._cnfg['sequence']
        seq = {
            'a': seq,
            'lk': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
            'rk': '3D-SPGR-SS' if seq=='3D-SPGR-SSI' else seq,
        }
        idx = {
            'a': 0, 
            'lk': 1, 
            'rk': 2,
        }
        for roi in idx.keys():
            s_ref = magnetization.Signal(seq[roi], **p)(
                R1=p[f'R10_{roi}'], 
                S0=1, 
                B1corr=p[f'B1corr_{roi}'], 
                TE=0,
            )
            p[f'S0_{roi}'] = np.mean(signal[idx[roi]][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: dict, signal: dict, free: dict, 
        bounds: dict, n0: int, staged: bool, **kwargs
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == '3D-SPGR-SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")

        if staged:

            # Optimize Aorta parameters
            free_stage = {k: v for k, v in free.items() if k in self._params('free_aorta')}
            train(self._predict_aorta, time[0], signal[0], self._pars, free_stage, **kwargs)

            # Optimize Kidney parameters
            free_stage = {k: v for k, v in free.items() if k in self._params('free_kidneys')}
            train(self._predict_kidneys, time[1:], signal[1:], self._pars, free_stage, **kwargs)

        # Joint Optimization
        return train(self._predict, time, signal, self._pars, free, **kwargs)

    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _plot(self, time, signal, xlim, fname, show):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_aorta()
        self._compute_signal_kidneys()

        if xlim is None: xlim = [self._t[0], self._t[-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(wspace=0.3)

        def plot_data1scan(sig, t, s, roi, ax, color):
            ax.set(xlabel='Time (min)', ylabel=f'{roi} signal (a.u.)', xlim=xlim)
            ax.plot(t / 60, s, marker='o', color=color[0], label='dict', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=color[1], linewidth=3.0, label='Prediction')
            ax.legend()

        plot_data1scan(self._Sa, time[0], signal[0], 'Aorta', ax1, ['lightcoral', 'darkred'])
        plot_data1scan(self._Sk['lk'], time[1], signal[1], 'Left kidney', ax3, ['cornflowerblue', 'darkblue'])
        plot_data1scan(self._Sk['rk'], time[2], signal[2], 'Right kidney', ax5, ['cornflowerblue', 'darkblue'])

        # Plot aorta
        cb_lk = self._Ck['lk'][0,:] / (p['vp_lk'] / (1 - p['H']))
        cb_rk = self._Ck['rk'][0,:] / (p['vp_rk'] / (1 - p['H']))

        ax2.set(xlabel='Time (min)', ylabel='Blood conc (mM)', xlim=xlim)
        ax2.plot(self._t / 60, 0 * self._t, color='gray')
        ax2.plot(self._t / 60, 1000 * self._ca, linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.plot(self._t / 60, 1000 * cb_lk, linestyle='--', color='lightcoral', linewidth=2.0, label='Left kidney')
        ax2.plot(self._t / 60, 1000 * cb_rk, linestyle='-.', color='lightcoral', linewidth=2.0, label='Right kidney')
        ax2.legend()

        def plot_conc_kidney(C, kid, ax):
            ax.set(xlabel='Time (min)', ylabel=f'{kid} conc (mM)', xlim=xlim)
            ax.plot(self._t / 60, 0 * self._t, color='gray')
            ax.plot(self._t / 60, 1000 * C[0, :], linestyle='-', color='darkred', linewidth=2.0, label='Blood')
            ax.plot(self._t / 60, 1000 * C[1, :], linestyle='-', color='darkcyan', linewidth=2.0, label='Tubuli')
            ax.plot(self._t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
            ax.legend()

        plot_conc_kidney(self._Ck['lk'], 'Left kidney', ax4)
        plot_conc_kidney(self._Ck['rk'], 'Right kidney', ax6)

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # Public API: dict Extraction
    # ==========================================

    def time(self) -> dict:
        """Internal time array

        Returns:
            Tuple: (aorta_time, portal_time, liver_time).
        """
        self._compute_time()
        return {
            'aorta': self._t, 
            'kidney_left': self._t, 
            'kidney_right': self._t,
        }

    def conc(self) -> dict:
        """Concentrations in aorta and kidney.

        Args:
            sum (bool, optional): If set to true, the kidney 
              concentrations are the sum over all compartments. If 
              set to false, the compartmental concentrations are 
              returned individually. Defaults to True.

        Returns:
            tuple: time points, aorta blood concentrations, left 
              kidney concentrations, right kidney concentrations.
        """
        self._compute_conc_aorta()
        self._compute_conc_kidneys()
        return {
            'aorta': self._ca, 
            'kidney_left': self._Ck['lk'], 
            'kidney_right': self._Ck['rk'],
        }

    def relax(self) -> dict:
        """Relaxation rates in aorta and kidney.

        Returns:
            tuple: time points, aorta relaxation rate, left kidney 
              relaxation rate, right kidney relaxation rate.
        """
        self._compute_relax_aorta()
        self._compute_relax_kidneys()
        return {
            'aorta': self._R1a, 
            'kidney_left': self._R1k['lk'], 
            'kidney_right': self._R1k['rk'],
        }
    
    def signal(self) -> dict:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal_aorta()
        self._compute_signal_kidneys()
        return {
            'aorta': self._Sa, 
            'kidney_left': self._Sk['lk'], 
            'kidney_right': self._Sk['rk'],
        }

    def predict(self, time) -> dict:
        """Predict the data at given time

        Args:
            time (tuple): Tuple of 3 arrays with time points for 
              aorta, left kidney and right kidney, in that order. 
              The three arrays can all be different in length and value.

        Returns:
            tuple: Tuple of 3 arrays with signals for aorta, left 
              kidney and right kidney, in that order. The three 
              arrays can all be different in length and value but 
              each has to have the same length as its corresponding 
              array of time points.
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        else:
            time = tuple(3 * [time])

        signal = self._predict(time)

        return {
            'aorta': signal[0],
            'kidney_left': signal[1],
            'kidney_right': signal[2],
        }

    def train(
        self, time: dict, signal: dict, free: dict = None, 
        bounds: dict = None, n0=1, staged=False, **kwargs
    ) -> Tuple[dict, dict, np.ndarray]:
        """Train the free parameters

       Args:
            time (tuple): (time_aorta, time_portal, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_portal, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific params.
            n0 (int, optional): Baseline points for S0 estimation. Defaults to 1.
            staged (bool, optional): If True, the training is performed in stages
            **kwargs: Passed to scipy.optimize.curve_fit via utils.train.

        Returns:
            vals, sdev, pcov: Values, standard deviations and 
              covariance matrix of free parameters

        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        else:
            time = tuple(3 * [time])
        signal = (
            signal['aorta'], 
            signal['kidney_left'], 
            signal['kidney_right'], 
        )
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)

    def plot(
        self, time: dict, signal: dict, xlim: list = None, 
        fname: str = None, show = True,
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 3 arrays with time points for aorta, 
              portal vein and liver, in that order. The two arrays can be 
              different in length and value.
            signal (array-like): tuple of 3 arrays with signals for aorta, 
              portal vein and liver, in that order. The arrays can be 
              different in length and value but each has to have the same 
              length as its corresponding array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        if isinstance(time, dict):
            time = (
                time['aorta'], 
                time['kidney_left'],
                time['kidney_right'], 
            )
        else:
            time = tuple(3 * [time])
        signal = (
            signal['aorta'], 
            signal['kidney_left'], 
            signal['kidney_right'], 
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
                time['aorta'], 
                time['kidney_left'], 
                time['kidney_right'], 
            )
        else:
            time = tuple(3 * [time])
        signal = np.concatenate((
            signal['aorta'], 
            signal['kidney_left'], 
            signal['kidney_right'], 
        ))
        signal_pred = np.concatenate(self._predict(time))
        cost = loss(signal_pred.reshape(1, -1), signal.reshape(1, -1), metric, nfree)
        return cost[0]




