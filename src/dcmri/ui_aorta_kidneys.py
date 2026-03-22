from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dcmri import lib, sig, utils, pk, pk_aorta, ui, kidney
from dcmri.lexicon import LEXICON
import dcmri.lexicon_utils as lexicon

# Shorthand notation for data type hint
Data = Tuple[np.ndarray, np.ndarray, np.ndarray]


class AortaKidneys(ui.SuperModel):
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

    Notes:

        In the table below, if **Bounds** is None, the parameter is fixed 
        during training. Otherwise it is allowed to vary between the 
        bounds given.

        .. _AortaKidneys-defaults:
        .. list-table:: AortaKidneys parameters. 
            :widths: 5 10 5 5 5 10
            :header-rows: 1

            * - Parameter
              - Description
              - Value
              - Unit
              - Bounds
              - Usage
            * - **General**
              - 
              - 
              - 
              - 
              - 
            * - dt
              - Prediction time step
              - 0.25
              - sec
              - None
              - Always
            * - tmax
              - Maximum time predicted
              - 120
              - sec
              - None
              - Always
            * - dose_tolerance
              - Stopping criterion whole body model
              - 0.1
              -
              - None
              - Always
            * - t0
              - Baseline duration
              - 0
              - 
              - None
              - Always
            * - field_strength
              - B0-field
              - 3
              - T
              - None
              - Always
            * - **Injection**
              - 
              -
              - 
              - 
              -
            * - weight
              - Subject weight
              - 70
              - kg
              - None
              - Always
            * - dose
              - Contrast agent dose
              - 0.0125
              - mL/kg
              - None
              - Always
            * - rate
              - Contrast agent injection rate
              - 1
              - mL/kg
              - None
              - Always
            * - **Sequence**
              - 
              -
              - 
              - 
              - 
            * - TR
              - Repetition time
              - 0.005
              - sec
              - None
              - sequence in ['SS', 'SSI']
            * - FA
              - Flip angle
              - 15
              - deg
              - None
              - sequence in ['SR', 'SS', 'SSI']
            * - TC
              - Time to k-space center
              - 0.1
              - sec
              - None
              - sequence == 'SR'
            * - TS
              - Sampling duration
              - 0
              - sec
              - None
              - Always
            * - TF
              - Inflow time
              - 0
              - sec
              - None
              - sequence == 'SSI'
            * - **Aorta**
              - 
              -
              - 
              - 
              -
            * - BAT
              - Bolus arrival time
              - 60
              - sec
              - [0, inf]
              - Always
            * - CO
              - Cardiac output
              - 100
              - mL/sec
              - [0, 300]
              - Always
            * - Thl
              - Heart-lung transit time
              - 10
              - sec
              - [0, 30]
              - Always
            * - Dhl
              - Heart-lung dispersion
              - 0.2
              - 
              - [0.05, 0.95]
              - heartlung in ['pfcomp', 'chain']
            * - To
              - Organs transit time
              - 20
              - sec
              - [0, 60]
              - Always
            * - Eo
              - Organs extraction fraction
              - 0.15
              - 
              - [0, 0.5]
              - organs == '2cxm'
            * - Toe
              - Organs extracellular transit time
              - 120
              - sec
              - [0, 800]
              - organs == '2cxm'
            * - Eb
              - Body extraction fraction
              - 0.05
              - 
              - [0.01, 0.15]
              - Always
            * - R10a
              - Arterial precontrast R1
              - 0.7
              - /sec
              - None
              - Always
            * - S0a
              - Arterial signal scale factor
              - 1
              - a.u.
              - None
              - Always
            * - **Kidneys**
              -
              -
              - 
              - 
              -
            * - H
              - Hematocrit
              - 0.45
              - 
              - None
              - Always
            * - RPF
              - Renal plasma flow
              - 20
              - mL/sec
              - [0, 100]
              - Always
            * - DRF
              - Differential renal function
              - 0.5
              - 
              - [0, 1.0]
              - Always
            * - DRPF
              - Differential renal plasma flow
              - 0.5
              - 
              - [0, 1.0]
              - kidneys == '2CF'
            * - FF
              - Filtration fraction
              - 0.1
              - 
              - [0, 0.3]
              - agent in ['gadoxetate', 'gadobenate']
            * - **Left kidney**
              - 
              -
              - 
              - 
              - 
            * - Ta_lk
              - Left kidney arterial delay
              - 0
              - sec
              - [0, 3]
              - Always
            * - vp_lk
              - Left kidney plasma volume
              - 0.15
              - mL/cm3
              - [0, 0.3]
              - Always
            * - Tt_lk
              - Left kidney tubular transit time
              - 120
              - sec
              - [0, inf]
              - Always
            * - R10_lk
              - Left kidney precontrast R1
              - 0.65
              - 1/sec
              - None
              - Always
            * - S0_lk
              - Left kidney signal scale factor
              - 1.0
              - a.u.
              - [0, inf]
              - Always
            * - vol_lk
              - Left kidney volume
              - 150
              - cm3
              - None
              - Always
            * - **Right kidney**
              - 
              -
              - 
              - 
              -
            * - Ta_rk
              - Right kidney arterial delay
              - 0
              - sec
              - [0, 3]
              - Always
            * - vp_rk
              - Right kidney plasma volume
              - 0.15
              - mL/cm3
              - [0, 0.3]
              - Always
            * - Tt_rk
              - Right kidney tubular transit time
              - 120
              - sec
              - [0, inf]
              - Always
            * - R10_rk
              - Right kidney precontrast R1
              - 0.65
              - /sec
              - None
              - Always
            * - S0_rk
              - Right kidney signal scale factor
              - 1.0
              - a.u.
              - [0, inf]
              - Always
            * - vol_rk
              - Right kidney volume
              - 150
              - cm3
              - None
              - Always


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
        ...     R10a=1/dc.T1(pars['B0'], 'blood'),
        ...     R10_lk=1/dc.T1(pars['B0'], 'kidney'),
        ...     R10_rk=1/dc.T1(pars['B0'], 'kidney'),
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

    def __init__(
        self, 
        organs='comp', 
        heartlung='pfcomp', 
        kidneys='2CF', 
        sequence='SS',
        agent='gadoterate', 
        **params,
    ):

        # Check configuration
        if organs not in ['comp','2cxm']:
            raise ValueError(
                f"{organs} is not a valid model for the organs. "
                "Current options are 'comp' and '2cxm'."
            )
        if heartlung not in ['comp', 'pfcomp', 'chain']:
            raise ValueError(
                f"{heartlung} is not a valid heart-lung system. "
                "Current options are 'comp', 'pfcomp' and 'chain'."
            )
        if kidneys not in ['2CF', 'HF']:
            raise ValueError(
                f"Kinetic model {kidneys} is not available."
            )
        if sequence not in ['SS', 'SR', 'SSI', 'lin']:
            raise ValueError(
                f"Sequence {sequence} is not available."
            )
        
        self._version = '1.0'
        self._cnfg = {
            'organs': organs, 
            'heartlung': heartlung, 
            'kidneys': kidneys, 
            'sequence': sequence, 
            'agent': agent
        }
        self._pars = lexicon.init(self._pars_list(), LEXICON)

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
        # Computed variables
        self._t = None
        self._c = {}
        self._r = {}
        self._s = {}
    
    def _pars_list(self, select='all'):

        organs = {
            'comp': ['To', 'Eb'],
            '2cxm': ['To', 'Eb', 'To_e', 'Eo']
        }[self._cnfg['organs']]

        heartlung = {
            'comp': ['Thl'],
            'pfcomp': ['Thl', 'Dhl'],
            'chain': ['Thl', 'Dhl'],
        }[self._cnfg['heartlung']]

        sequence = {
            'SR': ['FA', 'TR', 'TC', 'TP'],
            'SS': ['FA', 'TR'], 
            'lin': [],
            'SSI': ['FA', 'TR'],
        }[self._cnfg['sequence']]

        inflow = {
            'SR': [],
            'SS': [], 
            'lin': [],
            'SSI': ['TF'],            
        }[self._cnfg['sequence']]

        kidneys = {
            '2CF': ['DRPF'],
            'HF': [],
        }[self._cnfg['kidneys']]

        agent = ['FF'] if self._cnfg['agent'] in ['gadoxetate', 'gadobenate'] else []

        free = heartlung + organs + inflow + kidneys + agent
        pars_list = {
            'all': free + sequence + [
                'dt', 'tmax', 'dose_tolerance', 'field_strength',
                'agent', 'weight', 'dose', 'rate', 'TS',
                'H', 'BAT', 'CO', 
                'RPF', 'DRF',
                'Ta_lk', 'vp_lk', 'Tt_lk', 'vol_lk', 
                'Ta_rk', 'vp_rk', 'Tt_rk', 'vol_rk', 
                'R10_a', 'R10_lk', 'R10_rk',
                'S0_a', 'S0_lk', 'S0_rk',
                'B1corr_a', 'B1corr_lk', 'B1corr_rk',
            ],
            'free': free + [
                'BAT', 'CO', 'RPF', 'DRF',
                'Ta_lk', 'vp_lk', 'Tt_lk',
                'Ta_rk', 'vp_rk', 'Tt_rk',
            ],
            'free_aorta': heartlung + organs + inflow + [
                'BAT', 'CO'
            ],
            'free_kidneys': kidneys + agent + [
                'RPF', 'DRF',
                'Ta_lk', 'vp_lk', 'Tt_lk',
                'Ta_rk', 'vp_rk', 'Tt_rk'
            ],
            'sequence_a': sequence + inflow, 
            'sequence_k': sequence,
        }
        return pars_list[select]
    

    # ==========================================
    # Forward Model: Aorta
    # ==========================================

    def _set_time(self):
        p = self._pars
        self._t = np.arange(0, p['tmax'], p['dt'])

    def _compute_conc_aorta(self) -> np.ndarray:
        self._set_time()
        p = self._pars

        hl, orgs = self._cnfg['heartlung'], self._cnfg['organs']

        if hl=='comp':
            heartlung = ['comp', (p['Thl'],)]
        elif hl=='pfcomp':
            heartlung = ['pfcomp', (p['Thl'], p['Dhl'])]
        elif hl=='chain':
            heartlung = ['chain', (p['Thl'], p['Dhl'])]

        if orgs=='comp':
            organs = ['comp', (p['To'],)]
        elif orgs=='2cxm':
            organs = ['2cxm', ([p['To'], p['To_e']], p['Eo'])]

        conc = lib.ca_conc(p['agent'])
        Ji = lib.ca_injection(
            self._t, p['weight'], conc, p['dose'], p['rate'], p['BAT']
        )
        Jb = pk_aorta.flux_aorta(
            Ji, E=p['Eb'], dt=p['dt'], tol=p['dose_tolerance'],
            heartlung=heartlung, organs=organs,
        )
        self._c['a'] = Jb / p['CO']

    def _compute_relax_aorta(self):  
        self._compute_conc_aorta()
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._r['a'] = p['R10_a'] + rb * self._c['a']

    def _compute_signal_aorta(self):
        self._compute_relax_aorta()
        p = self._pars
        seq = self._cnfg['sequence']
        pars = {k: p[k] for k in self._pars_list('sequence_a')}
        if 'FA' in pars: pars['FA'] *= p['B1corr_a']
        self._s['a'] = sig.signal(seq, self._r['a'], p['S0_a'], **pars)

    def _predict_aorta(self, time):
        self._set_time()
        self._compute_signal_aorta()
        p = self._pars
        return utils.sample(time, self._t, self._s['a'], p['TS'])
    
    # ==========================================
    # Forward Model: Kidneys
    # ==========================================

    def _compute_conc_kidneys(self):
        p = self._pars

        if self._cnfg['agent'] in ['gadoxetate', 'gadobenate']:
            FF = p['FF']
        else:
            FF = p['Eb']/(1-p['Eb'])
        GFR = FF * p['RPF']
        Ft = {
            'lk': p['DRF'] * GFR / p['vol_lk'],
            'rk': (1 - p['DRF']) * GFR / p['vol_rk']
        }
        ca = {
            k: pk.flux_plug(self._c['a'], p[f'Ta_{k}'], dt=p['dt']) / (1 - p['H'])
            for k in ['lk', 'rk']
        }

        if self._cnfg['kidneys'] == '2CF':
            Fp = {
                'lk': p['DRPF'] * p['RPF'] / p[f'vol_lk'],
                'rk': (1 - p['DRPF']) * p['RPF'] / p['vol_rk']
            }
            for k in ['lk', 'rk']:
                self._c[k] = kidney.conc_kidney(
                    ca[k], Fp[k], p[f'vp_{k}'], Ft[k], p[f'Tt_{k}'], 
                    dt=p['dt'], kinetics='2CF', sum=False
                ) 

        if self._cnfg['kidneys'] == 'HF':
            for k in ['lk', 'rk']:
                self._c[k] = kidney.conc_kidney(
                    ca[k], p[f'vp_{k}'], Ft[k], p[f'Tt_{k}'], 
                    dt=p['dt'], kinetics='HF', sum=False,
                )

    def _compute_relax_kidneys(self):
        self._compute_conc_kidneys()
        p = self._pars
        rb = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        for k in ['lk', 'rk']:
            self._r[k] = p[f'R10_{k}'] + rb * self._c[k].sum(axis=0) 

    def _compute_signal_kidneys(self):
        self._compute_relax_kidneys()
        p = self._pars

        seq = 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence']

        for kid in ['lk', 'rk']:
            pars = {k: p[k] for k in self._pars_list('sequence_k')}
            if 'FA' in pars: pars['FA'] *= p[f'B1corr_{kid}']
            self._s[kid] = sig.signal(seq, self._r[kid], p[f'S0_{kid}'], **pars)

    def _predict_kidneys(self, time):
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        self._compute_signal_kidneys()
        return (
            utils.sample(time[0], self._t, self._s['lk'], p['TS']),
            utils.sample(time[1], self._t, self._s['rk'], p['TS']),
        )
    
    # ===========================================
    # Forward Model: Liver, Portal Vein and Aorta
    # ===========================================
    
    def _predict(self, time: Data) -> Data:
        p = self._pars
        p['tmax'] = p['dt'] + p['TS'] + np.max(np.concatenate(time))
        aorta = self._predict_aorta(time[0])
        kidneys = self._predict_kidneys(time[1:])
        return (aorta, kidneys[0], kidneys[1])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, time: Data, signal: Data, n0: int):
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
        seq = {
            'a': self._cnfg['sequence'],
            'lk': 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence'],
            'rk': 'SS' if self._cnfg['sequence']=='SSI' else self._cnfg['sequence'],
        }
        params = {
            'a': 'sequence_a',
            'lk': 'sequence_k',
            'rk': 'sequence_k',
        }
        idx = {
            'a': 0, 
            'lk': 1, 
            'rk': 2,
        }
        for roi in idx.keys():
            pars = {k: p[k] for k in self._pars_list(params[roi])}
            if 'FA' in pars: pars['FA'] *= p[f'B1corr_{roi}']
            s_ref = sig.signal(seq[roi], p[f'R10_{roi}'], 1, **pars)
            p[f'S0_{roi}'] = np.mean(signal[idx[roi]][:n0]) / s_ref if s_ref > 0 else 0

    def _train(
        self, time: Data, signal: Data, free: dict, 
        bounds: dict, n0: int, staged: bool, **kwargs
    ):
        self._estimate_parameters(time, signal, n0)
        free = self._set_free_pars(free, bounds) 

        # Extra conditions for SSI sequence
        if self._cnfg['sequence'] == 'SSI' and 'S0_a' not in free:
            raise ValueError("For SSI sequence, 'S0_a' must be a free parameter.")

        if staged:

            # Optimize Aorta parameters
            free_stage = {k: v for k, v in free.items() if k in self._pars_list('free_aorta')}
            utils.train(self._predict_aorta, time[0], signal[0], self._pars, free_stage, **kwargs)

            # Optimize Kidney parameters
            free_stage = {k: v for k, v in free.items() if k in self._pars_list('free_kidneys')}
            utils.train(self._predict_kidneys, time[1:], signal[1:], self._pars, free_stage, **kwargs)

        # Joint Optimization
        return utils.train(self._predict, time, signal, self._pars, free, **kwargs)

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
            ax.plot(t / 60, s, marker='o', color=color[0], label='Data', linestyle='None')
            ax.plot(self._t / 60, sig, linestyle='-', color=color[1], linewidth=3.0, label='Prediction')
            ax.legend()

        plot_data1scan(self._s['a'], time[0], signal[0], 'Aorta', ax1, ['lightcoral', 'darkred'])
        plot_data1scan(self._s['lk'], time[1], signal[1], 'Left kidney', ax3, ['cornflowerblue', 'darkblue'])
        plot_data1scan(self._s['rk'], time[2], signal[2], 'Right kidney', ax5, ['cornflowerblue', 'darkblue'])

        # Plot aorta
        cb_lk = self._c['lk'][0,:] / (p['vp_lk'] / (1 - p['H']))
        cb_rk = self._c['rk'][0,:] / (p['vp_rk'] / (1 - p['H']))

        ax2.set(xlabel='Time (min)', ylabel='Blood conc (mM)', xlim=xlim)
        ax2.plot(self._t / 60, 0 * self._t, color='gray')
        ax2.plot(self._t / 60, 1000 * self._c['a'], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
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

        plot_conc_kidney(self._c['lk'], 'Left kidney', ax4)
        plot_conc_kidney(self._c['rk'], 'Right kidney', ax6)

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> Data:
        """Internal time array

        Returns:
            Tuple: (aorta_time, portal_time, liver_time).
        """
        self._set_time()
        return self._t, self._t, self._t

    def conc(self) -> Data:
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
        return self._c['a'], self._c['lk'], self._c['rk']

    def relax(self) -> Data:
        """Relaxation rates in aorta and kidney.

        Returns:
            tuple: time points, aorta relaxation rate, left kidney 
              relaxation rate, right kidney relaxation rate.
        """
        self._compute_relax_aorta()
        self._compute_relax_kidneys()
        return self._r['a'], self._r['lk'], self._r['rk']
    
    def signal(self) -> Data:
        """Return signals in aorta and liver.

        Returns:
            tuple: signals for (aorta, portal, liver)
        """
        self._compute_signal_aorta()
        self._compute_signal_kidneys()
        return self._s['a'], self._s['lk'], self._s['rk']

    def predict(self, time) -> Data:
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
        return self._predict(time)

    def train(
        self, time: Data, signal: Data, free: dict = None, 
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
        return self._train(time, signal, free, bounds, n0, staged, **kwargs)

    def plot(
        self, time: Data, signal: Data, xlim: list = None, 
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
        self._plot(time, signal, xlim, fname, show)



# def _deriv_params(p):

#     # Kidneys
#     if 'FF' not in p:
#         p['FF'] = _div(p['Eb'], 1-p['Eb'])
#     if {'RPF', 'FF'}.issubset(p):   
#         p['GFR'] =  p['RPF'] * p['FF']
#     if {'DRPF', 'RPF'}.issubset(p): 
#         p['RPF_lk'] = p['DRPF'] * p['RPF']
#         p['RPF_rk'] = (1 - p['DRPF']) * p['RPF']
#     if {'DRF', 'GFR'}.issubset(p):
#         p['GFR_lk'] = p['DRF'] * p['GFR']
#         p['GFR_rk'] = (1 - p['DRF']) * p['GFR']

#     # Kidney LK
#     if {'RPF_lk', 'vol_lk'}.issubset(p):
#         p['Fp_lk'] = _div(p['RPF_lk'], p['vol_lk'])
#     if {'RPF_lk', 'GFR_lk', 'vp_lk', 'vol_lk'}.issubset(p):
#         p['Tp_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk']+p['GFR_lk'])
#     if {'RPF_lk', 'vp_lk', 'vol_lk'}.issubset(p):
#         p['Tv_lk'] = _div(p['vp_lk'] * p['vol_lk'], p['RPF_lk'])
#     if {'GFR_lk', 'vol_lk'}.issubset(p):
#         p['Ft_lk'] = _div(p['GFR_lk'], p['vol_lk'])
#     if {'GFR_lk', 'RPF_lk'}.issubset(p):
#         p['FF_lk'] = _div(p['GFR_lk'], p['RPF_lk'])
#         p['E_lk'] = _div(p['GFR_lk'], p['GFR_lk']+p['RPF_lk'])

#     # Kidney RK
#     if {'RPF_rk', 'vol_rk'}.issubset(p):
#         p['Fp_rk'] = _div(p['RPF_rk'], p['vol_rk'])
#     if {'RPF_rk', 'GFR_rk', 'vp_rk', 'vol_rk'}.issubset(p):
#         p['Tp_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk']+p['GFR_rk'])
#     if {'RPF_rk', 'vp_rk', 'vol_rk'}.issubset(p):
#         p['Tv_rk'] = _div(p['vp_rk'] * p['vol_rk'], p['RPF_rk'])
#     if {'GFR_rk', 'vol_rk'}.issubset(p):
#         p['Ft_rk'] = _div(p['GFR_rk'], p['vol_rk'])
#     if {'GFR_rk', 'RPF_rk'}.issubset(p):
#         p['FF_rk'] = _div(p['GFR_rk'], p['RPF_rk'])
#         p['E_rk'] = _div(p['GFR_rk'], p['GFR_rk']+p['RPF_rk'])

#     return p


# def _div(a, b):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, 0, np.divide(a, b))
