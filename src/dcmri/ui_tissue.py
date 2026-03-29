
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from dcmri import sig, utils, ui, tissue, lib, rel
from dcmri.lexicon import LEXICON
import dcmri.lexicon_utils as lexicon
from dcmri.signal_2_conc import SignalToConc




class Tissue(ui.SuperModel):
    """Vascular-interstitial tissue.

    This is the most common tissue type as found in for instance brain,
    cancer, lung, muscle, prostate, skin, and more. For more detail see
    :ref:`two-site-exchange`.

    Args:
        kinetics (str, optional): Tracer-kinetic model. Possible values are
         '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to 'HF'.
        water_exchange (str, optional): Water exchange regime. Any combination
          of two of the letters 'F', 'N', 'R' is allowed. Defaults to 'FF'.
        sequence (str, optional): imaging sequence. Possible values are 'SS'
          and 'SR'. Defaults to 'SS'.
        aif (array-like, optional): Signal-time curve in the blood of the
          feeding artery. If *aif* is not provided, the arterial
          blood concentration is *ca*. Defaults to None.
        ca (array-like, optional): Blood concentration in the arterial
          input. *ca* is ignored if *aif* is provided, but is required
          otherwise. Defaults to None.
        t (array-like, optional): Time points of the arterial input function.
          If *t* is not provided, the temporal sampling is uniform with
          interval *dt*. Defaults to None.
        dt (float, optional): Time interval between values of the arterial
          input function. *dt* is ignored if *t* is provided. Defaults to 1.0.
        free (dict, optional): Dictionary with free parameters and their
          bounds. If not provided, a default set of free parameters is used.
          Defaults to None.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. See tables :ref:`Tissue-parameters` and
          :ref:`Tissue-defaults` for a list of tissue parameters and their
          default values.

    See Also:
        `Liver`, `Kidney`

    Example:

        Fit an extended Tofts model to data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(CNR=50)

        Build a tissue and set the parameters to match the experimental
        conditions of the synthetic data:

        >>> tissue = dc.Tissue(
        ...     aif = aif,
        ...     dt = time[1],
        ...     r1 = dc.relaxivity(3, 'blood','gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     n0 = 15,
        ... )

        Train the tissue on the data:

        >>> tissue.train(time, roi)

        Print the optimized tissue parameters, their standard deviations and
        any derived parameters:

        >>> tissue.print_params(round_to=2)
        <BLANKLINE>
        --------------------------------
        Free parameters with their stdev
        --------------------------------
        <BLANKLINE>
        Blood volume (vb): 0.03 (0.0) mL/cm3
        Interstitial volume (vi): 0.2 (0.0) mL/cm3
        Permeability-surface area product (PS): 0.0 (0.0) mL/sec/cm3
        <BLANKLINE>
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        <BLANKLINE>
        Tissue Hematocrit (H): 0.45 
        Plasma volume (vp): 0.02 mL/cm3
        Interstitial mean transit time (Ti): 58.92 sec
        B1-corrected Flip Angle (FAcorr): 15 deg

        Plot the fit to the data and the reconstructed concentrations, using
        the noise-free ground truth as reference:

        >>> tissue.plot(time, roi, ref=gt)

    Notes:

        Table :ref:`Tissue-parameters` lists the parameters that are relevant 
        in each regime. Alternatively, you can use `dcmri.Tissue.info` to 
        print them out. 
        
        Table :ref:`Tissue-defaults` list all possible parameters and their 
        default settings. 

        .. _Tissue-parameters:
        .. list-table:: **Tissue parameters**
            :widths: 20 30 30
            :header-rows: 1

            * - Parameters
              - When to use
              - Further detail
            * - n0
              - Always
              - For estimating baseline signal
            * - r1, R10
              - Always
              - :ref:`relaxation-params`
            * - R10a, B1corr_a
              - When aif is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - S0, TS
              - Always
              - :ref:`params-per-sequence`
            * - FA, TR, B1corr
              - If **sequence** in ['SR','SS']
              - :ref:`params-per-sequence`
            * - TP, TC
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - Fb, PS, Ktrans, vb, H, vi,
                ve, vc, PSe, PSc.
              - Depends on **kinetics** and **water_exchange**
              - :ref:`tissue-kinetic-regimes`

        .. _Tissue-defaults:
        .. list-table:: **Parameter defaults**
            :widths: 5 10 10 10 10
            :header-rows: 1

            * - Parameter
              - Type
              - Value
              - Bounds
              - Free/Fixed
            * - r1
              - Relaxation
              - 5000.0
              - [0, inf]
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
            * - Fb
              - Kinetic
              - 0.01
              - [0, inf]
              - Free
            * - Ktrans
              - Kinetic
              - 0.002
              - [0, inf]
              - Free
            * - PS
              - Kinetic
              - 0.003
              - [0, inf]
              - Free
            * - PSc
              - Kinetic
              - 0.03
              - [0, inf]
              - Free
            * - PSe
              - Kinetic
              - 0.03
              - [0, inf]
              - Free
            * - vb
              - Kinetic
              - 0.1
              - [0, 1]
              - Free
            * - vc
              - Kinetic
              - 0.4
              - [0, 1]
              - Free
            * - ve
              - Kinetic
              - 0.355
              - [0, 1]
              - Free
            * - vi
              - Kinetic
              - 0.5
              - [0, 1]
              - Free
    """

    def __init__(
        self,
        shape=None, 
        kinetics='HF', 
        water_exchange='FF', 
        sequence='SS',
        inflow_sequence=None,
        **params
    ):
        # Set default configs
        if inflow_sequence is None:
            inflow_sequence = sequence

        # Check configuration
        if shape is not None:
            if len(shape) > 3:
                raise ValueError(
                    f"The 'shape' parameter specifies spatial dimensions "
                    "and must be 1-, 2- or 3 dimensional (or empty for 1D data)"
                )
        if kinetics not in ['U', 'FX', 'NX', 'NXP', 'WV', 'HFU', 'HF', '2CU', '2CX']:
            raise ValueError(
                f"Kinetic model {kinetics} is not available."
            )
        if water_exchange not in ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']:
            raise ValueError(
                f"Water exchange model {water_exchange} is not available."
            )
        if sequence not in ['SS', 'SR', 'IR', 'SPGR', 'free']:
            raise ValueError(
                f"Sequence {sequence} is not available."
            )
        
        self._version = '1.0'
        self._cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange, 
            'sequence': sequence, 
            'inflow_sequence': inflow_sequence, 
        }
        self._pars = lexicon.init(self._pars_list(), LEXICON)

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
        # Store the shape of the parameters provided by the user, 
        # which can be scalar, 1D, 2D or 3D. 
        # 
        # Internally, parameters are all converted to 1D arrays for 
        # simplicity. The shape is retained so that outputs can be 
        # converted back to their original format when they are 
        # returned to the user.
        # 
        # Signals can be provided as 1D (nt), 2D (nx, nt), 3D (nx, ny, nt) 
        # or 4D (nx, ny, nz, nt). Internally they are converted to 
        # 2D (n_samples, n) and then converted back to original shapes 
        # before returning to the user.

        # First we check if the user has provided a shape, either 
        # directly or indirectly.

        self._orig_shape: tuple = None
        
        if shape is not None:
            # If the user has provided a shape, use that
            self._orig_shape = shape
        else:
            # Else take the shape from any user-defined parameters
            for p in params:
                if p in self._pars_list('pixel_orig'):
                    self._orig_shape = np.array(p).shape
                    continue
        # If the user has not provided a shape directly or indirectly, 
        # we revert to the default (scalar)
        if self._orig_shape is None:
            self._orig_shape = ()

        # Convert all pixel parameters to flat 1D arrays
        for p in self._pars_list('pixel_orig'):
            if np.isscalar(self._pars[p]):
                self._pars[p] = np.array([self._pars[p]])
            else:
                self._pars[p] = np.array(self._pars[p]).reshape(-1)

        # Pixel parameters are silently converted to full arrays if needed
        for p in self._pars_list('pixel_orig'):
            if self._pars[p].size == 1:
                self._pars[p] = np.full(self._n_samples, self._pars[p][0])

        # Make sure that all pixel parameters have the shape (n_samples, )
        for p in self._pars_list('pixel_orig'):
            size = self._pars[p].size
            if size != self._n_samples:
                raise ValueError(f"Parameter {p} has size {size} but the number of samples is {self._n_samples}") 
            self._pars[p] = self._pars[p].reshape(self._n_samples)

        # Add any derived parameters
        tissue.derive_params(self._pars)
        self._pars['r1'] = lib.relaxivity(self._pars['field_strength'], 'blood', self._pars['agent'])

   
    @property
    def _n_samples(self):
        return 1 if self._orig_shape==() else np.prod(self._orig_shape)
    
    def _pars_list(self, select='all'):
        kin, wex, seq, iseq = self._cnfg['kinetics'], self._cnfg['water_exchange'], self._cnfg['sequence'], self._cnfg['inflow_sequence']

        params_signal_tissue = tissue.Signal(kin, wex, seq, iseq).params()
        params_relax_tissue = tissue.Relax(kin, wex).params()
        params_conc_tissue = tissue.Conc(kin).params()
        all_kinetic_pars = list(set(params_conc_tissue + params_relax_tissue))

        pars_list = {
            'all': params_signal_tissue + [
                'c_a', 'dt', 'field_strength', 'agent', 'TS',
                'R10', 'B1corr', 'R10_a',
            ],
            'free': [f for f in params_relax_tissue if f not in ['H', 'r1']], 
            'pixel': [f for f in all_kinetic_pars if f not in ['H', 'r1']] + ['S0', 'R10', 'B1corr'],
            'pixel_orig': [f for f in params_relax_tissue if f != 'H'] + ['S0', 'R10', 'B1corr'],
        }
        return pars_list[select]
    
    def _pixel_pars(self, x, pars):
        p = self._pars
        pixel_pars = self._pars_list('pixel')
        pars_x = {k: p[k][x] for k in pars if k in pixel_pars}
        pars_x |= {k: p[k] for k in pars if k not in pixel_pars}
        return pars_x

    def _set_new_shape(self, shape):
        self._orig_shape = shape

        # Scalar pixel parameters are silently converted to full arrays
        pixel_pars = lexicon.init(self._pars_list('pixel'), LEXICON)
        for p, v in pixel_pars.items():
            self._pars[p] = np.full(self._n_samples, v)

        # Add any derived parameters
        tissue.derive_params(self._pars)
    

    # ==========================================
    # Forward Model
    # ==========================================

    def _signal(self, x=None) -> np.ndarray: # (n_samples, nt)
        p = self._pars
        Sx = tissue.Signal(**self._cnfg)

        def _pixel_signal(x) -> np.ndarray: # (nt, )
            pars_x = self._pixel_pars(x, Sx.params())
            pars_x['FA'] = p['FA'] * p['B1corr'][x]
            pars_x['FAR'] = p['FA'] * p['B1corr'][x]
            S = Sx(p['c_a'], **pars_x)
            return S

        if x is None:
            # Compute whole array
            if self._n_samples==1:
                results = [_pixel_signal(0)]
            else:
                results = Parallel(n_jobs=-1)(delayed(_pixel_signal)(x) for x in range(self._n_samples))
        else: 
            # Compute one pixel
            results = [_pixel_signal(x)]
        
        return np.stack(results, axis=0) 

    def _time(self):
        p = self._pars
        return p['dt'] * np.arange(p['c_a'].size)

    def _predict(self, time, x=None):
        t = self._time()
        s = self._signal(x)
        if x is not None:
            s = s.reshape(-1)
        p = self._pars
        return utils.sample(time, t, s, p['TS'])
    
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, signal: np.ndarray, aif: ui.Input, n0: int):
        p = self._pars

        # Estimate S0
        seq = self._cnfg['sequence']
        iseq = self._cnfg['inflow_sequence']
        s0 = sig.Signal(seq, iseq)
    
        def s0_pixel(x):
            s0_pars = self._pixel_pars(x, s0.params())
            s0_pars.update(
                {
                    'S0': 1,
                    'FAR': p['FA'] * p['B1corr'][x],
                    'FA': p['FA'] * p['B1corr'][x],
                    'Fi': p['Fb'][x] if 'Fb' in p else 0,
                    'R1i': p['R10_a']
                }
            )
            s_ref = s0(p['R10'][x], **s0_pars)
            S0 = np.mean(signal[x, :n0]) / s_ref if s_ref > 0 else 0
            return S0 

        p['S0'] = np.array([s0_pixel(x) for x in range(self._n_samples)])

        if aif is not None:
            # Arterial concentration estimation
            pars = p.copy()
            pars['FA'] = p['FA'] * aif.B1corr
            ca = SignalToConc(seq)(aif.signal, aif.R10, n0=n0, **p)

            # Interpolate on internal time
            self._t = np.arange(0, aif.time[-1] + p['dt'], p['dt'])
            p['c_a'] = np.interp(self._t, aif.time, ca)


    def _train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: ui.Input, free: dict, bounds: dict, 
        n0: int, modsel: str, **kwargs
    ):
        self._estimate_parameters(signal, aif, n0)
        return None, None, None
    
        free = self._set_free_pars(free, bounds)

        # for n_samples > 1, free params must be a subset of pixel - check this
        if self._n_samples > 1:
            pixel_pars = self._pars_list('pixel')
            if not set(free.keys()).issubset(set(pixel_pars)):
                raise ValueError('For a pixel-based analysis, only pixel-based parameters can be free.')
        
        # No model selection - parallellize over pixels
        if modsel is None:
            def train_pixel(x):
                return utils.train(self._predict, time, signal[x,:], self._pars, free, x, **kwargs)
            
            if self._n_samples==1:
                results = [train_pixel(0)]
            else:
                # results = [train_pixel(x) for x in range(self._n_samples)]
                results = Parallel(n_jobs=-1)(delayed(train_pixel)(x) for x in range(self._n_samples))

        # Model selection with 1 sample - parallellize over models
        elif self._n_samples==1:
            results = [self._train_pixel_all_models(time, signal, free, modsel, 0, parallel=True, **kwargs)]

        # Model selection with multiple samples - parallellize over samples
        else:
            results = Parallel(n_jobs=-1)(
                delayed(self._train_pixel_all_models)(
                    time, signal, free, modsel, x, parallel=False,
                ) for x in range(self._n_samples)
            )

        vals = {p: np.stack([r[0][p] for r in results]) for p in free}
        sdev = {p: np.stack([r[1][p] for r in results]) for p in free}
        pcov = np.stack([r[2] for r in results]).reshape(self._n_samples, len(free), len(free))

        # Add any derived parameters
        tissue.derive_params(self._pars)

        return vals, sdev, pcov
    
    def _train_pixel_all_models(self, time, signal, free: dict, metric, x, parallel=True, **kwargs):

        def train_pixel_single_model(kin, wex):

            # Store original configuration
            kinetics, water_exchange = self._cnfg['kinetics'], self._cnfg['water_exchange']

            # Check if the sub-model is nested
            pars_relax_topmodel = tissue.Relax(kinetics, water_exchange).params()
            pars_relax_submodel = tissue.Relax(kin, wex).params()
            if not set(pars_relax_submodel).issubset(pars_relax_topmodel):
                return None

            # Identify the free parameters of the sub-model
            free_relax_submodel = {k: v for k, v in free.items() if k in pars_relax_submodel}
            free_norelax = {k: v for k, v in free.items() if k not in pars_relax_topmodel}
            free_submodel = free_norelax | free_relax_submodel

            # Set configuration to sub-model
            self._cnfg['kinetics'], self._cnfg['water_exchange'] = kin, wex
            
            # Train single pixel to submodel
            result = utils.train(self._predict, time, signal[x,:], self._pars, free_submodel, x, reset=True, **kwargs)
            
            # Compute cost
            s_pred = self._predict(time, x)
            cost = utils.loss(s_pred, signal[x,:], metric, len(free_submodel))

            # Reset original configuration
            self._cnfg['kinetics'], self._cnfg['water_exchange'] = kinetics, water_exchange
            
            return (kin, wex), result, cost
        
        kin_list = ['HF', 'U', 'FX', 'NX', 'NXP', 'WV', 'HFU', '2CU', '2CX']
        wex_list = ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']

        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(train_pixel_single_model)(kin, wex) 
                for kin in kin_list for wex in wex_list
            )
        else:
            results = [
                train_pixel_single_model(kin, wex) 
                for kin in kin_list for wex in wex_list
            ]

        # Rebuild dictionaries
        valid_results = [r for r in results if r is not None]
        cost_dict = {r[0]: r[2] for r in valid_results}
        result_dict = {r[0]: r[1] for r in valid_results}

        # Find the best model
        best_model = min(cost_dict, key=cost_dict.get)
        result = result_dict[best_model] + (best_model,)

        # Update state with optimized values
        for p, v in result[0].items(): 
            self._pars[p] = v

        return result


    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _concentration(self):
        p = self._pars
        Cx = tissue.Conc(self._cnfg['kinetics'])

        def _conc_pixel(x):
            pars_x = self._pixel_pars(x, Cx.params())
            return Cx(p['c_a'], dt=p['dt'], **pars_x)

        if self._n_samples==1:
            C = [_conc_pixel(0)]
        else:
            C = Parallel(n_jobs=-1)(delayed(_conc_pixel)(x) for x in range(self._n_samples))

        return np.stack(C)  # (n_samples, nc, nt)


    def _relaxation_rate(self):
        p = self._pars
        Rx = tissue.Relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])

        def _relax_pixel(x):
            pars_x = self._pixel_pars(x, Rx.params())
            return Rx(p['c_a'], dt=p['dt'], **pars_x)
        
        if self._n_samples==1:
            results = [_relax_pixel(0)]
        else:
            results = Parallel(n_jobs=-1)(delayed(_relax_pixel)(x) for x in range(self._n_samples))

        return np.stack(results)   # (n_samples, nc, nt)
    
    def _magnetization(self):
        p = self._pars
        Mx = tissue.Mz(**self._cnfg)

        def _magn_pixel(x):
            pars_x = self._pixel_pars(x, Mx.params())
            pars_x['FA'] = p['FA'] * p['B1corr'][x]
            return Mx(p['c_a'], dt=p['dt'], **pars_x)
        
        if self._n_samples==1:
            Mz = [_magn_pixel(0)]
        else:
            Mz = Parallel(n_jobs=-1)(delayed(_magn_pixel)(x) for x in range(self._n_samples))
   
        return np.stack(Mz)  # (n_samples, nc, nt) 


    def _plot(
        self, time, signal, sdev=None, round_to=None, xlim=None, 
        fname=None, show=True
    ):
        """Plot the model fit against data.

        Args:
            time (array-like, optional): Array with time points.
            signal (array-like, optional): Array with measured signals for
              each element of *time*.
            xlim (array_like, optional): 2-element array with lower and upper
              boundaries of the x-axis. Defaults to None.
            round_to (int, optional): Rounding for the model parameters.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to
              True.
        """
        clr = {
            'Plasma': 'darkred',
            'Interstitium': 'steelblue',
            'Extracellular': 'dimgrey',
            'Tissue': 'darkgrey',
            'Blood': 'darkred',
            'Extravascular': 'blue',
            'Tissue cells': 'lightblue',
            'Blood + Interstitium': 'purple',
        }
        plot_labels_kin = {
            '2CX': (['vb', 'vi'], ['Blood', 'Interstitium']),
            '2CU': (['vb', 'vi'], ['Blood', 'Interstitium']),
            'HF': (['vb', 'vi'], ['Blood', 'Interstitium']),
            'HFU': (['vb', 'vi'], ['Blood', 'Interstitium']),
            'FX': (['ve'], ['Extracellular']),
            'NX': (['vb'], ['Blood']), 
            'NXP': (['vb'], ['Blood']),
            'U': (['vb'], ['Blood']),
            'WV': (['vi'], ['Interstitium']),
        }
        def plot_labels_relax(kin, wex) -> list:

            if wex == 'FF':
                return ['Tissue']

            if wex in ['RR', 'NN', 'NR', 'RN']:
                if kin == 'WV':
                    return ['Interstitium', 'Tissue cells']
                else:
                    return ['Blood', 'Interstitium', 'Tissue cells']

            if wex in ['RF', 'NF']:
                if kin == 'WV':
                    return ['Extravascular']
                else:
                    return ['Blood', 'Extravascular']

            if wex in ['FR', 'FN']:
                if kin == 'WV':
                    return ['Interstitium', 'Tissue cells']
                else:
                    return ['Blood + Interstitium', 'Tissue cells']
                
        t = self._time()
        S = self._signal()
        R = self._relaxation_rate()
        C = self._concentration()
        Mz = self._magnetization()
        p = self._pars

        c = (R - R[:, :, 0][:, :, np.newaxis]) / p['r1']
        v = tissue.WaterVolumes(self._cnfg['kinetics'], self._cnfg['water_exchange'])(**p)

        if xlim is None: xlim = [np.amin(t), np.amax(t)]
        xlim=np.array(xlim) / 60

        if self._cnfg['water_exchange'] != 'FF':
            fig, ax = plt.subplots(2, 2, figsize=(10, 12))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            ax00 = ax[0, 0]
            ax01 = ax[0, 1]
            ax10 = ax[1, 0]
            ax11 = ax[1, 1]
            ax_text = None
        else:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            ax00 = ax[0]
            ax01 = ax[1]
            ax_text = ax[2]

        ax00.set_title('MRI signals')
        ax00.plot(time / 60, self._predict(time)[0,:], marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
        ax00.plot(time / 60, signal[0,:], marker='x', linestyle='None', color='darkblue', label='Data')
        ax00.plot(t / 60, S[0,:], linestyle='-', linewidth=3.0, color='darkblue', label='Model')
        ax00.set(ylabel='MRI signal (a.u.)', xlabel='Time (min)', xlim=xlim)
        ax00.legend()

        conc_comp, conc_label = plot_labels_kin[self._cnfg['kinetics']]
        relax_comp = plot_labels_relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])
    
        ax01.set_title('Tissue concentration in indicator compartments')
        ax01.plot(t / 60, 1000 * self._pars['c_a'], linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial blood')
        for k, vk in enumerate(conc_comp):
            # ck = C[k, ...] / p[vk] if p[vk] > 0 else 0 * C[k, ...]
            ax01.plot(t / 60, 1000 * C[0, k, ...], linestyle='-', linewidth=3.0, label=conc_label[k], color=clr[conc_label[k]])
        ax01.set(ylabel='Concentration (mM)', xlabel='Time (min)', xlim=xlim)
        ax01.legend()

        if self._cnfg['water_exchange'] != 'FF':
            ax11.set_title('Concentration in water compartments')
            for i in range(c.shape[1]):
                ax11.plot(t / 60, 1000 * c[0, i, :], linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
            ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
            ax11.legend()

            ax10.set_title('Magnetization in water compartments')
            for i in range(Mz.shape[1]):
                mi = Mz[0, i, :] / v[i] if v[i] > 0 else 0 * Mz[0, i, :]
                ax10.plot(t / 60, mi, linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
            ax10.set(xlabel='Time (min)', ylabel='Magnetization (a.u.)', xlim=xlim)
            ax10.legend()

        if ax_text is not None:

            if sdev is None:
                pars = self._pars_list('free')
            else:
                pars = list(sdev.keys())
                sdev = {k: float(v) for k, v in sdev.items()}

            vals = {k: p[k][0] for k in pars}
            msg = lexicon.string_params(vals, sdev, round_to)
            msg = "\n".join(list(msg.values()))
            ax_text.set_title('Free parameters')
            ax_text.axis("off")  # hide axes
            ax_text.text(0, 0.9, msg, fontsize=10, transform=ax_text.transAxes, ha="left", va="top")

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> np.ndarray:
        """Kidney signal time points"""
        return self._time()

    def conc(self):
        """Return the tissue concentration

        Returns:
            np.ndarray: Concentration in M

        Example:

            Build a tissue, and plot the tissue concentrations in each
            compartment:

        .. plot::
            :include-source:
            :context: close-figs

            >>> import dcmri as dc
            >>> import matplotlib.pyplot as plt

            >>> t, aif, _ = dc.fake_aif()
            >>> tissue = dc.Tissue('HFU', 'RR', aif=aif, t=t)
            >>> C = tissue.conc(sum=False)

            >>> _ = plt.figure()
            >>> _ = plt.plot(t/60, 1e3*C[0,:], label='Plasma')
            >>> _ = plt.plot(t/60, 1e3*C[1,:], label='Interstitium')
            >>> _ = plt.xlabel('Time (min)')
            >>> _ = plt.ylabel('Concentration (mM)')
            >>> _ = plt.legend()
            >>> _ = plt.show()
        """
        C = self._concentration()
        return C.reshape(self._orig_shape + C.shape[1:])
    

    def relax(self):
        """Compartmental relaxation rates, volume fractions and
        water-permeability matrix.

        tuple: relaxation rates of tissue compartments and their volumes.
            - **R1** (numpy.ndarray): in the fast water exchange limit, the
              relaxation rates are a 1D array. In all other situations,
              relaxation rates are a 2D-array with dimensions (k,n), where k is
              the number of compartments and n is the number of time points
              in ca.
            - **v** (numpy.ndarray or None): the volume fractions of the tissue
              compartments. Returns None in 'FF' regime.
            - **Fw** (numpy.ndarray or None): 2D array with water exchange
              rates between tissue compartments. Returns None in 'FF' regime.

        Example:

            Build a tissue, print its compartmental volumes and water
            permeability matrix, and plot the free relaxation rates of each
            compartment:

        .. plot::
            :include-source:
            :context: close-figs

            >>> import dcmri as dc
            >>> t, aif, _ = dc.fake_aif()
            >>> tissue = dc.Tissue('2CX', 'RR', aif=aif, t=t)
            >>> R1, v, Fw = tissue.relax()

            >>> v
            array([0.1, 0.3, 0.6])

            >>> Fw
            array([[0.02, 0.03, 0.  ],
                   [0.03, 0.  , 0.03],
                   [0.  , 0.03, 0.  ]])

            >>> import matplotlib.pyplot as plt
            >>> _ = plt.figure()
            >>> _ = plt.plot(t/60, R1[0,:], label='Blood')
            >>> _ = plt.plot(t/60, R1[1,:], label='Interstitium')
            >>> _ = plt.plot(t/60, R1[2,:], label='Cells')
            >>> _ = plt.xlabel('Time (min)')
            >>> _ = plt.ylabel('Relaxation rate (Hz)')
            >>> _ = plt.legend()
            >>> plt.show()

        """
        R1 = self._relaxation_rate()
        return R1.reshape(self._orig_shape + R1.shape[1:])
    
    def magn(self) -> np.ndarray:
        """Pseudocontinuous magnetization

        Returns:
            np.ndarray: the magnetization as a 1D array.
        """
        Mz = self._magnetization()
        return Mz.reshape(self._orig_shape + Mz.shape[1:])

    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        S = self._signal()
        return S.reshape(self._orig_shape + S.shape[1:])

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        S = self._predict(time)
        return S.reshape(self._orig_shape + S.shape[1:])

    def train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: ui.Input=None, free: dict=None, bounds: dict=None, 
        n0=1, modsel=None, **kwargs
    ):
        """Train the free parameters

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element
              of *time*.
            modsel (bool, optional): Of True, all submodels are tested and the best is selected.
            kwargs: any keyword parameters accepted by the specified fit
              method. For 'NNLS' these are all parameters accepted by
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            self
        """
        if modsel is not None:
            if modsel not in ['AIC', 'BIC']:
                raise ValueError("'modsel' must be either 'AIC' (Akaike Information Criterion) or 'BIC' (Baysian Information Criterion)")
        
        if aif is not None:
            n_times = aif.signal.size
        else:
            n_times = self._pars['c_a'].size
        if np.prod(signal.shape) != self._n_samples * n_times:
            self._set_new_shape(signal.shape[:-1])

        signal = signal.reshape(self._n_samples, -1)
        vals, sdev, pcov = self._train(time, signal, aif, free, bounds, n0, modsel, **kwargs)

        return vals, sdev, pcov

        vals = {k: v.reshape(self._orig_shape + v.shape[1:]) for k, v in vals.items()}
        sdev = {k: v.reshape(self._orig_shape + v.shape[1:]) for k, v in sdev.items()}
        pcov = pcov.reshape(self._orig_shape + pcov.shape[1:])
        return vals, sdev, pcov

    def plot(
        self, time: np.ndarray, signal: np.ndarray, sdev: dict=None,
        round_to=None, xlim=None, fname=None, show=True
    ):
        """Plot the model fit against data.

        Args:
            time (array-like, optional): Array with time points.
            signal (array-like, optional): Array with measured signals for
              each element of *time*.
            sdev (dict): Standard deviations of free parameters
            xlim (array_like, optional): 2-element array with lower and upper
              boundaries of the x-axis. Defaults to None.
            round_to (int, optional): Rounding for the model parameters.
            ref (tuple, optional): Tuple of optional test data in the form
              (x,y), where x is an array with x-values and y is an array with
              y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to
              True.
        """
        if signal.shape[:-1] != self._orig_shape:
            raise ValueError(f"Incompatible shapes: the model was initiated with shape {self._orig_shape}")
        signal = signal.reshape(self._n_samples, -1)
        self._plot(time, signal, sdev, round_to, xlim, fname, show)




# def _all_pars(kin, wex, seq, p):

#     #pars = _model_pars(kin, wex, seq)
#     p = {par: p[par] for par in pars}

#     try:
#         p['Fp'] = p['Fb'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['vp'] = p['vb'] * (1 - p['H'])
#     except KeyError:
#         pass
#     try:
#         p['Ktrans'] = _div(p['Fp'] * p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['ve'] = p['vi'] + p['vc']
#     except KeyError:
#         pass
#     try:
#         p['E'] = _div(p['PS'], p['Fp'] + p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Ti'] = _div(p['vi'], p['PS'])
#     except KeyError:
#         pass
#     try:
#         p['Tp'] = _div(p['vp'], p['PS'] + p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Tb'] = _div(p['vp'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Te'] = _div(p['ve'], p['Fp'])
#     except KeyError:
#         pass
#     try:
#         p['Twc'] = _div(1 - p['vb'] - p['vi'], p['PSc'])
#     except KeyError:
#         pass
#     try:
#         p['Twi'] = _div(p['vi'], p['PSc'] + p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['Twb'] = _div(p['vb'], p['PSe'])
#     except KeyError:
#         pass
#     try:
#         p['FAcorr'] = p['B1corr'] * p['FA']
#     except KeyError:
#         pass

#     return p


# def _div(a, b):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, 0, np.divide(a, b))
