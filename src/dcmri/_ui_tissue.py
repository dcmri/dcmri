
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import dcmri.pk
from dcmri import sig, utils, ui, tissue, lib
from dcmri.lexicon import LEXICON
import dcmri.lexicon_utils as lexicon




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
        kinetics='HF', 
        water_exchange='FF', 
        sequence='SS',
        **params
    ):
        # Check configuration
        if kinetics not in ['U', 'FX', 'NX', 'NXP', 'WV', 'HFU', 'HF', '2CU', '2CX']:
            raise ValueError(
                f"Kinetic model {kinetics} is not available."
            )
        if water_exchange not in ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']:
            raise ValueError(
                f"Water exchange model {water_exchange} is not available."
            )
        if sequence not in ['SS', 'SR']:
            raise ValueError(
                f"Sequence {sequence} is not available."
            )
        
        self._version = '1.0'
        self._cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange, 
            'sequence': sequence, 
        }
        self._pars = lexicon.init(self._pars_list(), LEXICON)

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")
            
        # Add any derived parameters
        self._compute_derived()
            
    def _pars_list(self, select='all'):
        kin, wex, seq = self._cnfg['kinetics'], self._cnfg['water_exchange'], self._cnfg['sequence']
        relax = tissue.params_relax_tissue(kin, wex)
        pars_seq = {
            'SR': ['FA', 'TR', 'TC', 'TP'],
            'SS': ['FA', 'TR'],
        }[seq]

        pars_list = {
            'all': relax + pars_seq + [
                'c_a', 'dt', 'field_strength', 'agent',
                'T_a', 'TS',
                'S0', 'R10', 'B1corr', 'noise_sdev',
            ],
            'free': ['T_a'] + [f for f in relax if f != 'H'],
            'conc': tissue.params_relax_tissue(kin),
            'relax': relax,
            'sequence': pars_seq,
            'signal': pars_seq + ['S0', 'noise_sdev']
        }
        return pars_list[select]
    
    def _compute_derived(self):
        p = self._pars
        if {'H', 'vb', 'vi'}.issubset(p):
            p['ve'] = (1 - p['H']) * p['vb'] + p['vi']
    
    # ==========================================
    # Forward Model
    # ==========================================   

    def _compute_concentration(self):
        p = self._pars
        kin_pars = {k: p[k] for k in self._pars_list('conc')}
        
        ca = dcmri.pk.flux_plug(p['c_a'], p['T_a'], dt=p['dt'])
        self._C = tissue.conc_tissue(
            ca, dt=p['dt'], sum=False, 
            kinetics=self._cnfg['kinetics'], **kin_pars
        )

    def _compute_relaxation_rate(self):
        p = self._pars

        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        kin_pars = {k: p[k] for k in self._pars_list('relax')}
        
        ca = dcmri.pk.flux_plug(p['c_a'], p['T_a'], dt=p['dt'])
        R = tissue.relax_tissue(
            ca, p['R10'], rp, dt=p['dt'],
            kinetics=self._cnfg['kinetics'], 
            water_exchange=self._cnfg['water_exchange'], **kin_pars
        )
        self._R1 = R['R1']
        self._v = R['v']
        self._Fw = R['Fw']
        self._c = R['c']
    
    def _compute_magnetization(self):
        p = self._pars

        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        kin_pars = {k: p[k] for k in self._pars_list('relax')}

        # TODO -> just one dict seq + kin?
        seq_pars = {k: p[k] for k in self._pars_list('sequence')}
        if 'FA' in seq_pars: seq_pars['FA'] *= p['B1corr']
        seq_pars['model'] = self._cnfg['sequence']

        ca = dcmri.pk.flux_plug(p['c_a'], p['T_a'], dt=p['dt'])
        Mz = tissue.Mz_tissue(
            ca, p['R10'], rp, dt=p['dt'],
            kinetics=self._cnfg['kinetics'],
            water_exchange=self._cnfg['water_exchange'],
            sequence=seq_pars, **kin_pars
        )
        self._Mz = Mz['Mz']
        self._R1 = Mz['R1']
        self._v = Mz['v']
        self._Fw = Mz['Fw']
        self._c = Mz['c']

    def _compute_signal(self):
        p = self._pars

        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        relax_pars = {k: p[k] for k in self._pars_list('relax')}

        # TODO -> just one dict seq + kin?
        seq_pars = {k: p[k] for k in self._pars_list('signal')}
        if 'FA' in seq_pars: seq_pars['FA'] *= p['B1corr']
        seq_pars['model'] = self._cnfg['sequence']

        ca = dcmri.pk.flux_plug(p['c_a'], p['T_a'], dt=p['dt'])
        self._S = tissue.signal_tissue(
            ca, p['R10'], rp, dt=p['dt'],
            kinetics=self._cnfg['kinetics'],
            water_exchange=self._cnfg['water_exchange'],
            sequence=seq_pars, **relax_pars
        )

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
        seq_pars = {k: p[k] for k in self._pars_list('sequence')}
        if 'FA' in seq_pars: seq_pars['FA'] *= p[f'B1corr']

        s_ref = sig.signal(self._cnfg['sequence'], p['R10'], 1, **seq_pars)
        p['S0'] = np.mean(signal[:n0]) / s_ref if s_ref > 0 else 0

        # Arterial concentration estimation
        if aif is not None:

            seq_pars = {k: p[k] for k in self._pars_list('sequence')}
            if 'FA' in seq_pars: seq_pars['FA'] *= aif['B1corr']

            r1 = lib.relaxivity(p['field_strength'], 'blood', p['agent'])

            ca = sig.conc(self._cnfg['sequence'], aif['signal'], aif['R10'], r1, n0=n0, **seq_pars)
            t = np.arange(0, np.max(aif['time']) + p['TS'] + p['dt'], p['dt'])
            p['c_a'] = np.interp(t, aif['time'], ca)


    def _train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: dict, free: dict, bounds: dict, 
        n0: int, modsel: str, **kwargs
    ):
        self._estimate_parameters(signal, n0, aif)
        free = self._set_free_pars(free, bounds)

        if modsel is None:
            result = utils.train(self._predict, time, signal, self._pars, free, **kwargs)

        else:
            result = self._train_modsel(time, signal, free, modsel, **kwargs)

        # Add any derived parameters
        self._compute_derived()

        return result

    
    def _train_modsel(self, time, signal, free, metric, **kwargs):
        kin_list = ['HF', 'U', 'FX', 'NX', 'NXP', 'WV', 'HFU', '2CU', '2CX']
        wex_list = ['FF', 'RF', 'NF', 'FR', 'RR', 'NR', 'FN', 'RN', 'NN']

        # Use Parallel to run the flattened loop
        results = Parallel(n_jobs=-1)(
            delayed(self._train_model)(kin, wex, time, signal, free, metric, **kwargs)
            for kin in kin_list for wex in wex_list
        )

        # Rebuild dictionaries
        valid_results = [r for r in results if r is not None]
        cost_dict = {r[0]: r[2] for r in valid_results}
        res_dict = {r[0]: r[1] for r in valid_results}

        # Find the best model
        best_model = min(cost_dict, key=cost_dict.get)
        result = res_dict[best_model] + (best_model,)

        # Update state with optimized values
        for p, v in result[0].items(): self._pars[p] = v

        return result


    def _train_model(self, kin, wex, time, signal, free, metric, **kwargs):
        # Check if parameters are valid for this subset
        pars_relax_subset = tissue.params_relax_tissue(kin, wex)
        if not set(pars_relax_subset).issubset(self._pars.keys()):
            return None

        # Update local config for this specific worker
        self._cnfg['kinetics'] = kin
        self._cnfg['water_exchange'] = wex
        
        # Filter free parameters
        free_norelax = {k: v for k, v in free.items() if k not in self._pars_list('relax')}
        free_relax_subset = {k: v for k, v in free.items() if k in pars_relax_subset}
        free_model = free_norelax | free_relax_subset
        
        # Run training and calculate cost
        res = utils.train(self._predict, time, signal, self._pars.copy(), free_model, **kwargs)
        c = self.cost(time, signal, metric, len(free))
        
        return (kin, wex), res, c


    # ==========================================
    # I/O and Reporting
    # ==========================================

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
            'U': ([], []),
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
                
        self._set_time()
        self._compute_concentration()
        self._compute_magnetization()
        self._compute_signal()
        t = self._t
        p = self._pars

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
        ax00.plot(time / 60, self._predict(time), marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
        ax00.plot(time / 60, signal, marker='x', linestyle='None', color='darkblue', label='Data')
        ax00.plot(self._t / 60, self._S, linestyle='-', linewidth=3.0, color='darkblue', label='Model')
        ax00.set(ylabel='MRI signal (a.u.)', xlabel='Time (min)', xlim=xlim)
        ax00.legend()

        conc_comp, conc_label = plot_labels_kin[self._cnfg['kinetics']]
        relax_comp = plot_labels_relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])

        C = self._C.reshape((1, -1)) if self._C.ndim==1 else self._C
        c = self._c.reshape((1, -1)) if self._c.ndim==1 else self._c
        Mz = self._Mz.reshape((1, -1)) if self._Mz.ndim==1 else self._Mz
        v = [self._v] if np.isscalar(self._v) else self._v
    
        ax01.set_title('Tissue concentration in indicator compartments')
        ax01.plot(t / 60, 1000 * self._pars['c_a'], linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial blood')
        for k, vk in enumerate(conc_comp):
            # ck = C[k, ...] / p[vk] if p[vk] > 0 else 0 * C[k, ...]
            ax01.plot(t / 60, 1000 * C[k, ...], linestyle='-', linewidth=3.0, label=conc_label[k], color=clr[conc_label[k]])
        ax01.set(ylabel='Concentration (mM)', xlabel='Time (min)', xlim=xlim)
        ax01.legend()

        if self._cnfg['water_exchange'] != 'FF':
            ax11.set_title('Concentration in water compartments')
            for i in range(c.shape[0]):
                ax11.plot(t / 60, 1000 * c[i, :], linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
            ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
            ax11.legend()

            ax10.set_title('Magnetization in water compartments')
            for i in range(Mz.shape[0]):
                mi = Mz[i, ...] / v[i] if v[i] > 0 else 0 * Mz[i, ...]
                ax10.plot(t / 60, mi, linestyle='-', linewidth=3.0, color=clr[relax_comp[i]], label=relax_comp[i])
            ax10.set(xlabel='Time (min)', ylabel='Magnetization (a.u.)', xlim=xlim)
            ax10.legend()

        if ax_text is not None:

            if sdev is None:
                pars = self._pars_list('free')
            else:
                pars = list(sdev.keys())

            vals = {k: p[k] for k in pars}
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
        self._set_time()
        return self._t

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
        self._compute_concentration()
        return self._C
    

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
        self._compute_relaxation_rate()
        return self._R1
    
    def magn(self) -> np.ndarray:
        """Pseudocontinuous magnetization

        Returns:
            np.ndarray: the magnetization as a 1D array.
        """
        self._compute_magnetization()
        return self._Mz

    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        self._compute_signal()
        return self._S

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        return self._predict(time)

    def train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: dict=None, free: dict=None, bounds: dict=None, 
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
                raise ValueError("'modsel' must be eith 'AIC' (Akaike Information Criterion) or 'BIC' (Baysian Information Criterion)")
        return self._train(time, signal, aif, free, bounds, n0, modsel, **kwargs)

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
