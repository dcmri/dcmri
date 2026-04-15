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

    Use `fake.tissue` to generate synthetic test data:

    >>> time, aif, roi, gt = dc.fake.tissue(CNR=50)

    Build a tissue and set the parameters to match the experimental
    conditions of the synthetic data:

    >>> tissue = dc.TissueX(
    ...     aif = aif,
    ...     dt = time[1],
    ...     r1 = dc.const.r1(3, 'blood','gadodiamide'),
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
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from dcmri import tissue_x
from dcmri.lexicon import QUANTITIES, string_params, init
from dcmri import relaxivity
from dcmri.inverse import SignalToConc
from dcmri.core import SuperModel, Input
from dcmri.kinetics import ConcTissueX
from dcmri.utils.misc import sample
from dcmri.utils.fit import loss, train_batch, format_batch_training


class TissueX(SuperModel):

    configs = deepcopy(tissue_x.Signal.configs)

    def __init__(
        self,
        kinetics='HF', 
        water_exchange='FF', 
        sequence='3D-SPGR-SS',
        transverse_relaxation='lin',
        shape=None, 
        **params
    ):
        cnfg = {
            'kinetics': kinetics, 
            'water_exchange': water_exchange, 
            'sequence': sequence, 
            'transverse_relaxation': transverse_relaxation,
        }
        self._version = '1.0'
        self._cnfg = self._set_config(**cnfg)
        self._pars = self._set_pars(**params)

        # Check configuration
        if shape is not None:
            if len(shape) > 3:
                raise ValueError(
                    f"The 'shape' parameter specifies spatial dimensions "
                    "and must be 1-, 2- or 3 dimensional (or empty for 1D data)"
                )       
            
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
        # 2D (n_pixels, n_times) or 3D (n_pixels, n_acquisitions, n_times) 
        # and then converted back to original shapes 
        # before returning to the user.

        # First we check if the user has provided a shape, either 
        # directly or indirectly.

        self._pixels_shape: tuple = None

        # Get the non-scalar shapes from any user-defined pixel parameters
        shapes = [np.array(v).shape for p, v in params.items() if p in self._params('pixel')]
        shapes = [s for s in shapes if np.prod(s) > 1]
        if len(set(shapes)) > 1:
            # Raise if more than 1 non-scalar shape is present
            raise ValueError(f"Pixel-based parameters {self._params('pixel')} must all have the same shape, or be scalars.")
        elif len(shapes) == 0: 
            # No pixel-parameters provided - take the user defined shape
            self._pixels_shape = shape
        elif shape is None:
            # Pixel-parameters but no shape provided - take from pixel parameters
            self._pixels_shape = shapes[0]
        elif shape != shapes[0]:
            # Both pixel-parameters and shape provided - check agreement
            raise ValueError(f"Shape provided is inconsistent with shapes of pixel-parameters {self._params('pixel')} provided.")
        else:
            # Both are provided and in agreement - pick one
            self._pixels_shape = shapes[0]

        # If the user has not provided a shape directly or indirectly, 
        # we revert to the default (scalar)
        if self._pixels_shape is None:
            self._pixels_shape = ()

        # Convert all pixel parameters to flat 1D arrays
        for p in self._params('pixel'):
            if np.isscalar(self._pars[p]):
                self._pars[p] = np.array([self._pars[p]])
            else:
                self._pars[p] = np.array(self._pars[p]).reshape(-1)

        # Pixel parameters are silently converted to full arrays if needed
        for p in self._params('pixel'):
            if self._pars[p].size == 1:
                self._pars[p] = np.full(self._shape[0], self._pars[p][0])
    
    @property
    def _shape(self):
        n_pixels = 1 if self._pixels_shape==() else np.prod(self._pixels_shape)
        n_times = self._pars['c_a'].size
        if self._cnfg['sequence'] in ['DE-EPI']:
            n_channels = 2
        else:
            n_channels = 1
        return (n_pixels, n_channels, n_times)

    
    def _params(self, select=None):
        if select is None:
            select = 'all'
        kin, wex, seq, t2s = self._cnfg['kinetics'], self._cnfg['water_exchange'], self._cnfg['sequence'], self._cnfg['transverse_relaxation']

        if select == 'all':
            return tissue_x.Signal(kin, wex, seq, t2s)._params() + ['c_a', 'dt', 'TS']
        
        elif select == 'pixel': # pixel-based parameters
            pars = (
                ConcTissueX(kin)._params()
                + tissue_x.WaterConc(kin, wex)._params()
                + tissue_x.ContrastConc(kin)._params()
                + tissue_x.WaterVolumes(kin, wex)._params()
                + tissue_x.WaterFlows(kin, wex)._params()
                + ['R10', 'R20', 'R20s', 'r2s', 'r2s_quad', 'r2s_vasc', 'r2s_ees']
                + ['S0', 'B1corr', 'noise_sdev']
            )
            return [p for p in list(set(pars)) if p != 'H' and p in self._params()]
        
        elif select == 'free': # default free parameters (subset of ppixel)
            pars = (
                ConcTissueX(kin)._params()
                + tissue_x.WaterConc(kin, wex)._params()
                + tissue_x.ContrastConc(kin)._params()
                + tissue_x.WaterVolumes(kin, wex)._params()
                + tissue_x.WaterFlows(kin, wex)._params()
                + ['r2s_vasc', 'r2s_ees']
            )
            return [p for p in list(set(pars)) if p != 'H' and p in self._params()]
    
    
    # ==========================================
    # Forward Model
    # ==========================================

    def _signal(self, x=None) -> np.ndarray: # (n_pixels, n_channels, n_times)
        p = self._pars

        def pixel_signal(x) -> np.ndarray: # (n_channels, n_times)
            kwargs_x = self._cnfg | self._pixel_pars(x)
            S = tissue_x.Signal(**kwargs_x)(p['c_a'])
            return S.reshape(-1, S.shape[-1])  # (n_channels, n_times)
        
        if x is None:
            if self._shape[0]==1:
                results = [pixel_signal(0)]
            else:
                results = Parallel(n_jobs=-1)(delayed(pixel_signal)(x) for x in range(self._shape[0]))
        else: 
            results = [pixel_signal(x)]
        
        return np.stack(results) # (n_pixels, n_channels, n_times)

    def _time(self):
        p = self._pars
        return p['dt'] * np.arange(p['c_a'].size) # (n_times, )

    def _predict(self, time, x=None):
        t = self._time()
        s = self._signal(x) # (n_pixels, n_channels, n_times)
        p = self._pars
        n_channels = s.shape[1]
        readouts = [sample(time, t, s[:,i,:], p['TS']) for i in range(n_channels)]
        return np.stack(readouts, axis=1) # (n_pixels, n_channels, n_times)
        
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, signal: np.ndarray, aif: Input, n0: int):
        # signal (n_pixels, n_channels, n_times)
        p = self._pars
        
        def s0_pixel(x):
            kwargs_x = self._cnfg | self._pixel_pars(x)
            s_ref = tissue_x.Signal(**kwargs_x)([0], S0=1).flatten()
            s_avr = np.mean(signal[x, :, :n0], axis=-1) 
            S0 = np.divide(s_avr, s_ref, out=np.zeros_like(s_avr), where=s_ref != 0)
            return S0

        S0 = np.stack([s0_pixel(x) for x in range(self._shape[0])]) # n_pixels, n_channels
        p['S0'] = np.mean(S0, axis=1)  # n_pixels - average over channels

        if aif is not None:
            seq = self._cnfg['sequence']
            ca = SignalToConc(seq, **p)(aif.signal, S0=None, R10=aif.R10, n0=n0, B1corr=aif.B1corr)
            # Interpolate on internal time
            self._t = np.arange(0, aif.time[-1] + p['dt'], p['dt'])
            p['c_a'] = np.interp(self._t, aif.time, ca)


    def _train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: Input, free: dict, bounds: dict, 
        n0: int, configs: list, select: str, **kwargs
    ):
        self._estimate_parameters(signal, aif, n0)
        #return None, None, None
        free = self._set_free_pars(free, bounds)

        # for n_pixels > 1, free params must be a subset of pixel - check this
        if self._shape[0] > 1:
            pixel_pars = self._params('pixel')
            if not set(free.keys()).issubset(set(pixel_pars)):
                raise ValueError('For a pixel-based analysis, only pixel-based parameters can be free.')
        
        # No model selection
        if configs is None:
            results = train_batch(self._predict, time, signal, self._pars, free, **kwargs)

        # Model selection
        else:
            results = self._train_batch_configurations(time, signal, free, configs, select, **kwargs)

        # Format outputs
        return format_batch_training(results, free)


    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _concentration(self):
        p = self._pars
        Cx = ConcTissueX(self._cnfg['kinetics'])

        def _conc_pixel(x):
            pars_x = self._pixel_pars(x)
            return Cx(p['c_a'], **pars_x)

        if self._shape[0]==1:
            C = [_conc_pixel(0)]
        else:
            C = Parallel(n_jobs=-1)(delayed(_conc_pixel)(x) for x in range(self._shape[0]))

        return np.stack(C)  # (n_pixels, n_compartments, n_times)


    def _relaxation_rate(self):
        p = self._pars
        Cx = ConcTissueX(self._cnfg['kinetics'])
        Rx = tissue_x.R1(self._cnfg['kinetics'], self._cnfg['water_exchange'])

        def _relax_pixel(x):
            pars_x = self._pixel_pars(x)
            C = Cx(p['c_a'], **pars_x)
            return Rx(C, **pars_x)
        
        if self._shape[0]==1:
            results = [_relax_pixel(0)]
        else:
            results = Parallel(n_jobs=-1)(delayed(_relax_pixel)(x) for x in range(self._shape[0]))

        return np.stack(results)   # (n_pixels, n_compartments, n_times)
    
    def _magnetization(self):
        p = self._pars
        Cx = ConcTissueX(self._cnfg['kinetics'])
        Rx = tissue_x.R1(self._cnfg['kinetics'], self._cnfg['water_exchange'])
        Mx = tissue_x.Mz(**self._cnfg)

        if 'Fb' in p:
            R1a = relaxivity.relax_t1(p['c_a'], p['R10_a'], p['r1'])
        else:
            R1a = None

        def _magn_pixel(x):
            pars_x = self._pixel_pars(x)
            C = Cx(p['c_a'], **pars_x)
            R1 = Rx(C, **pars_x)
            return Mx(R1, R1a, **pars_x)
        
        if self._shape[0]==1:
            Mz = [_magn_pixel(0)]
        else:
            Mz = Parallel(n_jobs=-1)(delayed(_magn_pixel)(x) for x in range(self._shape[0]))
   
        return np.stack(Mz)  # (n_pixels, n_compartments, n_times) 


    def _plot(
        self, time, signal, sdev=None, round_to=None, xlim=None, 
        fname=None, show=True
    ):
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
        C = self._concentration() # (n_pixels, n_compartments, n_times)
        R = self._relaxation_rate() # (n_pixels, n_compartments, n_times)
        Mz = self._magnetization() # (n_pixels, n_compartments, n_times)
        S = self._signal() # (n_pixels, n_channels, n_times)
        p = self._pars

        c = (R - R[:, :, 0][:, :, np.newaxis]) / p['r1']
        x = 0
        v = tissue_x.WaterVolumes(self._cnfg['kinetics'], self._cnfg['water_exchange'])(**self._pixel_pars(x))

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
        for ci in range(signal.shape[1]):
            ax00.plot(time / 60, self._predict(time)[0, ci, :], marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
            ax00.plot(time / 60, signal[0, ci, :], marker='x', linestyle='None', color='darkblue', label='Data')
            ax00.plot(t / 60, S[0, ci, :], linestyle='-', linewidth=3.0, color='darkblue', label='Model')
        ax00.set(ylabel='MRI signal (a.u.)', xlabel='Time (min)', xlim=xlim)
        ax00.legend()

        conc_comp, conc_label = plot_labels_kin[self._cnfg['kinetics']]
        relax_comp = plot_labels_relax(self._cnfg['kinetics'], self._cnfg['water_exchange'])
    
        ax01.set_title('Tissue concentration in indicator compartments')
        ax01.plot(t / 60, 1000 * self._pars['c_a'], linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial blood')
        for k, vk in enumerate(conc_comp):
            # ck = C[k, ...] / p[vk] if p[vk] > 0 else 0 * C[k, ...]
            ax01.plot(t / 60, 1000 * C[0, k, :], linestyle='-', linewidth=3.0, label=conc_label[k], color=clr[conc_label[k]])
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
                pars = self._params('free')
            else:
                pars = list(sdev.keys())
                sdev = {k: float(v) for k, v in sdev.items()}

            vals = {k: p[k][0] for k in pars}
            msg = string_params(vals, sdev, round_to)
            msg = "\n".join(list(msg.values()))
            ax_text.set_title('Free parameters')
            ax_text.axis("off")  # hide axes
            ax_text.text(0, 0.9, f'Kinetics: {self._cnfg['kinetics']}', fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
            ax_text.text(0, 0.85, f'Water exchange: {self._cnfg['water_exchange']}', fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
            ax_text.text(0, 0.8, f'Sequence: {self._cnfg['sequence']}', fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
            ax_text.text(0, 0.75, f'R2* model: {self._cnfg['transverse_relaxation']}', fontsize=10, transform=ax_text.transAxes, ha="left", va="top")
            ax_text.text(0, 0.6, msg, fontsize=10, transform=ax_text.transAxes, ha="left", va="top")

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    # ==========================================
    # Public API: Data Extraction
    # ==========================================

    def time(self) -> np.ndarray:
        """Kidney signal time points"""
        return self._time()

    def conc(self) -> np.ndarray:
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

            >>> t, aif, _ = dc.fake.aif()
            >>> tissue = dc.TissueX('HFU', 'RR', aif=aif, t=t)
            >>> C = tissue.conc(sum=False)

            >>> _ = plt.figure()
            >>> _ = plt.plot(t/60, 1e3*C[0,:], label='Plasma')
            >>> _ = plt.plot(t/60, 1e3*C[1,:], label='Interstitium')
            >>> _ = plt.xlabel('Time (min)')
            >>> _ = plt.ylabel('Concentration (mM)')
            >>> _ = plt.legend()
            >>> _ = plt.show()
        """
        C = self._concentration() # (n_pixels, n_compartments, n_times)
        C = C.reshape(self._pixels_shape + C.shape[1:]) # (nx, ny, nz, n_compartments, n_times)
        if C.shape[-2] == 1:
            return C[...,0,:]
        else:
            return C
    

    def relax(self) -> np.ndarray:
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
            >>> t, aif, _ = dc.fake.aif()
            >>> tissue = dc.TissueX('2CX', 'RR', aif=aif, t=t)
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
        R1 = self._relaxation_rate() # (n_pixels, n_compartments, n_times)
        R1 = R1.reshape(self._pixels_shape + R1.shape[1:]) # (nx, ny, nz, n_compartments, n_times)
        if R1.shape[-2] == 1:
            return R1[...,0,:]
        else:
            return R1
    
    def magn(self) -> np.ndarray:
        """Pseudocontinuous magnetization

        Returns:
            np.ndarray: the magnetization as a 1D array.
        """
        Mz = self._magnetization() # (n_pixels, n_compartments, n_times)
        Mz = Mz.reshape(self._pixels_shape + Mz.shape[1:]) # (nx, ny, nz, n_compartments, n_times)
        if Mz.shape[-2] == 1:
            return Mz[...,0,:]
        else:
            return Mz
        
    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        S = self._signal() # (n_pixels, n_channels, n_times)
        S = S.reshape(self._pixels_shape + S.shape[1:]) # (nx, ny, nz, n_compartments, n_times)
        if S.shape[-2] == 1:
            return S[...,0,:] # (nx, ny, nz, n_times)
        else:
            return S # (nx, ny, nz, n_channels, n_times) or # (nx, ny, nz, n_times)

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        S = self._predict(time) # (n_pixels, n_channels, n_times)
        S = S.reshape(self._pixels_shape + S.shape[1:]) # (nx, ny, nz, n_channels, n_times)
        if S.shape[-2] == 1: # one channel - squeeze out
            return S[...,0,:] # (nx, ny, nz, n_times)
        else:
            return S # (nx, ny, nz, n_channels, n_times) or # (nx, ny, nz, n_times)

    def train(
        self, time: np.ndarray, signal: np.ndarray, 
        aif: Input=None, free: dict=None, bounds: dict=None, 
        n0=1, configs: list=None, select='AIC', **kwargs
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

        if select not in ['AIC', 'BIC']:
            raise ValueError("'modsel' must be either 'AIC' (Akaike Information Criterion) or 'BIC' (Baysian Information Criterion)")
    
        if aif is not None:
            n_times = aif.signal.size
        else:
            n_times = self._pars['c_a'].size

        if self._cnfg['sequence'] in ['DE-EPI']:
            n_channels = 2
        else:
            n_channels = 1

        # Derive the shape from the data
        if np.prod(signal.shape) != self._shape[0] * n_channels * n_times:
            self._pixels_shape = signal.shape[:-1]
            # Reset pixel parameters
            pixel_pars = init(self._params('pixel'), QUANTITIES)
            for p, v in pixel_pars.items():
                self._pars[p] = np.full(self._shape[0], v)

        signal = signal.reshape(self._shape[0], n_channels, n_times)
        vals, sdev, pcov, model = self._train(time, signal, aif, free, bounds, n0, configs, select, **kwargs)

        # Convert to input shape
        vals = {k: v.reshape(self._pixels_shape + v.shape[1:]) for k, v in vals.items()}
        sdev = {k: v.reshape(self._pixels_shape + v.shape[1:]) for k, v in sdev.items()}
        if self._pixels_shape == ():
            pcov = pcov[0]
            model = model[0]
        else:
            pcov = pcov.reshape(self._pixels_shape)
            model = model.reshape(self._pixels_shape)

        if configs is None:
            return vals, sdev, pcov
        else:
             return vals, sdev, pcov, model

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
        signal = signal.reshape(self._shape)
        self._plot(time, signal, sdev, round_to, xlim, fname, show)

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
        signal = signal.reshape(self._shape[0], np.prod(self._shape[1:]))
        signal_pred = self._predict(time).reshape(self._shape[0], np.prod(self._shape[1:]))

        cost = loss(signal_pred, signal, metric, nfree)
        if self._pixels_shape == ():
            return cost[0]
        else:
            return cost