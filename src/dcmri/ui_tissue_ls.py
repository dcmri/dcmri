import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec

from dcmri import rel, sig, pk_inv, lib, utils, ui
from dcmri.lexicon import LEXICON
import dcmri.lexicon_utils as lexicon


class TissueLS(ui.SuperModel):
    """Array of linear and stationary tissues with a single inlet.

    These are generic model-free tissue types. Their response to 
    an indicator injection is proportional to the dose (linear) and 
    independent of the time of injection (stationary).

    Args:
        shape (array-like, required): shape of the tissue array (spatial dimensions only). 
          Any number of dimensions is allowed.
        aif (array-like, required): Signal-time curve in the blood of the
          feeding artery. 
        dt (float, optional): Time interval between values of the arterial
          input function. Defaults to 1.0.
        sequence (str, optional): imaging sequence. Possible values 
          are 'SS', 'SR' and 'lin' (linear). Defaults to 'SS'.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. 

    See Also:
        `TissueLS`, `TissueArray`

    Example:

        Fit a linear and stationary model to the synthetic test data:

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import dcmri as dc

        Generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue()

        The correct ground truth for ve in model-free analysis is the 
        extracellular part of the distribution space:

        >>> gt['ve'] = gt['vp'] + gt['vi'] if gt['PS'] > 0 else gt['vp']

        Build a tissue and set the constants to match the
        experimental conditions of the synthetic test data. 

        >>> tissue = dc.TissueLS(
        ...     dt = time[1],
        ...     sequence = 'SS',
        ...     r1 = dc.relaxivity(3, 'blood','gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10a = 1/dc.T1(3.0,'blood'),
        ...     R10 = 1/dc.T1(3.0,'muscle'),
        ... )

        Train the tissue on the data. Since have noise-free synthetic 
        data we use a lower tolerance than the default, which is optimized 
        for noisy data:

        >>> tissue.train(roi, aif, n0=10, tol=0.01)

        Plot the reconstructed signals along with the concentrations 
        and the impulse response function.

        >>> tissue.plot(roi)

    """
    def __init__(self, shape=None, sequence='SS', **params):

        # Check configuration
        if shape is not None:
            if len(shape) not in [1, 2, 3]:
                raise ValueError(
                    f"The 'shape' parameter specifies spatial dimensions "
                    "and must be 1-, 2- or 3 dimensional (or empty for 1D data)"
                )
        if sequence not in ['SS', 'SR', 'lin']:
            raise ValueError(
                f"Sequence {sequence} is not recognized. "
                f"Current options are 'SS', 'SR', 'lin'."
            )
        
        self._version = '1.0'
        self._cnfg = {'sequence': sequence}
        self._pars = lexicon.init(self._pars_list(), LEXICON)
        self._shape = shape

        # Override defaults with user-provided parameters
        for p, val in params.items():
            if p in self._pars:
                self._pars[p] = val
            else:
                raise ValueError(f"'{p}' is not a valid parameter for this configuration.")

        # Try to read the spatial shape from the data
        if self._pars['irf'].ndim > 1:
            # If the user has provided a multidimensional IRF, 
            # take the shape from that (and ignore the shape keyword)
            self._shape = self._pars['irf'].shape[:-1]
        else:
            # Else if one of the images is non-scalar take the shape from that
            for array_par in self._pars_list('pixel'):
                if not np.isscalar(self._pars[array_par]):
                    self._shape = self._pars[array_par].shape

        # If an IRF has not been provided then initialise with ones of the right shape
        n_times = self._pars['c_a'].shape[-1]
        if 'irf' not in params:
            if self._shape is None:
                n_samples = 1
            else:
                n_samples = np.prod(self._shape)
            self._pars['irf'] = 0.02 * np.ones((n_samples, n_times))
        else:
            if self._pars['irf'].ndim == 1:
                n_samples = 1
            else:
                n_samples = np.prod(self._pars['irf'].shape[:-1])
            self._pars['irf'] = self._pars['irf'].reshape(n_samples, -1)
        
        # Ensure 1D shape (n_samples, ) for all pixel-level parameters
        # If scalars are provided for array data, use it to initialise 
        # the array
        for array_par in self._pars_list('pixel'):
            if np.isscalar(self._pars[array_par]):
                self._pars[array_par] = np.full(n_samples, self._pars[array_par])
            else:
                self._pars[array_par] = self._pars[array_par].reshape(n_samples)
        

    def _pars_list(self, select='all'):
        pars_seq = {
            'SR': ['FA', 'TR', 'TC', 'TP'],
            'SS': ['FA', 'TR'],
            'lin': [],
        }[self._cnfg['sequence']]

        pars_list = {
            'all': pars_seq + [
                'c_a', 'irf', 'dt', 'field_strength', 'agent',
                'TS', 'S0', 'R10', 'B1corr', 'noise_sdev',
            ],
            'sequence': pars_seq,
            'pixel': ['R10', 'S0', 'B1corr'],
        }
        return pars_list[select]
    
    def _compute_concentration(self):
        p = self._pars

        if p['c_a'].size != p['irf'].shape[-1]:
            raise ValueError("Cannot compute concentrations as IRF and AIF have different number of time points")
 
        ca_mat = p['dt'] * pk_inv.convmat(p['c_a'])
        # Transpose here because for matrix multiplication 
        # we need (rows, columns) = (n_times, n_samples)
        conc = ca_mat @ p['irf'].T
        # Transpose back to get result in standard form (n_samples, n_times)
        self._C = conc.T
    
    def _compute_relaxation_rate(self): 
        self._compute_concentration()
        p = self._pars

        rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
        self._R1 = rel.relax(self._C, p['R10'], rp)

    def _compute_signal(self):
        self._compute_relaxation_rate()
        p = self._pars

        seq_name = self._cnfg['sequence']
        pars = {k: p[k] for k in self._pars_list('sequence')}
        n_samples, n_times = p['irf'].shape

        def _compute_pixel_signal(x):
            if 'FA' in pars: pars['FA'] = p['FA'] * p['B1corr'][x] 
            return sig.signal(seq_name, self._R1[x,:], p['S0'][x], **pars)

        if n_samples==1:
            self._S = _compute_pixel_signal(0)
        else:
            results = Parallel(n_jobs=-1)(delayed(_compute_pixel_signal)(x) for x in range(n_samples))
            self._S = np.array(results)

        self._S = self._S.reshape(n_samples, n_times)

    def _set_time(self):
        p = self._pars
        self._t = p['dt'] * np.arange(p['c_a'].size)

    def _predict(self, time: np.ndarray):
        self._set_time()
        self._compute_signal()
        p = self._pars

        return utils.sample(time, self._t, self._S, p['TS'])
  
    # ==========================================
    # Inverse Model: Training
    # ==========================================

    def _estimate_parameters(self, aif: ui.Input, n0: int):
        self._set_time()
        p = self._pars

        seq_name = self._cnfg['sequence']
        seq_pars = {k: p[k] for k in self._pars_list('sequence')}
        
        # Arterial concentration estimation
        if aif is not None:
            if 'FA' in seq_pars: seq_pars['FA'] = p['FA'] * aif.B1corr
            rp = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
            ca = sig.conc(seq_name, aif.signal, aif.R10, rp, n0=n0, **seq_pars)
            self._t = np.arange(0, aif.time[-1] + p['dt'], p['dt'])
            p['c_a'] = np.interp(self._t, aif.time, ca)


    def _train(self, time, signal, aif: ui.Input, n0=1, tol=0.1):
        self._set_time()
        p = self._pars

        self._estimate_parameters(aif, n0)

        # Compute tissue concentration
        seq_name = self._cnfg['sequence']
        seq_pars = {k: p[k] for k in self._pars_list('sequence')}

        n_samples, n_times = signal.shape[0], p['c_a'].size
        def _conc_pixel(x):
            r1 = lib.relaxivity(p['field_strength'], 'blood', p['agent'])
            if 'FA' in seq_pars: seq_pars['FA'] = p['FA'] * p['B1corr'][x]
            C_x = sig.conc(seq_name, signal[x,:], p['R10'][x], r1, n0=n0, **seq_pars)
            C_x[np.isnan(C_x)] = 0
            C_x = np.interp(self._t, time, C_x, right=0, left=0)
            # Compute S0 on the fly. This is not needed for deconvolution 
            # analysis but we are computing it anyway so that signal 
            # predictions can be verified agains data directly
            s_ref = sig.signal(seq_name, p['R10'][x], 1, **seq_pars)
            S0 = np.mean(signal[x,:n0]) / s_ref if s_ref > 0 else 0
            return C_x, np.array(S0)

        if n_samples==1:
            C, S0 = _conc_pixel(0)
        else:
            results = Parallel(n_jobs=-1)(delayed(_conc_pixel)(x) for x in range(n_samples))
            C = np.array([r[0] for r in results])
            S0 = np.array([r[1] for r in results])

        C = C.reshape(n_samples, n_times)
        p['S0'] = S0.reshape(n_samples)

        # Deconvolve with arterial concentration
        p['irf'] = pk_inv.deconv(C.T, p['c_a'], p['dt'], tol=tol).T

        return p['irf']

    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _params(self, iv=False, Hct=0.45):
        p = self._pars
        amax = np.max(p['irf'], axis=1)
        auc = np.sum(p['irf'], axis=1) * p['dt']
        ratio = np.zeros_like(auc, dtype=float)
        np.divide(auc, amax, out=ratio, where=amax != 0)

        if iv:
            # C(t) = Fb exp(-t/Tb) x ca(t)
            return {
                'S0': p['S0'],
                'Fb': amax,
                'vb': auc,
                'Tb': ratio,  
            }
        else:
            return {
                'S0': p['S0'],
                'Fp': amax * (1 - Hct),
                've': auc * (1 - Hct),
                'Te': ratio,
            }

    def _plot(self, time, signal, fname, show, round_to):
        self._set_time()
        self._compute_signal()
        p = self._pars

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig.subplots_adjust(wspace=0.3)

        # Plot predicted signals and measured signals
        ax[0].set_title('MRI signals')
        ax[0].set(ylabel='MRI signal (a.u.)', xlabel='Time (min)')
        ax[0].plot(
            self._t / 60, np.mean(self._S, axis=0), linestyle='-', 
            linewidth=3.0, color='cornflowerblue', 
            label='Tissue (predicted)',
        )
        ax[0].plot(
            time / 60, np.mean(signal, axis=0), marker='x', linestyle='None',
            color='darkblue', label='Tissue (measured)',
        )
        ax[0].legend()

        # Plot predicted concentrations and measured concentrations
        ax[1].set_title('Tissue concentrations')
        ax[1].set(ylabel='Concentration (mM)', xlabel='Time (min)')
        ax[1].plot(
            self._t / 60, 1000 * p['c_a'], linestyle='-', linewidth=5.0,
            color='lightcoral', label='Arterial blood',
        )
        ax[1].plot(
            self._t / 60, 1000 * np.mean(self._C, axis=0), linestyle='-', linewidth=3.0, 
            color='cornflowerblue', label='Tissue (predicted)', 
        )
        ax[1].legend()

        # Plot impulse response
        ax[2].set_title('Impulse response function')
        ax[2].set(ylabel='IRF (mL/sec/cm3)', xlabel='Time (min)')
        ax[2].plot(
            self._t / 60, np.mean(p['irf'], axis=0), linestyle='-', linewidth=3.0,
            color='cornflowerblue', label='IRF', 
        )
        ax[2].legend()

        # Plot text
        vals = {k: v[0] for k, v in self._params().items()}
        msg = lexicon.string_params(vals, round_to=round_to)
        msg = "\n".join(list(msg.values()))        
        ax[3].set_title('Free parameters')
        ax[3].axis("off")  # hide axes
        ax[3].text(0, 0.9, msg, fontsize=10, transform=ax[3].transAxes, ha="left", va="top")

        # Show and/or save plot
        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    def _plot_2d(self, time, signal, fname, show, vmin, vmax, truth=None):
        self._compute_signal()
        
        s_recon = self._S.reshape(self._shape + (-1,))
        signal_2d = signal.reshape(self._shape + (-1,))
        
        params = self._params()
        params = {k: v.reshape(self._shape) for k, v in params.items()}
        
        nrows = 2 if truth is not None else 1
        n_signal_pairs = 2 
        total_pairs = n_signal_pairs + len(params)
        
        # 1. Simplify Logic: [Image, Colorbar] only. 
        # The gap is now handled by 'wspace', not a 3rd column.
        col_ratios = [1.0, 0.05] * total_pairs
        total_cols = len(col_ratios)
        
        fig = plt.figure(figsize=(total_pairs * 3.5, nrows * 3.5))
        
        # 2. Use wspace=0.4 to create the "Buffer Space" between pairs
        # Since we have 2 columns per group, we only want the gap every 2nd column.
        # We do this by setting a global wspace and then overriding it if needed.
        gs = GridSpec(nrows, total_cols, figure=fig, width_ratios=col_ratios, 
                    wspace=0.4, hspace=0.2)

        v_max_all = 0.5 * max(np.amax(s_recon), np.amax(signal_2d))

        # --- 1. Signal Columns ---
        for c, func in enumerate([np.mean, np.max]):
            img_col = c * 2      # Indices are now 0, 2, 4...
            cbar_col = img_col + 1
            
            for r in range(nrows):
                data = s_recon if r == 0 else signal_2d
                ax = fig.add_subplot(gs[r, img_col])
                ax.set_title(f'{"Mean" if c==0 else "Max"}\n({"Recon" if r==0 else "Original"})', fontsize=9)
                im = ax.imshow(func(data, axis=-1), vmin=0, vmax=v_max_all, cmap='gray')
                
                # Remove ticks immediately upon creation
                ax.set_xticks([]); ax.set_yticks([])

                if r == 0:
                    cax = fig.add_subplot(gs[:, cbar_col])
                    fig.colorbar(im, cax=cax)

        # --- 2. Parameter Columns ---
        for i, (par, data) in enumerate(params.items()):
            img_col = (n_signal_pairs * 2) + (i * 2)
            cbar_col = img_col + 1
            
            v0 = vmin.get(par, np.percentile(data, 1)) if isinstance(vmin, dict) else np.percentile(data, 1)
            v1 = vmax.get(par, np.percentile(data, 99)) if isinstance(vmax, dict) else np.percentile(data, 99)
            
            for r in range(nrows):
                ax = fig.add_subplot(gs[r, img_col])
                ax.set_title(f"{par}\n({'Recon' if r==0 else 'Truth'})", fontsize=9)
                
                plot_data = data if r == 0 else truth[par].reshape(self._shape)
                im_p = ax.imshow(plot_data, vmin=v0, vmax=v1, cmap='gray')
                ax.set_xticks([]); ax.set_yticks([])

                if r == 0:
                    cax_p = fig.add_subplot(gs[:, cbar_col])
                    cb = fig.colorbar(im_p, cax=cax_p)
                    cb.ax.tick_params(labelsize=8)

        if fname: plt.savefig(fname, bbox_inches='tight', dpi=300)
        if show: plt.show()
        else: plt.close(fig)

    def _plot_3d(self, time, signal, fname, show, vmin, vmax):
        self._compute_signal()
        
        # 1. Prepare Data
        params = self._params()
        params = {k: v.reshape(self._shape) for k, v in params.items()}
        
        width, height, n_slices = self._shape
        aspect_ratio = 16/9

        for par, img in params.items():
            # Calculate Mosaic Grid (nrows x ncols)
            nrows = int(np.round(np.sqrt((width * n_slices) / (aspect_ratio * height))))
            nrows = max(1, nrows)
            ncols = int(np.ceil(n_slices / nrows))

            # 2. Create the "Big" 2D Canvas
            # We fill with NaNs so the "empty" tiles in the last row don't show up as 0
            mosaic = np.full((nrows * width, ncols * height), np.nan)

            # 3. Stitch slices into the canvas
            for i in range(n_slices):
                row = i // ncols
                col = i % ncols
                
                # Map the 2D slice into the 2D canvas
                # We transpose the slice (.T) to maintain standard medical orientation
                slice_2d = np.flip(img[:, :, i], 0)
                
                r_start, r_end = row * width, (row + 1) * width
                c_start, c_end = col * height, (col + 1) * height
                
                mosaic[r_start:r_end, c_start:c_end] = slice_2d

            # 4. Plot the single large array
            fig, ax = plt.subplots(figsize=(12, 12 * (nrows/ncols)))
            
            v0 = vmin.get(par) if isinstance(vmin, dict) else None
            v1 = vmax.get(par) if isinstance(vmax, dict) else None
            if v0 is None: v0 = np.nanpercentile(img, 1)
            if v1 is None: v1 = np.nanpercentile(img, 99)

            im = ax.imshow(mosaic, cmap='magma', vmin=v0, vmax=v1, origin='lower')
            ax.axis('off')
            ax.set_title(par, fontsize=14)

            # Add one clean colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 5. Save/Show
            if fname is not None:
                folder, filename = os.path.split(fname)
                name, ext = os.path.splitext(filename)
                fig.savefig(os.path.join(folder, f"{par}_{name}{ext}"), bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close(fig)


    # ==========================================
    # Inverse Model: Training
    # ==========================================


    def params(self, iv=False, Hct=0.45):
        """Parameters derived from the impulse response"""
        params = self._params(iv, Hct)
        if self._shape is None:
            return {k: v[0] for k, v in params.items()}
        else:
            return {k: v.reshape(self._shape) for k, v in params.items()}
    
    def time(self) -> np.ndarray:
        """Signal time points"""
        self._set_time()
        return self._t

    def conc(self):
        """Return the tissue concentration

        Returns:
            np.ndarray: Concentration in M
        """
        self._compute_concentration()
        if self._shape is None:
            return self._C[0,:]
        else:
            n_times = self._pars['irf'].shape[-1]
            return self._C.reshape(self._shape + (n_times, ))

    def relax(self):
        """Tissue relaxation rates

        Returns:
            np.ndarray: Concentration in M
        """
        self._compute_relaxation_rate()
        if self._shape is None:
            return self._R1[0,:]
        else:
            n_times = self._pars['irf'].shape[-1]
            return self._C.reshape(self._shape + (n_times, ))

    def signal(self) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        self._compute_signal()
        if self._shape is None:
            return self._S[0,:]
        else:
            n_times = self._pars['irf'].shape[-1]
            return self._S.reshape(self._shape + (n_times, ))

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Predicted Signal for each element of *time*.
        """
        S_pred = self._predict(time)
        if self._shape is None:
            return S_pred[0,:]
        else:
            n_times = len(time)
            return S_pred.reshape(self._shape + (n_times, ))

    def train(self, time, signal:np.ndarray, aif:ui.Input=None, n0=1, tol=0.1):
        """Train the free parameters

        Args:
            signal (array-like): Array with measured signals.
            aif (Input, optional): AIF signal, time and baseline R1.
            n0 (int, optional): Number of baseline time points.
            tol: cut-off value for the singular values in the 
                computation of the matrix pseudo-inverse.

        Returns:
            ndarray: impulse response function
        """ 
        if signal.ndim==1:
            signal = signal.reshape(1, -1) 
        else:
            n_times = len(time)
            signal = signal.reshape(-1, n_times)

        self._train(time, signal, aif, n0, tol)

        if self._shape is None:
            return self._pars['irf'][0,:]
        else:
            n_times = self._pars['irf'].shape[-1]
            return self._pars['irf'].reshape(self._shape + (n_times, ))
        
    def plot(
        self, time, signal: np.ndarray, round_to=None, fname=None, 
        show=True
    ):
        """Plot the model fit against data

        Args:
            time (array-like): Array with time points
            signal (array-like, optional): Array with measured signals.
            round_to (int, optional): Rounding for the model parameters.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. 
            show (bool, optional): If True, the plot is shown. 
        """
        if signal.ndim==1:
            signal = signal.reshape(1, -1) 
        else:
            n_times = len(time)
            signal = signal.reshape(-1, n_times)
        self._plot(time, signal, fname, show, round_to)

    def plot_2d(
        self, time, signal: np.ndarray, fname=None, 
        show=True, vmin=None, vmax=None, truth=None
    ):
        """Plot the model fit against data

        Args:
            time (array-like): Array with time points
            signal (array-like, optional): Array with measured signals.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. 
            show (bool, optional): If True, the plot is shown. 
        """
        if signal.ndim==1:
            raise ValueError("Cannot apply plot_2d() to a 1D signal. Please use plot() instead")
        else:
            n_times = len(time)
            signal = signal.reshape(-1, n_times)

        if len(self._shape)==1:
            raise ValueError("Cannot apply plot_2d() to a 1D signal. Please use plot() instead")
        elif len(self._shape)==3:
            raise ValueError("Cannot apply plot_2d() to a 3D signal. Please use plot_3d() instead")

        self._plot_2d(time, signal, fname, show, vmin, vmax, truth)

    def plot_3d(
        self, time, signal: np.ndarray, fname=None, 
        show=True, vmin=None, vmax=None,
    ):
        """Plot the model fit against data

        Args:
            time (array-like): Array with time points
            signal (array-like, optional): Array with measured signals.
            fname (path, optional): Filepath to save the image. If no value is
              provided, the image is not saved. 
            show (bool, optional): If True, the plot is shown. 
        """
        if signal.ndim==1:
            raise ValueError("Cannot apply plot_3d() to a 1D signal. Please use plot() instead")
        else:
            n_times = len(time)
            signal = signal.reshape(-1, n_times)

        if len(self._shape)==1:
            raise ValueError("Cannot apply plot_3d() to a 1D signal. Please use plot() instead")
        elif len(self._shape)==2:
            raise ValueError("Cannot apply plot_3d() to a 2D signal. Please use plot_2d() instead")

        self._plot_3d(time, signal, fname, show, vmin, vmax)
