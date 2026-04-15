import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec

from dcmri import magnetization
from dcmri.lexicon import string_params, SEQUENCES  
from dcmri.inverse import SignalToConc
from dcmri.core import SuperModel, Input
from dcmri.core import SuperFunc
from dcmri import relaxivity
from dcmri.utils.misc import sample
from dcmri.utils.fit import loss
from dcmri import convolution


def conc_ls(ca, irf, dt):
    ca = np.array(ca)
    if ca.size != irf.shape[-1]:
        raise ValueError("Cannot compute concentrations as IRF and AIF have different number of time points")
    
    ca_mat = dt * convolution.convmat(ca)
    # Reshape IRF to 2D (n_samples, n_times)
    shape = irf.shape
    irf = irf.reshape(-1, irf.shape[-1])
    # Convolve with transpose because for matrix multiplication 
    # we need (rows, columns) = (n_times, n_samples)
    conc = ca_mat @ irf.T
    # Transpose back to get result in standard form (n_samples, n_times)
    # Then convert back to original shape 
    return conc.T.reshape(shape)

def irf_ls(ca, c, dt, tol=1e-2):
    # Reshape c to 2D (n_samples, n_times)
    shape = c.shape
    c = c.reshape(-1, c.shape[-1])
    # Deconvolve with arterial concentration
    # Use transpose because for matrix multiplication 
    # we need (rows, columns) = (n_times, n_samples)
    irf = convolution.deconv(c.T, ca, dt, tol=tol)
    # Transpose back to get result in standard form (n_samples, n_times)
    # Then convert back to original shape 
    return irf.T.reshape(shape)
        

class Signal(SuperFunc):
    configs = {
        'sequence': [s for s, v in SEQUENCES.items() if v['steady-state']]
    }  
    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(**params)   

    def __call__(self, ca, **params):
        p = self._update_pars(**params)
        seq = self._cnfg['sequence']

        C = conc_ls(ca, p['irf'], p['dt'])

        signal = magnetization.Signal(seq, **p)

        if seq == 'SE-EPI':
            R2 = relaxivity.relax_t2(C, 0, p['r2'])
            return signal(R2=R2, TA=np.inf, PA=None)
        elif seq == 'GE-EPI':
            R2s = relaxivity.relax_t2s(C, 0, p['r2s']) 
            return signal(R2s=R2s, TA=np.inf, PA=None)
        elif seq == 'DE-EPI':
            R2 = relaxivity.relax_t2(C[0,:], 0, p['r2'])
            R2s = relaxivity.relax_t2s(C[1,:], 0, p['r2s'])
            return signal(R2=R2, R2s=R2s, TA=np.inf, PA=None)
        else:
            R1 = relaxivity.relax_t1(C, p['R10'], p['r1'])  
            return signal(R1=R1, TE=0)  
    
    def _params(self):
        seq = self._cnfg['sequence']
        p = ['irf', 'dt']
        p_excl = ['R1', 'R2s', 'R2', 'R1i', 'Fi', 'me', 'v', 'Fw']
        if seq == 'SE-EPI':
            p += ['r2']
            p_excl += ['TA', 'PA']
        elif seq == 'GE-EPI':
            p += ['r2s']
            p_excl += ['TA', 'PA']
        elif seq == 'DE-EPI':
            p += ['r2', 'r2s']
            p_excl += ['TA', 'PA']
        else:
            p += ['R10', 'r1']  
            p_excl += ['TE'] 
        p += [ps for ps in magnetization.Signal(seq)._params() if ps not in p_excl]
        return p
    
class BaselineSignal(SuperFunc):
    configs = {
        'sequence': [s for s, v in SEQUENCES.items() if v['steady-state']]
    }  
    def __init__(self, sequence='3D-SPGR-SS', **params):
        self._cnfg = self._set_config(sequence=sequence)
        self._pars = self._set_pars(**params)   

    def __call__(self, **params):
        p = self._update_pars(**params)
        seq = self._cnfg['sequence']

        signal = magnetization.Signal(seq, **p)

        if seq == 'SE-EPI':
            return signal(R2=0, TA=np.inf, PA=None)
        elif seq == 'GE-EPI': 
            return signal(R2s=0, TA=np.inf, PA=None)
        elif seq == 'DE-EPI':
            return signal(R2=0, R2s=0, TA=np.inf, PA=None)
        else:
            return signal(R1=p['R10'], TE=0)  
    
    def _params(self):
        seq = self._cnfg['sequence']
        p = []
        p_excl = ['R1', 'R2s', 'R2', 'R1i', 'Fi', 'me', 'v', 'Fw']
        if seq == 'SE-EPI':
            p_excl += ['TA', 'PA']
        elif seq == 'GE-EPI':
            p_excl += ['TA', 'PA']
        elif seq == 'DE-EPI':
            p_excl += ['TA', 'PA']
        else:
            p += ['R10']  
            p_excl += ['TE'] 
        p += [ps for ps in magnetization.Signal(seq)._params() if ps not in p_excl]
        return p
    

class TissueLS(SuperModel):
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

        >>> time, aif, roi, gt = dc.fake.tissue()

        The correct ground truth for ve in model-free analysis is the 
        extracellular part of the distribution space:

        >>> gt['ve'] = gt['vp'] + gt['vi'] if gt['PS'] > 0 else gt['vp']

        Build a tissue and set the constants to match the
        experimental conditions of the synthetic test data. 

        >>> tissue = dc.TissueLS(
        ...     dt = time[1],
        ...     sequence = 'SS',
        ...     r1 = dc.const.r1(3, 'blood','gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10a = 1/dc.const.T1(3.0,'blood'),
        ...     R10 = 1/dc.const.T1(3.0,'muscle'),
        ... )

        Train the tissue on the data. Since have noise-free synthetic 
        data we use a lower tolerance than the default, which is optimized 
        for noisy data:

        >>> tissue.train(roi, aif, n0=10, tol=0.01)

        Plot the reconstructed signals along with the concentrations 
        and the impulse response function.

        >>> tissue.plot(roi)
    """
    configs = {
        'sequence': [s for s, v in SEQUENCES.items() if v['steady-state']]
    }
    def __init__(
        self, 
        sequence='3D-SPGR-SS', 
        shape=None, 
        **params,
    ):
        cnfg = {
            'sequence': sequence,
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

        # Try to read the spatial shape from the data
        self._pixels_shape: tuple = None

        # Build a list of user-defined pixel shapes
        # Start with the non-scalar shapes from any user-defined pixel parameters
        shapes = [np.array(v).shape for p, v in params.items() if p in self._params('pixel')]
        shapes = [s for s in shapes if np.prod(s) > 1]
        # If the user has defined an IRF, add its pixel shape to the list
        if 'irf' in params:
            irf_pixel_shape = np.array(params['irf']).shape[:-1]
            if np.prod(irf_pixel_shape) > 1:
                shapes += [irf_pixel_shape]

        # Build pixels_shape from the user-defined shapes, checking for consistency
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

        # If an IRF has not been provided then initialise with ones of the right shape
        if 'irf' not in params:
            self._pars['irf'] = 0.02 * np.ones(self._shape)
        elif self._pars['irf'].size != np.prod(self._shape):
            raise ValueError(f"IRF has {self._pars['irf'].size} values but should have {np.prod(self._shape)} values to match the shape of the AIF.")
        else:
            self._pars['irf'] = self._pars['irf'].reshape(self._shape)
        
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
        seq = self._cnfg['sequence']
        if select == 'all':
            return ['c_a', 'TS'] + Signal(seq)._params()
        if select == 'pixel':
            return [p for p in self._params() if p in ['S0', 'B1corr', 'R10']]
  
    # ==========================================
    # Inverse Model
    # ==========================================

    def _estimate_parameters(self, aif: Input, n0: int):
        if aif is None:
            return
        p = self._pars
        seq = self._cnfg['sequence']

        ca = SignalToConc(seq, **p)(aif.signal, S0=None, R10=aif.R10, n0=n0, B1corr=aif.B1corr)
        # Interpolate on internal time
        t = np.arange(0, aif.time[-1] + p['dt'], p['dt'])
        p['c_a'] = np.interp(t, aif.time, ca)


    def _train(self, time, signal, aif: Input, n0=1, tol=0.1):
        self._estimate_parameters(aif, n0)

        t = self._time()
        p = self._pars
        seq = self._cnfg['sequence']

        # Compute tissue concentration
        def _conc_pixel(x):
            kwargs_x = self._pixel_pars(x)
            C_x = SignalToConc(seq, **kwargs_x)(signal[x, ...], S0=None, n0=n0)
            C_x = np.stack([np.interp(t, time, C_x[i,:], right=0, left=0) for i in range(C_x.shape[0])])
            # Compute S0 on the fly so signal predictions can be verified
            s_ref = BaselineSignal(seq, **kwargs_x)(S0=1)
            s_ref = np.mean(s_ref)
            s_avr = np.mean(signal[x, :, :n0]) 
            S0_x = np.divide(s_avr, s_ref, out=np.zeros_like(s_avr), where=s_ref != 0)
            return C_x, S0_x

        if self._shape[0]==1:
            results = [_conc_pixel(0)]
        else:
            results = Parallel(n_jobs=-1)(delayed(_conc_pixel)(x) for x in range(self._shape[0]))

        C = np.array([r[0] for r in results]).reshape(self._shape)
        p['S0'] = np.array([r[1] for r in results]).reshape(self._shape[0])

        # Deconvolve with arterial concentration
        p['irf'] = irf_ls(p['c_a'], C, p['dt'], tol=tol)


    # ==========================================
    # I/O and Reporting
    # ==========================================

    def _concentration(self):
        p = self._pars
        return conc_ls(p['c_a'], p['irf'], p['dt'])
    
    def _signal(self) -> np.ndarray: # (n_pixels, n_channels, n_times)
        p = self._pars

        def pixel_signal(x) -> np.ndarray: # (n_channels, n_times)
            kwargs_x = self._cnfg | self._pixel_pars(x)
            kwargs_x = {k: v for k, v in kwargs_x.items() if k != 'irf'} | {'irf': p['irf'][x, ...]}
            S = Signal(**kwargs_x)(p['c_a'])
            return S.reshape(-1, S.shape[-1])  # (n_channels, n_times)
        
        if self._shape[0]==1:
            results = [pixel_signal(0)]
        else:
            results = Parallel(n_jobs=-1)(delayed(pixel_signal)(x) for x in range(self._shape[0]))
        
        return np.stack(results) # (n_pixels, n_channels, n_times)

    def _time(self):
        p = self._pars
        return p['dt'] * np.arange(p['c_a'].size) # (n_times, )

    def _predict(self, time):
        t = self._time()
        s = self._signal() # (n_pixels, n_channels, n_times)
        p = self._pars
        n_channels = s.shape[1]
        readouts = [sample(time, t, s[:,i,:], p['TS']) for i in range(n_channels)]
        return np.stack(readouts, axis=1) # (n_pixels, n_channels, n_times)

    def _parameters(self, iv=False, Hct=0.45):
        p = self._pars
        amax = np.max(p['irf'], axis=-1)
        auc = np.sum(p['irf'], axis=-1) * p['dt']
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
        t = self._time()
        C = self._concentration() # (n_pixels, n_channels, n_times)
        S = self._signal() # (n_pixels, n_channels, n_times)
        p = self._pars

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig.subplots_adjust(wspace=0.3)

        # Plot predicted signals and measured signals
        ax[0].set_title('MRI signals')
        ax[0].set(ylabel='MRI signal (a.u.)', xlabel='Time (min)')
        for i in range(S.shape[1]):
            ax[0].plot(
                t / 60, S[0, i, :].T, linestyle='-', 
                linewidth=1.0, color='cornflowerblue', alpha=0.3,
            )
            ax[0].plot(
                t / 60, np.mean(S[:, i, :], axis=0), linestyle='-', 
                linewidth=3.0, color='cornflowerblue', 
                label='Tissue (predicted)',
            )
            ax[0].plot(
                time / 60, np.mean(signal[:, i, :], axis=0), marker='x', linestyle='None',
                color='darkblue', label='Tissue (measured)',
            )
        ax[0].legend()

        # Plot predicted concentrations and measured concentrations
        ax[1].set_title('Tissue concentrations')
        ax[1].set(ylabel='Concentration (mM)', xlabel='Time (min)')
        ax[1].plot(
            t / 60, 1000 * p['c_a'], linestyle='-', linewidth=5.0,
            color='lightcoral', label='Arterial blood',
        )
        for i in range(C.shape[1]):
            ax[1].plot(
                t / 60, 1000 * np.mean(C[:, i, :], axis=0), linestyle='-', linewidth=3.0, 
                color='cornflowerblue', label='Tissue (predicted)', 
            )
        ax[1].legend()

        # Plot impulse response
        ax[2].set_title('Impulse response function')
        ax[2].set(ylabel='IRF (mL/sec/cm3)', xlabel='Time (min)')
        for i in range(C.shape[1]):
            ax[2].plot(
                t / 60, np.mean(p['irf'][:, i, :], axis=0), linestyle='-', linewidth=3.0,
                color='cornflowerblue', label='IRF', 
            )
        ax[2].legend()

        # Plot text
        vals = {k: v[0] for k, v in self._parameters().items()}
        msg = string_params(vals, round_to=round_to)
        msg = "\n".join(list(msg.values()))        
        ax[3].set_title('Free parameters')
        ax[3].axis("off")  # hide axes
        ax[3].text(0, 0.9, msg, fontsize=10, transform=ax[3].transAxes, ha="left", va="top")

        # Show and/or save plot
        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    def _plot_2d(self, time, signal, fname, show, vmin, vmax, truth=None):
        channel = 0 # For multi-channel data this shows channel 0 only. TODO: Generalize
         
        s_recon = self._signal()
        
        shape_2d = self._pixels_shape + (self._shape[-2],)
        s_recon = s_recon.reshape(shape_2d + (self._shape[-1],))
        signal_2d = signal.reshape(shape_2d + (self._shape[-1],))
        
        params = self._parameters()
        params = {k: v.reshape(shape_2d) for k, v in params.items()}
        
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

        # --- 1. Signal Columns ---
        for c, func in enumerate([np.mean, np.max]):
            img_col = c * 2      # Indices are now 0, 2, 4...
            cbar_col = img_col + 1

            s_recon_c = func(s_recon[..., channel, :], axis=-1)
            signal_2d_c = func(signal_2d[..., channel, :], axis=-1)

            v_max_all = np.percentile([s_recon_c, signal_2d_c], 99)
            v_min_all = np.percentile([s_recon_c, signal_2d_c], 1)
            
            for r in range(nrows):
                data = s_recon_c if r == 0 else signal_2d_c
                ax = fig.add_subplot(gs[r, img_col])
                ax.set_title(f'{"Mean" if c==0 else "Max"}\n({"Recon" if r==0 else "Original"})', fontsize=9)
                im = ax.imshow(data, vmin=v_min_all, vmax=v_max_all, cmap='gray')
                
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
                
                plot_data = data if r == 0 else truth[par].reshape(shape_2d)
                im_p = ax.imshow(plot_data[..., channel], vmin=v0, vmax=v1, cmap='gray')
                ax.set_xticks([]); ax.set_yticks([])

                if r == 0:
                    cax_p = fig.add_subplot(gs[:, cbar_col])
                    cb = fig.colorbar(im_p, cax=cax_p)
                    cb.ax.tick_params(labelsize=8)

        if fname: plt.savefig(fname, bbox_inches='tight', dpi=300)
        if show: plt.show()
        else: plt.close(fig)

    def _plot_3d(self, time, signal, fname, show, vmin, vmax):
        # TODO: For multi-channel data this shows channel 0 only. 
        channel = 0

        # 1. Prepare Data
        shape_3d = self._pixels_shape + (self._shape[-2],)
        params = self._parameters()
        params = {k: v.reshape(shape_3d) for k, v in params.items()}
        
        width, height, n_slices, n_channels = shape_3d
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
                slice_2d = np.flip(img[:, :, i, channel], 0)
                
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


    def parameters(self, iv=False, Hct=0.45):
        """Parameters derived from the impulse response"""
        params = self._parameters(iv, Hct)
        return {k: v.reshape(self._pixels_shape) for k, v in params.items()}
    
    def time(self) -> np.ndarray:
        """Signal time points"""
        return self._time()

    def conc(self):
        """Return the tissue concentration

        Returns:
            np.ndarray: Concentration in M
        """
        C = self._concentration() # (n_pixels, n_compartments, n_times)
        C = C.reshape(self._pixels_shape + C.shape[1:]) # (nx, ny, nz, n_compartments, n_times)
        if C.shape[-2] == 1:
            return C[...,0,:]
        else:
            return C

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
            np.ndarray: Predicted Signal for each element of *time*.
        """
        S = self._predict(time) # (n_pixels, n_channels, n_times)
        S = S.reshape(self._pixels_shape + S.shape[1:]) # (nx, ny, nz, n_channels, n_times)
        if S.shape[-2] == 1: # one channel - squeeze out
            return S[...,0,:] # (nx, ny, nz, n_times)
        else:
            return S # (nx, ny, nz, n_channels, n_times) or # (nx, ny, nz, n_times)

    def train(self, time, signal:np.ndarray, aif:Input=None, n0=1, tol=0.1):
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
        if aif is not None:
            n_times = aif.signal.size
        else:
            n_times = self._pars['c_a'].size

        if self._cnfg['sequence'] in ['DE-EPI']:
            n_channels = 2
        else:
            n_channels = 1

        signal = signal.reshape(int(np.prod(self._pixels_shape)), n_channels, n_times)
        self._train(time, signal, aif, n0, tol)

        orig_shape = self._pixels_shape + self._shape[-2:]
        return self._pars['irf'].reshape(orig_shape)
    
        
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
        signal = signal.reshape(self._shape)
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
        if len(self._pixels_shape) in [0,1]:
            raise ValueError("Cannot apply plot_2d() to a 1D signal. Please use plot() instead")
        elif len(self._pixels_shape)==3:
            raise ValueError("Cannot apply plot_2d() to a 3D signal. Please use plot_3d() instead")

        signal = signal.reshape(self._shape)
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
        if len(self._pixels_shape) in [0,1]:
            raise ValueError("Cannot apply plot_3d() to a 1D signal. Please use plot() instead")
        elif len(self._pixels_shape)==2:
            raise ValueError("Cannot apply plot_3d() to a 2D signal. Please use plot_2d() instead")

        signal = signal.reshape(self._shape)
        self._plot_3d(time, signal, fname, show, vmin, vmax)

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
        signal = signal.reshape(self._shape)
        signal_pred = self._predict(time).reshape(self._shape[0], np.prod(self._shape[1:]))

        cost = loss(signal_pred, signal, metric, nfree)
        if self._pixels_shape == ():
            return cost[0]
        else:
            return cost
