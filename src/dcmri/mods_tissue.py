
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np

import dcmri.mods as mods
import dcmri.sig as sig
import dcmri.rel as rel
import dcmri.tissue as tissue
import dcmri.utils as utils


class TissueArray(mods.ArrayModel):
    """Pixel-based model for vascular-interstitial tissues.

    The model accepts the following parameters:

        - **shape** (array-like, default=(32,32)): shape of the pixel array, not including the time dimensions. Any number of dimensions are allowed.
        - **parallel** (bool, default=False): use parallel computing during training or not.
        - **verbose** (int, default=0): verbosity during computation. 0: no feedback, 1: show progress bar.

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery. If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Plasma concentration (M) in the arterial input. Must be provided when aif = None, ignored otherwise.

        **Acquisition parameters**

        - **t** (array-like, default=None): Time points (sec) of the aif. If t is not provided, the temporal sampling is uniform with interval dt.
        - **dt** (float, default=1.0): Time interval of the AIF in sec.
        - **relaxivity** (float, default=0.005): Contrast agent relaxivity 
          (Hz/M). 
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec.
        - **FA** (array, default=np.full(shape, 15)): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.
        - **TP** (float, default=0): Preparation delay in a saturation-
          recovery sequence.
        - **TS** (float, default=None): Sampling duration, or duration of the 
          readout for a single time point. If TS=None, the continuous signals 
          are sampled at the required time points by interpolation. If TS is a 
          finite value, the signals are averaged over a time TS around the 
          required time point. Defaults to None.

        **Tracer-kinetic parameters**

        - **kinetics** (str, default='HF'): Tracer-kinetic model (see below for options)
        - **Hct** (float, default=0.45): Hematocrit.
        - **Fp** (array, default=np.full(shape, 0.01)): Plasma flow, or flow of plasma into the plasma compartment (mL/sec/mL).
        - **Ktrans** (array, default=np.full(shape, 0.003)): Transfer constant: volume of arterial plasma cleared of indicator per unit time and per unit tissue (mL/sec/mL).
        - **vp** (array, default=np.full(shape, 0.1)): Plasma volume, or volume fraction of the plasma compartment (mL/mL).
        - **ve** (array, default=np.full(shape, 0.5)): Extravascular, extracellular volume: volume fraction of the interstitial compartment (mL/mL).
        - **Ktrans** (array, default=np.full(shape, 0.0023)): Volume transfer constant (mL/sec/mL).
        - **v** (array, default=np.full(shape, 0.6)): Extracellular volume fraction (mL/mL).

        **Water-kinetic parameters**

        - **water_exchange** (str, default='fast'): Water exchange regime ('fast', 'none' or 'any').
        - **PSe** (array, default=np.full(shape, 10)): Transendothelial water permeability-surface area product: PS for water across the endothelium (mL/sec/mL).
        - **PSc** (array, default=np.full(shape, 10)): Transcytolemmal water permeability-surface area product: PS for water across the cell wall (mL/sec/mL).

        **Signal parameters**

        - **sequence** (str, default='SS'): imaging sequence.
        - **R10a** (float, default=1): Precontrast arterial relaxation rate in 1/sec.
        - **R10** (array, default=np.full(shape, 1)): Precontrast tissue relaxation rate in 1/sec.
        - **S0** (array, default=np.full(shape, 1)): Scale factor for the MR signal (a.u.).

        **Prediction and training parameters**

        - **n0** (int): number of precontrast baseline signals.
        - **free** (array-like): list of free parameters. The default depends on the kinetics parameter.

    Args:
        params (dict, optional): override defaults for any of the parameters.

    Notes:

        Possible values for the **kinetics** argument, along with relevant parameters::

        - 'U': uptake tissue. Parameters: Fp
        - 'NX': no tracer exchange tissue. Parameters: Fp, vp
        - 'FX': fast tracer exchange tissue. Parameters: Fp, v
        - 'WV': weakly vascularized tissue - also known as *Tofts model*.Parameters: Ktrans, ve
        - 'HFU': high-flow uptake tissue - also known as *Patlak model*. Parameters: vp, PS
        - 'HF': high-flow tissue - also known as *extended Tofts model*, *extended Patlak model* or *general kinetic model*. Params = (vp, PS, ve, )
        - '2CU': two-compartment uptake tissue. Parameters: Fp, vp, PS
        - '2CX': two-compartment exchange tissue. Parameters: Fp, vp, PS, ve

    See Also:
        `Tissue`

    Example:

    Fit a coarse image of the brain using a 2-compartment exchange model.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import numpy as np
        >>> import dcmri as dc

        Use `fake_brain` to generate synthetic test data:

        >>> n=8
        >>> time, signal, aif, gt = dc.fake_brain(n)

        Build a tissue array model and set the constants to match the experimental conditions of the synthetic test data:

        >>> shape = (n,n)
        >>> model = dc.TissueArray(
        ...     shape = shape,
        ...     aif = aif,
        ...     dt = time[1],
        ...     relaxivity = dc.relaxivity(3, 'blood', 'gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10 = 1/gt['T1'],
        ...     n0 = 15,
        ...     kinetics = '2CX',
        ... )

        Train the model on the ROI data:

        >>> model.train(time, signal)

        Plot the reconstructed maps, along with their standard deviations and the ground truth for reference:

        >>> model.plot(time, signal, ref=gt)

        As the left panel shows, the measured signals are accurately reconstructed by the model. However, while these simulated data are noise-free, they are temporally undersampled (dt = 1.5 sec). As a result the reconstruction of the parameter maps (right panel) is not perfect - as can be seen by comparison against the ground truth, or, in the absence of a ground truth, by inspecting the standard deviations. PS for instance is showing some areas where the value is overestimated and the standard deviations large.

        The interstitial volume fraction ve is wrong in a large part of the image, but this is for another reason: much of this tissue has an intact blood brain barrier, and therefore therefore the properties of the extravascular space are fundamentally unmeasureable. This type of error is dangerous because it cannot be detected by inspecting the standard deviations. When no ground truth is available, this therefore risks a misinterpretation of the results. The risk of this type of error can be reduced by applying model selection.

    """

    def __init__(self, shape, kinetics='HF',
                 water_exchange='FF', sequence='SS', **params):

        # Array model params
        self.shape = shape
        self.parallel = False
        self.verbose = 0

        # Define model
        self.kinetics = kinetics
        self.water_exchange = water_exchange
        self.sequence = sequence
        _check_config(self)

        # Input function
        self.aif = None
        self.ca = None
        self.t = None
        self.dt = 1.0

        # Model parameters
        model_pars = _model_pars(kinetics, water_exchange, sequence)
        for p in model_pars:
            if PARAMS[p]['pixel_par']:
                setattr(self, p, np.full(shape, PARAMS[p]['init']))
            else:
                setattr(self, p, PARAMS[p]['init'])
        # TODO check in set_free - only pixel_pars can be free
        self.free = {
            p: PARAMS[p]['bounds'] for p in model_pars if (
            PARAMS[p]['default_free'] and PARAMS[p]['pixel_par'])
        } 

        # overide defaults
        self._set_defaults(**params)

        # sdevs
        for par in self.free:
            setattr(self, 'sdev_' + par, np.zeros(shape).astype(np.float32))


    def _pix(self, x):

        pars = _model_pars(self.kinetics, self.water_exchange, self.sequence)
        kwargs = {}
        for p in pars:
            if PARAMS[p]['pixel_par']:
                kwargs[p] = getattr(self, p)[x]
            else:
                kwargs[p] = getattr(self, p)

        return Tissue(

            # Set config
            kinetics=self.kinetics,
            water_exchange=self.water_exchange,
            sequence=self.sequence,

            # Input function
            aif=self.aif,
            ca=self.ca,
            t=self.t,
            dt=self.dt,

            # Parameters
            **kwargs,
        )

    def info(self):
        """Print information about the tissue"""

        print('Water exchange regime: ' + self.water_exchange)
        print('Kinetics: ' + self.kinetics)
        print('\n\n')
        print('Model parameters')
        print('----------------')
        for p in _model_pars(self.kinetics, 
                             self.water_exchange, self.sequence):
            print('Parameter: ' + str(p))
            print('--> Full name: ' + PARAMS[p]['name'])
            print('--> Units: ' + PARAMS[p]['unit'])


    def _train_curve(self, args):
        pix, p = super()._train_curve(args)
        self.S0[p] = pix.S0
        return pix, p

    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at given time points

        Args:
            time (array-like): 1D array with time points.

        Returns:
            ndarray: Array of predicted y-values.
        """
        return super().predict(time)

    def train(self, time: np.ndarray, signal: np.ndarray, **kwargs):
        """Train the free parameters

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension must be time.
            kwargs: any keyword parameters accepted by `Tissue.train`.

        Returns:
            TissueArray: A reference to the model instance.
        """
        return super().train(time, signal, **kwargs)

    # TODO: Add conc and relax functions

    def cost(self, time, signal, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension must be time.
            metric (str, optional): Which metric to use - options are 'RMS' (Root-mean-square), 'NRMS' (Normalized root-mean-square), 'AIC' (Akaike information criterion), 'cAIC' (Corrected Akaike information criterion for small models) or 'BIC' (Bayesian information criterion). Defaults to 'NRMS'.

        Returns:
            ndarray: goodness of fit in each element of the data array.
        """
        return super().cost(time, signal, metric)

    def params(self, *args):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if only one parameter is required.
        """
        p = _all_pars(self.kinetics, 
                      self.water_exchange, 
                      self.sequence, 
                      self.__dict__)
        if args == ():
            args = p.keys()
        pars = []
        for a in args:
            if a in p:
                v = p[a]
            else:
                v = getattr(self, a)
            pars.append(v)
        if len(pars) == 1:
            return pars[0]
        else:
            return pars

    def export_params(self):
        pars = _all_pars(self.kinetics, 
                         self.water_exchange, 
                         self.sequence, 
                         self.__dict__)
        pars = {p: [PARAMS[p]['name'], pars[p], PARAMS[p]['unit']] 
                for p in pars}
        return self._add_sdev(pars)

    def plot(self, time, signal, vmin={}, vmax={},
             cmap='gray', ref=None, fname=None, show=True):
        """Plot parameter maps

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension is time.
            vmin (dict, optional): Minimum values on display for given parameters. Defaults to {}.
            vmax (dict, optional): Maximum values on display for given parameters. Defaults to {}.
            cmap (str, optional): matplotlib colormap. Defaults to 'gray'.
            ref (dict, optional): Reference images - typically used to display ground truth data when available. Keys are 'signal' (array of data in the same shape as signal), and the parameter maps to show. Defaults to None.
            fname (str, optional): File path to save image. Defaults to None.
            show (bool, optional): Determine whether the image is shown or not. Defaults to True.

        Raises:
            NotImplementedError: Features that are not currently implemented.
        """
        if len(self.shape) == 1:
            raise NotImplementedError('Cannot plot 1D images.')
        yfit = self.predict(time)
        params = {par: getattr(self, par) for par in _kin_pars(self.kinetics)}
        params['S0'] = self.S0

        if len(self.shape) == 2:
            ncols = 2 + len(params)
            nrows = 2 if ref is None else 3
            fig = plt.figure(figsize=(ncols * 2, nrows * 2))
            figcols = fig.subfigures(
                1, 2, wspace=0.0, hspace=0.0, width_ratios=[2, ncols - 2])

            # Left panel: signal
            ax = figcols[0].subplots(nrows, 2)
            figcols[0].subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                for j in range(2):
                    ax[i, j].set_yticks([])
                    ax[i, j].set_xticks([])

            # Signal maps
            ax[0, 0].set_title('max(signal)')
            ax[0, 0].set_ylabel('reconstruction')
            ax[0, 0].imshow(np.amax(yfit, axis=-1), vmin=0,
                            vmax=0.5 * np.amax(signal), cmap=cmap)
            ax[1, 0].set_ylabel('data')
            ax[1, 0].imshow(np.amax(signal, axis=-1), vmin=0,
                            vmax=0.5 * np.amax(signal), cmap=cmap)
            if ref is not None:
                ax[2, 0].set_ylabel('ground truth')
                ax[2, 0].imshow(np.amax(ref['signal'], axis=-1),
                                vmin=0, vmax=0.5 * np.amax(signal), cmap=cmap)
            ax[0, 1].set_title('mean(signal)')
            ax[0, 1].imshow(np.mean(yfit, axis=-1), vmin=0,
                            vmax=0.5 * np.amax(signal), cmap=cmap)
            ax[1, 1].imshow(np.mean(signal, axis=-1), vmin=0,
                            vmax=0.5 * np.amax(signal), cmap=cmap)
            if ref is not None:
                ax[2, 1].imshow(np.mean(ref['signal'], axis=-1),
                                vmin=0, vmax=0.5 * np.amax(signal), cmap=cmap)

            # Right panel: free parameters
            ax = figcols[1].subplots(nrows, ncols - 2)
            figcols[1].subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                for j in range(ncols - 2):
                    ax[i, j].set_yticks([])
                    ax[i, j].set_xticks([])
            ax[0, 0].set_ylabel('reconstruction')
            ax[1, 0].set_ylabel('std devs')
            if ref is not None:
                ax[2, 0].set_ylabel('ground truth')
            for i, par in enumerate(params.keys()):
                v0 = vmin[par] if par in vmin else None
                v1 = vmax[par] if par in vmax else None
                ax[0, i].set_title(par)
                ax[0, i].imshow(params[par], vmin=v0, vmax=v1, cmap=cmap)
                if hasattr(self, 'sdev_' + par):
                    ax[1, i].imshow(getattr(self, 'sdev_' + par),
                                    vmin=v0, vmax=v1, cmap=cmap)
                else:
                    ax[1, i].imshow(
                        np.zeros(self.shape).astype(np.int16), cmap=cmap)
                if ref is not None:
                    ax[2, i].imshow(ref[par], vmin=v0, vmax=v1, cmap=cmap)
        if len(self.shape) == 3:
            raise NotImplementedError('3D plot not yet implemented')

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_signals(self, time, signal, cmap='gray',
                     ref=None, fname=None, show=True):
        """Plot measured and reconstructed dynamic signals.

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension is time.
            cmap (str, optional): matplotlib colormap. Defaults to 'gray'.
            ref (dict, optional): Reference images - typically used to display ground truth data when available. Keys are 'signal' (array of data in the same shape as signal), and the parameter maps to show. Defaults to None.
            fname (str, optional): File path to save image. Defaults to None.
            show (bool, optional): Determine whether the image is shown or not. Defaults to True.

        Raises:
            NotImplementedError: Features that are not currently implemented.
        """
        if len(self.shape) == 1:
            raise NotImplementedError('Cannot plot 1D images.')
        yfit = self.predict(time)

        if len(self.shape) == 2:
            ncols = 1
            nrows = 2 if ref is None else 3
            # fig = plt.figure(figsize=(ncols*2, nrows*2), layout='constrained')
            fig = plt.figure(figsize=(ncols * 2, nrows * 2))

            # Left panel: signal
            # figcols[0].suptitle('Signal', fontsize='x-large')
            ax = fig.subplots(nrows, ncols)
            # figcols[0].subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.00, wspace=0)
            fig.subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                ax[i].set_yticks([])
                ax[i].set_xticks([])
            # data animation
            ax[0].set_title('signal(time)')
            ax[0].set_ylabel('data')
            im = ax[0].imshow(signal[:, :, 0], cmap=cmap,
                              animated=True, vmin=0, vmax=0.5 * np.amax(signal))
            ims = []
            for i in range(signal.shape[-1]):
                im = ax[0].imshow(signal[:, :, i], cmap=cmap,
                                  animated=True, vmin=0, vmax=0.5 * np.amax(signal))
                ims.append([im])
            anim_data = ArtistAnimation(fig, ims, interval=50)
            # fit animation
            # ax[1,0].set_title('model fit', rotation='vertical', x=-0.1,y=0.5)
            ax[1].set_ylabel('model fit')
            im = ax[1].imshow(yfit[:, :, 0], cmap=cmap,
                              animated=True, vmin=0, vmax=0.5 * np.amax(signal))
            ims = []
            for i in range(yfit.shape[-1]):
                im = ax[1].imshow(yfit[:, :, i], cmap=cmap,
                                  animated=True, vmin=0, vmax=0.5 * np.amax(signal))
                ims.append([im])
            anim_fit = ArtistAnimation(fig, ims, interval=50)
            # truth animation
            if ref is not None:
                ax[2].set_ylabel('ground truth')
                im = ax[2].imshow(ref['signal'][:, :, 0], cmap=cmap,
                                  animated=True, vmin=0, vmax=0.5 * np.amax(signal))
                ims = []
                for i in range(ref['signal'].shape[-1]):
                    im = ax[2].imshow(ref['signal'][:, :, i], cmap=cmap,
                                      animated=True, vmin=0, vmax=0.5 * np.amax(signal))
                    ims.append([im])
                anim_truth = ArtistAnimation(fig, ims, interval=50)

        if len(self.shape) == 3:
            raise NotImplementedError('3D plot not yet implemented')

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_params(self, roi=None, vmin={}, vmax={},
                    ref=None, fname=None, show=True):
        """Show parameter distributions in regions of interest.

        Args:
            roi (dict, optional): Dictionary with masks for regions-of-interest to be shown in the plot. if none is provided, the entire array is shown. Defaults to None.
            vmin (dict, optional): Minimum values on display for given parameters. Defaults to {}.
            vmax (dict, optional): Maximum values on display for given parameters. Defaults to {}.
            cmap (str, optional): matplotlib colormap. Defaults to 'gray'.
            ref (dict, optional): Reference images - typically used to display ground truth data when available. Keys are 'signal' (array of data in the same shape as ydata), and the parameter maps to show. Defaults to None.
            fname (str, optional): File path to save image. Defaults to None.
            show (bool, optional): Determine whether the image is shown or not. Defaults to True.
        """
        params = {par: getattr(self, par) for par in _kin_pars(self.kinetics)}
        params['S0'] = self.S0

        ncols = len(params)
        if roi is None:
            nrows = 1
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(
                    2 * ncols, 2 * nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2,
                                right=0.8, top=0.9, bottom=0.1)
            for i, par in enumerate(params):
                ax[i].set_yticks([])
                ax[i].set_xticks([])
                ax[i].set_title(par, fontsize=8)

                data = params[par]
                if data.size == 0:
                    continue
                if ref is not None:
                    data = np.concatenate((data, ref[par]), axis=None)
                v0 = vmin[par] if par in vmin else np.amin(data)
                v1 = vmax[par] if par in vmax else np.amax(data)
                if v0 != v1:
                    hrange = [v0, v1]
                else:
                    hrange = [-1, 1] if v0 == 0 else [0.9 * v0, 1.1 * v0]

                if ref is not None:
                    ax[i].hist(ref[par], range=[
                               vmin[par], vmax[par]], label='Truth')
                ax[i].hist(params[par], range=[vmin[par],
                           vmax[par]], label='Reconstructed')
            ax[-1].legend(loc='center left',
                          bbox_to_anchor=(1, 0.5), fontsize=8)
        else:
            nrows = len(roi)
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(
                    2 * ncols, 2 * nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2,
                                right=0.8, top=0.9, bottom=0.1)
            i = 0
            for name, mask in roi.items():
                ax[i, 0].set_ylabel(
                    name, fontsize=8, rotation='horizontal', labelpad=30)
                for p, par in enumerate(params):
                    ax[i, p].set_yticks([])
                    ax[i, p].set_xticks([])
                    if i == 0:
                        ax[i, p].set_title(par, fontsize=8)

                    data = params[par][mask == 1]
                    if data.size == 0:
                        continue
                    if ref is not None:
                        data = np.concatenate(
                            (data, ref[par][mask == 1]), axis=None)
                    v0 = vmin[par] if par in vmin else np.amin(data)
                    v1 = vmax[par] if par in vmax else np.amax(data)
                    if v0 != v1:
                        hrange = [v0, v1]
                    else:
                        hrange = [-1, 1] if v0 == 0 else [0.9 * v0, 1.1 * v0]

                    if ref is not None:
                        ax[i, p].hist(ref[par][mask == 1],
                                      range=hrange, label='Truth')
                    ax[i, p].hist(params[par][mask == 1],
                                  range=hrange, label='Reconstructed')
                i += 1
            ax[0, -1].legend(loc='center left',
                             bbox_to_anchor=(1, 0.5), fontsize=8)

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_fit(self, time, signal,
                 hist_kwargs={}, roi=None, ref=None,
                 fname=None, show=True,
                 ):
        """Plot time curves and fits in representative pixels.

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension is time.
            hist_kwargs (dict, optional): Keyword arguments to be passed on the matlotlib's hist() finction. Defaults to {}.
            roi (dict, optional): Dictionary with masks for regions-of-interest to be shown in the plot. if none is provided, the entire array is shown. Defaults to None.
            ref (dict, optional): Reference images - typically used to display ground truth data when available. Keys are 'signal' (array of data in the same shape as signal), and the parameter maps to show. Defaults to None.
            fname (str, optional): File path to save image. Defaults to None.
            show (bool, optional): Determine whether the image is shown or not. Defaults to True.
        """
        nt = signal.shape[-1]
        signal = signal.reshape((-1, nt))
        yfit = self.predict(time).reshape((-1, nt))
        # rms = 100*np.linalg.norm(signal-yfit, axis=-1)/np.linalg.norm(signal, axis=-1)
        rms = np.linalg.norm(signal - yfit, axis=-1)
        cols = ['fit error histogram', '5th perc',
                '25th perc', 'median', '75th perc', '95th perc']
        if roi is None:
            nrows = 1
            ncols = 6
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(
                    2 * ncols, 2 * nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2,
                                right=0.8, top=0.9, bottom=0.1)
            for r in range(nrows):
                for c in range(ncols):
                    ax[r, c].set_xticks([])
                    ax[r, c].set_yticks([])
            for c in range(ncols):
                ax[0, c].set_title(cols[c], fontsize=8)
            _plot_roi(time, signal, yfit, ref,
                      hist_kwargs, rms, ax, 'all pixels')
        else:
            nrows = len(roi)
            ncols = 6
            fig, ax = plt.subplots(
                nrows, ncols, figsize=(
                    2 * ncols, 2 * nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2,
                                right=0.8, top=0.9, bottom=0.1)
            for r in range(nrows):
                for c in range(ncols):
                    ax[r, c].set_xticks([])
                    ax[r, c].set_yticks([])
            for c in range(ncols):
                ax[0, c].set_title(cols[c], fontsize=8)
            i = 0
            for name, mask in roi.items():
                _plot_roi(time, signal, yfit, ref, hist_kwargs,
                          rms, ax[i, :], name, mask=mask.ravel())
                i += 1
        legend = ax[0, -1].legend(loc='center left',
                                  bbox_to_anchor=(1, 0.5), fontsize=8)
        labels = ['Truth', 'Prediction', 'Data']
        for i, label in enumerate(legend.get_texts()):
            label.set_text(labels[i])

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


# TODO: iex and wex instead of kinetics and water_exchange? Water exchange
# is also kinetics.

class Tissue(mods.Model):
    """Model for general vascular-interstitial tissues. See sections on 
    :ref:`two-site-exchange` and :ref:`imaging-sequences` for more detail 
    on notations and definitions.

    Args:
        kinetics (str, optional): Tracer-kinetic model. Possible values are 
         '2CX', '2CU', 'HF', 'HFU', 'NX', 'FX', 'WV', 'U'. Defaults to 'HF'.
        water_exchange (str, optional): Water exchange regime, Any combination 
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
          not provided. 
          
    Relevant parameters and their defaults are listed in the tables below:

        .. list-table:: **Tissue parameters** 
            :widths: 20 30 30
            :header-rows: 1 

            * - Parameters
              - When to use
              - Further detail
            * - **n0**
              - Always
              - :ref:`model-fitting-params`
            * - **r1**, **R10**
              - Always
              - :ref:`relaxation-params`
            * - **R10a**, **Ha**, **B1corr_a**
              - When **aif** is provided
              - :ref:`relaxation-params`, :ref:`params-per-sequence`
            * - **S0**, **FA**, **TR**, **TS**, **B1corr**
              - Always
              - :ref:`params-per-sequence`
            * - **TP**, **TC**
              - If **sequence** is 'SR'
              - :ref:`params-per-sequence`
            * - **Fp**, **PS**, **Ktrans**, **vp**, **vb**, **H**, **vi**, 
                **ve**, **vc**, **PSe**, **PSc**.
              - Depends on **kinetics** and **water_exchange**
              - :ref:`kinetic-regimes`

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
              - 0.005
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
            * - Ha
              - Kinetic
              - 0.45
              - [0, 1]
              - Fixed
            * - Fp
              - Kinetic
              - 0.01
              - [0, inf]
              - Free
            * - Ktrans
              - Kinetic
              - 0.003
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
            * - vp
              - Kinetic
              - 0.055
              - [0, 1]
              - Free


    See Also:
        `Liver`, `Kidney`

    Example:

        Single time-curve analysis: fit extended Tofts model to data.

    .. plot::
        :include-source:
        :context: close-figs

        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(CNR=50)

        Build a tissue and set the parameters to match the experimental 
        conditions of the synthetic data:

        >>> model = dc.Tissue(
        ...     aif = aif,
        ...     dt = time[1],
        ...     relaxivity = dc.relaxivity(3, 'blood','gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     n0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right), 
        using the noise-free ground truth as reference:

        >>> model.plot(time, roi, ref=gt)

    """

    def __init__(self, 
                 kinetics='HF', water_exchange='FF', sequence='SS', 
                 aif=None, ca=None, t=None, dt=1.0, 
                 free=None, **params):

        # Set configuration
        self.kinetics = kinetics
        self.water_exchange = water_exchange
        self.sequence = sequence
        _check_config(self)

        # Input function
        self.aif = aif
        self.ca = ca
        self.t = t
        self.dt = dt

        # Model parameters
        model_pars = _model_pars(kinetics, water_exchange, sequence)
        for p in model_pars:
            setattr(self, p, PARAMS[p]['init']) # TODO _pars attr?
        free_pars = [p for p in model_pars if PARAMS[p]['default_free']]
        self.free = {p: PARAMS[p]['bounds'] for p in free_pars} # TODO make private _free
 
        # overide defaults
        self._set_defaults(free=free, **params)


    def info(self):
        """Print information about the tissue"""

        print('Water exchange regime: ' + self.water_exchange)
        print('Kinetics: ' + self.kinetics)
        print('\n\n')
        print('Model parameters')
        print('----------------')
        for p in _model_pars(self.kinetics, 
                             self.water_exchange, self.sequence):
            print('Parameter: ' + str(p))
            print('--> Full name: ' + PARAMS[p]['name'])
            print('--> Units: ' + PARAMS[p]['unit'])


    def time(self):
        """Return an array of time points

        Returns:
            np.ndarray: time points in seconds.
        """
        if self.t is None:
            if self.aif is not None:
                return self.dt * np.arange(np.size(self.aif))
            elif self.ca is not None:
                return self.dt * np.arange(np.size(self.ca))
            else:
                raise ValueError('Either aif or ca must be provided.')
        else:
            return self.t

    def _check_ca(self):

        # Arterial concentrations
        if self.ca is None:
            if self.aif is None:
                raise ValueError(
                    """Either aif or ca must be provided to predict 
                       signal data.""")
            else:
                if self.sequence == 'SR':
                    cb = sig.conc_src(self.aif, self.TC,
                                      1 / self.R10a, 
                                      self.relaxivity, self.n0)
                elif self.sequence == 'SS':
                    cb = sig.conc_ss(self.aif, self.TR, self.B1corr_a*self.FA,
                                     1 / self.R10a, 
                                     self.relaxivity, self.n0)
                self.ca = cb / (1 - self.Ha)

    def conc(self, sum=True):
        """Return the tissue concentration

        Args:
            sum (bool, optional): If True, returns the total concentrations. 
              Else returns the concentration in the individual compartments. 
              Defaults to True.

        Returns:
            np.ndarray: Concentration in M
        """
        self._check_ca()
        pars = _all_pars(self.kinetics, self.water_exchange, self.sequence, 
                         self.__dict__)
        pars = {p: pars[p] for p in _kin_pars(self.kinetics)}
        # pars = {p: getattr(self, p) for p in _kin_pars(self.kinetics)}
        return tissue.conc_tissue(
            self.ca, t=self.t, dt=self.dt, sum=sum, kinetics=self.kinetics, 
            **pars)

    def relax(self):
        """Compartmental rselaxation rates, volume fractions and 
        water-permeability matrix.

        tuple: relaxation rates of tissue compartments and their volumes.
            - **R1** (numpy.ndarray): in the fast water exchange limit, the 
              relaxation rates are a 1D array. In all other situations, 
              relaxation rates are a 2D-array with dimensions (k,n), where k is 
              the number of compartments and n is the number of time points 
              in ca.
            - **v** (numpy.ndarray or None): the volume fractions of the tissue 
              compartments. Returns None in 'FF' regime. 
            - **PSw** (numpy.ndarray or None): 2D array with water exchange 
              rates between tissue compartments. Returns None in 'FF' regime. 
        """
        # TODO: ADD diagonal element to PSw (flow term)!!
        # Fb = self.Fp/(1-self.Hct)
        # PSw = np.array([[Fb,self.PSe,0],[self.PSe,0,self.PSc],[0,self.PSc,0]])
        self._check_ca()
        pars = _tissue_pars(self.kinetics, self.water_exchange)
        pars = {p: getattr(self, p) for p in pars}
        # TODO in FF v should be a 1-element list (and signal needs to adjust
        # accordingly)
        R1, v, PSw = tissue.relax_tissue(self.ca, self.R10, 
                                    self.relaxivity, t=self.t, 
                                    dt=self.dt,
                                    kinetics=self.kinetics, 
                                    water_exchange=
                                    self.water_exchange.replace('N', 'R'), 
                                    **pars)
        return R1, v, PSw

    def signal(self, sum=True) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        R1, v, PSw = self.relax()
        if self.sequence == 'SR':
            if not sum:
                raise ValueError(
                    """Separate signals for signal model SR are 
                       not yet implemented.""")
            return sig.signal_sr(R1, self.S0, self.TR,
                                 self.B1corr*self.FA, self.TC, self.TP, v=v, 
                                 PSw=PSw)
        elif self.sequence == 'SS':
            return sig.signal_ss(R1, self.S0, self.TR,
                                 self.B1corr*self.FA, v=v, PSw=PSw, sum=sum)

    # TODO: make time optional (if not provided, assume equal to self.time())
    def predict(self, time: np.ndarray) -> np.ndarray:
        """Predict the data at specific time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        t = self.time()
        if np.amax(time) > np.amax(t):
            msg = """The acquisition window is longer than the duration 
                     of the AIF. The largest time point that can be 
                     predicted is """ + str(np.amax(t) / 60) + 'min.'
            raise ValueError(msg)
        sig = self.signal()
        return utils.sample(time, t, sig, self.TS)

    def train(self, time, signal, method='NLLS', **kwargs):
        """Train the free parameters

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element 
              of *time*.
            method (str, optional): Method to use for training. Currently the 
              only option is 'NNLS' (Non-negative least squares). 
              Default is 'NNLS'.
            kwargs: any keyword parameters accepted by the specified fit 
              method. For 'NNLS' these are all parameters accepted by 
              `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            Tissue: A reference to the model instance.
        """
        # Estimate S0
        if self.sequence == 'SR':
            Sref = sig.signal_sr(self.R10, 1, self.TR,
                                 self.B1corr*self.FA, self.TC, self.TP)
        elif self.sequence == 'SS':
            Sref = sig.signal_ss(self.R10, 1, self.TR, self.B1corr*self.FA)
        else:
            raise NotImplementedError(
                'Signal model ' + self.sequence + 'is not (yet) supported.')

        self.S0 = np.mean(signal[:self.n0]) / Sref if Sref > 0 else 0

        # If there is no signal, set all free parameters to zero
        if self.S0 == 0:
            for par in self.free:
                setattr(self, par, 0)
            return self

        if method == 'NLLS':
            return mods.train(self, time, signal, **kwargs)

        if method == 'PSMS':  # Work in progress
            # Fit the complete model
            mods.train(self, time, signal, **kwargs)
            # Fit an intravascular model with the same free parameters
            iv = deepcopy(self)
            # iv.kinetics = 'NX'
            setattr(iv, 'PS', 0)
            for par in ['ve', 'PS', 'PSc']:
                if par in iv.free:
                    iv.free.pop(par)
            mods.train(iv, time, signal, **kwargs)
            # If the intravascular model has a lower AIC, take the free
            # parameters from there
            if iv.cost(time, signal, metric='cAIC') < self.cost(
                    time, signal, metric='cAIC'):
                for par in self.free:
                    setattr(self, par, getattr(iv, par))
                # # If the tissue is 1-compartmental and the blood MTT > 30s
                # # Assume the observed compartment is actually interstitial.
                # if iv.vp/iv.Fp > 30:
                #     self.vp = 0
                #     self.Fp = 0
                #     self.PS = 0
                #     self.vi = iv.vp
                #     self.Ktrans = iv.Fp
                #     self.v = iv.vp

    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is not 
              provided, the values are not rounded. Defaults to None.

        Returns:
            list or float: values of parameter values, or a scalar value if 
            only one parameter is required.
        """
        p = _all_pars(self.kinetics, self.water_exchange, self.sequence, 
                      self.__dict__)
        if args == ():
            args = p.keys()
        pars = []
        for a in args:
            if a in p:
                v = p[a]
            else:
                if hasattr(self, a):
                    v = getattr(self, a)
                else:
                    msg = (a + ' is not a model parameter, and cannot be '
                           + 'derived from the model parameters.')
                    raise ValueError(msg)
            if round_to is not None:
                v = round(v, round_to)
            pars.append(v)
        if len(pars) == 1:
            return pars[0]
        else:
            return pars

    def export_params(self):
        pars = _all_pars(self.kinetics, self.water_exchange, self.sequence, 
                         self.__dict__)
        pars = {p: [PARAMS[p]['name'], pars[p], PARAMS[p]['unit']] for p in pars}
        return self._add_sdev(pars)

    def cost(self, time, signal, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element 
              of *time*.
            metric (str, optional): Which metric to use (see below for 
              options). Defaults to 'NRMS'. 

        Returns:
            float: goodness of fit.

        Possible metrics are:

        - 'RMS' (Root-mean-square)
        - 'NRMS' (Normalized root-mean-square) 
        - 'AIC' (Akaike information criterion) 
        - 'cAIC' (Corrected Akaike information criterion for small models) 
        - 'BIC' (Bayesian information criterion) 
        """
        return super().cost(time, signal, metric=metric)

    def plot(self, time=None, signal=None,
             xlim=None, ref=None, fname=None, show=True):
        """Plot the model fit against data.

        Args:
            time (array-like, optional): Array with time points.
            signal (array-like, optional): Array with measured signals for 
              each element of *time*.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form 
              (x,y), where x is an array with x-values and y is an array with 
              y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is 
              provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        t = self.time()
        if time is None:
            time = t

        if xlim is None:
            xlim = [np.amin(t), np.amax(t)]

        if self.water_exchange != 'FF':
            fig, ax = plt.subplots(2, 2, figsize=(10, 12))
            ax00 = ax[0, 0]
            ax01 = ax[0, 1]
            ax10 = ax[1, 0]
            ax11 = ax[1, 1]
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax00 = ax[0]
            ax01 = ax[1]

        ax00.set_title('MRI signals')
        if ref is not None:
            if 'signal' in ref:
                ax00.plot(t / 60, ref['signal'], linestyle='-', linewidth=3.0,
                          color='lightgray', label='Tissue ground truth')
        ax00.plot(time / 60, self.predict(time), marker='o', linestyle='None',
                  color='cornflowerblue', label='Predicted data')
        if signal is not None:
            ax00.plot(time / 60, signal, marker='x', linestyle='None',
                      color='darkblue', label='Data')
        ax00.plot(t / 60, self.predict(t), linestyle='-',
                  linewidth=3.0, color='darkblue', label='Model')
        ax00.set(ylabel='MRI signal (a.u.)', xlim=np.array(xlim) / 60)
        ax00.legend()

        C = self.conc(sum=False)
        if C.ndim==1:
            C = C.reshape((1,-1))
        v, comps = _plot_labels_kin(self.kinetics)
        pars = _all_pars(self.kinetics, self.water_exchange, self.sequence, 
                         self.__dict__)
        pars = {p: pars[p] for p in _kin_pars(self.kinetics)}
        ax01.set_title('Concentration in indicator compartments')
        if ref is not None:
            # ax01.plot(ref['t']/60, 1000*ref['C'], marker='o', 
            # linestyle='None', color='lightgray', label='Tissue ground truth')
            ax01.plot(ref['t'] / 60, 1000 * ref['cp'], marker='o', 
                      linestyle='None',
                      color='lightcoral', label='Arterial ground truth')
        ax01.plot(t / 60, 1000 * self.ca, linestyle='-', linewidth=5.0,
                  color='lightcoral', label='Arterial plasma')
        for k, vk in enumerate(v):
            if vk in pars:
                ck = C[k, :] / pars[vk]
                ax01.plot(t / 60, 1000 * ck, linestyle='-', linewidth=3.0,
                          label=comps[k], color=_clr(comps[k]))
        # ax01.plot(t/60, 1000*np.sum(C,axis=0), linestyle='-', 
        # linewidth=3.0, color='darkblue', label='Tissue')
        ax01.set(ylabel='Concentration (mM)', xlim=np.array(xlim) / 60)
        ax01.legend()

        if self.water_exchange != 'FF':

            R1, v, PSw = self.relax()
            c = rel.c_lin(R1, self.relaxivity)
            comps = _plot_labels_relax(self.kinetics, self.water_exchange)
            if R1.ndim == 1:
                c = c.reshape((1, len(c)))
            ax11.set_title('Concentration in water compartments')
            for i in range(c.shape[0]):
                ax11.plot(t / 60, 1000 * c[i, :], linestyle='-',
                          linewidth=3.0, color=_clr(comps[i]), label=comps[i])
            ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)',
                     xlim=np.array(xlim) / 60)
            ax11.legend()

            S = self.signal(sum=False)
            ax10.set_title('Magnetization in water compartments')
            for i in range(S.shape[0]):
                if np.isscalar(v):  # TODO renove this after ref of v
                    Si = S[i, ...] / v
                else:
                    Si = S[i, ...] / v[i]
                ax10.plot(t / 60, Si, linestyle='-', linewidth=3.0,
                          color=_clr(comps[i]), label=comps[i])
            ax10.set(xlabel='Time (min)',
                     ylabel='Magnetization (a.u.)', xlim=np.array(xlim) / 60)
            ax10.legend()

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


# Helper functions

def _clr(comp):
    if comp == 'Plasma':
        return 'darkred'
    if comp == 'Interstitium':
        return 'steelblue'
    if comp == 'Extracellular':
        return 'dimgrey'
    if comp == 'Tissue':
        return 'darkgrey'
    if comp == 'Blood':
        return 'darkred'
    if comp == 'Extravascular':
        return 'blue'
    if comp == 'Tissue cells':
        return 'lightblue'
    if comp == 'Blood + Interstitium':
        return 'purple'
    
def _plot_labels_kin(kin):

    if kin == '2CX':
        return ['vp', 'vi'], ['Plasma', 'Interstitium']
    if kin == '2CU':
        return ['vp', 'vi'], ['Plasma', 'Interstitium']
    if kin == 'HF':
        return ['vp', 'vi'], ['Plasma', 'Interstitium']
    if kin == 'HFU':
        return ['vp', 'vi'], ['Plasma', 'Interstitium']
    if kin == 'FX':
        return ['ve'], ['Extracellular']
    if kin == 'NX':
        return ['vp'], ['Plasma']
    if kin == 'U':
        return ['vp'], ['Plasma']
    if kin == 'WV':
        return ['vi'], ['Interstitium']


def _plot_labels_relax(kin, wex) -> list:

    if wex == 'FF':
        return ['Tissue']

    if wex in ['RR', 'NN', 'NR', 'RN']:
        if kin == 'WV':
            return ['Interstitium']
        elif kin in ['NX', 'U']:
            return ['Blood', 'Extravascular']
        else:
            return ['Blood', 'Interstitium', 'Tissue cells']

    if wex in ['RF', 'NF']:
        if kin == 'WV':
            return ['Interstitium']
        else:
            return ['Blood', 'Extravascular']

    if wex in ['FR', 'FN']:
        if kin == 'WV':
            return ['Interstitium', 'Tissue cells']
        else:
            return ['Blood + Interstitium', 'Tissue cells']



def _plot_roi(xdata, ydata, yfit, ref, hist_kwargs, rms, ax, name, mask=None):
    ax[0].set_ylabel(name, fontsize=8, rotation='horizontal', labelpad=30)
    if np.size(rms[mask == 1]) == 0:
        return
    perc = np.nanpercentile(rms[mask == 1], [5, 25, 50, 75, 95])
    if np.count_nonzero(~np.isnan(perc)) == 0:
        return
    if mask is None:
        inroi = np.ones(rms.shape) == 1
    else:
        inroi = mask == 1
    loc = [(rms == p) & inroi for p in perc]
    ax[0].hist(rms[inroi], **hist_kwargs)
    if ref is not None:
        style = {'color': 'lightsteelblue', 'linewidth': 5.0}
        yref = ref['signal'].reshape((-1, ydata.shape[-1]))
        ax[1].plot(xdata, np.mean(yref[loc[0], :], axis=0),
                   label='Truth (5th perc)', **style)
        ax[2].plot(xdata, np.mean(yref[loc[1], :], axis=0),
                   label='Truth (25th perc)', **style)
        ax[3].plot(xdata, np.mean(yref[loc[2], :], axis=0),
                   label='Truth (median)', **style)
        ax[4].plot(xdata, np.mean(yref[loc[3], :], axis=0),
                   label='Truth (75th perc)', **style)
        ax[5].plot(xdata, np.mean(yref[loc[4], :], axis=0),
                   label='Truth (95th perc)', **style)
    style = {'color': 'darkblue'}
    ax[1].plot(xdata, np.mean(yfit[loc[0], :], axis=0),
               label='Prediction (5th perc)', **style)
    ax[2].plot(xdata, np.mean(yfit[loc[1], :], axis=0),
               label='Prediction (25th perc)', **style)
    ax[3].plot(xdata, np.mean(yfit[loc[2], :], axis=0),
               label='Prediction (median)', **style)
    ax[4].plot(xdata, np.mean(yfit[loc[3], :], axis=0),
               label='Prediction (75th perc)', **style)
    ax[5].plot(xdata, np.mean(yfit[loc[4], :], axis=0),
               label='Prediction (95th perc)', **style)
    style = {'marker': 'o', 'markersize': 1,
             'linestyle': 'None', 'color': 'crimson'}
    ax[1].plot(xdata, np.mean(ydata[loc[0], :], axis=0),
               label='Data (5th perc)', **style)
    ax[2].plot(xdata, np.mean(ydata[loc[1], :], axis=0),
               label='Data (25th perc)', **style)
    ax[3].plot(xdata, np.mean(ydata[loc[2], :], axis=0),
               label='Data (median)', **style)
    ax[4].plot(xdata, np.mean(ydata[loc[3], :], axis=0),
               label='Data (75th perc)', **style)
    ax[5].plot(xdata, np.mean(ydata[loc[4], :], axis=0),
               label='Data (95th perc)', **style)


def _check_config(self: Tissue):

    if self.sequence not in ['SS', 'SR']:
        msg = 'Sequence ' + str(self.sequence) + ' is not available.'
        raise ValueError(msg)

    if self.kinetics not in ['2CX', '2CU', 'HF', 'HFU', 'WV', 'FX', 'NX', 'U']:
        msg = 'The value ' + str(self.kinetics) + \
            ' for the kinetics argument is not recognised.'
        msg += '\n possible values are 2CX, 2CU, HF, HFU, WV, FX, NX, U.'
        raise ValueError(msg)

    if self.water_exchange not in ['FF', 'NF',
                                   'RF', 'FN', 'NN', 'RN', 'FR', 'NR', 'RR']:
        msg = 'The value ' + str(self.water_exchange) + \
            ' for the water_exchange argument is not recognised.'
        msg += """\n It must be a 2-element string composed of characters 
                  N, F, R."""
        raise ValueError(msg)


# CONFIGURATIONS


def _model_pars(kin, wex, seq):
    pars = ['relaxivity']
    pars += ['Ha', 'R10a', 'B1corr_a']
    pars += _seq_pars(seq)
    pars += _tissue_pars(kin, wex)
    pars += ['R10','S0','n0']
    return pars


def _seq_pars(seq):
    if seq=='SS':
        return ['B1corr', 'FA','TR', 'TS']
    elif seq =='SR':
        return ['B1corr', 'FA', 'TR', 'TC', 'TP', 'TS']


def _tissue_pars(kin, wex) -> list:

    pars = _relax_pars(kin, wex)

    if wex == 'FF':
        return pars
    if wex == 'RR':
        if kin == 'WV':
            return ['PSc'] + pars
        elif kin in ['NX', 'U']:
            return ['PSe'] + pars
        else:
            return ['PSe', 'PSc'] + pars
    if wex == 'NN':
        return pars
    if wex == 'NR':
        if kin in ['NX', 'U']:
            return pars
        else:
            return ['PSc'] + pars
    if wex == 'RN':
        if kin == 'WV':
            return pars
        else:
            return ['PSe'] + pars
    if wex == 'RF':
        if kin == 'WV':
            return pars
        else:
            return ['PSe'] + pars
    if wex == 'NF':
        return pars
    if wex == 'FR':
        return ['PSc'] + pars
    if wex == 'FN':
        return pars

    
def _relax_pars(kin, wex) -> list:

    if wex == 'FF':
        return _kin_pars(kin)

    if kin == '2CX':
        return ['H', 'vb', 'vi', 'Fp', 'PS']
    if kin == 'HF':
        return ['H', 'vb', 'vi', 'PS']
    if kin == 'WV':
        return ['vi', 'Ktrans']

    if wex in ['RR', 'NN', 'NR', 'RN']:

        if kin == '2CU':
            return ['H', 'vb', 'vi', 'Fp', 'PS']
        if kin == 'HFU':
            return ['H', 'vb', 'vi', 'PS']
        if kin == 'FX':
            return ['H', 'vb', 'vi', 'Fp']
        if kin == 'NX':
            return ['H', 'vb', 'Fp']
        if kin == 'U':
            return ['vb', 'Fp']

    if wex in ['RF', 'NF']:

        if kin == '2CU':
            return ['H', 'vb', 'Fp', 'PS']
        if kin == 'HFU':
            return ['H', 'vb', 'PS']
        if kin == 'FX':
            return ['H', 'vb', 'vi', 'Fp']
        if kin == 'NX':
            return ['H', 'vb', 'Fp']
        if kin == 'U':
            return ['vb', 'Fp']

    if wex in ['FR', 'FN']:

        if kin == '2CU':
            return ['vc', 'vp', 'Fp', 'PS']
        if kin == 'HFU':
            return ['vc', 'vp', 'PS']
        if kin == 'FX':
            return ['vc', 've', 'Fp']
        if kin == 'NX':
            return ['vc', 'vp', 'Fp']
        if kin == 'U':
            return ['vc', 'Fp']
        
def _kin_pars(kin):

    if kin == '2CX':
        return ['vp', 'vi', 'Fp', 'PS']
    if kin == '2CU':
        return ['vp', 'Fp', 'PS']
    if kin == 'HF':
        return ['vp', 'vi', 'PS']
    if kin == 'HFU':
        return ['vp', 'PS']
    if kin == 'FX':
        return ['ve', 'Fp']
    if kin == 'NX':
        return ['vp', 'Fp']
    if kin == 'U':
        return ['Fp']
    if kin == 'WV':
        return ['vi', 'Ktrans']




PARAMS = {
    'relaxivity': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0,np.inf],
        'name': 'Contrast agent relaxivity',
        'unit': 'Hz/M',
        'pixel_par': False,
    },
    'Ha': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Arterial Hematocrit',
        'unit': '',
        'pixel_par': False,
    },
    'R10a': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial precontrast R1',
        'unit': 'Hz',
        'pixel_par': False,
    },
    'B1corr_a': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Arterial B1-correction factor', 
        'unit': '',
        'pixel_par': False,
    },
    'Fp': {
        'init': 0.01,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Plasma flow',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'PS': {
        'init': 0.003,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Permeability-surface area product',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'vp': {
        'init': 0.1*(1-0.45),
        'default_free': True,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Plasma volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'vi': {
        'init': 0.3,
        'default_free': True,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Interstitial volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'Ktrans': {
        'init': 0.003 * 0.01 / (0.003 + 0.01),
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Volume transfer constant',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    've': {
        'init': 0.1*(1-0.45) + 0.3,
        'default_free': True,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Extracellular volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'vb': {
        'init': 0.1,
        'default_free': True,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Blood volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'vc': {
        'init': 0.6,
        'default_free': True,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Intracellular volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1-1e-3],
        'name': 'Tissue Hematocrit',
        'unit': '',
        'pixel_par': True,
    },
    'PSe': {
        'init': 0.03,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Transendothelial water PS',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'PSc': {
        'init': 0.03,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Transcytolemmal water PS',
        'unit': 'mL/sec/cm3',
        'pixel_par': True,
    },
    'B1corr': {
        'init': 1,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue B1-correction factor',
        'unit': '',
        'pixel_par': True,
    },
    'FA': {
        'init': 15,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Flip angle',
        'unit': 'deg',
        'pixel_par': False,
    },
    'TR': {
        'init': 0.005,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Repetition time',
        'unit': 'sec',
        'pixel_par': False,
    },
    'TC': {
        'init': 0.2,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Time to k-space center',
        'unit': 'sec',
        'pixel_par': False,
    },
    'TP': {
        'init': 0.05,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Preparation delay',
        'unit': 'sec',
        'pixel_par': False,
    },
    'TS': {
        'init': None,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Sampling time',
        'unit': 'sec',
        'pixel_par': False,
    },
    'R10': {
        'init': 0.7,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Tissue precontrast R1',
        'unit': 'Hz',
        'pixel_par': True,
    },
    'S0': {
        'init': 1.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Signal scaling factor',
        'unit': 'a.u.',
        'pixel_par': True,
    },
    'n0': {
        'init': 1,
        'default_free': False,
        'bounds': None,
        'name': 'Number of precontrast acquisitions',
        'unit': '',
        'pixel_par': False,
    },

    # Derived parameters

    'FAcorr': {
        'name': 'B1-corrected Flip Angle',
        'unit': 'deg',
    },
    'E': {
        'name': 'Tracer extraction fraction',
        'unit': '',
    },
    'Ti': {
        'name': 'Interstitial mean transit time',
        'unit': 'sec',
    },
    'Tp': {
        'name': 'Plasma mean transit time',
        'unit': 'sec',
    },
    'Tb': {
        'name': 'Blood mean transit time',
        'unit': 'sec',
    },
    'Te': {
        'name': 'Extracellular mean transit time',
        'unit': 'sec',
    },
    'Twc': {
        'name': 'Intracellular water mean transit time',
        'unit': 'sec',
    },
    'Twi': {
        'name': 'Interstitial water mean transit time',
        'unit': 'sec',
    },
    'Twb': {
        'name': 'Intravascular water mean transit time',
        'unit': 'sec',
    },
}



def _all_pars(kin, wex, seq, p):

    p = {par: p[par] for par in _model_pars(kin, wex, seq)}

    try:
        p['Ktrans'] = _div(p['Fp'] * p['PS'], p['Fp'] + p['PS'])
    except KeyError:
        pass
    try:
        p['vp'] = p['vb']*(1 - p['H'])
    except KeyError:
        pass
    try:
        p['ve'] = p['vi'] + p['vc']
    except KeyError:
        pass
    try:
        p['E'] = _div(p['PS'], p['Fp'] + p['PS'])
    except KeyError:
        pass
    try:
        p['Ti'] = _div(p['vi'], p['PS'])
    except KeyError:
        pass
    try:
        p['Tp'] = _div(p['vp'], p['PS'] + p['Fp'])
    except KeyError:
        pass
    try:
        p['Tb'] = _div(p['vp'], p['Fp'])
    except KeyError:
        pass
    try:
        p['Te'] = _div(p['ve'], p['Fp'])
    except KeyError:
        pass
    try:
        p['Twc'] = _div(1 - p['vb'] - p['vi'], p['PSc'])
    except KeyError:
        pass
    try:
        p['Twi'] = _div(p['vi'], p['PSc'] + p['PSe'])
    except KeyError:
        pass
    try:
        p['Twb'] = _div(p['vb'], p['PSe'])
    except KeyError:
        pass
    try:
        p['FAcorr'] = p['B1corr']*p['FA']
    except KeyError:
        pass

    return p


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b == 0, 0, np.divide(a, b))
    
