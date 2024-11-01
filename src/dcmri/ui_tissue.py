
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np

import dcmri.ui as ui
import dcmri.sig as sig
import dcmri.rel as rel
import dcmri.tissue as tissue
import dcmri.utils as utils


class TissueArray(ui.ArrayModel):
    """Pixel-based vascular-interstitial tissue.

    This is the most common tissue type as found in for instance brain,
    cancer, lung, muscle, prostate, skin, and more. For more detail see
    :ref:`two-site-exchange`.

    Usage of this class is mostly identical to `dcmri.Tissue`.

    Args:
        shape (array-like): shape of the tissue array in dimensions. Any
          number of dimensions is allowed but the last must be time.
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
        parallel (bool, optional): If True, computations are parallelized.
          Defaults to False.
        verbose (int, optional): verbosity of the computation. With verbose=0,
          no feedback is given; with verbose=1, a status bar is shown.
          Defaults to 0.
        params (dict, optional): values for the parameters of the tissue,
          specified as keyword parameters. Defaults are used for any that are
          not provided. The parameters are the same as in `dcmri.Tissue`.

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

        Build a tissue array and set the constants to match the
        experimental conditions of the synthetic test data:

        >>> shape = (n,n)
        >>> tissue = dc.TissueArray(
        ...     shape,
        ...     kinetics = '2CX',
        ...     aif = aif,
        ...     dt = time[1],
        ...     r1 = dc.relaxivity(3, 'blood', 'gadodiamide'),
        ...     TR = 0.005,
        ...     FA = 15,
        ...     R10a = 1/dc.T1(3.0,'blood'),
        ...     R10 = 1/gt['T1'],
        ...     n0 = 10,
        ... )

        Train the tissue on the data:

        >>> tissue.train(time, signal)

        Plot the reconstructed maps, along with their standard deviations
        and the ground truth for reference:

        >>> tissue.plot(time, signal, ref=gt)
    """

    def __init__(self, shape,
                 kinetics='HF', water_exchange='FF', sequence='SS',
                 aif=None, ca=None, t=None, dt=1.0,
                 free=None, parallel=False, verbose=0, **params):

        # Array model params
        self.shape = shape
        self.parallel = parallel
        self.verbose = verbose

        # Define model
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
            if PARAMS[p]['pixel_par']:
                setattr(self, p, np.full(shape, PARAMS[p]['init']))
            else:
                setattr(self, p, PARAMS[p]['init'])
        self.free = {
            p: PARAMS[p]['bounds'] for p in model_pars if (
                PARAMS[p]['default_free'] and PARAMS[p]['pixel_par'])
        }

        # overide defaults
        self._set_defaults(free=free, **params)

        # sdevs
        for par in self.free:
            setattr(self, 'sdev_' + par, np.zeros(shape).astype(np.float32))

    def _params(self, p):
        return PARAMS[p]

    def _par_values(self, *args, **kwargs):
        return _par_values(self, *args, **kwargs)

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
        """
        Print detailed information about the tissue

        Example:

            List all parameters of a default tissue:

            >>> import dcmri as dc
            >>> tissue = dc.Tissue()
            >>> tissue.info()
            -------------
            Configuration
            -------------
            Kinetics: HF
            Water exchange regime: FF
            Imaging sequence: SS
            ----------
            Parameters
            ----------
            r1
            --> Full name: Contrast agent relaxivity
            --> Units: Hz/M
            --> Initial value: 5000.0
            --> Current value: 5000.0
            --> Free parameter: No
            --> Bounds: [0, inf]
            R10a
            --> Full name: Arterial precontrast R1
            --> Units: Hz
            --> Initial value: 0.7
            --> Current value: 0.7
            --> Free parameter: No
            --> Bounds: [0, inf]
            B1corr_a
            --> Full name: Arterial B1-correction factor
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No
            --> Bounds: [0, inf]
            S0
            --> Full name: Signal scaling factor
            --> Units: a.u.
            --> Initial value: 1.0
            --> Current value: 1.0
            --> Free parameter: No
            --> Bounds: [0, inf]
            B1corr
            --> Full name: Tissue B1-correction factor
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No
            --> Bounds: [0, inf]
            FA
            --> Full name: Flip angle
            --> Units: deg
            --> Initial value: 15
            --> Current value: 15
            --> Free parameter: No
            --> Bounds: [0, inf]
            TR
            --> Full name: Repetition time
            --> Units: sec
            --> Initial value: 0.005
            --> Current value: 0.005
            --> Free parameter: No
            --> Bounds: [0, inf]
            TS
            --> Full name: Sampling time
            --> Units: sec
            --> Initial value: 0
            --> Current value: 0
            --> Free parameter: No
            --> Bounds: [0, inf]
            H
            --> Full name: Tissue Hematocrit
            --> Units:
            --> Initial value: 0.45
            --> Current value: 0.45
            --> Free parameter: No
            --> Bounds: [0.001, 0.999]
            vb
            --> Full name: Blood volume
            --> Units: mL/cm3
            --> Initial value: 0.1
            --> Current value: 0.1
            --> Free parameter: Yes
            --> Bounds: [0.001, 0.999]
            vi
            --> Full name: Interstitial volume
            --> Units: mL/cm3
            --> Initial value: 0.3
            --> Current value: 0.3
            --> Free parameter: Yes
            --> Bounds: [0.001, 0.999]
            PS
            --> Full name: Permeability-surface area product
            --> Units: mL/sec/cm3
            --> Initial value: 0.003
            --> Current value: 0.003
            --> Free parameter: Yes
            --> Bounds: [0, inf]
            R10
            --> Full name: Tissue precontrast R1
            --> Units: Hz
            --> Initial value: 0.7
            --> Current value: 0.7
            --> Free parameter: No
            --> Bounds: [0, inf]
            n0
            --> Full name: Number of precontrast acquisitions
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No

        """
        info(self)

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
            signal (array-like): Array of signal curves. Any number of
              dimensions is allowed but the last dimension must be time.
            kwargs: any keyword parameters accepted by `Tissue.train`.

        Returns:
            TissueArray: A reference to the model instance.
        """
        return super().train(time, signal, **kwargs)

    # TODO: Add conc and relax functions

    def cost(self, time, signal, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element
              of *time*.
            metric (str, optional): Which metric to use.
              Possible metrics are 'RMS' (Root-mean-square); 'NRMS'
              (Normalized root-mean-square); 'AIC' (Akaike information
              criterion); 'cAIC' (Corrected Akaike information criterion for
              small models); 'BIC' (Bayesian information criterion). Defaults
              to 'NRMS'.

        Returns:
            ndarray: goodness of fit in each element of the data array.
        """
        return super().cost(time, signal, metric)

    def params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if
            only one parameter is required.

        """
        return super().params(*args, round_to=round_to)

    def export_params(self):
        pars = self._par_values(export=True)
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
        params = self._par_values(kin=True)
        if 'H' in params:
            del params['H']
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
                v0 = vmin[par] if par in vmin else np.percentile(params[par], 1)
                v1 = vmax[par] if par in vmax else np.percentile(params[par], 99)
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

        params = self._par_values(kin=True)
        if 'H' in params:
            del params['H']
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

class Tissue(ui.Model):
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
        Interstitial volume (vi): 0.2 (0.01) mL/cm3
        Permeability-surface area product (PS): 0.0 (0.0) mL/sec/cm3
        <BLANKLINE>
        ----------------------------
        Fixed and derived parameters
        ----------------------------
        <BLANKLINE>
        Tissue Hematocrit (H): 0.45
        Plasma volume (vp): 0.02 mL/cm3
        Interstitial mean transit time (Ti): 71.01 sec
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
            * - S0, FA, TR, TS, B1corr
              - Always
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

        # overide defaults
        self._set_defaults(free=free, **params)

    def _params(self):
        return PARAMS
    
    def _model_pars(self):
        return _model_pars(self.kinetics, self.water_exchange, self.sequence)

    def _par_values(self, *args, **kwargs):
        return _par_values(self, *args, **kwargs)

    def info(self):
        """
        Print detailed information about the tissue

        Example:

            List all parameters of a default tissue:

            >>> import dcmri as dc
            >>> tissue = dc.Tissue()
            >>> tissue.info()
            -------------
            Configuration
            -------------
            Kinetics: HF
            Water exchange regime: FF
            Imaging sequence: SS
            ----------
            Parameters
            ----------
            r1
            --> Full name: Contrast agent relaxivity
            --> Units: Hz/M
            --> Initial value: 5000.0
            --> Current value: 5000.0
            --> Free parameter: No
            --> Bounds: [0, inf]
            R10a
            --> Full name: Arterial precontrast R1
            --> Units: Hz
            --> Initial value: 0.7
            --> Current value: 0.7
            --> Free parameter: No
            --> Bounds: [0, inf]
            B1corr_a
            --> Full name: Arterial B1-correction factor
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No
            --> Bounds: [0, inf]
            S0
            --> Full name: Signal scaling factor
            --> Units: a.u.
            --> Initial value: 1.0
            --> Current value: 1.0
            --> Free parameter: No
            --> Bounds: [0, inf]
            B1corr
            --> Full name: Tissue B1-correction factor
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No
            --> Bounds: [0, inf]
            FA
            --> Full name: Flip angle
            --> Units: deg
            --> Initial value: 15
            --> Current value: 15
            --> Free parameter: No
            --> Bounds: [0, inf]
            TR
            --> Full name: Repetition time
            --> Units: sec
            --> Initial value: 0.005
            --> Current value: 0.005
            --> Free parameter: No
            --> Bounds: [0, inf]
            TS
            --> Full name: Sampling time
            --> Units: sec
            --> Initial value: 0
            --> Current value: 0
            --> Free parameter: No
            --> Bounds: [0, inf]
            H
            --> Full name: Tissue Hematocrit
            --> Units:
            --> Initial value: 0.45
            --> Current value: 0.45
            --> Free parameter: No
            --> Bounds: [0.001, 0.999]
            vb
            --> Full name: Blood volume
            --> Units: mL/cm3
            --> Initial value: 0.1
            --> Current value: 0.1
            --> Free parameter: Yes
            --> Bounds: [0.001, 0.999]
            vi
            --> Full name: Interstitial volume
            --> Units: mL/cm3
            --> Initial value: 0.3
            --> Current value: 0.3
            --> Free parameter: Yes
            --> Bounds: [0.001, 0.999]
            PS
            --> Full name: Permeability-surface area product
            --> Units: mL/sec/cm3
            --> Initial value: 0.003
            --> Current value: 0.003
            --> Free parameter: Yes
            --> Bounds: [0, inf]
            R10
            --> Full name: Tissue precontrast R1
            --> Units: Hz
            --> Initial value: 0.7
            --> Current value: 0.7
            --> Free parameter: No
            --> Bounds: [0, inf]
            n0
            --> Full name: Number of precontrast acquisitions
            --> Units:
            --> Initial value: 1
            --> Current value: 1
            --> Free parameter: No

        """
        info(self)

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
                    "Either aif or ca must be provided \
                    to predict signal data.")
            else:
                if self.sequence == 'SR':
                    self.ca = sig.conc_src(
                        self.aif, self.TC, 1 / self.R10a, self.r1, self.n0)
                elif self.sequence == 'SS':
                    self.ca = sig.conc_ss(
                        self.aif, self.TR, self.B1corr_a * self.FA,
                        1 / self.R10a, self.r1, self.n0)

    def conc(self, sum=True):
        """Return the tissue concentration

        Args:
            sum (bool, optional): If True, returns the total concentrations.
              Else returns the concentration in the individual compartments.
              Defaults to True.

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
        self._check_ca()
        pars = self._par_values(kin=True)
        return tissue.conc_tissue(
            self.ca, t=self.t, dt=self.dt, sum=sum, kinetics=self.kinetics,
            **pars)

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
            - **PSw** (numpy.ndarray or None): 2D array with water exchange
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
            >>> R1, v, PSw = tissue.relax()

            >>> v
            array([0.1, 0.3, 0.6])

            >>> PSw
            array([[0.  , 0.03, 0.  ],
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
        self._check_ca()
        pars = self._par_values(tiss=True)
        R1, v, PSw = tissue.relax_tissue(
            self.ca, self.R10, self.r1, t=self.t, dt=self.dt,
            kinetics=self.kinetics, water_exchange=self.water_exchange,
            **pars)
        return R1, v, PSw

    def signal(self, sum=True) -> np.ndarray:
        """Pseudocontinuous signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        self._check_ca()  # TODO do not precompute
        tpars = self._par_values(tiss=True)
        spars = self._par_values(seq=True)
        spars['model'] = self.sequence
        return tissue.signal_tissue(
            self.ca, self.R10, self.r1, t=self.t, dt=self.dt,
            kinetics=self.kinetics,
            water_exchange=self.water_exchange,
            sequence=spars,
            # inflow = {
            #     'R10a': self.R10a,
            #     'B1corr_a': self.B1corr_a,
            # },
            sum=sum, **tpars)

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
            raise ValueError(
                "The acquisition window is longer than the duration "
                "of the AIF. The largest time point that can be "
                "predicted is " + str(np.amax(t) / 60) + "min.")
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
                                 self.B1corr * self.FA, self.TC, self.TP)
        elif self.sequence == 'SS':
            Sref = sig.signal_ss(self.R10, 1, self.TR, self.B1corr * self.FA)
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
            return ui.train(self, time, signal, **kwargs)

        if method == 'PSMS':  # Work in progress
            # Fit the complete model
            ui.train(self, time, signal, **kwargs)
            # Fit an intravascular model with the same free parameters
            iv = deepcopy(self)
            # iv.kinetics = 'NX'
            setattr(iv, 'PS', 0)
            for par in ['ve', 'PS', 'PSc']:
                if par in iv.free:
                    iv.free.pop(par)
            ui.train(iv, time, signal, **kwargs)
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

        Example:

            Train a tissue on synthetic data and print the compartment
            volumes:

            >>> import dcmri as dc
            >>> t, aif, roi, _ = dc.fake_tissue()
            >>> tissue = dc.Tissue('HF','RR', t=t, aif=aif).train(t, roi)
            >>> tissue.params('vp', 'vi', round_to=2)
            [np.float64(0.03), np.float64(0.24)]
        """
        return super().params(*args, round_to=round_to)

    def export_params(self):
        """Return model parameters with their descriptions

        Returns:
            dict: Dictionary with one item for each model parameter. The key
            is the parameter symbol (short name), and the value is a
            4-element list with [parameter name, value, unit, sdev].

        Example:

            Train a tissue on synthetic data and print the blood volume after
            training:

            >>> import dcmri as dc
            >>> t, aif, roi, _ = dc.fake_tissue()
            >>> tissue = dc.Tissue('HF','RR', t=t, aif=aif).train(t, roi)
            >>> pars = tissue.export_params()
            >>> pars['vb']
            ['Blood volume', np.float64(0.05735355675475683), 'mL/cm3', np.float64(0.016039245654090793)]

        """
        return super().export_params()

    def print_params(self, round_to=None):
        """Print the model parameters and their uncertainties

        Args:
            round_to (int, optional): Round to how many digits. If this is
              not provided, the values are not rounded. Defaults to None.

        Example:

            Train a tissue on synthetic data and print the parameters:

            >>> import dcmri as dc
            >>> t, aif, roi, _ = dc.fake_tissue()
            >>> tissue = dc.Tissue('HF','RR', t=t, aif=aif).train(t, roi)
            >>> tissue.print_params(round_to=2)
            <BLANKLINE>
            --------------------------------
            Free parameters with their stdev
            --------------------------------
            <BLANKLINE>
            Transendothelial water PS (PSe): 0.0 (0.22) mL/sec/cm3
            Transcytolemmal water PS (PSc): 0.0 (1.39) mL/sec/cm3
            Blood volume (vb): 0.06 (0.02) mL/cm3
            Interstitial volume (vi): 0.24 (0.07) mL/cm3
            Permeability-surface area product (PS): 0.0 (0.0) mL/sec/cm3
            <BLANKLINE>
            ----------------------------
            Fixed and derived parameters
            ----------------------------
            <BLANKLINE>
            Tissue Hematocrit (H): 0.45
            Plasma volume (vp): 0.03 mL/cm3
            Interstitial mean transit time (Ti): 83.27 sec
            Intracellular water mean transit time (Twc): 1.3232576345772858e+16 sec
            Interstitial water mean transit time (Twi): 186652627701867.1 sec
            Intravascular water mean transit time (Twb): 47283940807705.62 sec
            B1-corrected Flip Angle (FAcorr): 15 deg
        """
        super().print_params(round_to=round_to)

    def cost(self, time, signal, metric='NRMS') -> float:
        """Return the goodness-of-fit

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element
              of *time*.
            metric (str, optional): Which metric to use.
              Possible metrics are 'RMS' (Root-mean-square); 'NRMS'
              (Normalized root-mean-square); 'AIC' (Akaike information
              criterion); 'cAIC' (Corrected Akaike information criterion for
              small models); 'BIC' (Bayesian information criterion). Defaults
              to 'NRMS'.

        Returns:
            float: goodness of fit.

        Example:

            Generate a fake dataset, build two tissues and use the Akaike
            Information Criterion to decide which configuration is most
            consistent with the data.

            >>> import dcmri as dc
            >>> t, aif, roi, _ = dc.fake_tissue()

            >>> tissue1 = dc.Tissue('2CX','FF', t=t, aif=aif).train(t, roi)
            >>> tissue2 = dc.Tissue('HFU','RR', t=t, aif=aif).train(t, roi)

            >>> tissue1.cost(t, roi, 'AIC')
            np.float64(-967.4477371121604)

            >>> tissue2.cost(t, roi, 'AIC')
            np.float64(-375.7766916566553)

            tissue1 achieves the lowest cost and is therefore the optimal
            configuration according to the Akaike Information Criterion.
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
        if C.ndim == 1:
            C = C.reshape((1, -1))
        v, comps = _plot_labels_kin(self.kinetics)
        pars = self._par_values()
        ax01.set_title('Concentration in indicator compartments')
        if ref is not None:
            # ax01.plot(ref['t']/60, 1000*ref['C'], marker='o',
            # linestyle='None', color='lightgray', label='Tissue ground truth')
            ax01.plot(ref['t'] / 60, 1000 * ref['cb'], marker='o',
                      linestyle='None',
                      color='lightcoral', label='Arterial ground truth')
        ax01.plot(t / 60, 1000 * self.ca, linestyle='-', linewidth=5.0,
                  color='lightcoral', label='Arterial blood')
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
            c = rel.c_lin(R1, self.r1)
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
    if comp == 'Tissue blood':
        return 'darkred'
    if comp == 'Extravascular':
        return 'blue'
    if comp == 'Tissue cells':
        return 'lightblue'
    if comp == 'Blood + Interstitium':
        return 'purple'


def _plot_labels_kin(kin):

    if kin == '2CX':
        return ['vb', 'vi'], ['Blood', 'Interstitium']
    if kin == '2CU':
        return ['vb', 'vi'], ['Blood', 'Interstitium']
    if kin == 'HF':
        return ['vb', 'vi'], ['Blood', 'Interstitium']
    if kin == 'HFU':
        return ['vb', 'vi'], ['Blood', 'Interstitium']
    if kin == 'FX':
        return ['ve'], ['Extracellular']
    if kin == 'NX':
        return ['vb'], ['Blood']
    if kin == 'U':
        return ['vb'], ['Blood']
    if kin == 'WV':
        return ['vi'], ['Interstitium']


def _plot_labels_relax(kin, wex) -> list:

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


def info(self):

    print('-------------')
    print('Configuration')
    print('-------------')
    print('Kinetics: ' + self.kinetics)
    print('Water exchange regime: ' + self.water_exchange)
    print('Imaging sequence: ' + self.sequence)
    print('----------')
    print('Parameters')
    print('----------')

    for p in _model_pars(self.kinetics,
                         self.water_exchange, self.sequence):
        free = 'Yes' if p in self.free else 'No'
        print(p)
        print('--> Full name: ' + PARAMS[p]['name'])
        print('--> Units: ' + PARAMS[p]['unit'])
        print('--> Initial value: ' + str(PARAMS[p]['init']))
        print('--> Current value: ' + str(getattr(self, p)))
        print('--> Free parameter: ' + free)
        if PARAMS[p]['bounds'] is not None:
            lb = str(PARAMS[p]['bounds'][0])
            ub = str(PARAMS[p]['bounds'][1])
            print('--> Bounds: [' + lb + ', ' + ub + ']')


def _par_values(self, *args, tiss=False, kin=False, seq=False,
                export=False):

    pars = _all_pars(
        self.kinetics, self.water_exchange, self.sequence, self.__dict__)
    if args != ():
        return {p: pars[p] for p in args}
    if tiss:
        return {p: pars[p] for p in
                tissue.params_tissue(self.kinetics, self.water_exchange)}
    if kin:
        return {p: pars[p] for p in 
                tissue.params_tissue(self.kinetics, 'FF')}
    if seq:
        return {p: pars[p] for p in _seq_pars(self.sequence)}
    if export:
        p0 = _model_pars(self.kinetics, self.water_exchange, self.sequence)
        p1 = tissue.params_tissue(self.kinetics, self.water_exchange)
        discard = set(p0) - set(p1) - set(self.free.keys())
        return {p: pars[p] for p in pars if p not in discard}

    return pars


# CONFIGURATIONS


def _model_pars(kin, wex, seq):
    pars = ['r1']
    pars += ['R10a', 'B1corr_a']
    pars += _seq_pars(seq)
    pars += ['TS']
    pars += tissue.params_tissue(kin, wex)
    pars += ['R10', 'n0']
    return pars

def _seq_pars(seq):
    if seq == 'SS':
        return ['S0', 'B1corr', 'FA', 'TR']
    elif seq == 'SR':
        return ['S0', 'B1corr', 'FA', 'TR', 'TC', 'TP']



PARAMS = {
    'r1': {
        'init': 5000.0,
        'default_free': False,
        'bounds': [0, np.inf],
        'name': 'Contrast agent relaxivity',
        'unit': 'Hz/M',
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
    'Fb': {
        'init': 0.02,
        'default_free': True,
        'bounds': [0, np.inf],
        'name': 'Blood flow',
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
    'vi': {
        'init': 0.3,
        'default_free': True,
        'bounds': [1e-3, 1 - 1e-3],
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
        'init': 0.1 * (1 - 0.45) + 0.3,
        'default_free': True,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Extracellular volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'vb': {
        'init': 0.1,
        'default_free': True,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Blood volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'vc': {
        'init': 0.6,
        'default_free': True,
        'bounds': [1e-3, 1 - 1e-3],
        'name': 'Intracellular volume',
        'unit': 'mL/cm3',
        'pixel_par': True,
    },
    'H': {
        'init': 0.45,
        'default_free': False,
        'bounds': [1e-3, 1 - 1e-3],
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
        'init': 0,
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
    'vp': {
        'name': 'Plasma volume',
        'unit': 'mL/cm3',
    },
    'Fp': {
        'name': 'Plasma flow',
        'unit': 'mL/cm3',
    },
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

    pars = _model_pars(kin, wex, seq)
    p = {par: p[par] for par in pars}

    try:
        p['Fp'] = p['Fb'] * (1 - p['H'])
    except KeyError:
        pass
    try:
        p['vp'] = p['vb'] * (1 - p['H'])
    except KeyError:
        pass
    try:
        p['Ktrans'] = _div(p['Fp'] * p['PS'], p['Fp'] + p['PS'])
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
        p['FAcorr'] = p['B1corr'] * p['FA']
    except KeyError:
        pass

    return p


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b == 0, 0, np.divide(a, b))
