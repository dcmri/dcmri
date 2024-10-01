
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np

import dcmri.mods as mods
import dcmri.sig as sig
import dcmri.lib as lib
import dcmri.tissue as tissue
import dcmri.utils as utils

try: 
    num_workers = int(len(os.sched_getaffinity(0)))
except: 
    num_workers = int(os.cpu_count())



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
        - **field_strength** (float, default=3.0): Magnetic field strength in T.
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses, in sec.
        - **FA** (array, default=np.full(shape, 15)): Nominal flip angle in degrees.
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence.
        - **TP** (float, default=0): Preparation delay in a saturation-recovery sequence.
        - **TS** (float, default=None): Sampling duration, or duration of the readout for a single time point. If not provided, and sampling is uniform, TS is assumed to be equal to the uniform time interval. TS is required if sampling is non-uniform.

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
        - **R10b** (float, default=1): Precontrast arterial relaxation rate in 1/sec. 
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
        ...     agent = 'gadodiamide',
        ...     TR = 0.005,
        ...     FA = np.full(shape, 15),
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

    def __init__(self, shape, kinetics='HF', water_exchange='FF', sequence='SS', **params):

        # Array model params
        self.shape = shape
        self.parallel = False
        self.verbose = 0

        # Define model
        self.kinetics = kinetics
        self.water_exchange = water_exchange
        self.sequence = sequence
        _check_inputs(self)

        #
        # Set defaults for all parameters
        # 

        # Acquisition parameters
        self.field_strength = 3.0
        self.agent = 'gadoterate'

        # Input function
        self.aif = None
        self.ca = None
        self.t = None
        self.dt = 1.0
        self.Hcta = 0.45
        self.R10b = 0.7 # TODO: b to a
        self.FAa = 15

        # Model parameters
        model_pars = _tissue_pars(kinetics, water_exchange)
        for p in model_pars:
            setattr(self, p, np.full(shape, _init(p)))
        self.free = {p:_bound(p) for p in model_pars}
        self.R10 = np.full(shape, 0.7)

        # Sequence parameters
        self.FA = np.full(shape, 15)
        self.TR = 0.005
        self.TC = 0.2
        self.TP = 0.05
        self.TS = None
        
        # Tissue properties
        self.S0 = np.full(shape, 1.0)

        # training parameters
        self.n0 = 1

        # overide defaults
        self._set_defaults(**params)

        # sdevs
        for par in self.free:
            setattr(self, 'sdev_' + par,  np.zeros(shape).astype(np.float32))

        _init_tissue(self)



    def _pix(self, p):

        pars = _tissue_pars(self.kinetics, self.water_exchange)
        pars = {par:getattr(self, par)[p] for par in pars}

        return Tissue(

            kinetics = self.kinetics,
            water_exchange = self.water_exchange,
            sequence = self.sequence,

            # Acquisition parameters
            field_strength = self.field_strength,
            agent = self.agent,

            # Input function
            aif = self.aif,
            ca = self.ca,
            t = self.t,
            dt = self.dt,
            Hcta = self.Hcta,
            R10b = self.R10b,
            FAa = self.FAa,
            
            # Model parameters
            free = self.free,
            R10 = self.R10[p],

            # Sequence parameters
            FA = self.FA[p],
            TR = self.TR,
            TC = self.TC,
            TP = self.TP,
            TS = self.TS,

            # Tissue properties
            S0 = self.S0[p],

            # training parameters
            n0 = self.n0,
            
            # Model parameters
            **pars,
        )


    def info(self):
        """Print information about the tisue"""
        info = '\n'
        info += 'Water exchange regime: ' + self.water_exchange + '\n'
        info += 'Kinetics: ' + self.kinetics + '\n\n'
        info += 'Model parameters\n'
        info += '----------------\n'
        for p in _tissue_pars(self.kinetics, self.water_exchange):
            info += 'Parameter: ' + str(p) + '\n'
            info += '--> Definition: ' + _name(p) + '\n'
            info += '--> Units: ' + _unit(p) + '\n'
        print(info)
    

    def _train_curve(self, args):
        pix, p = super()._train_curve(args)
        self.S0[p] = pix.S0 
        return pix, p

    
    def predict(self, time:np.ndarray)->np.ndarray:
        """Predict the data at given time points

        Args:
            time (array-like): 1D array with time points.

        Returns:
            ndarray: Array of predicted y-values.
        """
        return super().predict(time)


    def train(self, time:np.ndarray, signal:np.ndarray, **kwargs):
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

    def cost(self, time, signal, metric='NRMS')->float:
        """Return the goodness-of-fit

        Args:
            time (array-like): 1D array with time points.
            signal (array-like): Array of signal curves. Any number of dimensions is allowed but the last dimension must be time. 
            metric (str, optional): Which metric to use - options are 'RMS' (Root-mean-square), 'NRMS' (Normalized root-mean-square), 'AIC' (Akaike information criterion), 'cAIC' (Corrected Akaike information criterion for small models) or 'BIC' (Bayesian information criterion). Defaults to 'NRMS'.

        Returns:
            ndarray: goodness of fit in each element of the data array. 
        """
        return super().cost(time, signal, metric)
    
    def get_params(self, *args):
        """Return the parameter values

        Args:
            args (tuple): parameters to get

        Returns:
            list or float: values of parameter values, or a scalar value if only one parameter is required.
        """
        p = _all_pars(self.kinetics, self.water_exchange, self.__dict__)
        if args == ():
            args = p.keys()
        pars = []
        for a in args:
            if a in p:
                pars.append(p[a])
            else:
                pars.append(getattr(self, a))
        if len(pars)==1:
            return pars[0]
        else:
            return pars
    
    def export_params(self):
        pars = _all_pars(self.kinetics, self.water_exchange, self.__dict__)
        pars = {p: [_name(p), pars[p], _unit(p)] for p in pars}
        return self._add_sdev(pars)


    def plot(self, time, signal, vmin={}, vmax={}, cmap='gray', ref=None, fname=None, show=True):
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
        if len(self.shape)==1:
            raise NotImplementedError('Cannot plot 1D images.')
        yfit = self.predict(time)
        params = _std_conc_pars(self.kinetics, self.__dict__)
        params['S0'] = self.S0

        if len(self.shape)==2:
            ncols = 2+len(params)
            nrows = 2 if ref is None else 3
            fig = plt.figure(figsize=(ncols*2, nrows*2))
            figcols = fig.subfigures(1, 2, wspace=0.0, hspace=0.0, width_ratios=[2,ncols-2])

            # Left panel: signal
            ax = figcols[0].subplots(nrows,2)
            figcols[0].subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                for j in range(2):
                    ax[i,j].set_yticks([])
                    ax[i,j].set_xticks([])           

            # Signal maps
            ax[0,0].set_title('max(signal)')
            ax[0,0].set_ylabel('reconstruction')
            ax[0,0].imshow(np.amax(yfit, axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)
            ax[1,0].set_ylabel('data')
            ax[1,0].imshow(np.amax(signal, axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)
            if ref is not None:
                ax[2,0].set_ylabel('ground truth')
                ax[2,0].imshow(np.amax(ref['signal'], axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)
            ax[0,1].set_title('mean(signal)')
            ax[0,1].imshow(np.mean(yfit, axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)
            ax[1,1].imshow(np.mean(signal, axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)
            if ref is not None:
                ax[2,1].imshow(np.mean(ref['signal'], axis=-1), vmin=0, vmax=0.5*np.amax(signal), cmap=cmap)

            # Right panel: free parameters
            ax = figcols[1].subplots(nrows, ncols-2)
            figcols[1].subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                for j in range(ncols-2):
                    ax[i,j].set_yticks([])
                    ax[i,j].set_xticks([]) 
            ax[0,0].set_ylabel('reconstruction')
            ax[1,0].set_ylabel('std devs')
            if ref is not None:
                ax[2,0].set_ylabel('ground truth')
            for i, par in enumerate(params.keys()):
                v0 = vmin[par] if par in vmin else None
                v1 = vmax[par] if par in vmax else None
                ax[0,i].set_title(par)
                ax[0,i].imshow(params[par], vmin=v0, vmax=v1, cmap=cmap)
                if hasattr(self, 'sdev_' + par):
                    ax[1,i].imshow(getattr(self, 'sdev_' + par), vmin=v0, vmax=v1, cmap=cmap)  
                else:
                    ax[1,i].imshow(np.zeros(self.shape).astype(np.int16), cmap=cmap)
                if ref is not None:
                    ax[2,i].imshow(ref[par], vmin=v0, vmax=v1, cmap=cmap) 
        if len(self.shape)==3:         
            raise NotImplementedError('3D plot not yet implemented')
        
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()
        

    def plot_signals(self, time, signal, cmap='gray', ref=None, fname=None, show=True):
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
        if len(self.shape)==1:
            raise NotImplementedError('Cannot plot 1D images.')
        yfit = self.predict(time)

        if len(self.shape)==2:
            ncols = 1
            nrows = 2 if ref is None else 3
            #fig = plt.figure(figsize=(ncols*2, nrows*2), layout='constrained')
            fig = plt.figure(figsize=(ncols*2, nrows*2))

            # Left panel: signal
            #figcols[0].suptitle('Signal', fontsize='x-large')
            ax = fig.subplots(nrows,ncols)
            #figcols[0].subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.00, wspace=0)
            fig.subplots_adjust(hspace=0.0, wspace=0)
            for i in range(nrows):
                ax[i].set_yticks([])
                ax[i].set_xticks([])  
            # data animation
            ax[0].set_title('signal(time)')
            ax[0].set_ylabel('data')
            im = ax[0].imshow(signal[:,:,0], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal))
            ims = []
            for i in range(signal.shape[-1]):
                im = ax[0].imshow(signal[:,:,i], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal)) 
                ims.append([im]) 
            anim_data = ArtistAnimation(fig, ims, interval=50)
            # fit animation
            #ax[1,0].set_title('model fit', rotation='vertical', x=-0.1,y=0.5)
            ax[1].set_ylabel('model fit')
            im = ax[1].imshow(yfit[:,:,0], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal))
            ims = []
            for i in range(yfit.shape[-1]):
                im = ax[1].imshow(yfit[:,:,i], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal)) 
                ims.append([im]) 
            anim_fit = ArtistAnimation(fig, ims, interval=50)
            # truth animation
            if ref is not None:
                ax[2].set_ylabel('ground truth')
                im = ax[2].imshow(ref['signal'][:,:,0], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal))
                ims = []
                for i in range(ref['signal'].shape[-1]):
                    im = ax[2].imshow(ref['signal'][:,:,i], cmap=cmap, animated=True, vmin=0, vmax=0.5*np.amax(signal)) 
                    ims.append([im]) 
                anim_truth = ArtistAnimation(fig, ims, interval=50)               

        if len(self.shape)==3:         
            raise NotImplementedError('3D plot not yet implemented')
        
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


    def plot_params(self, roi=None, vmin={}, vmax={}, ref=None, fname=None, show=True):
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
        params = _std_conc_pars(self.kinetics, self.__dict__)
        params['S0'] = self.S0

        ncols = len(params)
        if roi is None:
            nrows = 1
            fig, ax = plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2, right=0.8, top=0.9, bottom=0.1)
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
                    hrange = [v0,v1]
                else:
                    hrange = [-1,1] if v0==0 else [0.9*v0, 1.1*v0]

                if ref is not None:
                    ax[i].hist(ref[par], range=[vmin[par], vmax[par]], label='Truth')
                ax[i].hist(params[par], range=[vmin[par], vmax[par]], label='Reconstructed')
            ax[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        else:
            nrows = len(roi)
            fig, ax = plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2, right=0.8, top=0.9, bottom=0.1)
            i=0
            for name, mask in roi.items():
                ax[i,0].set_ylabel(name, fontsize=8, rotation='horizontal', labelpad=30)
                for p, par in enumerate(params):
                    ax[i,p].set_yticks([])
                    ax[i,p].set_xticks([])  
                    if i==0:
                        ax[i,p].set_title(par, fontsize=8)

                    data = params[par][mask==1] 
                    if data.size == 0:
                        continue
                    if ref is not None:
                        data = np.concatenate((data, ref[par][mask==1]), axis=None)
                    v0 = vmin[par] if par in vmin else np.amin(data) 
                    v1 = vmax[par] if par in vmax else np.amax(data) 
                    if v0 != v1:
                        hrange = [v0,v1]
                    else:
                        hrange = [-1,1] if v0==0 else [0.9*v0, 1.1*v0]

                    if ref is not None:
                        ax[i,p].hist(ref[par][mask==1], range=hrange, label='Truth')
                    ax[i,p].hist(params[par][mask==1], range=hrange, label='Reconstructed')
                i+=1
            ax[0,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)


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
        signal = signal.reshape((-1,nt))
        yfit = self.predict(time).reshape((-1,nt))
        #rms = 100*np.linalg.norm(signal-yfit, axis=-1)/np.linalg.norm(signal, axis=-1)
        rms = np.linalg.norm(signal-yfit, axis=-1)
        cols = ['fit error histogram', '5th perc', '25th perc', 'median', '75th perc', '95th perc']
        if roi is None:
            nrows = 1
            ncols = 6
            fig, ax = plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2, right=0.8, top=0.9, bottom=0.1)
            for r in range(nrows):
                for c in range(ncols):
                    ax[r,c].set_xticks([]) 
                    ax[r,c].set_yticks([]) 
            for c in range(ncols):
                 ax[0,c].set_title(cols[c], fontsize=8)
            _plot_roi(time, signal, yfit, ref, hist_kwargs, rms, ax, 'all pixels')
        else: 
            nrows = len(roi)
            ncols = 6
            fig, ax = plt.subplots(nrows,ncols,figsize=(2*ncols,2*nrows))
            fig.subplots_adjust(hspace=0.0, wspace=0, left=0.2, right=0.8, top=0.9, bottom=0.1)
            for r in range(nrows):
                for c in range(ncols):
                    ax[r,c].set_xticks([]) 
                    ax[r,c].set_yticks([])
            for c in range(ncols):
                 ax[0,c].set_title(cols[c], fontsize=8)
            i=0
            for name, mask in roi.items():
                _plot_roi(time, signal, yfit, ref, hist_kwargs, rms, ax[i,:], name, mask=mask.ravel())
                i+=1
        legend = ax[0,-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        labels = ['Truth','Prediction','Data']
        for i, label in enumerate(legend.get_texts()):
            label.set_text(labels[i])

        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()





#TODO: iex and wex instead of kinetics and water_exchange? Water exchange is also kinetics.

class Tissue(mods.Model):
    """Model for general vascular-interstitial tissues.

    The model accepts the following parameters:

        **Input function**

        - **aif** (array-like, default=None): Signal-time curve in a feeding artery (arbitrary units). If AIF is set to None, then the parameter ca must be provided (arterial concentrations).
        - **ca** (array-like, default=None): Plasma concentration in the arterial input (M). Must be provided when aif = None, ignored otherwise. 

        **Acquisition parameters**

        - **t** (array-like, default=None): Time points of the aif (sec). If t is not provided, the temporal sampling is uniform with interval dt.
        - **dt** (float, default=1.0): Time interval of the AIF (sec).
        - **field_strength** (float, default=3.0): Magnetic field strength (T).
        - **agent** (str, default='gadoterate'): Contrast agent generic name.
        - **TR** (float, default=0.005): Repetition time, or time between excitation pulses (sec).
        - **FA** (float, default=15): Nominal flip angle (degrees).
        - **TC** (float, default=0.1): Time to the center of k-space in a saturation-recovery sequence (sec).
        - **TP** (float, default=0): Preparation delay in a saturation-recovery sequence (sec).
        - **TS** (float, default=None): Sampling duration, or duration of the readout for a single time point (sec). If not provided, and sampling is uniform, TS is assumed to be equal to the uniform time interval. TS is required if sampling is non-uniform.

        **Tracer-kinetic parameters**

        - **kinetics** (str, default='HF'): Tracer-kinetic model (see below for options)
        - **Hct** (float, default=0.45): Hematocrit.
        - **Fp** (float, default=0.01): Plasma flow, or flow of plasma into the plasma compartment (mL/sec/cm3).
        - **Ktrans** (float, default=0.003): Volume transfer constant: volume of arterial plasma cleared of indicator per unit time and per unit tissue (mL/sec/cm3).
        - **vp** (float, default=0.1): Plasma volume, or volume fraction of the plasma compartment (mL/cm3). 
        - **vi** (float, default=0.5): Interstitial volume: volume fraction of the interstitial compartment (mL/cm3).
        - **Ktrans** (float, default=0.0023): Volume transfer constant (mL/sec/cm3).
        - **ve** (float, default=0.6): Extracellular volume fraction (mL/cm3).

        **Water-kinetic parameters**

        - **water_exchange** (str, default='fast'): Water exchange regime ('fast', 'none' or 'any').
        - **PSe** (float, default=10): Transendothelial water permeability-surface area product: PS for water across the endothelium (mL/sec/cm3).
        - **PSc** (float, default=10): Transcytolemmal water permeability-surface area product: PS for water across the cell wall (mL/sec/cm3).

        **Signal parameters**

        - **sequence** (str, default='SS'): imaging sequence.
        - **R10** (float, default=0.7): Precontrast tissue relaxation rate (1/sec).
        - **R10b** (float, default=0.7): Precontrast blood relaxation rate (1/sec).
        - **S0** (float, default=1): Scale factor for the MR signal (arbitrary units).

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
        `Liver`, `Kidney`

    Example:

        Single time-curve analysis: fit extended Tofts model to data.

    .. plot::
        :include-source:
        :context: close-figs
    
        >>> import dcmri as dc

        Use `fake_tissue` to generate synthetic test data:

        >>> time, aif, roi, gt = dc.fake_tissue(CNR=50)
        
        Build a tissue model and set the constants to match the experimental conditions of the synthetic test data:

        >>> model = dc.Tissue(
        ...     aif = aif,
        ...     dt = time[1],
        ...     agent = 'gadodiamide',
        ...     TR = 0.005,
        ...     FA = 15,
        ...     n0 = 15,
        ... )

        Train the model on the ROI data:

        >>> model.train(time, roi)

        Plot the reconstructed signals (left) and concentrations (right) and compare the concentrations against the noise-free ground truth:

        >>> model.plot(time, roi, ref=gt)

    """ 

    def __init__(self, kinetics='HF', water_exchange='FF', sequence='SS', **params):

        # Set configuration
        self.kinetics = kinetics
        self.water_exchange = water_exchange
        self.sequence = sequence
        self.agent = 'gadoterate'
        _check_inputs(self)

        # Acquisition parameters
        self.field_strength = 3.0

        # Input function
        self.aif = None
        self.ca = None
        self.t = None
        self.dt = 1.0
        self.Hcta = 0.45
        self.R10b = 0.7
        self.FAa = 15

        # Model parameters
        model_pars = _tissue_pars(kinetics, water_exchange)
        for p in model_pars:
            setattr(self, p, _init(p))
        self.free = {p:_bound(p) for p in model_pars}
        self.R10 = 0.7
        
        # Sequence parameters
        self.FA = 15
        self.TR = 0.005
        self.TC = 0.2
        self.TP = 0.05
        self.TS = None # initialized as dt if not provided

        # Tissue properties
        self.S0 = 1.0
        
        # Training parameters
        self.n0 = 1
        
        # overide defaults
        self._set_defaults(**params)
        
        _init_tissue(self)


    def info(self):
        """Print information about the tissue"""
        info = '\n'
        info += 'Water exchange regime: ' + self.water_exchange + '\n'
        info += 'Kinetics: ' + self.kinetics + '\n\n'
        info += 'Model parameters\n'
        info += '----------------\n'
        for p in _tissue_pars(self.kinetics, self.water_exchange):
            info += 'Parameter: ' + str(p) + '\n'
            info += '--> Definition: ' + _name(p) + '\n'
            info += '--> Units: ' + _unit(p) + '\n'
        print(info)


    def time(self):
        """Return an array of time points

        Returns:
            np.ndarray: time points in seconds.
        """
        if self.t is None:
            if self.aif is not None:
                return self.dt*np.arange(np.size(self.aif))
            elif self.ca is not None:
                return self.dt*np.arange(np.size(self.ca))
            else:
                raise ValueError('Either aif or ca must be provided.')
        else:
            return self.t
        
    def _check_ca(self):
        
        # Arterial concentrations
        if self.ca is None:
            if self.aif is None:
                raise ValueError('Either aif or ca must be provided to predict signal data.')
            else:
                r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
                if self.sequence == 'SR':
                    cb = sig.conc_src(self.aif, self.TC, 1/self.R10b, r1, self.n0)
                elif self.sequence == 'SS':
                    cb = sig.conc_ss(self.aif, self.TR, self.FAa, 1/self.R10b, r1, self.n0) 
                self.ca = cb/(1-self.Hcta) 

    def conc(self, sum=True):
        """Return the tissue concentration

        Args:
            sum (bool, optional): If True, returns the total concentrations. Else returns the concentration in the individual compartments. Defaults to True.

        Returns:
            np.ndarray: Concentration in M
        """
        self._check_ca()
        pars =  _std_conc_pars(self.kinetics, self.__dict__) 
        return tissue.conc_tissue(self.ca, t=self.t, dt=self.dt, sum=sum, kinetics=self.kinetics, **pars)   

    def relax(self):
        """Return the tissue relaxation rate(s)

        Returns:
            np.ndarray: the free relaxation rate of all tissue compartments. In the fast water exchange limit, the relaxation rates are a 1D array. In all other situations, relaxation rates are a 2D-array with dimensions (k,n), where k is the number of compartments and n is the number of time points in ca. 
        """
        # TODO: ADD diagonal element to PSw (flow term)!!
        # Fb = self.Fp/(1-self.Hct)
        # PSw = np.array([[Fb,self.PSe,0],[self.PSe,0,self.PSc],[0,self.PSc,0]])
        self._check_ca()
        pars = _std_relax_pars(self.kinetics, self.water_exchange, self.__dict__)
        r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
        # TODO in FF v should be a 1-element list (and signal needs to adjust accordingly)
        R1,v = tissue.relax_tissue(self.ca, self.R10, r1, t=self.t, dt=self.dt, kinetics=self.kinetics, water_exchange=self.water_exchange.replace('N','R'), **pars)
        PSw = _PSw(self.kinetics, self.water_exchange, self.__dict__)
        return R1, PSw, v


    def signal(self, sum=True)->np.ndarray:
        """Return the signal

        Returns:
            np.ndarray: the signal as a 1D array.
        """
        R1, PSw, v = self.relax()
        if self.sequence == 'SR':
            if not sum:
                raise ValueError('Separate signals for signal model SR are not yet implemented.')
            return sig.signal_sr(R1, self.S0, self.TR, self.FA, self.TC, self.TP, v=v, PSw=PSw)
        elif self.sequence == 'SS':
            return sig.signal_ss(R1, self.S0, self.TR, self.FA, v=v, PSw=PSw, sum=sum)


    # TODO: make time optional (if not provided, assume equal to self.time())
    def predict(self, time:np.ndarray)->np.ndarray:
        """Predict the data at given time points

        Args:
            time (array-like): Array of time points.

        Returns:
            np.ndarray: Array of predicted data for each element of *time*.
        """
        t = self.time()
        if np.amax(time) > np.amax(t):
            msg = 'The acquisition window is longer than the duration of the AIF. \n'
            msg += 'The largest time point that can be predicted is ' + str(np.amax(t)/60) + 'min.'
            raise ValueError(msg)
        sig = self.signal()
        return utils.sample(time, t, sig, self.TS)

   
    def train(self, time, signal, method='NLLS', **kwargs):
        """Train the free parameters

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element of *time*.
            method (str, optional): Method to use for training. Currently the only option is 'NNLS' (Non-negative least squares) Default is 'NNLS'.
            kwargs: any keyword parameters accepted by the specified fit method. For 'NNLS' these are all parameters accepted by `scipy.optimize.curve_fit`, except for bounds.

        Returns:
            Tissue: A reference to the model instance.
        """
        # Estimate S0
        if self.sequence == 'SR':
            Sref = sig.signal_sr(self.R10, 1, self.TR, self.FA, self.TC, self.TP)
        elif self.sequence == 'SS':
            Sref = sig.signal_ss(self.R10, 1, self.TR, self.FA)
        else:
            raise NotImplementedError('Signal model ' + self.sequence + 'is not (yet) supported.') 
        
        self.S0 = np.mean(signal[:self.n0])/Sref if Sref>0 else 0

        # If there is no signal, set all free parameters to zero
        if self.S0 == 0:
            for par in self.free:
                setattr(self, par, 0)
            return self
    
        if method == 'NLLS':
            return mods.train(self, time, signal, **kwargs)
        
        if method == 'PSMS': # Work in progress
            # Fit the complete model
            mods.train(self, time, signal, **kwargs)
            # Fit an intravascular model with the same free parameters
            iv = deepcopy(self)
            #iv.kinetics = 'NX'
            iv._par['PS'] = 0
            for par in ['ve','PS','PSc']:
                if par in iv.free:
                    iv.free.pop(par)
            mods.train(iv, time, signal, **kwargs)
            # If the intravascular model has a lower AIC, take the free parameters from there
            if iv.cost(time, signal, metric='cAIC') < self.cost(time, signal, metric='cAIC'):
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

    def get_params(self, *args, round_to=None):
        """Return the parameter values

        Args:
            args (tuple): parameters to get
            round_to (int, optional): Round to how many digits. If this is not provided, the values are not rounded. Defaults to None.

        Returns:
            list or float: values of parameter values, or a scalar value if only one parameter is required.
        """
        p = _all_pars(self.kinetics, self.water_exchange, self.__dict__)
        if args == ():
            args = p.keys()
        pars = []
        for a in args:
            if a in p:
                v = p[a]
            else:
                v = getattr(self, a)
            if round_to is not None:
                v = round(v, round_to)
            pars.append(v)
        if len(pars)==1:
            return pars[0]
        else:
            return pars

    def export_params(self):
        pars = _all_pars(self.kinetics, self.water_exchange, self.__dict__)
        pars = {p: [_name(p), pars[p], _unit(p)] for p in pars}
        return self._add_sdev(pars)
    

    def cost(self, time, signal, metric='NRMS')->float:
        """Return the goodness-of-fit

        Args:
            time (array-like): Array with time points.
            signal (array-like): Array with measured signals for each element of *time*.
            metric (str, optional): Which metric to use - options are 'RMS' (Root-mean-square), 'NRMS' (Normalized root-mean-square), 'AIC' (Akaike information criterion), 'cAIC' (Corrected Akaike information criterion for small models) or 'BIC' (Baysian information criterion). Defaults to 'NRMS'.

        Returns:
            float: goodness of fit.
        """
        return super().cost(time, signal, metric=metric)
    
    
    def plot(self, time=None, signal=None,  
             xlim=None, ref=None,fname=None, show=True):
        """Plot the model fit against data.

        Args:
            time (array-like, optional): Array with time points.
            signal (array-like, optional): Array with measured signals for each element of *time*.
            xlim (array_like, optional): 2-element array with lower and upper boundaries of the x-axis. Defaults to None.
            ref (tuple, optional): Tuple of optional test data in the form (x,y), where x is an array with x-values and y is an array with y-values. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to True.
        """
        t = self.time()
        if time is None:
            time = t
        
        if xlim is None:
            xlim = [np.amin(t), np.amax(t)]

        if self.water_exchange!='FF':
            fig, ax = plt.subplots(2,2,figsize=(10,12))
            ax00 = ax[0,0]
            ax01 = ax[0,1]
            ax10 = ax[1,0]
            ax11 = ax[1,1]
        else:
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ax00 = ax[0]
            ax01 = ax[1]


        ax00.set_title('MRI signals')
        if ref is not None:
            if 'signal' in ref:
                ax00.plot(t/60, ref['signal'], linestyle='-', linewidth=3.0, color='lightgray', label='Tissue ground truth')
        ax00.plot(time/60, self.predict(time), marker='o', linestyle='None', color='cornflowerblue', label='Predicted data')
        if signal is not None:
            ax00.plot(time/60, signal, marker='x', linestyle='None', color='darkblue', label='Data')
        ax00.plot(t/60, self.predict(t), linestyle='-', linewidth=3.0, color='darkblue', label='Model')
        ax00.set(ylabel='MRI signal (a.u.)', xlim=np.array(xlim)/60)
        ax00.legend()


        C = self.conc(sum=False)
        v = _kin_vols(self.kinetics)
        comps = _kin_comps(self.kinetics)
        pars = _std_conc_pars(self.kinetics, self.__dict__)
        ax01.set_title('Concentration in indicator compartments')
        if ref is not None:
            #ax01.plot(ref['t']/60, 1000*ref['C'], marker='o', linestyle='None', color='lightgray', label='Tissue ground truth')
            ax01.plot(ref['t']/60, 1000*ref['cp'], marker='o', linestyle='None', color='lightcoral', label='Arterial ground truth')
        ax01.plot(t/60, 1000*self.ca, linestyle='-', linewidth=5.0, color='lightcoral', label='Arterial plasma')
        for k, vk in enumerate(v):
            if vk in pars:
                ck = C[k,:]/pars[vk]
                ax01.plot(t/60, 1000*ck, linestyle='-', linewidth=3.0, label=comps[k], color=_clr(comps[k]))
        #ax01.plot(t/60, 1000*np.sum(C,axis=0), linestyle='-', linewidth=3.0, color='darkblue', label='Tissue')
        ax01.set(ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
        ax01.legend()

        if self.water_exchange!='FF':

            R1, _, v = self.relax()
            r1 = lib.relaxivity(self.field_strength, 'blood', self.agent)
            c = sig.c_lin(R1, r1)
            comps = _relax_comps(self.kinetics, self.water_exchange)
            if R1.ndim==1:
                c = c.reshape((1,len(c)))
            ax11.set_title('Concentration in water compartments')
            for i in range(c.shape[0]):
                ax11.plot(t/60, 1000*c[i,:], linestyle='-', linewidth=3.0, color=_clr(comps[i]), label=comps[i])
            ax11.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xlim)/60)
            ax11.legend()


            S = self.signal(sum=False)
            ax10.set_title('Magnetization in water compartments')
            for i in range(S.shape[0]):
                if np.isscalar(v): # TODO renove this after ref of v
                    Si = S[i,...]/v
                else:
                    Si = S[i,...]/v[i]
                ax10.plot(t/60, Si, linestyle='-', linewidth=3.0, color=_clr(comps[i]), label=comps[i])
            ax10.set(xlabel='Time (min)', ylabel='Magnetization (a.u.)', xlim=np.array(xlim)/60)
            ax10.legend()
        
        if fname is not None:
            plt.savefig(fname=fname)
        if show:
            plt.show()
        else:
            plt.close()


# Helper functions

def _clr(comp):
    if comp=='Plasma':
        return 'darkred'
    if comp=='Interstitium':
        return 'steelblue'
    if comp=='Extracellular':
        return 'dimgrey'
    if comp=='Tissue':
        return 'darkgrey'
    if comp=='Blood':
        return 'darkred'
    if comp=='Extravascular':
        return 'blue'
    if comp=='Tissue cells':
        return 'lightblue'
    if comp=='Blood + Interstitium':
        return 'purple'



def _plot_roi(xdata, ydata, yfit, ref, hist_kwargs, rms, ax, name, mask=None):
    ax[0].set_ylabel(name, fontsize=8, rotation='horizontal', labelpad=30)
    if np.size(rms[mask==1])==0:
        return
    perc = np.nanpercentile(rms[mask==1], [5, 25, 50, 75, 95])
    if np.count_nonzero(~np.isnan(perc))==0:
        return
    if mask is None:
        inroi = np.ones(rms.shape)==1
    else:
        inroi = mask==1
    loc = [(rms==p) & inroi for p in perc]
    ax[0].hist(rms[inroi], **hist_kwargs)
    if ref is not None:
        style = {'color':'lightsteelblue', 'linewidth':5.0}
        yref = ref['signal'].reshape((-1,ydata.shape[-1]))
        ax[1].plot(xdata, np.mean(yref[loc[0],:], axis=0), label='Truth (5th perc)', **style)
        ax[2].plot(xdata, np.mean(yref[loc[1],:], axis=0), label='Truth (25th perc)', **style)
        ax[3].plot(xdata, np.mean(yref[loc[2],:], axis=0), label='Truth (median)', **style)
        ax[4].plot(xdata, np.mean(yref[loc[3],:], axis=0), label='Truth (75th perc)', **style)
        ax[5].plot(xdata, np.mean(yref[loc[4],:], axis=0), label='Truth (95th perc)', **style)
    style = {'color':'darkblue'}
    ax[1].plot(xdata, np.mean(yfit[loc[0],:], axis=0), label='Prediction (5th perc)', **style)
    ax[2].plot(xdata, np.mean(yfit[loc[1],:], axis=0), label='Prediction (25th perc)', **style)
    ax[3].plot(xdata, np.mean(yfit[loc[2],:], axis=0), label='Prediction (median)', **style)
    ax[4].plot(xdata, np.mean(yfit[loc[3],:], axis=0), label='Prediction (75th perc)', **style)
    ax[5].plot(xdata, np.mean(yfit[loc[4],:], axis=0), label='Prediction (95th perc)', **style)
    style = {'marker':'o', 'markersize':1, 'linestyle':'None', 'color':'crimson'}
    ax[1].plot(xdata, np.mean(ydata[loc[0],:], axis=0), label='Data (5th perc)', **style)
    ax[2].plot(xdata, np.mean(ydata[loc[1],:], axis=0), label='Data (25th perc)', **style)
    ax[3].plot(xdata, np.mean(ydata[loc[2],:], axis=0), label='Data (median)', **style)
    ax[4].plot(xdata, np.mean(ydata[loc[3],:], axis=0), label='Data (75th perc)', **style)
    ax[5].plot(xdata, np.mean(ydata[loc[4],:], axis=0), label='Data (95th perc)', **style)    


def _check_inputs(self:Tissue):
    if self.sequence not in ['SS','SR']:
        msg = 'Sequence ' + str(self.sequence) + ' is not available.'
        raise ValueError(msg)
    
    if self.kinetics not in ['2CX', '2CU', 'HF', 'HFU', 'WV', 'FX', 'NX', 'U']:
        msg = 'The value ' + str(self.kinetics) + ' for the kinetics argument is not recognised.'
        msg += '\n possible values are 2CX, 2CU, HF, HFU, WV, FX, NX, U.'
        raise ValueError(msg)
    
    if self.water_exchange not in ['FF','NF','RF','FN','NN','RN','FR','NR','RR']:
        msg = 'The value ' + str(self.water_exchange) + ' for the water_exchange argument is not recognised.'
        msg += '\n It must be a 2-element string composed of characters N, F, R.'
        raise ValueError(msg)


def _init_tissue(self:Tissue):

    # Set TS
    if self.TS is None:
        if self.t is None:
            # With uniform sampling, assume TS=dt
            self.TS = self.dt
        else:
            # Raise an error if sampling is not uniform
            # as the sampling duration is ambiguous in that case
            dt = np.unique(self.t[1:]-self.t[:-1])
            if dt.size > 1:
                raise ValueError('For non-uniform time points, the sampling duration TS must be specified explicitly.')
            else:
                # With uniform sampling, assume TS=dt
                self.TS = dt[0]






# CONFIGURATIONS

def _kin_pars(kin):

    if kin=='2CX': 
        return ['vp','vi','Fp','PS']
    if kin=='2CU': 
        return ['vp','Fp','PS']
    if kin=='HF': 
        return ['vp','vi','PS']
    if kin=='HFU': 
        return ['vp','PS']
    if kin=='FX': 
        return ['ve','Fp']
    if kin=='NX': 
        return ['vp','Fp']
    if kin=='U': 
        return ['Fp']
    if kin=='WV': 
        return ['vi','Ktrans']
    
        
def _relax_pars(kin, wex)->list:

    if wex=='FF':
        return _kin_pars(kin)
    
    if kin == '2CX':
        return ['H','vp','vi','Fp','PS']
    if kin == 'HF':
        return ['H','vp','vi','PS']
    if kin=='WV':
        return ['vi','Ktrans']
        
    if wex in ['RR','NN','NR','RN']:

        if kin == '2CU':
            return ['H','vp','vi','Fp','PS']
        if kin == 'HFU':
            return ['H','vp','vi','PS']
        if kin=='FX':
            return ['H','vp','ve','Fp']
        if kin=='NX':
            return ['H','vp','Fp']
        if kin=='U':
            return ['vb','Fp'] 

    if wex in ['RF','NF']:

        if kin == '2CU':
            return ['H','vp','Fp','PS']
        if kin == 'HFU':
            return ['H','vp','PS']
        if kin=='FX':
            return ['H','vp','ve','Fp']
        if kin=='NX':
            return ['H','vp','Fp']
        if kin=='U':
            return ['vb','Fp']

    if wex in ['FR','FN']:

        if kin == '2CU':
            return ['vc','vp','Fp','PS']
        if kin == 'HFU':
            return ['vc','vp','PS']
        if kin=='FX':
            return ['vc','ve','Fp']
        if kin=='NX':
            return ['vc','vp','Fp']
        if kin=='U':
            return ['vc','Fp']
        

def _tissue_pars(kin, wex)->list:

    pars = _relax_pars(kin, wex)

    if wex=='FF':
        return pars
    
    if wex=='RR':
        if kin=='WV':
            return ['PSc'] + pars
        elif kin in ['NX','U']:
            return ['PSe'] + pars
        else:
            return ['PSe','PSc'] + pars
        
    if wex=='NN':
        return pars
    
    if wex=='NR':
        if kin in ['NX','U']:
            return pars
        else:
            return ['PSc'] + pars 
         
    if wex=='RN':
        if kin=='WV':
            return pars
        else:
            return ['PSe'] + pars  
        
    if wex=='RF':
        if kin=='WV':
            return pars
        else:
            return ['PSe'] + pars
        
    if wex=='NF':
        return pars
    
    if wex=='FR':
        return ['PSc'] + pars
    
    if wex=='FN':
        return pars


def _init(p):
    if p=='Fp': 
        return 0.01
    if p=='PS':
        return 0.003
    if p=='vb':
        return 0.1
    if p=='vi':
        return 0.3
    if p=='H':
        return 0.45
    if p=='PSe':
        return 0.03
    if p=='PSc':
        return 0.03
    if p=='Ktrans':
        return _init('PS')*_init('Fp')/(_init('PS')+_init('Fp'))
    if p=='vc':
        return 1-_init('vb')-_init('vi')
    if p=='vp':
        return _init('vb')*(1-_init('H'))
    if p=='ve':
        return _init('vp')+_init('vi')

def _bound(p):
    if p in ['Ktrans','PS','Fp','PSe','PSc']:
        return [0, np.inf]
    else:
        return [1e-3, 1-1e-3] # TODO: needs solutions vor v=0 in signal models
    
def _name(p)->str:

    # Primary
    if 'Fp'==p:
        return 'Plasma flow'
    if 'Ktrans'==p:
        return 'Volume transfer constant'
    if 'PS'==p:
        return 'Permeability-surface area product'
    if 'vb'==p: 
        return 'Blood volume'
    if 'vi'==p:
        return 'Interstitial volume'
    if 'vc'==p:
        return 'Intracellular volume'
    if 'vp'==p:
        return 'Plasma volume'
    if 've'==p:
        return 'Extracellular volume'
    if 'PSe'==p: 
        return 'Transendothelial water PS'
    if 'PSc'==p: 
        return 'Transcytolemmal water PS'
    if 'H'==p:
        return 'Hematocrit'    

    # Derived
    if 'E'==p:
        return 'Tracer extraction fraction'
    if 'Ti'==p:
        return 'Interstitial mean transit time'
    if 'Tp'==p:
        return 'Plasma mean transit time'
    if 'Tb'==p:
        return 'Blood mean transit time'
    if 'Te'==p:
        return 'Extracellular mean transit time'
    if 'Twc'==p: 
        return 'Intracellular water mean transit time'
    if 'Twi'==p: 
        return 'Interstitial water mean transit time'
    if 'Twb'==p: 
        return 'Intravascular water mean transit time'

    raise ValueError(p + ' not recognized')


def _unit(p)->str:

    # Primary

    if 'Fp'==p:
        return 'mL/sec/cm3'
    if 'Ktrans'==p:
        return 'mL/sec/cm3'
    if 'PS'==p:
        return 'mL/sec/cm3'
    if 'vb'==p: 
        return 'mL/cm3'
    if 'vi'==p:
        return 'mL/cm3'
    if 'vc'==p:
        return 'mL/cm3'
    if 'H'==p:
        return ''
    if 'vp'==p:
        return 'mL/cm3'
    if 've'==p:
        return 'mL/cm3'
    if 'PSe'==p: 
        return  'mL/sec/cm3'
    if 'PSc'==p: 
        return  'mL/sec/cm3'
    
    # Derived

    if 'E'==p:
        return ''
    if 'Ti'==p:
        return 'sec'
    if 'Tp'==p:
        return 'sec'
    if 'Tb'==p:
        return 'sec'
    if 'Te'==p:
        return 'sec'
    if 'Twc'==p: 
        return  'sec'
    if 'Twi'==p: 
        return  'sec'  
    if 'Twb'==p: 
        return  'sec'
    
    # All others dimensionless
    return ''


def _std_conc_pars(kin, p): 
    return {par:p[par] for par in _kin_pars(kin)}


def _std_relax_pars(kin, wex, p):
    return {par:p[par] for par in _relax_pars(kin, wex)}

        
def _PSw(kin, wex, p):
    
    if wex=='FF':
        return None
    
    elif wex == 'RR':

        if kin=='WV':
            PSw = [[0, p['PSc']], [p['PSc'], 0]]
        elif kin in ['NX', 'U']:
            PSw = [[0, p['PSe']], [p['PSe'], 0]]
        else:
            PSw = [[0, p['PSe'], 0], [p['PSe'], 0, p['PSc']], [0, p['PSc'], 0]]  

    elif wex == 'RN':

        if kin=='WV':
            PSw = [[0, 0], [0, 0]]
        elif kin in ['NX', 'U']:
            PSw = [[0, p['PSe']], [p['PSe'], 0]]
        else:
            PSw = [[0, p['PSe'], 0], [p['PSe'], 0, 0], [0, 0, 0]] 

    elif wex == 'NR':

        if kin=='WV':
            PSw = [[0, p['PSc']], [p['PSc'], 0]]
        elif kin in ['NX', 'U']:
            PSw = [[0, 0], [0, 0]]
        else:
            PSw = [[0, 0, 0], [0, 0, p['PSc']], [0, p['PSc'], 0]] 

    elif wex == 'NN':

        if kin=='WV':
            PSw = [[0,0], [0,0]]
        elif kin in ['NX', 'U']:
            PSw = [[0,0], [0,0]]
        else:
            PSw = [[0,0,0], [0,0,0], [0,0,0]] 

    elif wex == 'RF':

        if kin=='WV':
            return None
        else:
            PSw = [[0, p['PSe']], [p['PSe'], 0]]

    elif wex == 'NF':

        if kin=='WV':
            return None
        else:
            PSw = [[0,0], [0,0]]

    elif wex == 'FR':

        PSw = [[0, p['PSc']], [p['PSc'], 0]]

    elif wex == 'FN':

        PSw = [[0,0],[0,0]]

    return np.array(PSw)
    

def _kin_vols(kin):

    if kin=='2CX': 
        return ['vp','vi']
    if kin=='2CU': 
        return ['vp','vi']
    if kin=='HF': 
        return ['vp','vi']
    if kin=='HFU': 
        return ['vp','vi']
    if kin=='FX': 
        return ['ve']
    if kin=='NX': 
        return ['vp']
    if kin=='U': 
        return ['vp']
    if kin=='WV': 
        return ['vi']


def _kin_comps(kin):

    if kin=='2CX': 
        return ['Plasma','Interstitium']
    if kin=='2CU': 
        return ['Plasma','Interstitium']
    if kin=='HF': 
        return ['Plasma','Interstitium']
    if kin=='HFU': 
        return ['Plasma','Interstitium']
    if kin=='FX': 
        return ['Extracellular']
    if kin=='NX': 
        return ['Plasma']
    if kin=='U': 
        return ['Plasma']
    if kin=='WV': 
        return ['Interstitium']
    

def _relax_comps(kin, wex)->list:

    if wex=='FF':
        return ['Tissue']
        
    if wex in ['RR','NN','NR','RN']:
        if kin=='WV':
            return ['Interstitium']
        elif kin in ['NX','U']:
            return ['Blood', 'Extravascular']
        else:
            return ['Blood', 'Interstitium', 'Tissue cells']
        
    if wex in ['RF','NF']:
        if kin=='WV':
            return ['Interstitium']
        else:
            return ['Blood', 'Extravascular']

    if wex in ['FR','FN']:
        if kin=='WV': 
            return ['Interstitium', 'Tissue cells']
        else:
            return ['Blood + Interstitium', 'Tissue cells']



def _add_fix_params(kin, wex, p):

    if kin=='2CX': 
        pass
    elif kin=='HF': 
        p['Fp'] = np.inf
    elif kin=='HFU': 
        p['Fp'] = np.inf
    elif kin=='FX': 
        p['Ktrans'] = np.inf
    elif kin=='NX': 
        p['Ktrans'] = 0
    elif kin=='WV': 
        p['vp'] = 0

    if wex[0]=='N':
        p['PSe'] = 0
    elif wex[0]=='F':
        p['PSe'] = np.inf

    if wex[1]=='N':
        p['PSc'] = 0
    elif wex[1]=='F':
        p['PSc'] = np.inf

    return p


def _all_pars(kin, wex, p):

    p = {par:p[par] for par in _tissue_pars(kin, wex)}
 
    try:
        p['Ktrans'] = _div(p['Fp']*p['PS'], p['Fp']+p['PS'])
    except:
        pass
    try:
        p['ve'] = p['vi'] + p['vc']
    except:
        pass
    try:
        p['vb'] = _div(p['vp'], 1-p['H'])
    except:
        pass   
    try:
        p['E'] = _div(p['PS'], p['Fp']+p['PS'])
    except:
        pass     
    try:
        p['Ti'] = _div(p['vi'], p['PS'])
    except:
        pass
    try:
        p['Tp'] = _div(p['vp'], p['PS']+p['Fp'])
    except:
        pass
    try:
        p['Tb'] = _div(p['vp'], p['Fp'])
    except:
        pass
    try:
        p['Te'] = _div(p['ve'], p['Fp'])
    except:
        pass
    try:
        p['Twc'] = _div(1-p['vb']-p['vi'], p['PSc'])
    except:
        pass
    try:
        p['Twi'] = _div(p['vi'], p['PSc']+p['PSe'])
    except:
        pass
    try:
        p['Twb'] = _div(p['vb'], p['PSe'])
    except:
        pass

    return p


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b==0, 0, np.divide(a, b))

