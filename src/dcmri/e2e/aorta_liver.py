"""Joint model for aorta and liver signals.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver.  

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    liver (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only single-inlet models 
        are allowed. Defaults to '1I-IC-HFD'.
    non_stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to None.
    sequence (str, optional): imaging sequence. Possible values are '3D-SPGR-SS'
        and 'SR'. Defaults to '3D-SPGR-SS'.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`AortaLiver-parameters` and
        :ref:`AortaLiver-defaults` for a list of parameters and their
        default values.

See Also:
    `AortaLiver2scan`

Example:

    Use the model to reconstruct concentrations from experimentally 
    derived signals.

.. plot::
    :include-source:
    :context: close-figs

    >>> import matplotlib.pyplot as plt
    >>> import dcmri as dc

    Use `fake.tissue` to generate synthetic test data from 
    experimentally-derived concentrations:

    Use `fake.liver` to generate synthetic test data:

    >>> time, aif, vif, roi, gt = dc.fake.liver()

    Since this model generates two time curves, the x- and y-data are 
    tuples:

    >>> time, signal = (time,time), (aif,roi)

    Build an aorta-liver model and parameters to match the 
    conditions of the fake liver data:

    >>> model = dc.AortaLiver(
    ...     dt = 0.5,
    ...     tmax = 180,
    ...     weight = 70,
    ...     agent = 'gadoxetate',
    ...     field_strength = 3.0,
    ...     dose = 0.2,
    ...     rate = 3,
    ...     TR = 0.005,
    ...     FA = 15,
    ... )

    Train the model on the data:

    >>> model.train(time, signal, n0=10, xtol=1e-3)

    Plot the reconstructed signals and concentrations and compare 
    against the experimentally derived data:

    >>> model.plot(time, signal)

    We can also have a look at the model parameters after training:

    >>> model.print_params(round_to=3)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    First bolus arrival time (BAT): 13.231 (0.266) sec
    Cardiac output (CO): 102.893 (4.182) mL/sec
    Heart-lung mean transit time (T(hl)): 16.285 (0.409) sec
    Heart-lung dispersion (D(hl)): 0.324 (0.016)
    Organs blood mean transit time (To): 19.578 (5.583) sec
    Organs extraction fraction (Eo): 0.363 (0.075)
    Organs extravascular mean transit time (Toe): 46.775 (53.05) sec
    Body extraction fraction (Eb): 0.029 (0.168)
    Apparent liver extracellular volume fraction (ve_app): 0.307 (0.564) mL/cm3
    Extracellular mean transit time (Te): 44.748 (79.644) sec
    Extracellular dispersion (De): 0.916 (0.148)
    Hepatic plasma clearance (Ktrans): 0.002 (0.003) mL/sec/cm3
    Hepatocellular mean transit time (Th): 712.855 (5899.078) sec
    ----------------------------
    Fixed and derived parameters
    ----------------------------
    Aorta first baseline R1 (R1ba): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.169 a.u.
    Liver first baseline R1 (R1bl): 1.33 Hz
    Liver first signal scale factor (S0l): 150.0 a.u.
    Liver volume (vol): 1000 cm3
    Biliary tissue excretion rate (Kbh): 0.001 mL/sec/cm3
"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.tools import print_params, export_params
from dcmri.core.quantities import QUANTITIES
from dcmri.utils.fit import train, train_bat, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.models.aorta_liver import AortaLiverModel


class AortaLiver(SuperRoiModel):
    """Joint model or aorta and liver signals.

    A whole-body model to simultaneously predict signals in 
    aorta and liver.  

    Args:
        kinetics (str, optional): Tracer-kinetic model.
        non_stationary (str, optional): Stationarity regime of liver transporters.
        sequence (str, optional): imaging sequence.
        params (dict, optional): override parameter defaults.

    See Also:
        `AortaLiver2scan`
    """
    def __init__(self, data: dict=None, **config):
        if data is None:
            data = {}

        self._version = '1.0'
        self._model = AortaLiverModel(**config)

        # Initialise model parameters
        pars = self._model.lexicon_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if p in QUANTITIES and QUANTITIES[p]['group']=='phys'} # needs to be more general
            params_free |= {p for p in ['BAT'] if p in params}
            return params_free
        return params

    def _predict(self, time: tuple):
        pred = self._model(self._pars)
        return pred['S_a'][:, :, :len(time[0])], pred['S_l'][:, :, :len(time[1])]

    def _train(
        self, time: tuple, signal: tuple, free: dict, 
        bounds: dict, n0: int, **kwargs
    ):
        p = self._pars

        # Estimate BAT 
        bat = estimate_bat(time[0], signal[0], n0)
        p['BAT'] = max(bat - p['Thl'], 0)

        # Estimate baseline
        if self._model.config['calibrate']:
            for i, roi in enumerate(['a', 'l']):
                p[f"Sb_{roi}"] = signal[i][..., :n0]

        # Perform training
        free = self._set_free_pars(free, bounds) 
        return train_bat(self._predict, time, signal, self._pars, free, **kwargs)


    def _plot(
        self, time: tuple, signal: tuple, xlim: list, fname: str, show: bool
    ):
        prediction = self._model(self._pars)

        if xlim is None: 
            xlim = [prediction['t'][0], prediction['t'][-1]]
        xlim = np.array(xlim)/60
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot signals
        def plot_data(roi, ts, s, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    ax.plot(ts / 60, s[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(prediction[f'tS_{roi}'] / 60, prediction[f'S_{roi}'][i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data('a', time[0], signal[0], ax1, ['lightcoral', 'darkred'])
        plot_data('l', time[1], signal[1], ax3, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(prediction['t'] / 60, 0 * prediction['t'], color='gray')
        ax2.plot(prediction['t'] / 60, 1000 * prediction['C_a'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(prediction['t'] / 60, 0 * prediction['t'], color='gray')
        if prediction['C_l'].shape[0]==2:
            ax4.plot(prediction['t'] / 60, 1000 * prediction['C_l'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(prediction['t'] / 60, 1000 * prediction['C_l'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(prediction['t'] / 60, 1000 * prediction['C_l'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        else:
            ax4.plot(prediction['t'] / 60, 1000 * prediction['C_l'], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()


    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> np.ndarray:
        """Predicts the data."""
        return self._model(self._pars)
    
    def train(
        self, data: dict, free: dict = None, 
        bounds: dict = None, n0=10, **kwargs
    ) -> tuple:
        """Train the model free parameters.

        Args:
            time (tuple): (time_aorta, time_liver) arrays.
            signal (tuple): (signal_aorta, signal_liver) arrays.
            free (dict, optional): Free parameters and their bounds.
            bounds (dict, optional): Override default bounds for specific parameters.
            n0 (int, optional): Number of baseline time points for S0 estimation.
            **kwargs: Arguments passed to scipy.optimize.curve_fit.

        Returns:
            vals, sdev, pcov: Values, standard deviations and covariance matrix of free parameters
        """
        time = (data['tS_a'], data['tS_l'])
        signal = (data['S_a'], data['S_l'])

        self._pars['tmax'] = self._pars['dt'] + np.max(time[0]) + (time[0][-1] - time[0][-2])
        return self._train(time, signal, free, bounds, n0, **kwargs)

    def plot(
        self, data: dict, xlim=None, fname=None, show=True,
    ):
        """Plot the model fit against data

        Args:
            time (tuple): tuple of 2 arrays with time points for aorta and 
              liver, in that order. The two arrays can be different in length 
              and value.
            signal (array-like): tuple of 2 arrays with signals for aorta and 
              liver, in that order. The arrays can be different in length and 
              value but each has to have the same length as its corresponding 
              array of time points.
            xlim (array_like, optional): 2-element array with lower and upper 
              boundaries of the x-axis. Defaults to None.
            fname (path, optional): Filepath to save the image. If no value 
              is provided, the image is not saved. Defaults to None.
            show (bool, optional): If True, the plot is shown. Defaults to 
              True.
        """
        time = (data['tS_a'], data['tS_l'])
        signal = (data['S_a'], data['S_l'])

        self._pars['tmax'] = self._pars['dt'] + np.max(time[0]) + (time[0][-1] - time[0][-2])
        self._plot(time, signal, xlim, fname, show)

    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
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
        time = (data['tS_a'], data['tS_l'])
        signal = (data['S_a'], data['S_l'])

        self._pars['tmax'] = self._pars['dt'] + np.max(time[0]) + (time[0][-1] - time[0][-2])
        pred = self._predict(time)
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)
        
    def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._model._config['liver'])
        return export_params(pars, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    def print_params(self, *args, round_to=None, group=None, 
                     fixed_only=False, free_only=False, deriv=False):
        """Pretty print model parameters"""
        pars = self._pars
        if deriv:
            pars = dpars_liver(pars, self._model._config['liver'])
        if args != ():
            pars = {k: v for k, v in self._pars.items() if k in args}
        if fixed_only:
            pars = {k: v for k, v in pars.items() if k not in self._params('free')}
        if free_only:
            pars = {k: v for k, v in pars.items() if k in self._params('free')}
        print_params(pars, round_to=round_to, group=group)