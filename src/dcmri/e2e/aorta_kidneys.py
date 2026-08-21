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
    ...     R1ba=1/dc.const.T1(pars['B0'], 'blood'),
    ...     R1b_lk=1/dc.const.T1(pars['B0'], 'kidney'),
    ...     R1b_rk=1/dc.const.T1(pars['B0'], 'kidney'),
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

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.roi_model import SuperRoiModel
from dcmri.core.quantities import QUANTITIES
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.models.aorta_kidneys import AortaKidneysModel


class AortaKidneys(SuperRoiModel):
    """Joint model or aorta, portal vein and liver signals.

    A whole-body model to simultaneously predict signals in 
    aorta, portal vein and liver.  
    """
    def __init__(self, data: dict=None, **config):
        if data is None:
            data = {}

        self._version = '1.0'
        self._model = AortaKidneysModel(**config)

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
        return pred['S_a'][:, :, :len(time[0])], pred['S_lk'][:, :, :len(time[1])], pred['S_rk'][:, :, :len(time[2])]

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
            for i, roi in enumerate(['a', 'lk', 'rk']):
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
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        
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
        plot_data('lk', time[1], signal[1], ax3, ['cornflowerblue', 'darkblue'])
        plot_data('rk', time[2], signal[2], ax5, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(prediction['t'] / 60, 0 * prediction['t'], color='gray')
        ax2.plot(prediction['t'] / 60, 1000 * prediction['C_a'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.plot(prediction['t'] / 60, 1000 * prediction['c_lk'][0], linestyle='--', color='lightcoral', linewidth=2.0, label='Left Kidney')
        ax2.plot(prediction['t'] / 60, 1000 * prediction['c_rk'][0], linestyle='-.', color='lightcoral', linewidth=2.0, label='Right Kidney')
        ax2.legend()

        def plot_conc_kidney(C, kid, ax):
            ax.set(xlabel='Time (min)', ylabel=f'{kid} conc (mM)', xlim=xlim)
            ax.plot(prediction['t'] / 60, 0 * prediction['t'], color='gray')
            ax.plot(prediction['t'] / 60, 1000 * C[0], linestyle='-', color='darkred', linewidth=2.0, label='Blood')
            ax.plot(prediction['t'] / 60, 1000 * C[1], linestyle='-', color='darkcyan', linewidth=2.0, label='Tubuli')
            ax.plot(prediction['t'] / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
            ax.legend()

        plot_conc_kidney(prediction['C_lk'], 'Left kidney', ax4)
        plot_conc_kidney(prediction['C_rk'], 'Right kidney', ax6)

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
        time = (data['tS_a'], data['tS_lk'], data['tS_rk'])
        signal = (data['S_a'], data['S_lk'], data['S_rk'])

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
        time = (data['tS_a'], data['tS_lk'], data['tS_rk'])
        signal = (data['S_a'], data['S_lk'], data['S_rk'])

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
        time = (data['tS_a'], data['tS_lk'], data['tS_rk'])
        signal = (data['S_a'], data['S_lk'], data['S_rk'])

        self._pars['tmax'] = self._pars['dt'] + np.max(time[0]) + (time[0][-1] - time[0][-2])
        pred = self._predict(time)
        signal = np.concatenate(signal)
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)