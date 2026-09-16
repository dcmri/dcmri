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


from dcmri.core.tools import get_quantity, get_bounds
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.models.aorta_portal_liver import AortaPortalLiverModel


class AortaPortalLiver():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = AortaPortalLiverModel(**config)

        # Initialise model parameters
        pars = self._model.dummy_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if get_quantity(p)['group']=='phys'} 
            params_free |= {p for p in ['BAT', 'BAT_1', 'BAT_2'] if p in params}
            return params_free
        return params

    def _predict(self, time: tuple):
        pred = self._model(self._pars)
        return (
            pred['S_ao'][:, :, :len(time[0])].reshape(-1),
            pred['S_pv'][:, :, :len(time[1])].reshape(-1), 
            pred['S_li'][:, :, :len(time[2])].reshape(-1)
        )

    # ==========================================
    # User Interface
    # ==========================================

    def params(self, group=None) -> list:
        """Return a list of model parameters"""
        return self._params(group)

    def predict(self) -> dict:
        """Predicts the data."""
        return self._model(self._pars)
    
    def train(
            self, data: dict, free: dict = None, 
            bounds: dict = None, n0=1, **kwargs) -> tuple:

        p = self._pars
        
        # Estimate BAT 
        bat = estimate_bat(data['tS_ao'], data['S_ao'], n0)
        p['BAT'] = max(bat - p['T_hl'], 0)

        # Set calibration data
        if self._model.config['calibrate']:
            for roi in ['ao', 'pv', 'li']:
                p[f"Scal_{roi}"] = data[f"S_{roi}"][..., :n0]
                p[f'iScal_{roi}'] = np.arange(n0)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)

        time = (data['tS_ao'], data['tS_pv'], data['tS_li'])
        signal = (data['S_ao'], data['S_pv'], data['S_li'])
        return train_bat(self._predict, time, signal, p, free, **kwargs)


    def plot(self, data: dict, xlim=None, fname=None, show=True):
        prediction = self._model(self._pars)

        if xlim is None: 
            xlim = [prediction['tR'][0], prediction['tR'][-1]]
        xlim = np.array(xlim) / 60
        
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes
        
        # Plot signals
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(si.shape[0]):
                for j in range(si.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(prediction['tS_ao'], prediction['S_ao'], data['tS_ao'], data['S_ao'], ax1, ['lightcoral', 'darkred'])
        plot_data(prediction['tS_li'], prediction['S_li'], data['tS_li'], data['S_li'], ax5, ['cornflowerblue', 'darkblue'])
        plot_data(prediction['tS_pv'], prediction['S_pv'], data['tS_pv'], data['S_pv'], ax3, ['orchid', 'purple'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax2.plot(prediction['tC'] / 60, 1000 * prediction['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(ylabel='Concentration (mM)', xlim=xlim)
        ax4.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_pv'][0], linestyle='-', color='purple', linewidth=2.0, label='Portal vein')
        ax4.legend()

        ax6.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax6.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax6.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
        ax6.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
        ax6.plot(prediction['tC'] / 60, 1000 * prediction['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax6.legend()

        if fname: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()

    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        time = (data['tS_ao'], data['tS_pv'],  data['tS_li'])
        signal = (data['S_ao'], data['S_pv'], data['S_li'])

        pred = self._predict(time)
        signal = np.concatenate([s.reshape(-1) for s in signal])
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)