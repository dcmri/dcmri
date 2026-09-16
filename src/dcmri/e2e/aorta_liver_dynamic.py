"""Joint model for aorta and liver signals measured over two scans.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver, measured over two separate scans.

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    kinetics (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only single-inlet models 
        are allowed. Defaults to '1I-IC-HFD'.
    stationary (str, optional): For intracellular tracers - stationarity 
        regime of the hepatocytes. The options are 'UE', 'E', 'U' or None. 
        For more detail see :ref:`liver-tissues`. Defaults to 'UE'.
    stationary (str, optional): Stationarity regime of the hepatocytes. 
        The options are 'UE', 'E', 'U' or None. For more detail 
        see :ref:`liver-tissues`. Defaults to 'UE'.
    sequence (str, optional): imaging sequence. Possible values are 'SS'
        and 'SR'. Defaults to 'SS'.
    params (dict, optional): values for the parameters of the tissue,
        specified as keyword parameters. Defaults are used for any that are
        not provided. See tables :ref:`AortaLiver2scan-parameters` and
        :ref:`AortaLiver2scan-defaults` for a list of parameters and their
        default values.

See Also:
    `AortaLiver`

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

    >>> time, aif, roi, gt = dc.fake.tissue2scan(R1b=1/dc.const.T1(3.0,'liver'))

    Since this model generates four time curves, the x- and y-data are 
    tuples:

    >>> time = (time[0], time[1], time[0], time[1])
    >>> signal = (aif[0], aif[1], roi[0], roi[1])

    Build an aorta-liver model and parameters to match the conditions of 
    the fake tissue data:

    >>> model = dc.AortaLiver2scan(
    ...     dt = 0.5,
    ...     tmax = 420,
    ...     weight = 70,
    ...     agent = 'gadodiamide',
    ...     dose = 0.2,
    ...     dose2 = 0.2,
    ...     rate = 3,
    ...     field_strength = 3.0,
    ...     TR = 0.005,
    ...     FA = 15,
    ...     FA2 = 15,
    ...     TS = 0.5,
    ...     Th_i = 120,
    ...     Th_f = 120,
    ... )

    In this case we have defined different initial values for Th as 
    the defaults are optimized for the slow passage through hepatocytes. 
    We also need to reset the parameter bounds:

    >>> model.free['Th_i'] = [0, np.inf]
    >>> model.free['Th_f'] = [0, np.inf]

    Train the model on the data:

    >>> model.train(time, signal, n0=10, xtol=1e-3)

    Plot the reconstructed signals and concentrations and compare against 
    the experimentally derived data:

    >>> model.plot(time, signal)

    We can also have a look at the model parameters after training:

    >>> model.print_params(round_to=3)
    --------------------------------
    Free parameters with their stdev
    --------------------------------
    Aorta second signal scale factor (S02a): 195.824 (2.025) a.u.
    Liver second signal scale factor (S02l): 297.854 (4.9) a.u.
    Second bolus arrival time (BAT2): 254.512 (0.137) sec
    First bolus arrival time (BAT): 14.288 (0.132) sec
    Cardiac output (CO): 203.199 (5.406) mL/sec
    Heart-lung mean transit time (Thl): 15.236 (0.263) sec
    Heart-lung dispersion (Dhl): 0.381 (0.009)
    Organs blood mean transit time (To): 23.761 (3.052) sec
    Organs extraction fraction (Eo): 0.287 (0.053)
    Organs extravascular mean transit time (Toe): 50.274 (17.44) sec
    Body extraction fraction (Eb): 0.078 (0.015)
    Apparent liver extracellular volume fraction (ve_app): 0.053 (0.008) mL/cm3
    Extracellular mean transit time (Te): 1.298 (0.552) sec
    Extracellular dispersion (De): 1.0 (0.7)
    Initial hepatic plasma clearance (Ktrans_i): 0.005 (0.001) mL/sec/cm3
    Final hepatic plasma clearance (Ktrans_f): 0.005 (0.001) mL/sec/cm3
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
    ----------------------------
    Fixed and derived parameters
    ----------------------------
    Aorta first baseline R1 (R1ba): 0.614 Hz
    Aorta first signal scale factor (S0a): 100.117 a.u.
    Liver first baseline R1 (R1bl): 1.33 Hz
    Liver first signal scale factor (S0(l)): 150.003 a.u.
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec
"""

import matplotlib.pyplot as plt
import numpy as np

from dcmri.utils.fit import train_bat, loss
from dcmri.core.tools import get_quantity, get_bounds
from dcmri.models.aorta_liver_dynamic import AortaLiverDynamicModel
from dcmri.inverse.lib import estimate_bat


class AortaLiverDynamic():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = AortaLiverDynamicModel(**config)

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
            pred['S_1_ao'][:, :, :len(time[0])].reshape(-1), 
            pred['S_2_ao'][:, :, :len(time[1])].reshape(-1), 
            pred['S_1_li'][:, :, :len(time[2])].reshape(-1), 
            pred['S_2_li'][:, :, :len(time[3])].reshape(-1), 
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

    def train(self, data: dict, free: dict = None, bounds: dict = None, n0=1, **kwargs) -> tuple:
        p = self._pars

        # Estimate BAT
        bat = estimate_bat(data['tS_1_ao'], data['S_1_ao'], n0)
        bat2 = estimate_bat(data['tS_2_ao'], data['S_2_ao'], n0)
        p['BAT'] = max(bat - p['T_hl'], 0)
        p['BAT_1'] = max(bat - p['T_hl'], 0)
        p['BAT_2'] = max(bat2 - p['T_hl'], 0)

        # Set calibration data
        if self._model.config['calibrate']:
            p[f"Scal_1_ao"] = data['S_1_ao'][..., :n0]
            p[f"Scal_2_ao"] = data['S_2_ao'][..., :n0]
            p[f"Scal_1_li"] = data['S_1_li'][..., :n0]
            p[f"Scal_2_li"] = data['S_2_li'][..., :n0]
            for roi in ['ao', 'li']:
                for scan in [1, 2]:
                    p[f'iScal_{scan}_{roi}'] = np.arange(n0)

        # Perform training
        free = get_bounds(free, bounds, free_pars=self._params('free'), value=p)

        time = (data['tS_1_ao'], data['tS_2_ao'], data['tS_1_li'], data['tS_2_li'])
        signal = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])
        return train_bat(self._predict, time, signal, self._pars, free, **kwargs)


    def plot(self, data: dict, xlim: list = None, fname: str = None, show=True):
        prediction = self._model(self._pars)

        if xlim is None: 
            xlim = [prediction['tR'][0], prediction['tR'][-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)

        # Plot signals
        def _plot_data2scan(roi, ts, s, ax, color):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            for i in range(s[0].shape[0]):
                for j in range(s[0].shape[1]):
                    ax.plot(ts[0] / 60, s[0][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(ts[1] / 60, s[1][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(prediction[f'tS_1_{roi}'] / 60, prediction[f'S_1_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
                    ax.plot(prediction[f'tS_2_{roi}'] / 60, prediction[f'S_2_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan('ao',(data['tS_1_ao'], data['tS_2_ao']), (data['S_1_ao'], data['S_2_ao']), ax1, ['lightcoral', 'darkred'])
        _plot_data2scan('li',(data['tS_1_li'], data['tS_2_li']), (data['S_1_li'], data['S_2_li']), ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        ax2.plot(prediction['tC'] / 60, 1000 * prediction['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(prediction['tC'] / 60, 0 * prediction['tC'], color='gray')
        if prediction['C_li'].shape[0]==2:
            ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        else:
            ax4.plot(prediction['tC'] / 60, 1000 * prediction['C_li'], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname is not None: plt.savefig(fname=fname)
        if show: plt.show()
        else: plt.close()

    def cost(self, data: dict, metric: str = 'NRMS', nfree=None) -> float:
        time = (data['tS_1_ao'], data['tS_2_ao'], data['tS_1_li'], data['tS_2_li'])
        signal = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])

        pred = self._predict(time)
        signal = np.concatenate([s.reshape(-1) for s in signal])
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)


    

    # WIP below here


    # def export_params(self, sdev=None, group=None, num_only=False, deriv=False, scalar_only=False):
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     return export_params(pars, lexicon=QUANTITIES, sdev=sdev, num_only=num_only, scalar_only=scalar_only, group=group)

    # def print_params(self, *args, round_to=None, group=None, 
    #                  fixed_only=False, free_only=False, deriv=False):
    #     """Pretty print model parameters"""
    #     pars = self._pars
    #     if deriv:
    #         pars = dpars_liver(pars, self._cnfg['kinetics'])
    #     if args != ():
    #         pars = {k: v for k, v in self._pars.items() if k in args}
    #     if fixed_only:
    #         pars = {k: v for k, v in pars.items() if k not in self._params('free')}
    #     if free_only:
    #         pars = {k: v for k, v in pars.items() if k in self._params('free')}
    #     print_params(pars, round_to=round_to, group=group, lexicon=QUANTITIES)


