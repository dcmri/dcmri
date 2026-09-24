"""Joint model for aorta and liver signals measured over two scans.

This model uses a whole-body model to simultaneously predict signals in 
aorta and liver, measured over two separate scans.

For more detail on the whole-body model, see :ref:`whole-body-tissues`. 
For more detail on the liver model, see :ref:`liver-tissues`. 

Args:
    kinetics (str, optional): Tracer-kinetic liver model. See table 
        :ref:`table-liver-models` for options - only single-inlet models 
        are allowed. Defaults to '1I-IC_HFD'.
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
    Second bolus arrival time (BAT_2): 254.512 (0.137) sec
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
    Liver first signal scale factor (S0l): 150.003 a.u.
    Initial hepatocellular mean transit time (Th_i): 70.022 (12.142) sec
    Final hepatocellular mean transit time (Th_f): 72.227 (8.407) sec

"""

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.tools import get_quantity, get_bounds
from dcmri.utils.fit import train_bat, loss
from dcmri.inverse.lib import estimate_bat
from dcmri.forward.aorta_liver_drug import ForwardAortaLiverDrug
# from dcmri.core.quantities import QUANTITIES
# from dcmri.core.tools import export_params


class AortaLiverDrug():

    def __init__(self, data: dict=None, **config):
        self._version = '1.0'
        self._model = ForwardAortaLiverDrug(**config)

        # Initialise model parameters
        pars = self._model.dummy_data()
        if data is not None:
            pars |= data
        self._pars = self._model.input_data(pars)

    def _params(self, group=None):
        params = self._model.mapped_inputs()
        if group == 'free':
            params_free = {p for p in params if get_quantity(p)['group']=='phys'} 
            params_free |= {p for p in ['BAT_1', 'BAT_2', 'BAT_3', 'BAT_4'] if p in params}
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

    def predict(self) -> tuple:
        """Predicts the data."""
        return self._model(self._pars)

    def train(self, data: dict, free=None, bounds:dict=None, n0=[1, 1], **kwargs) -> tuple:
        p = self._pars

        # Estimate BAT
        bat1 = estimate_bat(data['tS_1_ao'], data['S_1_ao'], n0)
        bat2 = estimate_bat(data['tS_2_ao'], data['S_2_ao'], n0)

        p['BAT_1'] = max(bat1 - p['T_hl'], 0)
        if self._model.config['bolus'] == 'single':
            p['BAT_2'] = max(bat2 - p['T_hl'], 0)
        else:
            p['BAT_3'] = max(bat2 - p['T_hl'], 0)
            # p['BAT_2'] = p['BAT_1'] + p['bdel_1'] # These parameters beed to replace BAT_3 and BAT_4 in formward model
            # p['BAT_4'] = p['BAT_3'] + p['bdel_2']

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


    def plot(self, data: dict, xlim=None, clim=None, fname=None, show=True):
        prediction = self._model(self._pars)
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('Control visit')
        ax2.set_title('Treatment visit')
        ax3.set_title('Control visit')
        ax4.set_title('Treatment visit')

        def plot_data2scan(t, s, ti, si, ax, xl, yl, color):
            if xl is None: 
                xl = [0, t[-1]]
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xl)/60, ylim=yl)
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        ylim_a = [
            0.9 * min(prediction[f'S_1_ao'].min(), prediction[f'S_2_ao'].min(), data[f'S_1_ao'].min(), data[f'S_2_ao'].min()), 
            1.1 * max(prediction[f'S_1_ao'].max(), prediction[f'S_2_ao'].max(), data[f'S_1_ao'].max(), data[f'S_2_ao'].max()),
        ]
        ylim_l = [
            0.9 * min(prediction[f'S_1_li'].min(), prediction[f'S_2_li'].min(), data[f'S_1_li'].min(), data[f'S_2_li'].min()), 
            1.1 * max(prediction[f'S_1_li'].max(), prediction[f'S_2_li'].max(), data[f'S_1_li'].max(), data[f'S_2_li'].max()),
        ]

        plot_data2scan(prediction[f'tS_1_ao'], prediction[f'S_1_ao'], data['tS_1_ao'], data['S_1_ao'], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(prediction[f'tS_1_li'], prediction[f'S_1_li'], data['tS_1_li'], data['S_1_li'], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(prediction[f'tS_2_ao'], prediction[f'S_2_ao'], data['tS_2_ao'], data['S_2_ao'], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(prediction[f'tS_2_li'], prediction[f'S_2_li'], data['tS_2_li'], data['S_2_li'], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c[0,:], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(prediction['C_1_ao'].max(), prediction['C_2_ao'].max())]
        else:
            ylim = [0, clim[0] * 1000]

        plot_conc_aorta(prediction['tC_1'], prediction['C_1_ao'], ax3, xlim, ylim)
        plot_conc_aorta(prediction['tC_2'], prediction['C_2_ao'], ax4, xlim, ylim)

        def plot_conc_liver(t, C, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * C[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax.plot(t / 60, 1000 * C[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax.plot(t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')       
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(prediction['C_1_li'].max(), prediction['C_2_li'].max())]
        else:
            ylim = [0, clim[1] * 1000]

        plot_conc_liver(prediction['tC_1'], prediction['C_1_li'], ax7, xlim, ylim)
        plot_conc_liver(prediction['tC_2'], prediction['C_2_li'], ax8, xlim, ylim)

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()


    def cost(self, data: dict, metric: str = 'NRMS', nfree=None) -> float:
        time = (data['tS_1_ao'], data['tS_2_ao'], data['tS_1_li'], data['tS_2_li'])
        signal = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])

        pred = self._predict(time)
        signal = np.concatenate([s.reshape(-1) for s in signal])
        signal_pred = np.concatenate(pred)
        return loss(signal_pred, signal, metric, nfree)




    # WIP below here



#     def export_params(self, sdev=None, desc=False) -> dict:
#         """Parameters with values, definition and units"""
#         pars_deriv, sdev_deriv = _deriv_params(self._pars, sdev)
#         if desc:
#             pars_deriv = pars_deriv | self._desc()
#         pars = self._pars | pars_deriv
#         sdev = sdev | sdev_deriv if sdev is not None else sdev_deriv
#         return export_params(pars, sdev=sdev, lexicon=QUANTITIES)


#     def _desc(self):
#         t = self._time_dict() 
#         C = self._conc_dict()
#         R1, _ = self._relax_dict()
#         S = self._signal_dict()

#         pars = {}

#         for visit in ['ctrl', 'drug']:

#             # Compute AUC over 3hrs
#             BAT = self._pars[f'{visit[0]}_BAT']

#             tAUCb = (BAT < t[visit, 'aorta']) & (t[visit, 'aorta'] < BAT + 180 * 60)
#             tAUCl = (BAT < t[visit, 'liver']) & (t[visit, 'liver'] < BAT + 180 * 60)
#             AUC_Cb = np.trapezoid(C[visit, 'aorta'][tAUCb], t[visit, 'aorta'][tAUCb]) 
#             AUC_Cl = np.trapezoid(C[visit, 'liver'].sum(axis=0)[tAUCl], t[visit, 'liver'][tAUCl])

#             # Compute AUC over 35min
#             tAUCb = (BAT < t[visit, 'aorta']) & (t[visit, 'aorta'] < BAT + 35 * 60)
#             tAUCl = (BAT < t[visit, 'liver']) & (t[visit, 'liver'] < BAT + 35 * 60)
#             AUC35_Cb = np.trapezoid(C[visit, 'aorta'][tAUCb], t[visit, 'aorta'][tAUCb]) 
#             AUC35_Cl = np.trapezoid(C[visit, 'liver'].sum(axis=0)[tAUCl], t[visit, 'liver'][tAUCl])

#             # Compute relative enhancement at 20mins
#             tRE = BAT + 20*60
#             R1b = R1[visit, 'aorta']
#             R1l = R1[visit, 'liver']
#             RE_R1b = (R1b[t[visit, 'aorta'] < tRE][-1] - R1b[0])/R1b[0]
#             RE_R1l = (R1l[t[visit, 'liver'] < tRE][-1] - R1l[0])/R1l[0]

#             S0b = np.mean(S[visit, 'aorta'][t[visit, 'aorta'] < BAT - 30])
#             S0l = np.mean(S[visit, 'liver'][t[visit, 'liver'] < BAT - 30])
#             RE_Sb = (S[visit, 'aorta'][t[visit, 'aorta'] < tRE][-1] - S0b)/S0b
#             RE_Sl = (S[visit, 'liver'][t[visit, 'liver'] < tRE][-1] - S0l)/S0l

#             pars = pars | {
#                 f'{visit[0]}_AUC_Cb' : AUC_Cb, 
#                 f'{visit[0]}_AUC_Cl': AUC_Cl,
#                 f'{visit[0]}_AUC35_Cb': AUC35_Cb,
#                 f'{visit[0]}_AUC35_Cl': AUC35_Cl, 
#                 f'{visit[0]}_RE_R1b': RE_R1b,
#                 f'{visit[0]}_RE_R1l': RE_R1l,
#                 f'{visit[0]}_RE_Sb': RE_Sb, 
#                 f'{visit[0]}_RE_Sl': RE_Sl,
#             } 

#         return pars



# def _div(a, b):
#     with np.errstate(divide='ignore'):
#         return np.divide(a, b)
    
# def _deriv_params(p, sdev=None):

#     fCO_l = p[f'fCO_l']
#     Fb = fCO_l * p[f'CO'] / p[f'c_vol']

#     Fpl = Fb * (1 - p['H'])
#     Te = p[f've'] / Fpl

#     c_El = p[f'c_khe'] / (p[f'c_khe'] + Fpl)
#     d_El = p[f'd_khe'] / (p[f'd_khe'] + Fpl)

#     CL = p[f'GFR']
#     Fpk = (1 - fCO_l) * p[f'CO'] * (1 - p['H'])
#     Eg = CL / (CL + Fpk)

#     # c_Eb
#     CL = p['c_khe'] * p[f'c_vol'] + p[f'GFR']
#     Eb = CL / (CL + p[f'CO'] * (1 - p['H']))
#     c_Eb = np.mean(Eb)
#     # c_Eb = p[f'c_Eb']

#     # d_Eb
#     CL = p['d_khe'] * p[f'd_vol'] + p[f'GFR']
#     Eb = CL / (CL + p[f'CO'] * (1 - p['H']))
#     d_Eb = np.mean(Eb)
#     # d_Eb = p[f'd_Eb']
    
#     vh = 1 - p[f've'] / (1 - p['H'])
#     c_khe = p['c_khe']
#     #c_kbh = vh * p['c_Kbh']
#     c_kbh = p['c_kbh']

#     d_khe = p['d_khe'] 
#     d_kbh = p['d_kbh']
#     #d_kbh = vh * p['d_Kbh']
#     c_CL = c_khe * p['c_vol']
#     d_CL = d_khe * p['d_vol']
#     p_deriv = {
#         'c_Th': _div(vh, c_kbh),
#         'd_Th': _div(vh, d_kbh),
#         'c_El': c_El,
#         'd_El': d_El,
#         'Eg': Eg,
#         'Te': Te,
#         'Fb': Fb,
#         'c_Eb': c_Eb,
#         'd_Eb': d_Eb,
#         'r_Eb': _div(d_Eb - c_Eb, c_Eb),
#         'a_Eb': d_Eb - c_Eb,
#         'vh': vh,
#         'c_kbh': c_kbh,
#         'd_kbh': d_kbh,
#         'c_CL': c_CL,
#         'd_CL': d_CL,
#         'r_khe': _div(d_khe - c_khe, c_khe),
#         'r_kbh': _div(d_kbh - c_kbh, c_kbh),
#         'r_CL': _div(d_CL - c_CL, c_CL),
#         'a_khe': d_khe - c_khe,
#         'a_kbh': d_kbh - c_kbh,
#         'a_CL': d_CL - c_CL,
#     }
#     sd_deriv = {}
#     if sdev is not None:
#         pass
#     return p_deriv, sd_deriv