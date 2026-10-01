from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.tools import get_bounds
from dcmri.utils.fit import loss
from dcmri.inverse.aorta_liver_drug import InverseAortaLiverDrug as Inverse


class AortaLiverDrug():
    @classmethod
    def all_configs(cls, sample: int = None, seed: int = None, valid=False):
        return Inverse.all_configs(sample, seed, valid)
    
    def __init__(self, state: dict=None, **config):
        self._inverse = Inverse(**config)
        self._forward = self._inverse.forward
        self._state = self._forward.dummy_data(state)

    def state(self):
        return deepcopy(self._state)

    def predict(self) -> np.ndarray:
        return self._forward(self._state)
    
    def train(self, data: dict, pfree:dict=None, bounds: dict=None, nb=5, **kwargs):
        # Get free parameters
        default_pfree = self._inverse.pfree()     
        pfree = get_bounds(pfree, bounds, free_pars=default_pfree)

        # Apply inverse model
        inputs = self._state | data | {'pfree': pfree, 'nb': nb}
        result = self._inverse(inputs, **kwargs)

        # Update state
        self._state |= result['popt']

        return result

    def cost(self, data: dict, metric: str='NRMS', nfree=None) -> float:
        pred = self._forward(self._state)

        signal_data = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])
        signal_pred = (pred['S_1_ao'], pred['S_2_ao'], pred['S_1_li'], pred['S_2_li'])

        signal_data = np.concatenate([s.reshape(-1) for s in signal_data])
        signal_pred = np.concatenate([s.reshape(-1) for s in signal_pred])

        return loss(signal_pred, signal_data, metric, nfree)

    def plot(self, data: dict, xlim=None, clim=None, fname=None, show=True):
        pred = self._forward(self._state)
        
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
            0.9 * min(pred[f'S_1_ao'].min(), pred[f'S_2_ao'].min(), data[f'S_1_ao'].min(), data[f'S_2_ao'].min()), 
            1.1 * max(pred[f'S_1_ao'].max(), pred[f'S_2_ao'].max(), data[f'S_1_ao'].max(), data[f'S_2_ao'].max()),
        ]
        ylim_l = [
            0.9 * min(pred[f'S_1_li'].min(), pred[f'S_2_li'].min(), data[f'S_1_li'].min(), data[f'S_2_li'].min()), 
            1.1 * max(pred[f'S_1_li'].max(), pred[f'S_2_li'].max(), data[f'S_1_li'].max(), data[f'S_2_li'].max()),
        ]

        plot_data2scan(pred[f'tS_1_ao'], pred[f'S_1_ao'], data['tS_1_ao'], data['S_1_ao'], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_1_li'], pred[f'S_1_li'], data['tS_1_li'], data['S_1_li'], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(pred[f'tS_2_ao'], pred[f'S_2_ao'], data['tS_2_ao'], data['S_2_ao'], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_2_li'], pred[f'S_2_li'], data['tS_2_li'], data['S_2_li'], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c[0,:], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(pred['C_1_ao'].max(), pred['C_2_ao'].max())]
        else:
            ylim = [0, clim[0] * 1000]

        plot_conc_aorta(pred['tC_1'], pred['C_1_ao'], ax3, xlim, ylim)
        plot_conc_aorta(pred['tC_2'], pred['C_2_ao'], ax4, xlim, ylim)

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
            ylim = [0, 1.1 * 1000 * max(pred['C_1_li'].max(), pred['C_2_li'].max())]
        else:
            ylim = [0, clim[1] * 1000]

        plot_conc_liver(pred['tC_1'], pred['C_1_li'], ax7, xlim, ylim)
        plot_conc_liver(pred['tC_2'], pred['C_2_li'], ax8, xlim, ylim)

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()






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