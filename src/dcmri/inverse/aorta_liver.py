# +--------------------------------------------------------------------------------------------------+
# |                             InverseAortaLiver - all configs (n = 20)                             |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | inflow            | none, pool                                                      | none       |
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                               |            |
# | tof_corr          | False, True                                                     | False      |
# | magnitude         | False, True                                                     | True       |
# | trigger           | False, True                                                     | False      |
# | calibrate         | False, True                                                     | False      |
# | water_exchange    | F, N, R                                                         | F          |
# | baseline          | literature, measured                                            | literature |
# | bolus             | double, single                                                  | single     |
# | heartlung         | chain, comp, pfcomp                                             | pfcomp     |
# | organs            | 2cxm, comp                                                      | comp       |
# | lagut             | comp, pass, plucom                                              | comp       |
# | liver             | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF, 1I-IC-U                       | 1I-EC      |
# | non_stationary    | E, None, U, UE                                                  | None       |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_li  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_li  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_li | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train_bat
from dcmri.forward.aorta_liver import ForwardAortaLiver
from dcmri.kinetics.functions_liver import dpars_liver

configs = deepcopy(ForwardAortaLiver.configs)
defaults = deepcopy(ForwardAortaLiver.defaults)

configs['bolus'].discard('dual') # Only meaningful for split protocols

ROIS = ['ao', 'li']

class InverseAortaLiver(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'E_li', 'dose_1', 'TP', 'Ei_li', 'field_strength', 'me', 'B1corr_ao', 'ffa', 'tS_li', 'T_gu', 'dt', 'TR', 'tS_ao', 'vol_ao', 'kf_e2h', 'rate', 'TD', 'TA', 'dose', 'CO', 'R1_b', 'tacq', 'ki_e2h', 'PA', 'Tf_h', 'T_h', 'S_ao', 'S0_ao', 'S_li', 'v_e_li', 'Ti_h', 'TE2', 'D_hl', 'GFR', 'R1_li', 'bdel', 'S0_li', 'iStrig_ao', 'Nk0', 'pfree', 'R1_h', 'NSR_li', 'E_or', 'tstart', 'SA', 'v_h', 'R1_e', 'T_b_or', 'rate_1', 'iz', 'T_hl', 'T_la', 'Ef_li', 'agent', 'T_e_or', 'Nph', 'vol_li', 'fCO_li', 'FA', 'Nz', 'iStrig_li', 'weight', 'PSw', 'dose_tolerance', 'B1corr_li', 'nb', 'BAT', 'NSR_ao', 'v_li', 'rate_2', 'H', 'dose_2', 'k_e2h', 'TE', 'TE1', 'TF'}
    _all_outputs = {'pder', 'loss', 'pcov', 'psdev', 'popt'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = ForwardAortaLiver(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        nt = {roi: len(time[i]) for i, roi in enumerate(ROIS)}
        return tuple([pred[f'S_{roi}'][:, :, :nt[roi]].reshape(-1) for roi in ROIS])

    def _preproc(self, p):
        # Reshape signal if needed
        for roi in ROIS:
            if p[f'S_{roi}'].ndim == 1:
                p[f'S_{roi}'] = p[f'S_{roi}'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            for roi in ROIS:
                p[f"Scal_{roi}"] = p[f"S_{roi}"][..., :p['nb']]
                p[f'iScal_{roi}'] = np.arange(p['nb'])

    def _pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        pfree |= {'BAT'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def _pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['phys', 'body']}
        p = dpars_liver(p, kinetics=self.config['liver'])
        return p

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  
        self._preproc(p)

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = tuple([data[f'tS_{roi}'] for roi in ROIS])
        signal = tuple([data[f'S_{roi}'].reshape(-1) for roi in ROIS])
        p = train_bat(self._predict, time, signal, p, p['pfree'], **kwargs)

        p['pder'] = self._pder(p | p['popt'])

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'pfree'}
        for roi in ROIS:
            inputs |= {f'tS_{roi}', f'S_{roi}'}
        if self.config['calibrate']:
            inputs |= {'nb'}
            for roi in ROIS:
                inputs -= {f'Scal_{roi}', f'iScal_{roi}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'pder', 'loss'}
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': {'CO': (10, 300), 'BAT': (-60, 60)},
        }
        for roi in ROIS:        
            p |= {
                f'tS_{roi}': pred[f'tS_{roi}'],
                f'S_{roi}': pred[f'S_{roi}'], 
            }
        return self.input_data(p, data)

    def plot(self, data: dict, xlim=None, fname=None, show=True):
        p = self.map_data(data)
        self._preproc(p)

        pred = self.forward(p)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        xlim = np.array(xlim) / 60
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)
        
        # Plot signals
        def plot_data(t, s, ti, si, ax, clr):
            ax.set_title('MRI Signal Prediction')
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=clr[0], alpha=0.5, label='Data')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=clr[1], linewidth=3, label='Prediction')                
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Signal (a.u.)')
            ax.legend()

        plot_data(pred['tS_ao'], pred['S_ao'], data['tS_ao'], data['S_ao'], ax1, ['lightcoral', 'darkred'])
        plot_data(pred['tS_li'], pred['S_li'], data['tS_li'], data['S_li'], ax3, ['cornflowerblue', 'darkblue'])
        
        # Plot concentrations
        ax2.set(ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax2.plot(pred['tC'] / 60, 1000 * pred['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
        ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
        ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()