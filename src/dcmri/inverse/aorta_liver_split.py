# +--------------------------------------------------------------------------------------------------+
# |                         InverseAortaLiverSplit - all configs (n = 20)                          |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                               |            |
# | tof_corr          | False, True                                                     | False      |
# | inflow            | none, pool                                                      | none       |
# | magnitude         | False, True                                                     | True       |
# | trigger           | False, True                                                     | False      |
# | calibrate         | False, True                                                     | False      |
# | water_exchange    | F, N, R                                                         | F          |
# | baseline          | literature, measured                                            | literature |
# | bolus             | double, dual, single                                            | single     |
# | heartlung         | chain, comp, pfcomp                                             | pfcomp     |
# | organs            | 2cxm, comp                                                      | comp       |
# | lagut             | comp, pass, plucom                                              | comp       |
# | liver             | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF                                | 1I-EC      |
# | non_stationary    | E, None, U, UE                                                  | None       |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_li  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_li  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_li | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             InverseAortaLiverSplit - all inputs (n = 91)                                                            |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                 | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT            | sec        | bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_1          | sec        | 1st bolus arrival time                                               | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                               | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                                          | Indicator       | gadoterate |               |       |           |
# | bdel           | sec        | delay in a double injection                                          | Indicator       | 30         | (-30, 30)     |       |           |
# | dose           | mL/kg      | contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s       | injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_1_ao       |            | 1st noise-to-signal ratio in the aorta                               | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_1_li       |            | 1st noise-to-signal ratio in the liver                               | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_ao       |            | 2nd noise-to-signal ratio in the aorta                               | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_li       |            | 2nd noise-to-signal ratio in the liver                               | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_1_ao        | a.u.       | 1st signal scaling factor in the aorta                               | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | a.u.       | 1st signal scaling factor in the liver                               | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | a.u.       | 2nd signal scaling factor in the aorta                               | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | a.u.       | 2nd signal scaling factor in the liver                               | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S_1_ao         | a.u.       | 1st signal in the aorta                                              | Signal          | 1.0        | (0, 5)        |       |           |
# | S_1_li         | a.u.       | 1st signal in the liver                                              | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_ao         | a.u.       | 2nd signal in the aorta                                              | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_li         | a.u.       | 2nd signal in the liver                                              | Signal          | 1.0        | (0, 5)        |       |           |
# | iStrig_1_ao    |            | 1st indices of the signal trigger in the aorta                       | Signal          | None       |               |       |           |
# | iStrig_1_li    |            | 1st indices of the signal trigger in the liver                       | Signal          | None       |               |       |           |
# | iStrig_2_ao    |            | 2nd indices of the signal trigger in the aorta                       | Signal          | None       |               |       |           |
# | iStrig_2_li    |            | 2nd indices of the signal trigger in the liver                       | Signal          | None       |               |       |           |
# | nb             | a.u.       | number of baseline time points                                       | Signal          | 1          |               |       |           |
# | pfree          | a.u.       | set of free parameters                                               | Signal          | 1          |               |       |           |
# | tS_1_ao        | sec        | 1st signal time points in the aorta                                  | Signal          | 0.0        |               |       |           |
# | tS_1_li        | sec        | 1st signal time points in the liver                                  | Signal          | 0.0        |               |       |           |
# | tS_2_ao        | sec        | 2nd signal time points in the aorta                                  | Signal          | 0.0        |               |       |           |
# | tS_2_li        | sec        | 2nd signal time points in the liver                                  | Signal          | 0.0        |               |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                                           | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space              | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space                            | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                        | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                         | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg        | saturation Slab Flip Angle                                           | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                                     | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                                            | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                             | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                            | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                                    | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                                      | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                              | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition                            | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq_1         | sec        | 1st acquisition duration                                             | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tacq_2         | sec        | 2nd acquisition duration                                             | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart_1       | sec        | 1st start of the acquisition                                         | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_2       | sec        | 2nd start of the acquisition                                         | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_1_ao    |            | 1st B1-correction factor in the aorta                                | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_1_li    |            | 1st B1-correction factor in the liver                                | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_ao    |            | 2nd B1-correction factor in the aorta                                | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_li    |            | 2nd B1-correction factor in the liver                                | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                               | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_e           | Hz         | tissue R1 in extracellular space                                     | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_h           | Hz         | tissue R1 in hepatocytes                                             | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                            | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                                       | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs                       | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_li           |            | extraction fraction in the liver                                     | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_or           |            | extraction fraction in the organs                                    | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | Ef_li          |            | final extraction fraction in the liver                               | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_li          |            | initial extraction fraction in the liver                             | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | GFR            | mL/sec     | glomerular filtration rate                                           | Physiological   | 2          | (0, 10)       |       |           |
# | H              |            | hematocrit                                                           | Physiological   | 0.45       | (0, 1)        |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                              | Physiological   | 0.03       | (0, 100)      |       |           |
# | TF             | sec        | inflow time                                                          | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                             | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_or         | sec        | mean transit time in extracellular space of the organs               | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec        | mean transit time in the gut                                         | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_h            | sec        | mean transit time in hepatocytes                                     | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                             | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec        | mean transit time in the liver artery                                | Physiological   | 30         | (0.1, 60)     |       |           |
# | Tf_h           | sec        | final mean transit time in hepatocytes                               | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_h           | sec        | initial mean transit time in hepatocytes                             | Physiological   | 1800       | (600, 36000)  |       |           |
# | fCO_li         |            | fraction of the cardiac output in the liver                          | Physiological   | 0.1        | (0, 0.5)      |       |           |
# | ffa            |            | arterial flow fraction                                               | Physiological   | 0.2        | (0, 1)        |       |           |
# | k_e2h          | mL/sec/cm3 | tissue transfer rate from extracellular space to hepatocytes         | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_e2h         | mL/sec/cm3 | final tissue transfer rate from extracellular space to hepatocytes   | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_e2h         | mL/sec/cm3 | initial tissue transfer rate from extracellular space to hepatocytes | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | v_e_li         | mL/cm3     | volume fraction in extracellular space of the liver                  | Physiological   | 0.3        | (0.01, 0.6)   |       |           |
# | v_h            | mL/cm3     | volume fraction in hepatocytes                                       | Physiological   | 1          | (0, 1)        |       |           |
# | v_li           | mL/cm3     | volume fraction in the liver                                         | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                                       | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec        | pseudo-continuous time step                                          | Hyperparameters | 0.5        |               |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_ao         | cm3        | ROI volume in the aorta                                              | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_li         | cm3        | ROI volume in the liver                                              | Whole-body      | 1000       | (0, 10000)    |       |           |
# | weight         | kg         | body weight                                                          | Whole-body      | 70         | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                  InverseAortaLiverSplit - all outputs (n = 4)                                 |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | Key   | Unit | Name                                           | Group           | Init | Bounds | DICOM | OSIPI |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | loss  | a.u. | loss value of optimized model                  | Signal          | 1    |        |       |       |
# | pcov  | a.u. | dictionary with covariances of free parameters | Signal          | 1    |        |       |       |
# | popt  | a.u. | dictionary of optimized free parameter values  | Signal          | 1    |        |       |       |
# | psdev | a.u. | dictionary with parameter standard deviations  | Signal          | 1    |        |       |       |
# +-----------------------------------------------------------------------------------------------------------------+

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train_bat, train
from dcmri.forward.aorta_liver_split import ForwardAortaLiverSplit as Forward
from dcmri.kinetics.functions_liver import dpars_liver

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

ROIS = ['ao', 'li']
SCANS = [1, 2]

class InverseAortaLiverSplit(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'BAT_2', 'Ei_li', 'tS_2_li', 'T_e_or', 'T_b_or', 'iStrig_1_ao', 'S_2_li', 'Tf_h', 'GFR', 'tacq_2', 'dose', 'NSR_2_ao', 'agent', 'tacq_1', 'tS_1_li', 'T_h', 'S0_2_ao', 'B1corr_2_ao', 'fCO_li', 'H', 'E_li', 'B1corr_1_li', 'rate_1', 'TF', 'BAT_1', 'T_hl', 'S0_2_li', 'TE1', 'vol_li', 'dose_2', 'Nk0', 'rate_2', 'vol_ao', 'dose_tolerance', 'iStrig_2_ao', 'rate', 'nb', 'PSw', 'R1_h', 'tstart_1', 'Ti_h', 'R1_e', 'TE', 'tstart_2', 'tS_1_ao', 'CO', 'Nz', 'me', 'S0_1_li', 'bdel', 'Ef_li', 'Nph', 'iStrig_1_li', 'ki_e2h', 'B1corr_2_li', 'TD', 'B1corr_1_ao', 'S_2_ao', 'PA', 'NSR_2_li', 'kf_e2h', 'field_strength', 'SA', 'pfree', 'TR', 'k_e2h', 'TE2', 'weight', 'iz', 'T_la', 'R1_b', 'NSR_1_ao', 'FA', 'S_1_li', 'v_li', 'E_or', 'TP', 'dt', 'NSR_1_li', 'D_hl', 'ffa', 'BAT', 'T_gu', 'S_1_ao', 'v_h', 'tS_2_ao', 'dose_1', 'TA', 'iStrig_2_li', 'S0_1_ao', 'v_e_li'}
    _all_outputs = {'pcov', 'popt', 'psdev', 'pder', 'loss'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self._forward(self._pars)
        return (
            pred['S_1_ao'][:, :, :len(time[0])].reshape(-1), 
            pred['S_2_ao'][:, :, :len(time[1])].reshape(-1), 
            pred['S_1_li'][:, :, :len(time[2])].reshape(-1), 
            pred['S_2_li'][:, :, :len(time[3])].reshape(-1), 
        )

    def _preproc(self, p):
        # Reshape signal if needed
        for roi in ROIS:
            for scan in SCANS:
                if p[f'S_{scan}_{roi}'].ndim == 1:
                    p[f'S_{scan}_{roi}'] = p[f'S_{scan}_{roi}'].reshape(1, 1, -1)

        # Set calibration signal
        if self.config['calibrate']:
            for roi in ROIS:
                for scan in SCANS:
                    p[f'Scal_{scan}_{roi}'] = p[f'S_{scan}_{roi}'][..., :p[f'nb_{scan}']]
                    p[f'iScal_{scan}_{roi}'] = np.arange(p[f'nb_{scan}'])

    def _pfree(self):
        inputs = self._forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        pfree |= {p for p in ['BAT', 'BAT_1', 'BAT_2'] if p in inputs}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
        pfree -= {'v_li', 'H', 'GFR'}
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

        # if self.config['bolus'] == 'dual':
        #     bats = ['BAT_1', 'BAT_2']
        # else:
        #     bats = ['BAT']

        # Compute inverse
        self._pars = p
        time = (p['tS_1_ao'], p['tS_2_ao'], p['tS_1_li'], p['tS_2_li'])
        signal = (p['S_1_ao'], p['S_2_ao'], p['S_1_li'], p['S_2_li'])
        # output = train_bat(self._predict, time, signal, p, p['pfree'], bats=bats, **kwargs)
        output = train(self._predict, time, signal, p, p['pfree'], **kwargs)
        output['pder'] = self._pder(p | output['popt'])

        return self.map_results(output)

    def inputs(self) -> set:
        inputs = self._forward.mapped_inputs()
        inputs |= {'pfree'}
        for roi in ROIS:
            for scan in SCANS:
                inputs |= {f'tS_{scan}_{roi}', f'S_{scan}_{roi}'}
        if self.config['calibrate']:
            inputs |= {'nb_1', 'nb_2'}
            for roi in ROIS:
                for scan in SCANS:
                    inputs -= {f'Scal_{scan}_{roi}', f'iScal_{scan}_{roi}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'pder', 'loss'}
    
    def test_data(self, data: dict=None): 
        pfree = {
            'CO': (10, 300), 
            'BAT': (-60, 60),  
            'BAT_1': (-60, 60), 
            'BAT_2': (-60, 60)
        }

        p = self.init_data()
        p |= self._forward.test_data()

        pred = self._forward(p)
        p |= {
            'nb_1': 5,
            'nb_2': 5,
            'pfree': self._forward.filter_data(pfree)
        }
        for roi in ROIS:  
            for scan in SCANS:      
                p |= {
                    f'tS_{scan}_{roi}': pred[f'tS_{scan}_{roi}'],
                    f'S_{scan}_{roi}': pred[f'S_{scan}_{roi}'], 
                }
        
        return self.input_data(p, data)


    def plot(self, data: dict, xlim: list = None, fname: str = None, show=True):
        p = self.map_data(data)
        self._preproc(p)
        pred = self._forward(p)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
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
                    ax.plot(pred[f'tS_1_{roi}'] / 60, pred[f'S_1_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
                    ax.plot(pred[f'tS_2_{roi}'] / 60, pred[f'S_2_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan('ao',(p['tS_1_ao'], p['tS_2_ao']), (p['S_1_ao'], p['S_2_ao']), ax1, ['lightcoral', 'darkred'])
        _plot_data2scan('li',(p['tS_1_li'], p['tS_2_li']), (p['S_1_li'], p['S_2_li']), ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax2.plot(pred['tC'] / 60, 1000 * pred['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        if pred['C_li'].shape[0]==2:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        else:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()