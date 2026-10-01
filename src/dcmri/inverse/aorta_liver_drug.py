# +--------------------------------------------------------------------------------------------------+
# |                           InverseAortaLiverDrug - all configs (n = 20)                           |
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
# | liver             | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF                                | 1I-EC      |
# | non_stationary    | E, None, U, UE                                                  | None       |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_li  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_li  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_li | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                InverseAortaLiverDrug - all inputs (n = 103)                                                               |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                     | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT_1          | sec        | 1st bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                                              | Indicator       | gadoterate |               |       |           |
# | bdel           | sec        | delay in a double injection                                              | Indicator       | 30         | (-30, 30)     |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_3         | mL/kg      | 3rd contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_4         | mL/kg      | 4th contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_3         | mL/s       | 3rd injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_4         | mL/s       | 4th injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_1_ao       |            | 1st noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_1_li       |            | 1st noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_ao       |            | 2nd noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_li       |            | 2nd noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_1_ao        | a.u.       | 1st signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | a.u.       | 1st signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | a.u.       | 2nd signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | a.u.       | 2nd signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S_1_ao         | a.u.       | 1st signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_1_li         | a.u.       | 1st signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_ao         | a.u.       | 2nd signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_li         | a.u.       | 2nd signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | iStrig_1_ao    |            | 1st indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_1_li    |            | 1st indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_2_ao    |            | 2nd indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_2_li    |            | 2nd indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | nb             | a.u.       | number of baseline time points                                           | Signal          | 1          |               |       |           |
# | pfree          | a.u.       | set of free parameters                                                   | Signal          | 1          |               |       |           |
# | tS_1_ao        | sec        | 1st signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_1_li        | sec        | 1st signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# | tS_2_ao        | sec        | 2nd signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_2_li        | sec        | 2nd signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                                               | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space                  | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space                                | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                            | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                             | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg        | saturation Slab Flip Angle                                               | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                                         | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                                           | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                                                | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                                 | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                                | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                                        | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                                          | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                                  | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition                                | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq_1         | sec        | 1st acquisition duration                                                 | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tacq_2         | sec        | 2nd acquisition duration                                                 | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart_1       | sec        | 1st start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_2       | sec        | 2nd start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_1_ao    |            | 1st B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_1_li    |            | 1st B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_ao    |            | 2nd B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_li    |            | 2nd B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                                   | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_e           | Hz         | tissue R1 in extracellular space                                         | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_h           | Hz         | tissue R1 in hepatocytes                                                 | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                                | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                                           | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs                           | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_1_li         |            | 1st extraction fraction in the liver                                     | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_2_li         |            | 2nd extraction fraction in the liver                                     | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_or           |            | extraction fraction in the organs                                        | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | Ef_1_li        |            | 1st final extraction fraction in the liver                               | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ef_2_li        |            | 2nd final extraction fraction in the liver                               | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_1_li        |            | 1st initial extraction fraction in the liver                             | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_2_li        |            | 2nd initial extraction fraction in the liver                             | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | GFR            | mL/sec     | glomerular filtration rate                                               | Physiological   | 2          | (0, 10)       |       |           |
# | H              |            | hematocrit                                                               | Physiological   | 0.45       | (0, 1)        |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                                  | Physiological   | 0.03       | (0, 100)      |       |           |
# | TF             | sec        | inflow time                                                              | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_1_h          | sec        | 1st mean transit time in hepatocytes                                     | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_2_h          | sec        | 2nd mean transit time in hepatocytes                                     | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                                 | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_or         | sec        | mean transit time in extracellular space of the organs                   | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec        | mean transit time in the gut                                             | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                                 | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec        | mean transit time in the liver artery                                    | Physiological   | 30         | (0.1, 60)     |       |           |
# | Tf_1_h         | sec        | 1st final mean transit time in hepatocytes                               | Physiological   | 1800       | (600, 36000)  |       |           |
# | Tf_2_h         | sec        | 2nd final mean transit time in hepatocytes                               | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_1_h         | sec        | 1st initial mean transit time in hepatocytes                             | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_2_h         | sec        | 2nd initial mean transit time in hepatocytes                             | Physiological   | 1800       | (600, 36000)  |       |           |
# | fCO_li         |            | fraction of the cardiac output in the liver                              | Physiological   | 0.1        | (0, 0.5)      |       |           |
# | ffa            |            | arterial flow fraction                                                   | Physiological   | 0.2        | (0, 1)        |       |           |
# | k_1_e2h        | mL/sec/cm3 | 1st tissue transfer rate from extracellular space to hepatocytes         | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | k_2_e2h        | mL/sec/cm3 | 2nd tissue transfer rate from extracellular space to hepatocytes         | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_1_e2h       | mL/sec/cm3 | 1st final tissue transfer rate from extracellular space to hepatocytes   | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_2_e2h       | mL/sec/cm3 | 2nd final tissue transfer rate from extracellular space to hepatocytes   | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_1_e2h       | mL/sec/cm3 | 1st initial tissue transfer rate from extracellular space to hepatocytes | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_2_e2h       | mL/sec/cm3 | 2nd initial tissue transfer rate from extracellular space to hepatocytes | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | v_e_li         | mL/cm3     | volume fraction in extracellular space of the liver                      | Physiological   | 0.3        | (0.01, 0.6)   |       |           |
# | v_h            | mL/cm3     | volume fraction in hepatocytes                                           | Physiological   | 1          | (0, 1)        |       |           |
# | v_li           | mL/cm3     | volume fraction in the liver                                             | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                                           | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec        | pseudo-continuous time step                                              | Hyperparameters | 0.5        |               |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_1_ao       | cm3        | 1st ROI volume in the aorta                                              | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_1_li       | cm3        | 1st ROI volume in the liver                                              | Whole-body      | 1000       | (0, 10000)    |       |           |
# | vol_2_ao       | cm3        | 2nd ROI volume in the aorta                                              | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_2_li       | cm3        | 2nd ROI volume in the liver                                              | Whole-body      | 1000       | (0, 10000)    |       |           |
# | weight         | kg         | body weight                                                              | Whole-body      | 70         | (0, 300)      |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                   InverseAortaLiverDrug - all outputs (n = 4)                                   |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | Key   | Unit | Name                                           | Group           | Init | Bounds | DICOM | OSIPI |
# +-------+------+------------------------------------------------+-----------------+------+--------+-------+-------+
# | loss  | a.u. | loss value of optimized model                  | Signal          | 1    |        |       |       |
# | pcov  | a.u. | dictionary with covariances of free parameters | Signal          | 1    |        |       |       |
# | popt  | a.u. | dictionary of optimized free parameter values  | Signal          | 1    |        |       |       |
# | psdev | a.u. | dictionary with parameter standard deviations  | Signal          | 1    |        |       |       |
# +-----------------------------------------------------------------------------------------------------------------+


from copy import deepcopy
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train_bat
from dcmri.inverse.lib import estimate_bat
from dcmri.forward.aorta_liver_drug import ForwardAortaLiverDrug as Forward

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

configs['bolus'].discard('dual') # Only meaningful for split protocols

ROIS = ['ao', 'li']
SCANS = [1, 2]

class InverseAortaLiverDrug(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'T_e_or', 'S0_2_li', 'tS_2_li', 'k_2_e2h', 'S0_1_li', 'T_b_or', 'FA', 'iz', 'vol_1_li', 'CO', 'TE2', 'PSw', 'S0_2_ao', 'T_gu', 'dose_4', 'GFR', 'NSR_1_li', 'H', 'dose_1', 'T_hl', 'S0_1_ao', 'BAT_1', 'tstart_2', 'Nph', 'Nz', 'rate_3', 'dose_tolerance', 'tacq_1', 'field_strength', 'E_1_li', 'TF', 'v_li', 'TA', 'tS_2_ao', 'Ti_2_h', 'D_hl', 'dt', 'T_1_h', 'NSR_2_li', 'Ei_1_li', 'Tf_1_h', 'T_2_h', 'TE', 'R1_h', 'bdel', 'iStrig_1_ao', 'Nk0', 'ffa', 'agent', 'tacq_2', 'weight', 'S_2_li', 'vol_1_ao', 'rate_1', 'vol_2_ao', 'tS_1_li', 'TE1', 'S_1_ao', 'B1corr_2_ao', 'fCO_li', 'tstart_1', 'ki_2_e2h', 'Ef_1_li', 'TP', 'E_or', 'S_2_ao', 'B1corr_1_li', 'iStrig_2_ao', 'TR', 'tS_1_ao', 'vol_2_li', 'iStrig_1_li', 'nb', 'NSR_1_ao', 'me', 'v_h', 'rate_4', 'v_e_li', 'ki_1_e2h', 'TD', 'T_la', 'B1corr_2_li', 'kf_2_e2h', 'k_1_e2h', 'E_2_li', 'NSR_2_ao', 'rate_2', 'Ti_1_h', 'iStrig_2_li', 'R1_e', 'Ei_2_li', 'kf_1_e2h', 'SA', 'B1corr_1_ao', 'BAT_2', 'Ef_2_li', 'Tf_2_h', 'dose_2', 'S_1_li', 'PA', 'R1_b', 'pfree', 'dose_3'}
    _all_outputs = {'loss', 'pcov', 'popt', 'psdev'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        return (
            pred['S_1_ao'].reshape(-1), 
            pred['S_2_ao'].reshape(-1), 
            pred['S_1_li'].reshape(-1), 
            pred['S_2_li'].reshape(-1), 
        )

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  

        # Set calibration signal
        if self.config['calibrate']:
            for roi in ROIS:
                for scan in SCANS:
                    p[f"Scal_{scan}_{roi}"] = p[f"S_{scan}_{roi}"][..., :p['nb']]
                    p[f'iScal_{scan}_{roi}'] = np.arange(p['nb'])

        # # Estimate BAT
        # bat1 = estimate_bat(data['tS_1_ao'], data['S_1_ao'], p['nb'])
        # bat2 = estimate_bat(data['tS_2_ao'], data['S_2_ao'], p['nb'])

        # p['BAT_1'] = max(bat1 - p['T_hl'], 0)
        # p['BAT_2'] = max(bat2 - p['T_hl'], 0)

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        signal = (data['S_1_ao'], data['S_2_ao'], data['S_1_li'], data['S_2_li'])
        p = train_bat(self._predict, None, signal, p, p['pfree'], bats=['BAT_1', 'BAT_2'], **kwargs)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'nb', 'pfree'}
        for roi in ROIS:
            for scan in SCANS:
                inputs |= {f'tS_{scan}_{roi}', f'S_{scan}_{roi}'}
        # inputs -= {'BAT_1', 'BAT_2'}
        # for roi in ROIS:
        #     for scan in SCANS:
                inputs -= {f'Scal_{scan}_{roi}', f'iScal_{scan}_{roi}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'loss'}
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pfree = {'CO': (10, 300), 'BAT_1': (-60, 60), 'BAT_2': (-60, 60)}

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data(pfree),
        }

        for roi in ROIS:  
            for scan in SCANS:      
                p |= {
                    f'tS_{scan}_{roi}': pred[f'tS_{scan}_{roi}'],
                    f'S_{scan}_{roi}': pred[f'S_{scan}_{roi}'], 
                }
            
        return self.input_data(p, data)

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        pfree |= {'BAT_1', 'BAT_2'}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
        return {p: get_quantity(p)['bounds'] for p in pfree}