# +--------------------------------------------------------------------------------------------------+
# |                       InverseAortaLiverDynamicDrug - all configs (n = 20)                        |
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

# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            InverseAortaLiverDynamicDrug - all inputs (n = 133)                                                            |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                     | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT_1          | sec        | 1st bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_3          | sec        | 3rd bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_4          | sec        | 4th bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
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
# | NSR_3_ao       |            | 3rd noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_3_li       |            | 3rd noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_4_ao       |            | 4th noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_4_li       |            | 4th noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_1_ao        | a.u.       | 1st signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | a.u.       | 1st signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | a.u.       | 2nd signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | a.u.       | 2nd signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_3_ao        | a.u.       | 3rd signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_3_li        | a.u.       | 3rd signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_4_ao        | a.u.       | 4th signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_4_li        | a.u.       | 4th signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S_1_ao         | a.u.       | 1st signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_1_li         | a.u.       | 1st signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_ao         | a.u.       | 2nd signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_2_li         | a.u.       | 2nd signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_3_ao         | a.u.       | 3rd signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_3_li         | a.u.       | 3rd signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_4_ao         | a.u.       | 4th signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_4_li         | a.u.       | 4th signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | iStrig_1_ao    |            | 1st indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_1_li    |            | 1st indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_2_ao    |            | 2nd indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_2_li    |            | 2nd indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_3_ao    |            | 3rd indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_3_li    |            | 3rd indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_4_ao    |            | 4th indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_4_li    |            | 4th indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | nb             | a.u.       | number of baseline time points                                           | Signal          | 1          |               |       |           |
# | pfree          | a.u.       | set of free parameters                                                   | Signal          | 1          |               |       |           |
# | tS_1_ao        | sec        | 1st signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_1_li        | sec        | 1st signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# | tS_2_ao        | sec        | 2nd signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_2_li        | sec        | 2nd signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# | tS_3_ao        | sec        | 3rd signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_3_li        | sec        | 3rd signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# | tS_4_ao        | sec        | 4th signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_4_li        | sec        | 4th signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
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
# | tacq_3         | sec        | 3rd acquisition duration                                                 | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tacq_4         | sec        | 4th acquisition duration                                                 | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart_1       | sec        | 1st start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_2       | sec        | 2nd start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_3       | sec        | 3rd start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_4       | sec        | 4th start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_1_ao    |            | 1st B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_1_li    |            | 1st B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_ao    |            | 2nd B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_li    |            | 2nd B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_3_ao    |            | 3rd B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_3_li    |            | 3rd B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_4_ao    |            | 4th B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_4_li    |            | 4th B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
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
# |                                InverseAortaLiverDynamicDrug - all outputs (n = 4)                               |
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
from dcmri.forward.aorta_liver_dynamic_drug import ForwardAortaLiverDynamicDrug as Forward

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

ROIS = ['ao', 'li']
SCANS = [1, 2, 3, 4]

class InverseAortaLiverDynamicDrug(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'S_4_li', 'B1corr_2_ao', 'Ei_2_li', 'S_3_li', 'dt', 'tstart_4', 'iStrig_4_li', 'kf_1_e2h', 'tacq_1', 'Ti_2_h', 'nb', 'iStrig_3_li', 'B1corr_3_li', 'k_1_e2h', 'T_b_or', 'iStrig_4_ao', 'GFR', 'T_1_h', 'TE1', 'T_hl', 'BAT_1', 'kf_2_e2h', 'T_e_or', 'weight', 'vol_2_li', 'v_li', 'iStrig_2_li', 'vol_2_ao', 'k_2_e2h', 'NSR_1_ao', 'Tf_2_h', 'B1corr_1_ao', 'S0_3_ao', 'ki_2_e2h', 'agent', 'S0_2_li', 'SA', 'rate_4', 'TE2', 'tS_1_ao', 'dose_4', 'NSR_3_ao', 'E_or', 'tS_4_li', 'TD', 'iStrig_3_ao', 'bdel', 'vol_1_li', 'CO', 'H', 'BAT_4', 'rate_2', 'dose_2', 'B1corr_1_li', 'S_3_ao', 'S_2_li', 'v_e_li', 'B1corr_4_li', 'S_4_ao', 'tS_4_ao', 'fCO_li', 'R1_b', 'S_2_ao', 'tacq_4', 'E_2_li', 'field_strength', 'PSw', 'T_2_h', 'B1corr_2_li', 'me', 'Nz', 'TA', 'iStrig_1_li', 'tS_1_li', 'v_h', 'BAT_2', 'tstart_1', 'B1corr_3_ao', 'tstart_2', 'NSR_2_ao', 'Nph', 'tacq_2', 'BAT_3', 'NSR_3_li', 'Nk0', 'S0_1_ao', 'TF', 'vol_1_ao', 'tstart_3', 'Ti_1_h', 'PA', 'tS_2_li', 'iStrig_2_ao', 'S_1_li', 'Ei_1_li', 'tS_2_ao', 'ki_1_e2h', 'S0_1_li', 'tS_3_li', 'ffa', 'dose_1', 'pfree', 'Ef_2_li', 'TP', 'dose_tolerance', 'dose_3', 'Ef_1_li', 'T_gu', 'R1_e', 'NSR_2_li', 'TE', 'E_1_li', 'rate_1', 'iStrig_1_ao', 'S_1_ao', 'T_la', 'B1corr_4_ao', 'tacq_3', 'S0_4_ao', 'NSR_1_li', 'S0_4_li', 'tS_3_ao', 'NSR_4_ao', 'Tf_1_h', 'iz', 'NSR_4_li', 'R1_h', 'TR', 'S0_2_ao', 'rate_3', 'FA', 'S0_3_li', 'D_hl'}
    _all_outputs = {'popt', 'psdev', 'loss', 'pcov'}

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

            pred['S_3_ao'].reshape(-1), 
            pred['S_4_ao'].reshape(-1), 
            pred['S_3_li'].reshape(-1), 
            pred['S_4_li'].reshape(-1),
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
        # bat = estimate_bat(data['tS_1_ao'], data['S_1_ao'], p['nb'])
        # p['BAT_1'] = max(bat - p['T_hl'], 0)

        if self.config['bolus'] in ['single', 'double']:
            # bat = estimate_bat(data['tS_3_ao'], data['S_3_ao'], p['nb'])
            # p['BAT_2'] = max(bat - p['T_hl'], 0)
            bats = ['BAT_1', 'BAT_2']
        else:
            # bat = estimate_bat(data['tS_2_ao'], data['S_2_ao'], p['nb'])
            # p['BAT_2'] = max(bat - p['T_hl'], 0)
            # bat = estimate_bat(data['tS_3_ao'], data['S_3_ao'], p['nb'])
            # p['BAT_3'] = max(bat - p['T_hl'], 0)
            # bat = estimate_bat(data['tS_4_ao'], data['S_4_ao'], p['nb'])
            # p['BAT_4'] = max(bat - p['T_hl'], 0)
            bats = ['BAT_1', 'BAT_2', 'BAT_3', 'BAT_4']

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        signal = (
            data['S_1_ao'], 
            data['S_2_ao'], 
            data['S_1_li'], 
            data['S_2_li'], 
            
            data['S_3_ao'], 
            data['S_4_ao'], 
            data['S_3_li'], 
            data['S_4_li'],
        )
        self._pars = p
        p = train_bat(self._predict, None, signal, p, p['pfree'], bats=bats, **kwargs)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'nb', 'pfree'}
        for roi in ROIS:
            for scan in SCANS:
                inputs |= {f'tS_{scan}_{roi}', f'S_{scan}_{roi}'}
        # inputs -= {f'BAT_{i}' for i in [1, 2, 3, 4]}
        # for roi in ROIS:
        #     for scan in SCANS:
                inputs -= {f'Scal_{scan}_{roi}', f'iScal_{scan}_{roi}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'loss'}
    
    def dummy_data(self, data: dict=None): 
        pfree = {
            'CO': (10, 300), 
            'BAT': (-60, 60),  
            'BAT_1': (-60, 60), 
            'BAT_2': (-60, 60),
            'BAT_3': (-60, 60), 
            'BAT_4': (-60, 60)
        }

        p = self.init_data()
        p |= self.forward.dummy_data()

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
        if self.config['bolus'] in ['single', 'double']:
            pfree |= {'BAT_1', 'BAT_2'}
        else:
            pfree |= {'BAT_1', 'BAT_2', 'BAT_3', 'BAT_4'}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
            pfree |= {'S0_3_ao', 'S0_3_li', 'S0_4_ao', 'S0_4_li'}
        return {p: get_quantity(p)['bounds'] for p in pfree}