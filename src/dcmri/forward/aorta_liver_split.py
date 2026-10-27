# +--------------------------------------------------------------------------------------------------+
# |                              AortaLiverSplit - all configs (n = 20)                              |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS, ZTE-3D-SPGR-SSI              |            |
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
# | liver             | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF, 1I-IC-U                       | 1I-EC      |
# | non_stationary    | E, None, U, UE                                                  | None       |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_li  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_li  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_li | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                 AortaLiverSplit - all inputs (n = 105)                                                                |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | Key            | Value      | Unit       | Name                                                                 | Group           | Bounds        | DICOM | OSIPI     |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | B1corr_1_ao    | 1          |            | 1st B1-correction factor in the aorta                                | electromagnetic | (0, 5)        |       |           |
# | B1corr_1_li    | 1          |            | 1st B1-correction factor in the liver                                | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_ao    | 1          |            | 2nd B1-correction factor in the aorta                                | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_li    | 1          |            | 2nd B1-correction factor in the liver                                | electromagnetic | (0, 5)        |       |           |
# | me             | 1          | A cm2/mL   | equilibrium magnetization                                            | electromagnetic | (0, 5)        |       |           |
# | R1_b           | 0.65       | Hz         | tissue R1 in the blood                                               | electromagnetic | (0, 5)        |       |           |
# | R1_e           | 0.65       | Hz         | tissue R1 in extracellular space                                     | electromagnetic | (0, 5)        |       |           |
# | R1_h           | 0.65       | Hz         | tissue R1 in hepatocytes                                             | electromagnetic | (0, 5)        |       |           |
# | R1_li          | 0.65       | Hz         | tissue R1 in the liver                                               | electromagnetic | (0, 5)        |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | dose_tolerance | 0.1        |            | dose tolerance                                                       | hyperparameters |               |       |           |
# | dt             | 0.5        | sec        | pseudo-continuous time step                                          | hyperparameters |               |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | agent          | gadoterate |            | contrast agent generic name                                          | indicator       |               |       |           |
# | BAT            | 30         | sec        | bolus arrival time                                                   | indicator       | (-30, 30)     |       |           |
# | BAT_1          | 30         | sec        | 1st bolus arrival time                                               | indicator       | (-30, 30)     |       |           |
# | BAT_2          | 30         | sec        | 2nd bolus arrival time                                               | indicator       | (-30, 30)     |       |           |
# | bdel           | 30         | sec        | delay in a double injection                                          | indicator       | (-30, 30)     |       |           |
# | dose           | 0.1        | mL/kg      | contrast agent dose                                                  | indicator       | (0, 0.2)      |       |           |
# | dose_1         | 0.1        | mL/kg      | 1st contrast agent dose                                              | indicator       | (0, 0.2)      |       |           |
# | dose_2         | 0.1        | mL/kg      | 2nd contrast agent dose                                              | indicator       | (0, 0.2)      |       |           |
# | rate           | 1          | mL/sec     | injection rate                                                       | indicator       | (0, 10)       |       |           |
# | rate_1         | 1          | mL/sec     | 1st injection rate                                                   | indicator       | (0, 10)       |       |           |
# | rate_2         | 1          | mL/sec     | 2nd injection rate                                                   | indicator       | (0, 10)       |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | CO             | 100        | mL/sec     | cardiac output                                                       | physiological   | (0, 500)      |       |           |
# | D_hl           | 0.2        |            | transit time dispersion in the heart and Lungs                       | physiological   | (0.01, 0.99)  |       |           |
# | E_li           | 0.1        |            | extraction fraction in the liver                                     | physiological   | (0.0, 1.0)    |       |           |
# | E_or           | 0.15       |            | extraction fraction in the organs                                    | physiological   | (0, 0.5)      |       |           |
# | Ef_li          | 0.1        |            | final extraction fraction in the liver                               | physiological   | (0.0, 1.0)    |       |           |
# | Ei_li          | 0.1        |            | initial extraction fraction in the liver                             | physiological   | (0.0, 1.0)    |       |           |
# | fCO_li         | 0.1        |            | fraction of the cardiac output in the liver                          | physiological   | (0, 0.5)      |       |           |
# | ffa            | 0.2        |            | arterial flow fraction                                               | physiological   | (0, 1)        |       |           |
# | H              | 0.45       |            | hematocrit                                                           | physiological   | (0, 1)        |       |           |
# | k_e2h          | 0.0025     | mL/sec/cm3 | tissue transfer rate from extracellular space to hepatocytes         | physiological   | (0, 0.006)    |       |           |
# | kf_e2h         | 0.0025     | mL/sec/cm3 | final tissue transfer rate from extracellular space to hepatocytes   | physiological   | (0, 0.006)    |       |           |
# | ki_e2h         | 0.0025     | mL/sec/cm3 | initial tissue transfer rate from extracellular space to hepatocytes | physiological   | (0, 0.006)    |       |           |
# | PSw            | 0.03       | mL/sec/cm3 | water permeability-surface area product                              | physiological   | (0, 100)      |       |           |
# | T_b_or         | 20         | sec        | mean transit time in blood of the organs                             | physiological   | (0, 60)       |       |           |
# | T_e_or         | 120        | sec        | mean transit time in extracellular space of the organs               | physiological   | (0, 800)      |       |           |
# | T_gu           | 30         | sec        | mean transit time in the gut                                         | physiological   | (0.1, 60)     |       |           |
# | T_h            | 14400      | sec        | mean transit time in hepatocytes                                     | physiological   | (600, 21600)  |       |           |
# | T_hl           | 10         | sec        | mean transit time in the heart and Lungs                             | physiological   | (0, 30)       |       |           |
# | T_la           | 30         | sec        | mean transit time in the liver artery                                | physiological   | (0.0, 60)     |       |           |
# | TF             | 0.25       | sec        | inflow time                                                          | physiological   | (0, 1)        |       |           |
# | Tf_h           | 14400      | sec        | final mean transit time in hepatocytes                               | physiological   | (600, 21600)  |       |           |
# | Ti_h           | 14400      | sec        | initial mean transit time in hepatocytes                             | physiological   | (600, 21600)  |       |           |
# | v_e            | 0.3        | mL/cm3     | volume fraction in extracellular space                               | physiological   | (0.01, 0.6)   |       |           |
# | v_e_li         | 0.25       | mL/cm3     | volume fraction in extracellular space of the liver                  | physiological   | (0.01, 0.5)   |       |           |
# | v_h            | 1          | mL/cm3     | volume fraction in hepatocytes                                       | physiological   | (0, 1)        |       |           |
# | v_li           | 1          | mL/cm3     | volume fraction in the liver                                         | physiological   | (0, 1)        |       |           |
# | vr_or          | 0.9        |            | Venous return in the organs                                          | physiological   | (0, 1)        |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | FA_1           | 15         | deg        | 1st flip angle                                                       | sequence        | (0, 180)      |       |           |
# | FA_2           | 15         | deg        | 2nd flip angle                                                       | sequence        | (0, 180)      |       |           |
# | field_strength | 3          | T          | magnetic field strength                                              | sequence        | (0, 20)       |       |           |
# | iz_1           | 0          |            | 1st slice number in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | iz_2           | 0          |            | 2nd slice number in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | Nk0_1          | 64         |            | 1st number of acquired phase lines to the center of k-space          | sequence        | (0, 1000)     |       |           |
# | Nk0_2          | 64         |            | 2nd number of acquired phase lines to the center of k-space          | sequence        | (0, 1000)     |       |           |
# | Nph_1          | 128        |            | 1st number of acquired phase lines in k-space                        | sequence        | (0, 1000)     |       |           |
# | Nph_2          | 128        |            | 2nd number of acquired phase lines in k-space                        | sequence        | (0, 1000)     |       |           |
# | Nz_1           | 64         |            | 1st number of slices in a multi-slice acquisition                    | sequence        | (0, 1000)     |       |           |
# | Nz_2           | 64         |            | 2nd number of slices in a multi-slice acquisition                    | sequence        | (0, 1000)     |       |           |
# | PA_1           | 90         | deg        | 1st preparation Pulse Flip Angle                                     | sequence        | (0, 180)      |       |           |
# | PA_2           | 90         | deg        | 2nd preparation Pulse Flip Angle                                     | sequence        | (0, 180)      |       |           |
# | SA_1           | 0          | deg        | 1st saturation Slab Flip Angle                                       | sequence        | (0, 180)      |       |           |
# | SA_2           | 0          | deg        | 2nd saturation Slab Flip Angle                                       | sequence        | (0, 180)      |       |           |
# | TA_1           | 2.0        | sec        | 1st acquisition time                                                 | sequence        | (0, 30)       |       |           |
# | TA_2           | 2.0        | sec        | 2nd acquisition time                                                 | sequence        | (0, 30)       |       |           |
# | tacq_1         | 240        | sec        | 1st acquisition duration                                             | sequence        | (0, 10000.0)  |       |           |
# | tacq_2         | 240        | sec        | 2nd acquisition duration                                             | sequence        | (0, 10000.0)  |       |           |
# | TD_1           | 0.05       | sec        | 1st prepulse delay                                                   | sequence        | (0, 1)        |       |           |
# | TD_2           | 0.05       | sec        | 2nd prepulse delay                                                   | sequence        | (0, 1)        |       |           |
# | TE1_1          | 0.001      | sec        | 1st first echo time in a multi-echo sequence                         | sequence        | (0, 1)        |       |           |
# | TE1_2          | 0.001      | sec        | 2nd first echo time in a multi-echo sequence                         | sequence        | (0, 1)        |       |           |
# | TE2_1          | 0.005      | sec        | 1st second echo time in a multi-echo sequence                        | sequence        | (0, 1)        |       |           |
# | TE2_2          | 0.005      | sec        | 2nd second echo time in a multi-echo sequence                        | sequence        | (0, 1)        |       |           |
# | TE_1           | 0.001      | sec        | 1st echo time                                                        | sequence        | (0, 10)       |       |           |
# | TE_2           | 0.001      | sec        | 2nd echo time                                                        | sequence        | (0, 10)       |       |           |
# | TP_1           | 0.05       | sec        | 1st preparation delay                                                | sequence        | (0, 1)        |       |           |
# | TP_2           | 0.05       | sec        | 2nd preparation delay                                                | sequence        | (0, 1)        |       |           |
# | TR_1           | 0.005      | sec        | 1st repetition time                                                  | sequence        | (0, 1)        |       |           |
# | TR_2           | 0.005      | sec        | 2nd repetition time                                                  | sequence        | (0, 1)        |       |           |
# | tstart_1       | 0          | sec        | 1st start of the acquisition                                         | sequence        | (0, 10000.0)  |       |           |
# | tstart_2       | 0          | sec        | 2nd start of the acquisition                                         | sequence        | (0, 10000.0)  |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iScal_1_ao     | 0          |            | 1st indices of calibration signal in the aorta                       | signal          |               |       |           |
# | iScal_1_li     | 0          |            | 1st indices of calibration signal in the liver                       | signal          |               |       |           |
# | iScal_2_ao     | 0          |            | 2nd indices of calibration signal in the aorta                       | signal          |               |       |           |
# | iScal_2_li     | 0          |            | 2nd indices of calibration signal in the liver                       | signal          |               |       |           |
# | iStrig_1_ao    | None       |            | 1st indices of the signal trigger in the aorta                       | signal          |               |       |           |
# | iStrig_1_li    | None       |            | 1st indices of the signal trigger in the liver                       | signal          |               |       |           |
# | iStrig_2_ao    | None       |            | 2nd indices of the signal trigger in the aorta                       | signal          |               |       |           |
# | iStrig_2_li    | None       |            | 2nd indices of the signal trigger in the liver                       | signal          |               |       |           |
# | NSR_1_ao       | 0.0        |            | 1st noise-to-signal ratio in the aorta                               | signal          | (0, 100000.0) |       |           |
# | NSR_1_li       | 0.0        |            | 1st noise-to-signal ratio in the liver                               | signal          | (0, 100000.0) |       |           |
# | NSR_2_ao       | 0.0        |            | 2nd noise-to-signal ratio in the aorta                               | signal          | (0, 100000.0) |       |           |
# | NSR_2_li       | 0.0        |            | 2nd noise-to-signal ratio in the liver                               | signal          | (0, 100000.0) |       |           |
# | S0_1_ao        | 1.0        | a.u.       | 1st signal scaling factor in the aorta                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | 1.0        | a.u.       | 1st signal scaling factor in the liver                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | 1.0        | a.u.       | 2nd signal scaling factor in the aorta                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | 1.0        | a.u.       | 2nd signal scaling factor in the liver                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | Scal_1_ao      | 1.0        | a.u.       | 1st calibration signal in the aorta                                  | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_1_li      | 1.0        | a.u.       | 1st calibration signal in the liver                                  | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_ao      | 1.0        | a.u.       | 2nd calibration signal in the aorta                                  | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_li      | 1.0        | a.u.       | 2nd calibration signal in the liver                                  | signal          | (0, 5)        |       | Q.MS1.002 |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_ao         | 1          | cm3        | ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_li         | 1000       | cm3        | ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                          | whole-body      | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------------------------+
# |                                          AortaLiverSplit - all outputs (n = 41)                                         |
# +---------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | Key     | Value | Unit     | Name                                       | Group           | Bounds  | DICOM | OSIPI     |
# +---------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | M_1_ao  | 1     | A/cm     | 1st magnetization in the aorta             | electromagnetic | (0, 5)  |       |           |
# | M_1_li  | 1     | A/cm     | 1st magnetization in the liver             | electromagnetic | (0, 5)  |       |           |
# | M_2_ao  | 1     | A/cm     | 2nd magnetization in the aorta             | electromagnetic | (0, 5)  |       |           |
# | M_2_li  | 1     | A/cm     | 2nd magnetization in the liver             | electromagnetic | (0, 5)  |       |           |
# | R1_ao   | 0.65  | Hz       | tissue R1 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R1_li   | 0.65  | Hz       | tissue R1 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R1i_ao  | 0.65  | Hz       | inlet R1 in the aorta                      | electromagnetic | (0, 5)  |       |           |
# | R1i_li  | 0.65  | Hz       | inlet R1 in the liver                      | electromagnetic | (0, 5)  |       |           |
# | R2_ao   | 2.0   | Hz       | tissue R2 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R2_li   | 2.0   | Hz       | tissue R2 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R2s_ao  | 20    | Hz       | tissue R2* in the aorta                    | electromagnetic | (0, 5)  |       |           |
# | R2s_li  | 20    | Hz       | tissue R2* in the liver                    | electromagnetic | (0, 5)  |       |           |
# | tM_1_ao | 0.0   | sec      | 1st magnetization time points in the aorta | electromagnetic |         |       |           |
# | tM_1_li | 0.0   | sec      | 1st magnetization time points in the liver | electromagnetic |         |       |           |
# | tM_2_ao | 0.0   | sec      | 2nd magnetization time points in the aorta | electromagnetic |         |       |           |
# | tM_2_li | 0.0   | sec      | 2nd magnetization time points in the liver | electromagnetic |         |       |           |
# | tR      | 0.0   | sec      | relaxation rate time points                | electromagnetic |         |       |           |
# +---------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | C_ao    | 0.005 | mmol/cm3 | tissue concentration in the aorta          | indicator       | (0, 1)  |       |           |
# | C_li    | 0.005 | mmol/cm3 | tissue concentration in the liver          | indicator       | (0, 1)  |       |           |
# | ci_ao   | 0.005 | mmol/mL  | inlet concentration in the aorta           | indicator       |         |       |           |
# | ci_li   | 0.005 | mmol/mL  | inlet concentration in the liver           | indicator       |         |       |           |
# | J_ao    | 1     | mmol/sec | indicator flux in the aorta                | indicator       | (0, 10) |       |           |
# | J_la    | 1     | mmol/sec | indicator flux in the liver artery         | indicator       | (0, 10) |       |           |
# | J_lag   | 1     | mmol/sec | indicator flux in the liver artery and gut | indicator       | (0, 10) |       |           |
# | J_li    | 1     | mmol/sec | indicator flux in the liver                | indicator       | (0, 10) |       |           |
# | J_or    | 1     | mmol/sec | indicator flux in the organs               | indicator       | (0, 10) |       |           |
# | J_pv    | 1     | mmol/sec | indicator flux in the portal vein          | indicator       | (0, 10) |       |           |
# | J_vc    | 1     | mmol/sec | indicator flux in the vena cava            | indicator       | (0, 10) |       |           |
# | tC      | 0.0   | sec      | concentration time points                  | indicator       |         |       |           |
# +---------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | S0_1_ao | 1.0   | a.u.     | 1st signal scaling factor in the aorta     | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_1_li | 1.0   | a.u.     | 1st signal scaling factor in the liver     | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_2_ao | 1.0   | a.u.     | 2nd signal scaling factor in the aorta     | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_2_li | 1.0   | a.u.     | 2nd signal scaling factor in the liver     | signal          | (0, 5)  |       | Q.MS1.010 |
# | S_1_ao  | 1.0   | a.u.     | 1st signal in the aorta                    | signal          | (0, 5)  |       |           |
# | S_1_li  | 1.0   | a.u.     | 1st signal in the liver                    | signal          | (0, 5)  |       |           |
# | S_2_ao  | 1.0   | a.u.     | 2nd signal in the aorta                    | signal          | (0, 5)  |       |           |
# | S_2_li  | 1.0   | a.u.     | 2nd signal in the liver                    | signal          | (0, 5)  |       |           |
# | tS_1_ao | 0.0   | sec      | 1st signal time points in the aorta        | signal          |         |       |           |
# | tS_1_li | 0.0   | sec      | 1st signal time points in the liver        | signal          |         |       |           |
# | tS_2_ao | 0.0   | sec      | 2nd signal time points in the aorta        | signal          |         |       |           |
# | tS_2_li | 0.0   | sec      | 2nd signal time points in the liver        | signal          |         |       |           |
# +-------------------------------------------------------------------------------------------------------------------------+

from copy import deepcopy
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import extend_varname, group
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityLiver
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeLiver
from dcmri.signal.modules_tissue import ConcToRelax, RelaxToSignal
from dcmri.bloch.functions_sequences import channels

rois, scans = ['ao', 'li'], [1, 2]
tissue_rel = {'ao': RelaxivityArtery, 'li': RelaxivityLiver}
tissue_wex = {'ao': WaterExchangeArtery, 'li': WaterExchangeLiver}

CONFIGS = deepcopy(RelaxToSignal.configs | ConcToRelax.configs | WaterExchangeArtery.configs | WaterExchangeLiver.configs | RelaxivityArtery.configs | RelaxivityLiver.configs | ConcAortaLiver.configs)
DEFAULTS = deepcopy(RelaxToSignal.defaults | ConcToRelax.defaults | WaterExchangeArtery.defaults | WaterExchangeLiver.defaults | RelaxivityArtery.defaults | RelaxivityLiver.defaults | ConcAortaLiver.defaults)
CMAP = {roi: {} for roi in rois}

# ROI-specific configurations
roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

for key in roi_configs:
    config = CONFIGS.pop(key)
    default = DEFAULTS.pop(key)
    for roi in rois:
        CONFIGS[f'{key}_{roi}'] = config
        DEFAULTS[f'{key}_{roi}'] = default
        CMAP[roi] |= {key: f'{key}_{roi}'}

CONFIGS['inflow'].discard('inlet')


class AortaLiverSplit(Module):
    """Whole-body model for the aorta and liver signal acquired over 2 separate acquisitions."""

    configs = CONFIGS
    defaults = DEFAULTS

    _all_inputs = {'S0_2_ao', 'Scal_2_li', 'v_e_li', 'Nph_1', 'NSR_1_ao', 'B1corr_2_ao', 'iStrig_1_li', 'TA_2', 'T_la', 'Nz_2', 'NSR_1_li', 'Tf_h', 'vr_or', 'rate', 'SA_1', 'E_li', 'dose_1', 'SA_2', 'B1corr_1_li', 'Scal_2_ao', 'dose_2', 'R1_e', 'TE1_2', 'S0_2_li', 'iStrig_1_ao', 'Nk0_1', 'TP_2', 'me', 'v_li', 'dose', 'kf_e2h', 'E_or', 'agent', 'Nph_2', 'v_e', 'BAT_1', 'ki_e2h', 'PA_2', 'Scal_1_ao', 'Ti_h', 'TE_1', 'fCO_li', 'iStrig_2_ao', 'Ef_li', 'T_e_or', 'iz_1', 'B1corr_2_li', 'NSR_2_ao', 'CO', 'ffa', 'vol_li', 'tstart_2', 'Nk0_2', 'NSR_2_li', 'TF', 'iStrig_2_li', 'iz_2', 'PSw', 'tacq_2', 'bdel', 'v_h', 'T_h', 'Ei_li', 'PA_1', 'iScal_2_li', 'iScal_2_ao', 'FA_2', 'rate_1', 'R1_b', 'TE1_1', 'T_gu', 'R1_h', 'iScal_1_ao', 'B1corr_1_ao', 'tstart_1', 'tacq_1', 'dt', 'Scal_1_li', 'vol_ao', 'TD_2', 'TD_1', 'TP_1', 'dose_tolerance', 'T_b_or', 'S0_1_li', 'T_hl', 'FA_1', 'iScal_1_li', 'TR_1', 'TE2_2', 'BAT', 'R1_li', 'field_strength', 'TE2_1', 'rate_2', 'S0_1_ao', 'D_hl', 'TE_2', 'k_e2h', 'weight', 'H', 'Nz_1', 'TR_2', 'TA_1', 'BAT_2'}
    _all_outputs = {'S0_2_ao', 'R1i_li', 'S_1_li', 'tS_2_li', 'tS_1_li', 'tM_2_ao', 'tR', 'tM_1_li', 'S_2_li', 'J_li', 'S0_2_li', 'J_vc', 'S_2_ao', 'M_1_li', 'J_la', 'ci_li', 'M_1_ao', 'tS_2_ao', 'tS_1_ao', 'C_li', 'tC', 'J_pv', 'S_1_ao', 'S0_1_li', 'J_or', 'R2s_ao', 'R2_ao', 'J_ao', 'R1_li', 'R2s_li', 'R1_ao', 'S0_1_ao', 'R2_li', 'C_ao', 'J_lag', 'M_2_li', 'R1i_ao', 'tM_1_ao', 'ci_ao', 'tM_2_li', 'M_2_ao'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)

        # Liver never has tof_corr
        config = {
            'ao': self.config,
            'li': self.config | {'tof_corr': False}
        }

        # Configure modules
        self._conc = ConcAortaLiver(**self.config)
        self._tissue_rel = {}
        self._tissue_wex = {}
        self._conc_to_relax = {}
        self._relax_to_signal = {}
        
        for roi in rois:
            iomap_roi = {'F_b_ar': 'F_b_ao'}
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in tissue_rel[roi].all_outputs() | tissue_wex[roi].all_outputs()}
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in {'C', 'ci'} | ConcToRelax.all_outputs() - {'tR'}}

            self._tissue_rel[roi] = tissue_rel[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])
            self._conc_to_relax[roi] = ConcToRelax(iomap=iomap_roi, cmap=CMAP[roi], **config[roi])
            self._tissue_wex[roi] = tissue_wex[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])

            for scan in scans:
                iomap_roi |= {k: extend_varname(k, index=scan) for k in RelaxToSignal.all_inputs() if group(k)=='sequence'}
                iomap_roi |= {k: extend_varname(k, index=scan, roi=roi) for k in RelaxToSignal.all_outputs() | {'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr'}}
                self._relax_to_signal[roi, scan] = RelaxToSignal(iomap=iomap_roi, cmap=CMAP[roi], **self.config)

        self.map_io(imap, omap, iomap)

    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)  

        p['tmax'] = p['tstart_2'] + p['tacq_2'] + p['dt']

        p |= self._conc(p)
        for roi in rois:
            p |= self._tissue_rel[roi](p) 
            p |= self._conc_to_relax[roi](p)
            p |= self._tissue_wex[roi](p) 
            for scan in scans:
                p |= self._relax_to_signal[roi, scan](p)

        return self.map_results(p)
    
    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        for roi in rois:
            inputs |= self._tissue_rel[roi].mapped_inputs() 
            inputs |= self._conc_to_relax[roi].mapped_inputs()
            inputs |= self._tissue_wex[roi].mapped_inputs() 
            for scan in [1, 2]:
                inputs |= self._relax_to_signal[roi, scan].mapped_inputs()

        inputs -= {'tmax'} 
        inputs -= self._conc.new_mapped_outputs()
        for roi in rois:
            inputs -= self._tissue_rel[roi].new_mapped_outputs()
            inputs -= (self._conc_to_relax[roi].new_mapped_outputs() - {'R1_li'})
            inputs -= self._tissue_wex[roi].new_mapped_outputs()
            # for scan in scans:
            #     inputs -= self._relax_to_signal[roi, scan].new_mapped_outputs()

        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in rois:
            outputs |= self._conc_to_relax[roi].mapped_outputs() 
            for scan in scans:
                outputs |= self._relax_to_signal[roi, scan].mapped_outputs()
            outputs -= {f'F_b_{roi}'}
        return outputs

    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self._conc.test_data()

        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        for roi in rois:
            for scan in [1,2]:
                p |= {
                    f'iScal_{scan}_{roi}': np.arange(n0, dtype=int),
                    f'Scal_{scan}_{roi}': Scal, 
                }
        p |= {
            'tstart_1': 0,
            'tacq_1': 60, 
            'tstart_2': 120,
            'tacq_2': 90,
            'BAT': 30,
            'bdel': 120,
            'BAT_1': 30,
            'BAT_2': 150, 
        }  
        return self.input_data(p, data)