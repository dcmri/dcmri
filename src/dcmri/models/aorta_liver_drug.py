import numpy as np

from dcmri.core.tools import extend_varname, increment_varindex
from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityLiver
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeLiver
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

# +--------------------------------------------------------------------------------------------------+
# |                            AortaLiverDrugModel - all configs (n = 20)                            |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | inflow            | False, True                                                     | False      |
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                               |            |
# | tof_corr          | False, True                                                     | False      |
# | magnitude         | False, True                                                     | True       |
# | trigger           | False, True                                                     | False      |
# | calibrate         | False, True                                                     | False      |
# | baseline          | literature, measured                                            | literature |
# | compartments      | ('e', 'h'), ('li',)                                             | ('li',)    |
# | bolus             | dual, single                                                    | single     |
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

# +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                 AortaLiverDrugModel - all inputs (n = 104)                                                                 |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                      | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT_1          | sec        | 1st bolus arrival time                                                    | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                                    | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_3          | sec        | 3rd bolus arrival time                                                    | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_4          | sec        | 4th bolus arrival time                                                    | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                                               | Indicator       | gadoterate |               |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                                   | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                                   | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_3         | mL/kg      | 3rd contrast agent dose                                                   | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_4         | mL/kg      | 4th contrast agent dose                                                   | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                        | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                        | Indicator       | 1          | (0, 10)       |       |           |
# | rate_3         | mL/s       | 3rd injection rate                                                        | Indicator       | 1          | (0, 10)       |       |           |
# | rate_4         | mL/s       | 4th injection rate                                                        | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_1_ao       |            | 1st noise-to-signal ratio in the aorta                                    | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_1_li       |            | 1st noise-to-signal ratio in the liver                                    | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_ao       |            | 2nd noise-to-signal ratio in the aorta                                    | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_2_li       |            | 2nd noise-to-signal ratio in the liver                                    | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_1_ao        | a.u.       | 1st signal scaling factor in the aorta                                    | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | a.u.       | 1st signal scaling factor in the liver                                    | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | a.u.       | 2nd signal scaling factor in the aorta                                    | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | a.u.       | 2nd signal scaling factor in the liver                                    | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal_1_ao      | a.u.       | 1st calibration signal in the aorta                                       | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_1_li      | a.u.       | 1st calibration signal in the liver                                       | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_ao      | a.u.       | 2nd calibration signal in the aorta                                       | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_li      | a.u.       | 2nd calibration signal in the liver                                       | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal_1_ao     |            | 1st indices of calibration signal in the aorta                            | Signal          | 0          |               |       |           |
# | iScal_1_li     |            | 1st indices of calibration signal in the liver                            | Signal          | 0          |               |       |           |
# | iScal_2_ao     |            | 2nd indices of calibration signal in the aorta                            | Signal          | 0          |               |       |           |
# | iScal_2_li     |            | 2nd indices of calibration signal in the liver                            | Signal          | 0          |               |       |           |
# | iStrig_1_ao    |            | 1st indices of the signal trigger in the aorta                            | Signal          | None       |               |       |           |
# | iStrig_1_li    |            | 1st indices of the signal trigger in the liver                            | Signal          | None       |               |       |           |
# | iStrig_2_ao    |            | 2nd indices of the signal trigger in the aorta                            | Signal          | None       |               |       |           |
# | iStrig_2_li    |            | 2nd indices of the signal trigger in the liver                            | Signal          | None       |               |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                                                | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space                   | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space                                 | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                             | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                              | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg        | saturation Slab Flip Angle                                                | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                                          | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                                            | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                                                 | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                                  | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                                 | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                                         | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                                           | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                                   | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition                                 | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq_1         | sec        | 1st acquisition duration                                                  | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tacq_2         | sec        | 2nd acquisition duration                                                  | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart_1       | sec        | 1st start of the acquisition                                              | Sequence        | 0          | (0, 10000.0)  |       |           |
# | tstart_2       | sec        | 2nd start of the acquisition                                              | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_1_ao    |            | 1st B1-correction factor in the aorta                                     | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_1_li    |            | 1st B1-correction factor in the liver                                     | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_ao    |            | 2nd B1-correction factor in the aorta                                     | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_2_li    |            | 2nd B1-correction factor in the liver                                     | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                                    | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_e           | Hz         | tissue R1 in extracellular                                                | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_h           | Hz         | tissue R1 in hepatocytes                                                  | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                                 | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                                            | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs                            | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_1_li         |            | 1st extraction fraction in the liver                                      | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_2_li         |            | 2nd extraction fraction in the liver                                      | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_or           |            | extraction fraction in the organs                                         | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | Ef_1_li        |            | 1st final extraction fraction in the liver                                | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ef_2_li        |            | 2nd final extraction fraction in the liver                                | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_1_li        |            | 1st initial extraction fraction in the liver                              | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_2_li        |            | 2nd initial extraction fraction in the liver                              | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | GFR            | mL/sec     | glomerular filtration rate                                                | Physiological   | 2          | (0, 10)       |       |           |
# | H              |            | hematocrit                                                                | Physiological   | 0.45       | (0, 1)        |       |           |
# | PSw_e2h        | mL/sec/cm3 | water permeability-surface area product from extracellular to hepatocytes | Physiological   | 0.03       | (0, 100)      |       |           |
# | PSw_h2e        | mL/sec/cm3 | water permeability-surface area product from hepatocytes to extracellular | Physiological   | 0.03       | (0, 100)      |       |           |
# | TF             | sec        | inflow time                                                               | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_1_h          | sec        | 1st mean transit time in hepatocytes                                      | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_2_h          | sec        | 2nd mean transit time in hepatocytes                                      | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                                  | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_or         | sec        | mean transit time in extracellular of the organs                          | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec        | mean transit time in the gut                                              | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                                  | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec        | mean transit time in the liver artery                                     | Physiological   | 30         | (0.1, 60)     |       |           |
# | Tf_1_h         | sec        | 1st final mean transit time in hepatocytes                                | Physiological   | 1800       | (600, 36000)  |       |           |
# | Tf_2_h         | sec        | 2nd final mean transit time in hepatocytes                                | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_1_h         | sec        | 1st initial mean transit time in hepatocytes                              | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_2_h         | sec        | 2nd initial mean transit time in hepatocytes                              | Physiological   | 1800       | (600, 36000)  |       |           |
# | fCO_li         |            | fraction of the cardiac output in the liver                               | Physiological   | 0.1        | (0, 0.5)      |       |           |
# | ffa            |            | arterial flow fraction                                                    | Physiological   | 0.2        | (0, 1)        |       |           |
# | k_1_e2h        | mL/sec/cm3 | 1st tissue transfer rate from extracellular to hepatocytes                | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | k_2_e2h        | mL/sec/cm3 | 2nd tissue transfer rate from extracellular to hepatocytes                | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_1_e2h       | mL/sec/cm3 | 1st final tissue transfer rate from extracellular to hepatocytes          | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_2_e2h       | mL/sec/cm3 | 2nd final tissue transfer rate from extracellular to hepatocytes          | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_1_e2h       | mL/sec/cm3 | 1st initial tissue transfer rate from extracellular to hepatocytes        | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_2_e2h       | mL/sec/cm3 | 2nd initial tissue transfer rate from extracellular to hepatocytes        | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | v_e_li         | mL/cm3     | volume fraction in extracellular of the liver                             | Physiological   | 0.3        | (0.01, 0.6)   |       |           |
# | v_h            | mL/cm3     | volume fraction in hepatocytes                                            | Physiological   | 1          | (0, 1)        |       |           |
# | v_li           | mL/cm3     | volume fraction in the liver                                              | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                                            | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec        | pseudo-continuous time step                                               | Hyperparameters | 0.5        |               |       |           |
# +----------------+------------+---------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_1_ao       | cm3        | 1st ROI volume in the aorta                                               | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_1_li       | cm3        | 1st ROI volume in the liver                                               | Whole-body      | 1000       | (0, 10000)    |       |           |
# | vol_2_ao       | cm3        | 2nd ROI volume in the aorta                                               | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_2_li       | cm3        | 2nd ROI volume in the liver                                               | Whole-body      | 1000       | (0, 10000)    |       |           |
# | weight         | kg         | body weight                                                               | Whole-body      | 70         | (0, 300)      |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------+
# |                                          AortaLiverDrugModel - all outputs (n = 62)                                          |
# +----------+----------+------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | Key      | Unit     | Name                                           | Group           | Init  | Bounds  | DICOM | OSIPI     |
# +----------+----------+------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | C_1_ao   | mmol/cm3 | 1st tissue concentration in the aorta          | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_1_li   | mmol/cm3 | 1st tissue concentration in the liver          | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_2_ao   | mmol/cm3 | 2nd tissue concentration in the aorta          | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_2_li   | mmol/cm3 | 2nd tissue concentration in the liver          | Indicator       | 0.005 | (0, 1)  |       |           |
# | J_1_ao   | mmol/sec | 1st indicator flux in the aorta                | Indicator       | 1     | (0, 10) |       |           |
# | J_1_la   | mmol/sec | 1st indicator flux in the liver artery         | Indicator       | 1     | (0, 10) |       |           |
# | J_1_lag  | mmol/sec | 1st indicator flux in the liver artery and gut | Indicator       | 1     | (0, 10) |       |           |
# | J_1_li   | mmol/sec | 1st indicator flux in the liver                | Indicator       | 1     | (0, 10) |       |           |
# | J_1_or   | mmol/sec | 1st indicator flux in the organs               | Indicator       | 1     | (0, 10) |       |           |
# | J_1_pv   | mmol/sec | 1st indicator flux in the portal vein          | Indicator       | 1     | (0, 10) |       |           |
# | J_1_ve   | mmol/sec | 1st indicator flux in the vein                 | Indicator       | 1     | (0, 10) |       |           |
# | J_2_ao   | mmol/sec | 2nd indicator flux in the aorta                | Indicator       | 1     | (0, 10) |       |           |
# | J_2_la   | mmol/sec | 2nd indicator flux in the liver artery         | Indicator       | 1     | (0, 10) |       |           |
# | J_2_lag  | mmol/sec | 2nd indicator flux in the liver artery and gut | Indicator       | 1     | (0, 10) |       |           |
# | J_2_li   | mmol/sec | 2nd indicator flux in the liver                | Indicator       | 1     | (0, 10) |       |           |
# | J_2_or   | mmol/sec | 2nd indicator flux in the organs               | Indicator       | 1     | (0, 10) |       |           |
# | J_2_pv   | mmol/sec | 2nd indicator flux in the portal vein          | Indicator       | 1     | (0, 10) |       |           |
# | J_2_ve   | mmol/sec | 2nd indicator flux in the vein                 | Indicator       | 1     | (0, 10) |       |           |
# | ci_1_ao  | mmol/mL  | 1st inlet concentration in the aorta           | Indicator       | 0.005 |         |       |           |
# | ci_1_li  | mmol/mL  | 1st inlet concentration in the liver           | Indicator       | 0.005 |         |       |           |
# | ci_2_ao  | mmol/mL  | 2nd inlet concentration in the aorta           | Indicator       | 0.005 |         |       |           |
# | ci_2_li  | mmol/mL  | 2nd inlet concentration in the liver           | Indicator       | 0.005 |         |       |           |
# | tC_1     | sec      | 1st concentration time points                  | Indicator       | 0.0   |         |       |           |
# | tC_2     | sec      | 2nd concentration time points                  | Indicator       | 0.0   |         |       |           |
# +----------+----------+------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | S0_1_ao  | a.u.     | 1st signal scaling factor in the aorta         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_1_li  | a.u.     | 1st signal scaling factor in the liver         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_2_ao  | a.u.     | 2nd signal scaling factor in the aorta         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_2_li  | a.u.     | 2nd signal scaling factor in the liver         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S_1_ao   | a.u.     | 1st signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_1_li   | a.u.     | 1st signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_2_ao   | a.u.     | 2nd signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_2_li   | a.u.     | 2nd signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# +----------+----------+------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | M_1_ao   | A/cm     | 1st magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_1_li   | A/cm     | 1st magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_2_ao   | A/cm     | 2nd magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_2_li   | A/cm     | 2nd magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | R1_1_ao  | Hz       | 1st tissue R1 in the aorta                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_1_li  | Hz       | 1st tissue R1 in the liver                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_2_ao  | Hz       | 2nd tissue R1 in the aorta                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_2_li  | Hz       | 2nd tissue R1 in the liver                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_1_ao | Hz       | 1st inlet R1 in the aorta                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_1_li | Hz       | 1st inlet R1 in the liver                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_2_ao | Hz       | 2nd inlet R1 in the aorta                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_2_li | Hz       | 2nd inlet R1 in the liver                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R2_1_ao  | Hz       | 1st tissue R2 in the aorta                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_1_li  | Hz       | 1st tissue R2 in the liver                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_2_ao  | Hz       | 2nd tissue R2 in the aorta                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_2_li  | Hz       | 2nd tissue R2 in the liver                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2s_1_ao | Hz       | 1st tissue R2* in the aorta                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_1_li | Hz       | 1st tissue R2* in the liver                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_2_ao | Hz       | 2nd tissue R2* in the aorta                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_2_li | Hz       | 2nd tissue R2* in the liver                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | tM_1_ao  | sec      | 1st magnetization time points in the aorta     | Electromagnetic | 0.0   |         |       |           |
# | tM_1_li  | sec      | 1st magnetization time points in the liver     | Electromagnetic | 0.0   |         |       |           |
# | tM_2_ao  | sec      | 2nd magnetization time points in the aorta     | Electromagnetic | 0.0   |         |       |           |
# | tM_2_li  | sec      | 2nd magnetization time points in the liver     | Electromagnetic | 0.0   |         |       |           |
# | tR_1     | sec      | 1st relaxation rate time points                | Electromagnetic | 0.0   |         |       |           |
# | tR_2     | sec      | 2nd relaxation rate time points                | Electromagnetic | 0.0   |         |       |           |
# | tS_1_ao  | sec      | 1st signal time points in the aorta            | Electromagnetic | 0.0   |         |       |           |
# | tS_1_li  | sec      | 1st signal time points in the liver            | Electromagnetic | 0.0   |         |       |           |
# | tS_2_ao  | sec      | 2nd signal time points in the aorta            | Electromagnetic | 0.0   |         |       |           |
# | tS_2_li  | sec      | 2nd signal time points in the liver            | Electromagnetic | 0.0   |         |       |           |
# +------------------------------------------------------------------------------------------------------------------------------+


visits, rois = [1, 2], ['ao', 'li']
tissue_rel = {'ao': RelaxivityArtery, 'li': RelaxivityLiver}
tissue_wex = {'ao': WaterExchangeArtery, 'li': WaterExchangeLiver}
    
# ROI-specific configurations
roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

configs = ConcToSignal.configs | WaterExchangeArtery.configs | WaterExchangeLiver.configs | RelaxivityArtery.configs | RelaxivityLiver.configs | ConcAortaLiver.configs
defaults = ConcToSignal.defaults | WaterExchangeArtery.defaults | WaterExchangeLiver.defaults | RelaxivityArtery.defaults | RelaxivityLiver.defaults | ConcAortaLiver.defaults
cmap = {roi: {} for roi in rois}

for key in roi_configs:
    config = configs.pop(key)
    default = defaults.pop(key)
    for roi in rois:
        configs[f'{key}_{roi}'] = config
        defaults[f'{key}_{roi}'] = default
        cmap[roi] |= {key: f'{key}_{roi}'}


class AortaLiverDrugModel(Module):
    """Whole-body model for the aorta and liver signal acquired over 2 separate acquisitions."""

    configs = configs
    defaults = defaults

    _all_inputs = None
    _all_outputs = None
    _n_configs = None # valid configs counted
    
    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        for visit in visits:
            p['tmax'] = p[f'tstart_{visit}'] + p[f'tacq_{visit}'] + p['dt']

            p |= self._conc[visit](p)
            for roi in rois:
                p |= self._tissue_rel[visit, roi](p) 
                p |= self._tissue_wex[visit, roi](p) 
                p |= self._conc_to_signal[visit, roi](p)

        return self.map_results(p)
    
    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        config_sig = {
            'ao': self.config,
            'li': self.config | {'tof_corr': False}  # Only aorta has tof_corr
        }
        self._conc = {}
        self._tissue_rel = {}
        self._tissue_wex = {}
        self._conc_to_signal = {}

        for visit in visits:
            iomap = {}
            if visit==2: # variables that already have an index are incremented with 2 at the second visit
                iomap |= {k: increment_varindex(k, 2) for k in {'BAT_1', 'BAT_2', 'dose_1', 'dose_2', 'rate_1', 'rate_2'}}
            iomap |= {k: extend_varname(k, index=visit) for k in {'BAT', 'dose', 'rate', 'E_li', 'Ef_li', 'Ei_li', 'T_h', 'Tf_h', 'Ti_h', 'k_e2h', 'kf_e2h', 'ki_e2h', 'vol_ao', 'vol_li'}}
            iomap |= {k: extend_varname(k, index=visit) for k in ConcAortaLiver.all_outputs() - {'F_b_ao', 'F_b_li'}}
            self._conc[visit] = ConcAortaLiver(iomap=iomap, **self.config)

            for roi in rois:
                iomap |= {'F_b_ar': 'F_b_ao'}
                iomap |= {k: extend_varname(k, index=visit) for k in {'R_1_b', 'R1_li'}}
                iomap |= {k: extend_varname(k, roi=roi) for k in tissue_rel[roi].all_outputs() | tissue_wex[roi].all_outputs() | {'v_e'}}
                self._tissue_rel[visit, roi] = tissue_rel[roi](iomap=iomap, cmap=cmap[roi], **self.config)
                self._tissue_wex[visit, roi] = tissue_wex[roi](iomap=iomap, cmap=cmap[roi], **self.config)

                iomap |= {k: extend_varname(k, index=visit, roi=roi) for k in {'C', 'ci', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr', 'S', 'M', 'R1', 'R1i', 'R2', 'R2s', 'tM', 'tS'}}
                iomap |= {k: extend_varname(k, index=visit) for k in {'tC', 'tR', 'tacq', 'tstart'}}
                self._conc_to_signal[visit, roi] = ConcToSignal(iomap=iomap, cmap=cmap[roi], **config_sig[roi]) 

        self.map_io(imap, omap)


    def inputs(self) -> set:
        inputs = set()
        for visit in visits:
            inputs |= self._conc[visit].mapped_inputs()
            for roi in rois:
                inputs |= self._tissue_rel[visit, roi].mapped_inputs() 
                inputs |= self._tissue_wex[visit, roi].mapped_inputs() 
                inputs |= self._conc_to_signal[visit, roi].mapped_inputs()

        inputs -= {'tmax'} 
        for visit in visits:
            inputs -= self._conc[visit].new_mapped_outputs()
            for roi in rois:
                inputs -= self._tissue_rel[visit, roi].new_mapped_outputs()
                inputs -= self._tissue_wex[visit, roi].new_mapped_outputs()
                inputs -= self._conc_to_signal[visit, roi].new_mapped_outputs()

        return inputs 
    
    def outputs(self):
        outputs = set()
        for visit in visits:
            outputs |= self._conc[visit].mapped_outputs()
            for roi in rois:
                outputs |= self._conc_to_signal[visit, roi].mapped_outputs() 
                outputs -= {f'F_b_{roi}'}
        return outputs

    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        for visit in visits:
            for roi in rois:
                data |= {
                    f'iScal_{visit}_{roi}': np.arange(n0, dtype=int),
                    f'Scal_{visit}_{roi}': Scal, 
                }

        # data['E_2_li'] /= 10 # create some effect
        return data