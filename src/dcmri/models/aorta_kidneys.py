import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcAortaKidneys
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityKidney
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeKidney
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels


# +--------------------------------------------------------------------------------------------------+
# |                             AortaKidneysModel - all configs (n = 21)                             |
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
# | compartments      | ('b', 'u', 'c'), ('b', 'uc'), ('bc', 'u'), ('bu', 'c'), ('ki',) | ('ki',)    |
# | bolus             | dual, single                                                    | single     |
# | heartlung         | chain, comp, pfcomp                                             | pfcomp     |
# | organs            | 2cxm, comp                                                      | comp       |
# | kidneys           | 2CF, 2CFU, 2PF, 2PFU, CPF, FN, HF, HFU                          | 2CF        |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_lk  | None, lin                                                       | lin        |
# | t1_relaxation_rk  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_lk  | None, lin                                                       | None       |
# | t2_relaxation_rk  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_lk | None, lin, quad                                                 | lin        |
# | t2s_relaxation_rk | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                 AortaKidneysModel - all inputs (n = 101)                                                                 |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                                   | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | BAT            | sec        | bolus arrival time                                                     | Indicator       | 30         | (-30, 30)      |       |           |
# | BAT_1          | sec        | 1st bolus arrival time                                                 | Indicator       | 30         | (-30, 30)      |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                                 | Indicator       | 30         | (-30, 30)      |       |           |
# | agent          |            | contrast agent generic name                                            | Indicator       | gadoterate |                |       |           |
# | dose           | mL/kg      | contrast agent dose                                                    | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                                | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                                | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | rate           | mL/s       | injection rate                                                         | Indicator       | 1          | (0, 10)        |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                     | Indicator       | 1          | (0, 10)        |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                     | Indicator       | 1          | (0, 10)        |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR_ao         |            | noise-to-signal ratio in the aorta                                     | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | NSR_lk         |            | noise-to-signal ratio in the left kidney                               | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | NSR_rk         |            | noise-to-signal ratio in the right kidney                              | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0_ao          | a.u.       | signal scaling factor in the aorta                                     | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | S0_lk          | a.u.       | signal scaling factor in the left kidney                               | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | S0_rk          | a.u.       | signal scaling factor in the right kidney                              | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | Scal_ao        | a.u.       | calibration signal in the aorta                                        | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | Scal_lk        | a.u.       | calibration signal in the left kidney                                  | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | Scal_rk        | a.u.       | calibration signal in the right kidney                                 | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | iScal_ao       |            | indices of calibration signal in the aorta                             | Signal          | 0          |                |       |           |
# | iScal_lk       |            | indices of calibration signal in the left kidney                       | Signal          | 0          |                |       |           |
# | iScal_rk       |            | indices of calibration signal in the right kidney                      | Signal          | 0          |                |       |           |
# | iStrig_ao      |            | indices of the signal trigger in the aorta                             | Signal          | None       |                |       |           |
# | iStrig_lk      |            | indices of the signal trigger in the left kidney                       | Signal          | None       |                |       |           |
# | iStrig_rk      |            | indices of the signal trigger in the right kidney                      | Signal          | None       |                |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                                             | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space                | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space                              | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                          | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                           | Sequence        | 90         | (0, 180)       |       |           |
# | SA             | deg        | saturation Slab Flip Angle                                             | Sequence        | 0          | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                                       | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                                         | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                                              | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                               | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                              | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                                      | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                                        | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                                | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition                              | Sequence        | 0          | (0, 1000)      |       |           |
# | tacq           | sec        | acquisition duration                                                   | Sequence        | 240        | (0, 10000.0)   |       |           |
# | tstart         | sec        | start of the acquisition                                               | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr_ao      |            | B1-correction factor in the aorta                                      | Electromagnetic | 1          | (0, 5)         |       |           |
# | B1corr_lk      |            | B1-correction factor in the left kidney                                | Electromagnetic | 1          | (0, 5)         |       |           |
# | B1corr_rk      |            | B1-correction factor in the right kidney                               | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                                 | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_bc          | Hz         | tissue R1 in blood and cells                                           | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_bu          | Hz         | tissue R1 in blood and tubuli                                          | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                                     | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ki          | Hz         | tissue R1 in the kidney                                                | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_u           | Hz         | tissue R1 in tubuli                                                    | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_uc          | Hz         | tissue R1 in tubuli and cells                                          | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                              | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                                         | Physiological   | 100        | (0, 500)       |       |           |
# | DRPF           |            | Differential renal plasma flow                                         | Physiological   | 0.5        | (0, 1)         |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs                         | Physiological   | 0.2        | (0.01, 0.99)   |       |           |
# | E_li           |            | extraction fraction in the liver                                       | Physiological   | 0.1        | (0.0, 1.0)     |       |           |
# | E_or           |            | extraction fraction in the organs                                      | Physiological   | 0.15       | (0, 0.5)       |       |           |
# | FF_lk          |            | filtration fraction in the left kidney                                 | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | FF_rk          |            | filtration fraction in the right kidney                                | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | F_u_lk         | mL/sec/cm3 | flow per unit tissue in tubuli of the left kidney                      | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | F_u_rk         | mL/sec/cm3 | flow per unit tissue in tubuli of the right kidney                     | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | H              |            | hematocrit                                                             | Physiological   | 0.45       | (0, 1)         |       |           |
# | PSw_b2c        | mL/sec/cm3 | water permeability-surface area product from blood to cells            | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_b2u        | mL/sec/cm3 | water permeability-surface area product from blood to tubuli           | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_b2uc       | mL/sec/cm3 | water permeability-surface area product from blood to tubuli and cells | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_bc2u       | mL/sec/cm3 | water permeability-surface area product from blood and cells to tubuli | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_bu2c       | mL/sec/cm3 | water permeability-surface area product from blood and tubuli to cells | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_c2b        | mL/sec/cm3 | water permeability-surface area product from cells to blood            | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_c2bu       | mL/sec/cm3 | water permeability-surface area product from cells to blood and tubuli | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_c2u        | mL/sec/cm3 | water permeability-surface area product from cells to tubuli           | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_u2b        | mL/sec/cm3 | water permeability-surface area product from tubuli to blood           | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_u2bc       | mL/sec/cm3 | water permeability-surface area product from tubuli to blood and cells | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_u2c        | mL/sec/cm3 | water permeability-surface area product from tubuli to cells           | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSw_uc2b       | mL/sec/cm3 | water permeability-surface area product from tubuli and cells to blood | Physiological   | 0.03       | (0, 100)       |       |           |
# | TF             | sec        | inflow time                                                            | Physiological   | 0.5        | (0, 10)        |       |           |
# | T_ar           | sec        | mean transit time in the artery                                        | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                               | Physiological   | 20         | (0, 60)        |       |           |
# | T_e_or         | sec        | mean transit time in extracellular of the organs                       | Physiological   | 120        | (0, 800)       |       |           |
# | T_gu           | sec        | mean transit time in the gut                                           | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                               | Physiological   | 10         | (0, 30)        |       |           |
# | T_u_lk         | sec        | mean transit time in tubuli of the left kidney                         | Physiological   | 120        | (0, 600)       |       |           |
# | T_u_rk         | sec        | mean transit time in tubuli of the right kidney                        | Physiological   | 120        | (0, 600)       |       |           |
# | fCO_ki         |            | fraction of the cardiac output in the kidney                           | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | h_u_lk         | Hz         | transit time distribution in tubuli of the left kidney                 | Physiological   | 0          | (0.1, 60)      |       |           |
# | h_u_rk         | Hz         | transit time distribution in tubuli of the right kidney                | Physiological   | 0          | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                                           | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_bc           | mL/cm3     | volume fraction in blood and cells                                     | Physiological   | 1          | (0, 1)         |       |           |
# | v_bu           | mL/cm3     | volume fraction in blood and tubuli                                    | Physiological   | 1          | (0, 1)         |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                               | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_ki           | mL/cm3     | volume fraction in the kidney                                          | Physiological   | 1          | (0, 1)         |       |           |
# | v_p_lk         | mL/cm3     | volume fraction in plasma of the left kidney                           | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_p_rk         | mL/cm3     | volume fraction in plasma of the right kidney                          | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_u            | mL/cm3     | volume fraction in tubuli                                              | Physiological   | 1          | (0, 1)         |       |           |
# | v_uc           | mL/cm3     | volume fraction in tubuli and cells                                    | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                                         | Hyperparameters | 0.1        |                |       |           |
# | dt             | sec        | pseudo-continuous time step                                            | Hyperparameters | 0.5        |                |       |           |
# +----------------+------------+------------------------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | vol_ao         | cm3        | ROI volume in the aorta                                                | Whole-body      | 10         | (0.0, 1000)    |       |           |
# | vol_lk         | cm3        | ROI volume in the left kidney                                          | Whole-body      | 150        | (0.0, 1000)    |       |           |
# | vol_rk         | cm3        | ROI volume in the right kidney                                         | Whole-body      | 150        | (0.0, 1000)    |       |           |
# | weight         | kg         | body weight                                                            | Whole-body      | 70         | (0, 300)       |       |           |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------+
# |                                           AortaKidneysModel - all outputs (n = 42)                                          |
# +--------+----------+-------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | Key    | Unit     | Name                                            | Group           | Init  | Bounds  | DICOM | OSIPI     |
# +--------+----------+-------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | C_ao   | mmol/cm3 | tissue concentration in the aorta               | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_lk   | mmol/cm3 | tissue concentration in the left kidney         | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_rk   | mmol/cm3 | tissue concentration in the right kidney        | Indicator       | 0.005 | (0, 1)  |       |           |
# | J_ao   | mmol/sec | indicator flux in the aorta                     | Indicator       | 1     | (0, 10) |       |           |
# | J_lk   | mmol/sec | indicator flux in the left kidney               | Indicator       | 1     | (0, 10) |       |           |
# | J_or   | mmol/sec | indicator flux in the organs                    | Indicator       | 1     | (0, 10) |       |           |
# | J_rk   | mmol/sec | indicator flux in the right kidney              | Indicator       | 1     | (0, 10) |       |           |
# | J_ve   | mmol/sec | indicator flux in the vein                      | Indicator       | 1     | (0, 10) |       |           |
# | ci_ao  | mmol/mL  | inlet concentration in the aorta                | Indicator       | 0.005 |         |       |           |
# | ci_lk  | mmol/mL  | inlet concentration in the left kidney          | Indicator       | 0.005 |         |       |           |
# | ci_rk  | mmol/mL  | inlet concentration in the right kidney         | Indicator       | 0.005 |         |       |           |
# | tC     | sec      | concentration time points                       | Indicator       | 0.0   |         |       |           |
# +--------+----------+-------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | S0_ao  | a.u.     | signal scaling factor in the aorta              | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_lk  | a.u.     | signal scaling factor in the left kidney        | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_rk  | a.u.     | signal scaling factor in the right kidney       | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S_ao   | a.u.     | signal in the aorta                             | Signal          | 1.0   | (0, 5)  |       |           |
# | S_lk   | a.u.     | signal in the left kidney                       | Signal          | 1.0   | (0, 5)  |       |           |
# | S_rk   | a.u.     | signal in the right kidney                      | Signal          | 1.0   | (0, 5)  |       |           |
# +--------+----------+-------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | M_ao   | A/cm     | magnetization in the aorta                      | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_lk   | A/cm     | magnetization in the left kidney                | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_rk   | A/cm     | magnetization in the right kidney               | Electromagnetic | 1     | (0, 5)  |       |           |
# | R1_ao  | Hz       | tissue R1 in the aorta                          | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_lk  | Hz       | tissue R1 in the left kidney                    | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_rk  | Hz       | tissue R1 in the right kidney                   | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_ao | Hz       | inlet R1 in the aorta                           | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_lk | Hz       | inlet R1 in the left kidney                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_rk | Hz       | inlet R1 in the right kidney                    | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R2_ao  | Hz       | tissue R2 in the aorta                          | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_lk  | Hz       | tissue R2 in the left kidney                    | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_rk  | Hz       | tissue R2 in the right kidney                   | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2s_ao | Hz       | tissue R2* in the aorta                         | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_lk | Hz       | tissue R2* in the left kidney                   | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_rk | Hz       | tissue R2* in the right kidney                  | Electromagnetic | 20    | (0, 5)  |       |           |
# | tM_ao  | sec      | magnetization time points in the aorta          | Electromagnetic | 0.0   |         |       |           |
# | tM_lk  | sec      | magnetization time points in the left kidney    | Electromagnetic | 0.0   |         |       |           |
# | tM_rk  | sec      | magnetization time points in the right kidney   | Electromagnetic | 0.0   |         |       |           |
# | tR_ao  | sec      | relaxation rate time points in the aorta        | Electromagnetic | 0.0   |         |       |           |
# | tR_lk  | sec      | relaxation rate time points in the left kidney  | Electromagnetic | 0.0   |         |       |           |
# | tR_rk  | sec      | relaxation rate time points in the right kidney | Electromagnetic | 0.0   |         |       |           |
# | tS_ao  | sec      | signal time points in the aorta                 | Electromagnetic | 0.0   |         |       |           |
# | tS_lk  | sec      | signal time points in the left kidney           | Electromagnetic | 0.0   |         |       |           |
# | tS_rk  | sec      | signal time points in the right kidney          | Electromagnetic | 0.0   |         |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------+

_ALL_INPUTS = {'Scal_lk', 'fCO_ki', 'rate_2', 'E_li', 'v_b', 'Scal_rk', 'R1_bu', 'me', 'PSw_b2c', 'H', 'T_b_or', 'FF_lk', 'iScal_rk', 'vol_rk', 'CO', 'iz', 'v_bu', 'iStrig_ao', 'TF', 'B1corr_ao', 'Nph', 'T_u_rk', 'Nz', 'PSw_u2bc', 'v_uc', 'R1_b', 'R1_ki', 'v_p_lk', 'PSw_bc2u', 'PSw_b2u', 'NSR_rk', 'F_u_rk', 'h_u_rk', 'BAT_2', 'PSw_bu2c', 'dose_2', 'T_ar', 'T_gu', 'PSw_u2c', 'TE2', 'D_hl', 'BAT', 'field_strength', 'iScal_ao', 'DRPF', 'B1corr_rk', 'B1corr_lk', 'v_bc', 'S0_ao', 'dose_1', 'TE', 'dose', 'vol_lk', 'rate_1', 'h_u_lk', 'Nk0', 'weight', 'iStrig_rk', 'T_u_lk', 'PA', 'R1_uc', 'agent', 'TP', 'F_u_lk', 'NSR_lk', 'S0_rk', 'PSw_b2uc', 'T_hl', 'NSR_ao', 'TE1', 'dt', 'TA', 'R1_c', 'PSw_c2u', 'S0_lk', 'Scal_ao', 'vol_ao', 'BAT_1', 'T_e_or', 'v_p_rk', 'SA', 'PSw_u2b', 'tstart', 'FA', 'R1_bc', 'PSw_c2b', 'v_ki', 'v_u', 'TD', 'TR', 'R1_u', 'rate', 'FF_rk', 'dose_tolerance', 'v_c', 'PSw_uc2b', 'PSw_c2bu', 'iStrig_lk', 'iScal_lk', 'tacq', 'E_or'}
_ALL_OUTPUTS = {'R2_rk', 'tR_rk', 'tS_rk', 'C_lk', 'C_rk', 'tC', 'J_ao', 'M_ao', 'R1_lk', 'R2_lk', 'S0_rk', 'ci_rk', 'R2s_lk', 'R1i_lk', 'tR_ao', 'ci_lk', 'ci_ao', 'M_lk', 'J_ve', 'tM_lk', 'S0_lk', 'M_rk', 'R2s_ao', 'tS_lk', 'R1_rk', 'C_ao', 'R1_ao', 'tR_lk', 'S_ao', 'tM_ao', 'S_rk', 'tM_rk', 'R2s_rk', 'J_rk', 'J_lk', 'S0_ao', 'R1i_ao', 'tS_ao', 'S_lk', 'R1i_rk', 'J_or', 'R2_ao'}

rois = ['ao', 'lk', 'rk']
tissue_rel = {'ao': RelaxivityArtery, 'lk': RelaxivityKidney, 'rk': RelaxivityKidney}
tissue_wex = {'ao': WaterExchangeArtery, 'lk': WaterExchangeKidney, 'rk': WaterExchangeKidney}

# ROI-specific inputs and outputs
roi_io = tissue_rel['ao'].all_outputs() | tissue_rel['lk'].all_outputs() | tissue_rel['rk'].all_outputs()
roi_io |= tissue_wex['ao'].all_outputs() | tissue_wex['lk'].all_outputs() | tissue_wex['rk'].all_outputs()
roi_io |= {'C', 'ci', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr'} # ConcToSignal inputs
roi_io |= ConcToSignal.all_outputs()

iomap = {
    roi: {k:f'{k}_{roi}' for k in roi_io}
    for roi in rois
}
iomap['ao'] |= {'F_b_ar': 'F_b_ao'}
iomap['lk'] |= {'F_b_ki': 'F_b_lk'}
iomap['rk'] |= {'F_b_ki': 'F_b_rk'}

# ROI-specific configurations
configs = ConcToSignal.configs | WaterExchangeArtery.configs | WaterExchangeKidney.configs | RelaxivityArtery.configs | RelaxivityKidney.configs | ConcAortaKidneys.configs
defaults = ConcToSignal.defaults | WaterExchangeArtery.defaults | WaterExchangeKidney.defaults | RelaxivityArtery.defaults | RelaxivityKidney.defaults | ConcAortaKidneys.defaults

roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

for key in roi_configs:
    config = configs.pop(key)
    default = defaults.pop(key)
    for roi in rois:
        configs[f'{key}_{roi}'] = config
        defaults[f'{key}_{roi}'] = default

cmap = {
    roi: {k:f'{k}_{roi}' for k in roi_configs}
    for roi in rois
}

class AortaKidneysModel(Module):
    """Whole-body model for the aorta and kidneys signal."""

    configs = configs
    defaults = defaults

    _all_inputs = _ALL_INPUTS 
    _all_outputs = _ALL_OUTPUTS

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Only aorta has tof_corr
        config = {
            'ao': self.config,
            'lk': self.config | {'tof_corr': False},
            'rk': self.config | {'tof_corr': False}
        }

        self._conc = ConcAortaKidneys(**self.config)
        self._tissue_rel = {
            roi: tissue_rel[roi](iomap=iomap[roi], cmap=cmap[roi], **config[roi])
            for roi in rois
        }
        self._tissue_wex = {
            roi: tissue_wex[roi](iomap=iomap[roi], cmap=cmap[roi], **config[roi])
            for roi in rois
        }
        self._conc_to_signal = {
            roi: ConcToSignal(iomap=iomap[roi], cmap=cmap[roi], **config[roi]) 
            for roi in rois
        }
        self.map_io(imap, omap)


    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p['tmax'] = p['tstart'] + p['tacq'] + p['dt']

        p |= self._conc(p)
        for roi in rois:
            p |= self._tissue_rel[roi](p) 
            p |= self._tissue_wex[roi](p) 
            p |= self._conc_to_signal[roi](p)

        return self.map_results(p)
        
    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        for roi in rois:
            inputs |= self._tissue_rel[roi].mapped_inputs() 
            inputs |= self._tissue_wex[roi].mapped_inputs() 
            inputs |= self._conc_to_signal[roi].mapped_inputs()

        inputs -= {'tmax'} 
        inputs -= self._conc.new_mapped_outputs()
        for roi in rois:
            inputs -= self._tissue_rel[roi].new_mapped_outputs()
            inputs -= self._tissue_wex[roi].new_mapped_outputs()
            inputs -= self._conc_to_signal[roi].new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in rois:
            outputs |= self._conc_to_signal[roi].mapped_outputs() 
            outputs -= {f'F_b_{roi}'}
        return outputs
    
    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        for roi in rois:
            data |= {
                f'iScal_{roi}': np.arange(n0, dtype=int),
                f'Scal_{roi}': Scal, 
            }
        return data