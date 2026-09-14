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
# | water_exchange    | F, N, R                                                         | F          |
# | baseline          | literature, measured                                            | literature |
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

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                          AortaKidneysModel - all inputs (n = 85)                                                          |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | BAT            | sec        | bolus arrival time                                      | Indicator       | 30         | (-30, 30)      |       |           |
# | BAT_1          | sec        | 1st bolus arrival time                                  | Indicator       | 30         | (-30, 30)      |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                  | Indicator       | 30         | (-30, 30)      |       |           |
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |                |       |           |
# | dose           | mL/kg      | contrast agent dose                                     | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)       |       |           |
# | rate           | mL/s       | injection rate                                          | Indicator       | 1          | (0, 10)        |       |           |
# | rate_1         | mL/s       | 1st injection rate                                      | Indicator       | 1          | (0, 10)        |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                      | Indicator       | 1          | (0, 10)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR_ao         |            | noise-to-signal ratio in the aorta                      | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | NSR_lk         |            | noise-to-signal ratio in the left kidney                | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | NSR_rk         |            | noise-to-signal ratio in the right kidney               | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0_ao          | a.u.       | signal scaling factor in the aorta                      | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | S0_lk          | a.u.       | signal scaling factor in the left kidney                | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | S0_rk          | a.u.       | signal scaling factor in the right kidney               | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | Scal_ao        | a.u.       | calibration signal in the aorta                         | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | Scal_lk        | a.u.       | calibration signal in the left kidney                   | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | Scal_rk        | a.u.       | calibration signal in the right kidney                  | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | iScal_ao       |            | indices of calibration signal in the aorta              | Signal          | 0          |                |       |           |
# | iScal_lk       |            | indices of calibration signal in the left kidney        | Signal          | 0          |                |       |           |
# | iScal_rk       |            | indices of calibration signal in the right kidney       | Signal          | 0          |                |       |           |
# | iStrig_ao      |            | indices of the signal trigger in the aorta              | Signal          | None       |                |       |           |
# | iStrig_lk      |            | indices of the signal trigger in the left kidney        | Signal          | None       |                |       |           |
# | iStrig_rk      |            | indices of the signal trigger in the right kidney       | Signal          | None       |                |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                              | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)       |       |           |
# | SA             | deg        | saturation Slab Flip Angle                              | Sequence        | 0          | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)      |       |           |
# | tacq           | sec        | acquisition duration                                    | Sequence        | 240        | (0, 10000.0)   |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr_ao      |            | B1-correction factor in the aorta                       | Electromagnetic | 1          | (0, 5)         |       |           |
# | B1corr_lk      |            | B1-correction factor in the left kidney                 | Electromagnetic | 1          | (0, 5)         |       |           |
# | B1corr_rk      |            | B1-correction factor in the right kidney                | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ki          | Hz         | tissue R1 in the kidney                                 | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_u           | Hz         | tissue R1 in tubuli                                     | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                          | Physiological   | 100        | (0, 500)       |       |           |
# | DRPF           |            | Differential renal plasma flow                          | Physiological   | 0.5        | (0, 1)         |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs          | Physiological   | 0.2        | (0.01, 0.99)   |       |           |
# | E_li           |            | extraction fraction in the liver                        | Physiological   | 0.1        | (0.0, 1.0)     |       |           |
# | E_or           |            | extraction fraction in the organs                       | Physiological   | 0.15       | (0, 0.5)       |       |           |
# | FF_lk          |            | filtration fraction in the left kidney                  | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | FF_rk          |            | filtration fraction in the right kidney                 | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | F_u            | mL/sec/cm3 | flow per unit tissue in tubuli                          | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | F_u_lk         | mL/sec/cm3 | flow per unit tissue in tubuli of the left kidney       | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | F_u_rk         | mL/sec/cm3 | flow per unit tissue in tubuli of the right kidney      | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | H              |            | hematocrit                                              | Physiological   | 0.45       | (0, 1)         |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                 | Physiological   | 0.03       | (0, 100)       |       |           |
# | TF             | sec        | inflow time                                             | Physiological   | 0.5        | (0, 10)        |       |           |
# | T_ar           | sec        | mean transit time in the artery                         | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                | Physiological   | 20         | (0, 60)        |       |           |
# | T_e_or         | sec        | mean transit time in extracellular space of the organs  | Physiological   | 120        | (0, 800)       |       |           |
# | T_gu           | sec        | mean transit time in the gut                            | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                | Physiological   | 10         | (0, 30)        |       |           |
# | T_u_lk         | sec        | mean transit time in tubuli of the left kidney          | Physiological   | 120        | (0, 600)       |       |           |
# | T_u_rk         | sec        | mean transit time in tubuli of the right kidney         | Physiological   | 120        | (0, 600)       |       |           |
# | fCO_ki         |            | fraction of the cardiac output in the kidney            | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | h_u_lk         | Hz         | transit time distribution in tubuli of the left kidney  | Physiological   | 0          | (0.1, 60)      |       |           |
# | h_u_rk         | Hz         | transit time distribution in tubuli of the right kidney | Physiological   | 0          | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                            | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_ki           | mL/cm3     | volume fraction in the kidney                           | Physiological   | 1          | (0, 1)         |       |           |
# | v_p_lk         | mL/cm3     | volume fraction in plasma of the left kidney            | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_p_rk         | mL/cm3     | volume fraction in plasma of the right kidney           | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_u            | mL/cm3     | volume fraction in tubuli                               | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                          | Hyperparameters | 0.1        |                |       |           |
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |                |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | vol_ao         | cm3        | ROI volume in the aorta                                 | Whole-body      | 10         | (0.0, 1000)    |       |           |
# | vol_lk         | cm3        | ROI volume in the left kidney                           | Whole-body      | 150        | (0.0, 1000)    |       |           |
# | vol_rk         | cm3        | ROI volume in the right kidney                          | Whole-body      | 150        | (0.0, 1000)    |       |           |
# | weight         | kg         | body weight                                             | Whole-body      | 70         | (0, 300)       |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------------+
# |                                              AortaKidneysModel - all outputs (n = 42)                                              |
# +--------+------------+----------------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | Key    | Unit       | Name                                               | Group           | Init  | Bounds    | DICOM | OSIPI     |
# +--------+------------+----------------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | C_ao   | mmol/cm3   | tissue concentration in the aorta                  | Indicator       | 0.005 | (0, 1)    |       |           |
# | C_lk   | mmol/cm3   | tissue concentration in the left kidney            | Indicator       | 0.005 | (0, 1)    |       |           |
# | C_rk   | mmol/cm3   | tissue concentration in the right kidney           | Indicator       | 0.005 | (0, 1)    |       |           |
# | J_ao   | mmol/sec   | indicator flux in the aorta                        | Indicator       | 1     | (0, 10)   |       |           |
# | J_lk   | mmol/sec   | indicator flux in the left kidney                  | Indicator       | 1     | (0, 10)   |       |           |
# | J_or   | mmol/sec   | indicator flux in the organs                       | Indicator       | 1     | (0, 10)   |       |           |
# | J_rk   | mmol/sec   | indicator flux in the right kidney                 | Indicator       | 1     | (0, 10)   |       |           |
# | J_ve   | mmol/sec   | indicator flux in the vein                         | Indicator       | 1     | (0, 10)   |       |           |
# | ci_ao  | mmol/mL    | inlet concentration in the aorta                   | Indicator       | 0.005 |           |       |           |
# | ci_lk  | mmol/mL    | inlet concentration in the left kidney             | Indicator       | 0.005 |           |       |           |
# | ci_rk  | mmol/mL    | inlet concentration in the right kidney            | Indicator       | 0.005 |           |       |           |
# | tC     | sec        | concentration time points                          | Indicator       | 0.0   |           |       |           |
# +--------+------------+----------------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | S0_ao  | a.u.       | signal scaling factor in the aorta                 | Signal          | 1.0   | (0, 5)    |       | Q.MS1.010 |
# | S0_lk  | a.u.       | signal scaling factor in the left kidney           | Signal          | 1.0   | (0, 5)    |       | Q.MS1.010 |
# | S0_rk  | a.u.       | signal scaling factor in the right kidney          | Signal          | 1.0   | (0, 5)    |       | Q.MS1.010 |
# | S_ao   | a.u.       | signal in the aorta                                | Signal          | 1.0   | (0, 5)    |       |           |
# | S_lk   | a.u.       | signal in the left kidney                          | Signal          | 1.0   | (0, 5)    |       |           |
# | S_rk   | a.u.       | signal in the right kidney                         | Signal          | 1.0   | (0, 5)    |       |           |
# | tS_ao  | sec        | signal time points in the aorta                    | Signal          | 0.0   |           |       |           |
# | tS_lk  | sec        | signal time points in the left kidney              | Signal          | 0.0   |           |       |           |
# | tS_rk  | sec        | signal time points in the right kidney             | Signal          | 0.0   |           |       |           |
# +--------+------------+----------------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | M_ao   | A/cm       | magnetization in the aorta                         | Electromagnetic | 1     | (0, 5)    |       |           |
# | M_lk   | A/cm       | magnetization in the left kidney                   | Electromagnetic | 1     | (0, 5)    |       |           |
# | M_rk   | A/cm       | magnetization in the right kidney                  | Electromagnetic | 1     | (0, 5)    |       |           |
# | R1_ao  | Hz         | tissue R1 in the aorta                             | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1_lk  | Hz         | tissue R1 in the left kidney                       | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1_rk  | Hz         | tissue R1 in the right kidney                      | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1i_ao | Hz         | inlet R1 in the aorta                              | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1i_lk | Hz         | inlet R1 in the left kidney                        | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1i_rk | Hz         | inlet R1 in the right kidney                       | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R2_ao  | Hz         | tissue R2 in the aorta                             | Electromagnetic | 2.0   | (0, 5)    |       |           |
# | R2_lk  | Hz         | tissue R2 in the left kidney                       | Electromagnetic | 2.0   | (0, 5)    |       |           |
# | R2_rk  | Hz         | tissue R2 in the right kidney                      | Electromagnetic | 2.0   | (0, 5)    |       |           |
# | R2s_ao | Hz         | tissue R2* in the aorta                            | Electromagnetic | 20    | (0, 5)    |       |           |
# | R2s_lk | Hz         | tissue R2* in the left kidney                      | Electromagnetic | 20    | (0, 5)    |       |           |
# | R2s_rk | Hz         | tissue R2* in the right kidney                     | Electromagnetic | 20    | (0, 5)    |       |           |
# | tM_ao  | sec        | magnetization time points in the aorta             | Electromagnetic | 0.0   |           |       |           |
# | tM_lk  | sec        | magnetization time points in the left kidney       | Electromagnetic | 0.0   |           |       |           |
# | tM_rk  | sec        | magnetization time points in the right kidney      | Electromagnetic | 0.0   |           |       |           |
# | tR     | sec        | relaxation rate time points                        | Electromagnetic | 0.0   |           |       |           |
# +--------+------------+----------------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | F_u_lk | mL/sec/cm3 | flow per unit tissue in tubuli of the left kidney  | Physiological   | 0.005 | (0, 0.05) |       |           |
# | F_u_rk | mL/sec/cm3 | flow per unit tissue in tubuli of the right kidney | Physiological   | 0.005 | (0, 0.05) |       |           |
# +------------------------------------------------------------------------------------------------------------------------------------+

import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import extend_varname
from dcmri.kinetics.modules_conc import ConcAortaKidneys
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityKidney
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeKidney
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels


rois = ['ao', 'lk', 'rk']
tissue_rel = {'ao': RelaxivityArtery, 'lk': RelaxivityKidney, 'rk': RelaxivityKidney}
tissue_wex = {'ao': WaterExchangeArtery, 'lk': WaterExchangeKidney, 'rk': WaterExchangeKidney}

# ROI-specific configurations
roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

CONFIGS = ConcToSignal.configs | WaterExchangeArtery.configs | WaterExchangeKidney.configs | RelaxivityArtery.configs | RelaxivityKidney.configs | ConcAortaKidneys.configs
DEFAULTS = ConcToSignal.defaults | WaterExchangeArtery.defaults | WaterExchangeKidney.defaults | RelaxivityArtery.defaults | RelaxivityKidney.defaults | ConcAortaKidneys.defaults
CMAP = {roi: {} for roi in rois}

for key in roi_configs:
    config = CONFIGS.pop(key)
    default = DEFAULTS.pop(key)
    for roi in rois:
        CONFIGS[f'{key}_{roi}'] = config
        DEFAULTS[f'{key}_{roi}'] = default
        CMAP[roi] |= {key: f'{key}_{roi}'}


class AortaKidneysModel(Module):
    """Whole-body model for the aorta and kidneys signal."""

    configs = CONFIGS
    defaults = DEFAULTS

    _all_inputs = {'R1_ki', 'fCO_ki', 'TA', 'iStrig_ao', 'FF_rk', 'h_u_rk', 'tacq', 'DRPF', 'T_hl', 'dose', 'D_hl', 'NSR_rk', 'R1_b', 'F_u_rk', 'TF', 'tstart', 'Nk0', 'rate_2', 'B1corr_rk', 'vol_lk', 'T_ar', 'Scal_ao', 'F_u', 'h_u_lk', 'R1_u', 'E_or', 'iScal_lk', 'BAT', 'TE1', 'R1_c', 'SA', 'T_gu', 'S0_rk', 'B1corr_ao', 'vol_ao', 'BAT_1', 'T_e_or', 'weight', 'me', 'Scal_lk', 'NSR_lk', 'iScal_ao', 'Nz', 'FA', 'F_u_lk', 'dose_1', 'T_u_lk', 'Nph', 'CO', 'TE', 'B1corr_lk', 'NSR_ao', 'v_c', 'E_li', 'rate_1', 'vol_rk', 'TE2', 'dose_2', 'iz', 'v_p_rk', 'agent', 'v_u', 'T_u_rk', 'S0_lk', 'Scal_rk', 'T_b_or', 'PSw', 'TD', 'PA', 'v_ki', 'TR', 'rate', 'v_b', 'iStrig_rk', 'v_p_lk', 'BAT_2', 'iScal_rk', 'dose_tolerance', 'dt', 'H', 'field_strength', 'TP', 'iStrig_lk', 'S0_ao', 'FF_lk'}
    _all_outputs = {'ci_rk', 'M_lk', 'S_rk', 'J_lk', 'F_u_rk', 'tC', 'J_rk', 'R2_ao', 'M_ao', 'R1_ao', 'tM_ao', 'S_ao', 'S0_rk', 'R2s_lk', 'R1_lk', 'R2_rk', 'C_lk', 'J_ao', 'J_ve', 'F_u_lk', 'M_rk', 'R2_lk', 'tS_ao', 'S_lk', 'R1i_lk', 'R1i_ao', 'tM_rk', 'ci_ao', 'tM_lk', 'R1i_rk', 'R2s_rk', 'S0_lk', 'tS_lk', 'R1_rk', 'ci_lk', 'R2s_ao', 'J_or', 'C_rk', 'tS_rk', 'C_ao', 'tR', 'S0_ao'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p['tmax'] = p['tstart'] + p['tacq'] + p['dt']

        p |= self._conc(p)
        for roi in rois:
            p |= self._tissue_rel[roi](p) 
            p |= self._tissue_wex[roi](p) 
            p |= self._conc_to_signal[roi](p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)

        # Only aorta has tof_corr
        config = {
            'ao': self.config,
            'lk': self.config | {'tof_corr': False},
            'rk': self.config | {'tof_corr': False}
        }
        self._conc = ConcAortaKidneys(**self.config)
        self._tissue_rel = {}
        self._tissue_wex = {}
        self._conc_to_signal = {}

        roimap = {'ao': 'ar', 'lk': 'ki', 'rk': 'ki'}

        for roi in rois:
            iomap_roi = {f"F_b_{roimap[roi]}": f"F_b_{roi}"}
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in tissue_rel[roi].all_outputs() | tissue_wex[roi].all_outputs()} 
            
            self._tissue_rel[roi] = tissue_rel[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])
            self._tissue_wex[roi] = tissue_wex[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])

            vars = {'C', 'ci', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr', 'S', 'M', 'R1', 'R1i', 'R2', 'R2s', 'tM', 'tS'}
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in vars}

            self._conc_to_signal[roi] = ConcToSignal(iomap=iomap_roi, cmap=CMAP[roi], **config[roi]) 

        self.map_io(imap, omap, iomap)

   
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