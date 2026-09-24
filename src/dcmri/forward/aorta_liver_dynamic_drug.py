# +--------------------------------------------------------------------------------------------------+
# |                        ForwardAortaLiverDynamicDrug - all configs (n = 20)                         |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                               |            |
# | tof_corr          | False, True                                                     | False      |
# | inflow            | False, True                                                     | False      |
# | magnitude         | False, True                                                     | True       |
# | trigger           | False, True                                                     | False      |
# | calibrate         | False, True                                                     | False      |
# | water_exchange    | F, N, R                                                         | F          |
# | baseline          | literature, measured                                            | literature |
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

# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             ForwardAortaLiverDynamicDrug - all inputs (n = 130)                                                             |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                     | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+--------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT_1          | sec        | 1st bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_3          | sec        | 3rd bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_4          | sec        | 4th bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                                              | Indicator       | gadoterate |               |       |           |
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
# | Scal_1_ao      | a.u.       | 1st calibration signal in the aorta                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_1_li      | a.u.       | 1st calibration signal in the liver                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_ao      | a.u.       | 2nd calibration signal in the aorta                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_li      | a.u.       | 2nd calibration signal in the liver                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_3_ao      | a.u.       | 3rd calibration signal in the aorta                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_3_li      | a.u.       | 3rd calibration signal in the liver                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_4_ao      | a.u.       | 4th calibration signal in the aorta                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_4_li      | a.u.       | 4th calibration signal in the liver                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal_1_ao     |            | 1st indices of calibration signal in the aorta                           | Signal          | 0          |               |       |           |
# | iScal_1_li     |            | 1st indices of calibration signal in the liver                           | Signal          | 0          |               |       |           |
# | iScal_2_ao     |            | 2nd indices of calibration signal in the aorta                           | Signal          | 0          |               |       |           |
# | iScal_2_li     |            | 2nd indices of calibration signal in the liver                           | Signal          | 0          |               |       |           |
# | iScal_3_ao     |            | 3rd indices of calibration signal in the aorta                           | Signal          | 0          |               |       |           |
# | iScal_3_li     |            | 3rd indices of calibration signal in the liver                           | Signal          | 0          |               |       |           |
# | iScal_4_ao     |            | 4th indices of calibration signal in the aorta                           | Signal          | 0          |               |       |           |
# | iScal_4_li     |            | 4th indices of calibration signal in the liver                           | Signal          | 0          |               |       |           |
# | iStrig_1_ao    |            | 1st indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_1_li    |            | 1st indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_2_ao    |            | 2nd indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_2_li    |            | 2nd indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_3_ao    |            | 3rd indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_3_li    |            | 3rd indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_4_ao    |            | 4th indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_4_li    |            | 4th indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
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

# +------------------------------------------------------------------------------------------------------------------------------+
# |                                      ForwardAortaLiverDynamicDrug - all outputs (n = 82)                                       |
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
# | S0_3_ao  | a.u.     | 3rd signal scaling factor in the aorta         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_3_li  | a.u.     | 3rd signal scaling factor in the liver         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_4_ao  | a.u.     | 4th signal scaling factor in the aorta         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_4_li  | a.u.     | 4th signal scaling factor in the liver         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S_1_ao   | a.u.     | 1st signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_1_li   | a.u.     | 1st signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_2_ao   | a.u.     | 2nd signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_2_li   | a.u.     | 2nd signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_3_ao   | a.u.     | 3rd signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_3_li   | a.u.     | 3rd signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_4_ao   | a.u.     | 4th signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_4_li   | a.u.     | 4th signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# | tS_1_ao  | sec      | 1st signal time points in the aorta            | Signal          | 0.0   |         |       |           |
# | tS_1_li  | sec      | 1st signal time points in the liver            | Signal          | 0.0   |         |       |           |
# | tS_2_ao  | sec      | 2nd signal time points in the aorta            | Signal          | 0.0   |         |       |           |
# | tS_2_li  | sec      | 2nd signal time points in the liver            | Signal          | 0.0   |         |       |           |
# | tS_3_ao  | sec      | 3rd signal time points in the aorta            | Signal          | 0.0   |         |       |           |
# | tS_3_li  | sec      | 3rd signal time points in the liver            | Signal          | 0.0   |         |       |           |
# | tS_4_ao  | sec      | 4th signal time points in the aorta            | Signal          | 0.0   |         |       |           |
# | tS_4_li  | sec      | 4th signal time points in the liver            | Signal          | 0.0   |         |       |           |
# +----------+----------+------------------------------------------------+-----------------+-------+---------+-------+-----------+
# | M_1_ao   | A/cm     | 1st magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_1_li   | A/cm     | 1st magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_2_ao   | A/cm     | 2nd magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_2_li   | A/cm     | 2nd magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_3_ao   | A/cm     | 3rd magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_3_li   | A/cm     | 3rd magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_4_ao   | A/cm     | 4th magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_4_li   | A/cm     | 4th magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
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
# | tM_3_ao  | sec      | 3rd magnetization time points in the aorta     | Electromagnetic | 0.0   |         |       |           |
# | tM_3_li  | sec      | 3rd magnetization time points in the liver     | Electromagnetic | 0.0   |         |       |           |
# | tM_4_ao  | sec      | 4th magnetization time points in the aorta     | Electromagnetic | 0.0   |         |       |           |
# | tM_4_li  | sec      | 4th magnetization time points in the liver     | Electromagnetic | 0.0   |         |       |           |
# | tR_1     | sec      | 1st relaxation rate time points                | Electromagnetic | 0.0   |         |       |           |
# | tR_2     | sec      | 2nd relaxation rate time points                | Electromagnetic | 0.0   |         |       |           |
# +------------------------------------------------------------------------------------------------------------------------------+

from itertools import product

import numpy as np

from dcmri.core.tools import extend_varname, increment_varindex, parse_varname
from dcmri.core.module import Module
from dcmri.forward.aorta_liver_dynamic import ForwardAortaLiverDynamic
from dcmri.bloch.functions_sequences import channels

visits, scans, rois = [1, 2], [1, 2], ['ao', 'li']

configs = ForwardAortaLiverDynamic.configs
defaults = ForwardAortaLiverDynamic.defaults
configs['inflow'].discard('inlet')

class ForwardAortaLiverDynamicDrug(Module):
    """Whole-body model for the aorta and liver signal acquired over 2 separate acquisitions."""

    configs = configs
    defaults = defaults

    _all_inputs = {'dose_3', 'iScal_3_li', 'dose_4', 'NSR_1_ao', 'GFR', 'dose_tolerance', 'FA', 'Scal_3_ao', 'iStrig_1_ao', 'tacq_1', 'Scal_2_ao', 'iz', 'TE', 'iScal_4_li', 'v_li', 'NSR_2_ao', 'SA', 'kf_1_e2h', 'iStrig_4_li', 'Nph', 'TD', 'BAT_4', 'ki_2_e2h', 'R1_h', 'TE1', 'S0_2_ao', 'B1corr_3_li', 'agent', 'S0_4_li', 'Nz', 'field_strength', 'vol_1_ao', 'B1corr_4_li', 'T_la', 'iStrig_1_li', 'B1corr_2_li', 'tacq_3', 'v_h', 'T_hl', 'tacq_4', 'H', 'rate_4', 'S0_2_li', 'vol_2_ao', 'T_b_or', 'iScal_2_ao', 'Scal_1_li', 'dose_1', 'PSw', 'rate_2', 'R1_b', 'iStrig_2_ao', 'TA', 'Scal_3_li', 'tacq_2', 'k_1_e2h', 'tstart_3', 'Scal_4_ao', 'dose_2', 'iStrig_3_ao', 'NSR_4_li', 'tstart_2', 'Tf_1_h', 'Ef_1_li', 'TP', 'B1corr_3_ao', 'tstart_1', 'NSR_1_li', 'S0_1_ao', 'kf_2_e2h', 'Ef_2_li', 'PA', 'B1corr_1_li', 'v_e_li', 'NSR_4_ao', 'iScal_1_ao', 'S0_1_li', 'NSR_2_li', 'ki_1_e2h', 'iScal_1_li', 'iScal_3_ao', 'B1corr_4_ao', 'BAT_3', 'iStrig_4_ao', 'Tf_2_h', 'me', 'B1corr_2_ao', 'k_2_e2h', 'Scal_2_li', 'vol_1_li', 'BAT_2', 'Ei_1_li', 'weight', 'D_hl', 'fCO_li', 'iScal_2_li', 'T_e_or', 'Nk0', 'rate_3', 'TE2', 'S0_4_ao', 'iStrig_2_li', 'NSR_3_li', 'TR', 'S0_3_li', 'dt', 'BAT_1', 'rate_1', 'TF', 'ffa', 'Scal_1_ao', 'iScal_4_ao', 'E_1_li', 'Ti_2_h', 'T_gu', 'T_1_h', 'Ei_2_li', 'NSR_3_ao', 'E_or', 'tstart_4', 'Ti_1_h', 'vol_2_li', 'Scal_4_li', 'T_2_h', 'R1_e', 'S0_3_ao', 'E_2_li', 'B1corr_1_ao', 'CO', 'iStrig_3_li'}
    _all_outputs = {'tS_2_ao', 'tM_3_li', 'J_2_lag', 'S_3_li', 'M_1_ao', 'J_1_ao', 'M_3_li', 'ci_1_li', 'R1i_2_li', 'R1i_2_ao', 'C_2_li', 'tS_1_li', 'J_1_ve', 'J_1_lag', 'J_2_pv', 'S0_2_ao', 'R2s_1_ao', 'C_2_ao', 'S0_4_li', 'tC_2', 'ci_2_ao', 'ci_1_ao', 'R1_2_ao', 'S_1_ao', 'R2s_1_li', 'R2s_2_li', 'M_2_li', 'tS_4_li', 'S_3_ao', 'tR_2', 'R1_1_li', 'C_1_li', 'R2_2_ao', 'S0_2_li', 'tS_3_ao', 'M_4_li', 'R2_2_li', 'tS_1_ao', 'tM_4_li', 'J_1_li', 'J_1_or', 'S_4_ao', 'S_2_li', 'R1i_1_ao', 'tR_1', 'tS_2_li', 'tS_4_ao', 'S0_1_ao', 'J_2_ao', 'J_2_la', 'tC_1', 'ci_2_li', 'J_2_ve', 'M_1_li', 'R1i_1_li', 'S0_1_li', 'R2s_2_ao', 'J_2_or', 'tM_4_ao', 'tM_2_ao', 'C_1_ao', 'tM_1_li', 'S_1_li', 'R1_1_ao', 'M_3_ao', 'S0_4_ao', 'tM_1_ao', 'tS_3_li', 'R2_1_ao', 'J_1_pv', 'S0_3_li', 'tM_2_li', 'R1_2_li', 'S_4_li', 'M_2_ao', 'J_1_la', 'S_2_ao', 'J_2_li', 'tM_3_ao', 'S0_3_ao', 'R2_1_li', 'M_4_ao'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        for visit in visits:
            p |= self._aol[visit](p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Index inputs and outputs
        visit_inputs = {'BAT', 'dose', 'rate'}
        visit_inputs |= {'E_li', 'Ef_li', 'Ei_li', 'T_h', 'Tf_h', 'Ti_h', 'k_e2h', 'kf_e2h', 'ki_e2h', 'vol_ao', 'vol_li'} 

        vars = {'BAT', 'dose', 'rate', 'tacq', 'tstart'}
        visit_scan_inputs = {extend_varname(k, index=i) for k, i in product(vars, scans)}

        vars = {'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr'}
        visit_scan_inputs |= {extend_varname(k, roi=roi, index=i) for k, roi, i in product(vars, rois, scans)} 

        vars = ForwardAortaLiverDynamic.all_outputs()
        visit_outputs = {o for o in vars if parse_varname(o)['index'] is None}
        visit_scan_outputs = {o for o in vars if parse_varname(o)['index'] is not None}

        self._aol = {}

        for visit in visits:
            imap_visit = {k: extend_varname(k, index=visit) for k in visit_inputs}
            omap_visit = {k: extend_varname(k, index=visit) for k in visit_outputs}
            if visit==2: 
                imap_visit |= {k: increment_varindex(k, 2) for k in visit_scan_inputs} 
                omap_visit |= {k: increment_varindex(k, 2) for k in visit_scan_outputs}
            
            self._aol[visit] = ForwardAortaLiverDynamic(imap=imap_visit, omap=omap_visit, **self.config)

        self.map_io(imap, omap)

    def inputs(self) -> set:
        inputs = set()
        for visit in visits:
            inputs |= self._aol[visit].mapped_inputs()
        return inputs 
    
    def outputs(self):
        outputs = set()
        for visit in visits:
            outputs |= self._aol[visit].mapped_outputs()
        return outputs

    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1
        scan_index = {(1, 1): 1, (1, 2): 2, (2, 1): 3, (2, 2): 4} # index for each (visit, scan) pair

        for roi in rois:
            for visit in visits:
                for scan in scans:
                    idx = scan_index[visit, scan]
                    data |= {
                        f'iScal_{idx}_{roi}': np.arange(n0, dtype=int),
                        f'Scal_{idx}_{roi}': Scal, 
                    }
        data |= {
            'tacq_1': 90,
            'tstart_2': 120,
            'tacq_2': 60,
            'tacq_3': 90,
            'tstart_4': 120,
            'tacq_4': 60,           
        }

        # data['E_2_li'] /= 10 # create some effect
        return data