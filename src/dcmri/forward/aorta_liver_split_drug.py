# +--------------------------------------------------------------------------------------------------+
# |                            AortaLiverSplitDrug - all configs (n = 20)                            |
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

# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                 AortaLiverSplitDrug - all inputs (n = 175)                                                                |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | Key            | Value      | Unit       | Name                                                                     | Group           | Bounds        | DICOM | OSIPI     |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | B1corr_1_ao    | 1          |            | 1st B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_1_li    | 1          |            | 1st B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_ao    | 1          |            | 2nd B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_li    | 1          |            | 2nd B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_3_ao    | 1          |            | 3rd B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_3_li    | 1          |            | 3rd B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_4_ao    | 1          |            | 4th B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_4_li    | 1          |            | 4th B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
# | me             | 1          | A cm2/mL   | equilibrium magnetization                                                | electromagnetic | (0, 5)        |       |           |
# | R1_b           | 0.65       | Hz         | tissue R1 in the blood                                                   | electromagnetic | (0, 5)        |       |           |
# | R1_e           | 0.65       | Hz         | tissue R1 in extracellular space                                         | electromagnetic | (0, 5)        |       |           |
# | R1_h           | 0.65       | Hz         | tissue R1 in hepatocytes                                                 | electromagnetic | (0, 5)        |       |           |
# | R1_li          | 0.65       | Hz         | tissue R1 in the liver                                                   | electromagnetic | (0, 5)        |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | dose_tolerance | 0.1        |            | dose tolerance                                                           | hyperparameters |               |       |           |
# | dt             | 0.5        | sec        | pseudo-continuous time step                                              | hyperparameters |               |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | agent          | gadoterate |            | contrast agent generic name                                              | indicator       |               |       |           |
# | BAT_1          | 30         | sec        | 1st bolus arrival time                                                   | indicator       | (-30, 30)     |       |           |
# | BAT_2          | 30         | sec        | 2nd bolus arrival time                                                   | indicator       | (-30, 30)     |       |           |
# | BAT_3          | 30         | sec        | 3rd bolus arrival time                                                   | indicator       | (-30, 30)     |       |           |
# | BAT_4          | 30         | sec        | 4th bolus arrival time                                                   | indicator       | (-30, 30)     |       |           |
# | bdel           | 30         | sec        | delay in a double injection                                              | indicator       | (-30, 30)     |       |           |
# | dose_1         | 0.1        | mL/kg      | 1st contrast agent dose                                                  | indicator       | (0, 0.2)      |       |           |
# | dose_2         | 0.1        | mL/kg      | 2nd contrast agent dose                                                  | indicator       | (0, 0.2)      |       |           |
# | dose_3         | 0.1        | mL/kg      | 3rd contrast agent dose                                                  | indicator       | (0, 0.2)      |       |           |
# | dose_4         | 0.1        | mL/kg      | 4th contrast agent dose                                                  | indicator       | (0, 0.2)      |       |           |
# | rate_1         | 1          | mL/sec     | 1st injection rate                                                       | indicator       | (0, 10)       |       |           |
# | rate_2         | 1          | mL/sec     | 2nd injection rate                                                       | indicator       | (0, 10)       |       |           |
# | rate_3         | 1          | mL/sec     | 3rd injection rate                                                       | indicator       | (0, 10)       |       |           |
# | rate_4         | 1          | mL/sec     | 4th injection rate                                                       | indicator       | (0, 10)       |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | CO             | 100        | mL/sec     | cardiac output                                                           | physiological   | (0, 500)      |       |           |
# | D_hl           | 0.2        |            | transit time dispersion in the heart and Lungs                           | physiological   | (0.01, 0.99)  |       |           |
# | E_1_li         | 0.1        |            | 1st extraction fraction in the liver                                     | physiological   | (0.0, 1.0)    |       |           |
# | E_2_li         | 0.1        |            | 2nd extraction fraction in the liver                                     | physiological   | (0.0, 1.0)    |       |           |
# | E_or           | 0.15       |            | extraction fraction in the organs                                        | physiological   | (0, 0.5)      |       |           |
# | Ef_1_li        | 0.1        |            | 1st final extraction fraction in the liver                               | physiological   | (0.0, 1.0)    |       |           |
# | Ef_2_li        | 0.1        |            | 2nd final extraction fraction in the liver                               | physiological   | (0.0, 1.0)    |       |           |
# | Ei_1_li        | 0.1        |            | 1st initial extraction fraction in the liver                             | physiological   | (0.0, 1.0)    |       |           |
# | Ei_2_li        | 0.1        |            | 2nd initial extraction fraction in the liver                             | physiological   | (0.0, 1.0)    |       |           |
# | fCO_li         | 0.1        |            | fraction of the cardiac output in the liver                              | physiological   | (0, 0.5)      |       |           |
# | ffa            | 0.2        |            | arterial flow fraction                                                   | physiological   | (0, 1)        |       |           |
# | H              | 0.45       |            | hematocrit                                                               | physiological   | (0, 1)        |       |           |
# | k_1_e2h        | 0.0025     | mL/sec/cm3 | 1st tissue transfer rate from extracellular space to hepatocytes         | physiological   | (0, 0.006)    |       |           |
# | k_2_e2h        | 0.0025     | mL/sec/cm3 | 2nd tissue transfer rate from extracellular space to hepatocytes         | physiological   | (0, 0.006)    |       |           |
# | kf_1_e2h       | 0.0025     | mL/sec/cm3 | 1st final tissue transfer rate from extracellular space to hepatocytes   | physiological   | (0, 0.006)    |       |           |
# | kf_2_e2h       | 0.0025     | mL/sec/cm3 | 2nd final tissue transfer rate from extracellular space to hepatocytes   | physiological   | (0, 0.006)    |       |           |
# | ki_1_e2h       | 0.0025     | mL/sec/cm3 | 1st initial tissue transfer rate from extracellular space to hepatocytes | physiological   | (0, 0.006)    |       |           |
# | ki_2_e2h       | 0.0025     | mL/sec/cm3 | 2nd initial tissue transfer rate from extracellular space to hepatocytes | physiological   | (0, 0.006)    |       |           |
# | PSw            | 0.03       | mL/sec/cm3 | water permeability-surface area product                                  | physiological   | (0, 100)      |       |           |
# | T_1_h          | 14400      | sec        | 1st mean transit time in hepatocytes                                     | physiological   | (600, 21600)  |       |           |
# | T_2_h          | 14400      | sec        | 2nd mean transit time in hepatocytes                                     | physiological   | (600, 21600)  |       |           |
# | T_b_or         | 20         | sec        | mean transit time in blood of the organs                                 | physiological   | (0, 60)       |       |           |
# | T_e_or         | 120        | sec        | mean transit time in extracellular space of the organs                   | physiological   | (0, 800)      |       |           |
# | T_gu           | 30         | sec        | mean transit time in the gut                                             | physiological   | (0.1, 60)     |       |           |
# | T_hl           | 10         | sec        | mean transit time in the heart and Lungs                                 | physiological   | (0, 30)       |       |           |
# | T_la           | 30         | sec        | mean transit time in the liver artery                                    | physiological   | (0.0, 60)     |       |           |
# | TF             | 0.25       | sec        | inflow time                                                              | physiological   | (0, 1)        |       |           |
# | Tf_1_h         | 14400      | sec        | 1st final mean transit time in hepatocytes                               | physiological   | (600, 21600)  |       |           |
# | Tf_2_h         | 14400      | sec        | 2nd final mean transit time in hepatocytes                               | physiological   | (600, 21600)  |       |           |
# | Ti_1_h         | 14400      | sec        | 1st initial mean transit time in hepatocytes                             | physiological   | (600, 21600)  |       |           |
# | Ti_2_h         | 14400      | sec        | 2nd initial mean transit time in hepatocytes                             | physiological   | (600, 21600)  |       |           |
# | v_e            | 0.3        | mL/cm3     | volume fraction in extracellular space                                   | physiological   | (0.01, 0.6)   |       |           |
# | v_e_li         | 0.25       | mL/cm3     | volume fraction in extracellular space of the liver                      | physiological   | (0.01, 0.5)   |       |           |
# | v_h            | 1          | mL/cm3     | volume fraction in hepatocytes                                           | physiological   | (0, 1)        |       |           |
# | v_li           | 1          | mL/cm3     | volume fraction in the liver                                             | physiological   | (0, 1)        |       |           |
# | vr_or          | 0.9        |            | Venous return in the organs                                              | physiological   | (0, 1)        |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | FA_1           | 15         | deg        | 1st flip angle                                                           | sequence        | (0, 180)      |       |           |
# | FA_2           | 15         | deg        | 2nd flip angle                                                           | sequence        | (0, 180)      |       |           |
# | FA_3           | 15         | deg        | 3rd flip angle                                                           | sequence        | (0, 180)      |       |           |
# | FA_4           | 15         | deg        | 4th flip angle                                                           | sequence        | (0, 180)      |       |           |
# | field_strength | 3          | T          | magnetic field strength                                                  | sequence        | (0, 20)       |       |           |
# | iz_1           | 0          |            | 1st slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | iz_2           | 0          |            | 2nd slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | iz_3           | 0          |            | 3rd slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | iz_4           | 0          |            | 4th slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | Nk0_1          | 64         |            | 1st number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nk0_2          | 64         |            | 2nd number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nk0_3          | 64         |            | 3rd number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nk0_4          | 64         |            | 4th number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nph_1          | 128        |            | 1st number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nph_2          | 128        |            | 2nd number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nph_3          | 128        |            | 3rd number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nph_4          | 128        |            | 4th number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nz_1           | 64         |            | 1st number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | Nz_2           | 64         |            | 2nd number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | Nz_3           | 64         |            | 3rd number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | Nz_4           | 64         |            | 4th number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | PA_1           | 90         | deg        | 1st preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | PA_2           | 90         | deg        | 2nd preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | PA_3           | 90         | deg        | 3rd preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | PA_4           | 90         | deg        | 4th preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | SA_1           | 0          | deg        | 1st saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | SA_2           | 0          | deg        | 2nd saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | SA_3           | 0          | deg        | 3rd saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | SA_4           | 0          | deg        | 4th saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | TA_1           | 2.0        | sec        | 1st acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | TA_2           | 2.0        | sec        | 2nd acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | TA_3           | 2.0        | sec        | 3rd acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | TA_4           | 2.0        | sec        | 4th acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | tacq_1         | 240        | sec        | 1st acquisition duration                                                 | sequence        | (0, 10000.0)  |       |           |
# | tacq_2         | 240        | sec        | 2nd acquisition duration                                                 | sequence        | (0, 10000.0)  |       |           |
# | tacq_3         | 240        | sec        | 3rd acquisition duration                                                 | sequence        | (0, 10000.0)  |       |           |
# | tacq_4         | 240        | sec        | 4th acquisition duration                                                 | sequence        | (0, 10000.0)  |       |           |
# | TD_1           | 0.05       | sec        | 1st prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TD_2           | 0.05       | sec        | 2nd prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TD_3           | 0.05       | sec        | 3rd prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TD_4           | 0.05       | sec        | 4th prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TE1_1          | 0.001      | sec        | 1st first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE1_2          | 0.001      | sec        | 2nd first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE1_3          | 0.001      | sec        | 3rd first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE1_4          | 0.001      | sec        | 4th first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE2_1          | 0.005      | sec        | 1st second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE2_2          | 0.005      | sec        | 2nd second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE2_3          | 0.005      | sec        | 3rd second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE2_4          | 0.005      | sec        | 4th second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE_1           | 0.001      | sec        | 1st echo time                                                            | sequence        | (0, 10)       |       |           |
# | TE_2           | 0.001      | sec        | 2nd echo time                                                            | sequence        | (0, 10)       |       |           |
# | TE_3           | 0.001      | sec        | 3rd echo time                                                            | sequence        | (0, 10)       |       |           |
# | TE_4           | 0.001      | sec        | 4th echo time                                                            | sequence        | (0, 10)       |       |           |
# | TP_1           | 0.05       | sec        | 1st preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TP_2           | 0.05       | sec        | 2nd preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TP_3           | 0.05       | sec        | 3rd preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TP_4           | 0.05       | sec        | 4th preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TR_1           | 0.005      | sec        | 1st repetition time                                                      | sequence        | (0, 1)        |       |           |
# | TR_2           | 0.005      | sec        | 2nd repetition time                                                      | sequence        | (0, 1)        |       |           |
# | TR_3           | 0.005      | sec        | 3rd repetition time                                                      | sequence        | (0, 1)        |       |           |
# | TR_4           | 0.005      | sec        | 4th repetition time                                                      | sequence        | (0, 1)        |       |           |
# | tstart_1       | 0          | sec        | 1st start of the acquisition                                             | sequence        | (0, 10000.0)  |       |           |
# | tstart_2       | 0          | sec        | 2nd start of the acquisition                                             | sequence        | (0, 10000.0)  |       |           |
# | tstart_3       | 0          | sec        | 3rd start of the acquisition                                             | sequence        | (0, 10000.0)  |       |           |
# | tstart_4       | 0          | sec        | 4th start of the acquisition                                             | sequence        | (0, 10000.0)  |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iScal_1_ao     | 0          |            | 1st indices of calibration signal in the aorta                           | signal          |               |       |           |
# | iScal_1_li     | 0          |            | 1st indices of calibration signal in the liver                           | signal          |               |       |           |
# | iScal_2_ao     | 0          |            | 2nd indices of calibration signal in the aorta                           | signal          |               |       |           |
# | iScal_2_li     | 0          |            | 2nd indices of calibration signal in the liver                           | signal          |               |       |           |
# | iScal_3_ao     | 0          |            | 3rd indices of calibration signal in the aorta                           | signal          |               |       |           |
# | iScal_3_li     | 0          |            | 3rd indices of calibration signal in the liver                           | signal          |               |       |           |
# | iScal_4_ao     | 0          |            | 4th indices of calibration signal in the aorta                           | signal          |               |       |           |
# | iScal_4_li     | 0          |            | 4th indices of calibration signal in the liver                           | signal          |               |       |           |
# | iStrig_1_ao    | None       |            | 1st indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_1_li    | None       |            | 1st indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_2_ao    | None       |            | 2nd indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_2_li    | None       |            | 2nd indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_3_ao    | None       |            | 3rd indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_3_li    | None       |            | 3rd indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_4_ao    | None       |            | 4th indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_4_li    | None       |            | 4th indices of the signal trigger in the liver                           | signal          |               |       |           |
# | NSR_1_ao       | 0.0        |            | 1st noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_1_li       | 0.0        |            | 1st noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | NSR_2_ao       | 0.0        |            | 2nd noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_2_li       | 0.0        |            | 2nd noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | NSR_3_ao       | 0.0        |            | 3rd noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_3_li       | 0.0        |            | 3rd noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | NSR_4_ao       | 0.0        |            | 4th noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_4_li       | 0.0        |            | 4th noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | S0_1_ao        | 1.0        | a.u.       | 1st signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | 1.0        | a.u.       | 1st signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | 1.0        | a.u.       | 2nd signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | 1.0        | a.u.       | 2nd signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_3_ao        | 1.0        | a.u.       | 3rd signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_3_li        | 1.0        | a.u.       | 3rd signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_4_ao        | 1.0        | a.u.       | 4th signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_4_li        | 1.0        | a.u.       | 4th signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | Scal_1_ao      | 1.0        | a.u.       | 1st calibration signal in the aorta                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_1_li      | 1.0        | a.u.       | 1st calibration signal in the liver                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_ao      | 1.0        | a.u.       | 2nd calibration signal in the aorta                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_2_li      | 1.0        | a.u.       | 2nd calibration signal in the liver                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_3_ao      | 1.0        | a.u.       | 3rd calibration signal in the aorta                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_3_li      | 1.0        | a.u.       | 3rd calibration signal in the liver                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_4_ao      | 1.0        | a.u.       | 4th calibration signal in the aorta                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_4_li      | 1.0        | a.u.       | 4th calibration signal in the liver                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_1_ao       | 1          | cm3        | 1st ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_1_li       | 1000       | cm3        | 1st ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | vol_2_ao       | 1          | cm3        | 2nd ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_2_li       | 1000       | cm3        | 2nd ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                              | whole-body      | (0, 300)      |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------+
# |                                          AortaLiverSplitDrug - all outputs (n = 82)                                          |
# +----------+-------+----------+------------------------------------------------+-----------------+---------+-------+-----------+
# | Key      | Value | Unit     | Name                                           | Group           | Bounds  | DICOM | OSIPI     |
# +----------+-------+----------+------------------------------------------------+-----------------+---------+-------+-----------+
# | M_1_ao   | 1     | A/cm     | 1st magnetization in the aorta                 | electromagnetic | (0, 5)  |       |           |
# | M_1_li   | 1     | A/cm     | 1st magnetization in the liver                 | electromagnetic | (0, 5)  |       |           |
# | M_2_ao   | 1     | A/cm     | 2nd magnetization in the aorta                 | electromagnetic | (0, 5)  |       |           |
# | M_2_li   | 1     | A/cm     | 2nd magnetization in the liver                 | electromagnetic | (0, 5)  |       |           |
# | M_3_ao   | 1     | A/cm     | 3rd magnetization in the aorta                 | electromagnetic | (0, 5)  |       |           |
# | M_3_li   | 1     | A/cm     | 3rd magnetization in the liver                 | electromagnetic | (0, 5)  |       |           |
# | M_4_ao   | 1     | A/cm     | 4th magnetization in the aorta                 | electromagnetic | (0, 5)  |       |           |
# | M_4_li   | 1     | A/cm     | 4th magnetization in the liver                 | electromagnetic | (0, 5)  |       |           |
# | R1_1_ao  | 0.65  | Hz       | 1st tissue R1 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R1_1_li  | 0.65  | Hz       | 1st tissue R1 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R1_2_ao  | 0.65  | Hz       | 2nd tissue R1 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R1_2_li  | 0.65  | Hz       | 2nd tissue R1 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R1i_1_ao | 0.65  | Hz       | 1st inlet R1 in the aorta                      | electromagnetic | (0, 5)  |       |           |
# | R1i_1_li | 0.65  | Hz       | 1st inlet R1 in the liver                      | electromagnetic | (0, 5)  |       |           |
# | R1i_2_ao | 0.65  | Hz       | 2nd inlet R1 in the aorta                      | electromagnetic | (0, 5)  |       |           |
# | R1i_2_li | 0.65  | Hz       | 2nd inlet R1 in the liver                      | electromagnetic | (0, 5)  |       |           |
# | R2_1_ao  | 2.0   | Hz       | 1st tissue R2 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R2_1_li  | 2.0   | Hz       | 1st tissue R2 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R2_2_ao  | 2.0   | Hz       | 2nd tissue R2 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R2_2_li  | 2.0   | Hz       | 2nd tissue R2 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R2s_1_ao | 20    | Hz       | 1st tissue R2* in the aorta                    | electromagnetic | (0, 5)  |       |           |
# | R2s_1_li | 20    | Hz       | 1st tissue R2* in the liver                    | electromagnetic | (0, 5)  |       |           |
# | R2s_2_ao | 20    | Hz       | 2nd tissue R2* in the aorta                    | electromagnetic | (0, 5)  |       |           |
# | R2s_2_li | 20    | Hz       | 2nd tissue R2* in the liver                    | electromagnetic | (0, 5)  |       |           |
# | tM_1_ao  | 0.0   | sec      | 1st magnetization time points in the aorta     | electromagnetic |         |       |           |
# | tM_1_li  | 0.0   | sec      | 1st magnetization time points in the liver     | electromagnetic |         |       |           |
# | tM_2_ao  | 0.0   | sec      | 2nd magnetization time points in the aorta     | electromagnetic |         |       |           |
# | tM_2_li  | 0.0   | sec      | 2nd magnetization time points in the liver     | electromagnetic |         |       |           |
# | tM_3_ao  | 0.0   | sec      | 3rd magnetization time points in the aorta     | electromagnetic |         |       |           |
# | tM_3_li  | 0.0   | sec      | 3rd magnetization time points in the liver     | electromagnetic |         |       |           |
# | tM_4_ao  | 0.0   | sec      | 4th magnetization time points in the aorta     | electromagnetic |         |       |           |
# | tM_4_li  | 0.0   | sec      | 4th magnetization time points in the liver     | electromagnetic |         |       |           |
# | tR_1     | 0.0   | sec      | 1st relaxation rate time points                | electromagnetic |         |       |           |
# | tR_2     | 0.0   | sec      | 2nd relaxation rate time points                | electromagnetic |         |       |           |
# +----------+-------+----------+------------------------------------------------+-----------------+---------+-------+-----------+
# | C_1_ao   | 0.005 | mmol/cm3 | 1st tissue concentration in the aorta          | indicator       | (0, 1)  |       |           |
# | C_1_li   | 0.005 | mmol/cm3 | 1st tissue concentration in the liver          | indicator       | (0, 1)  |       |           |
# | C_2_ao   | 0.005 | mmol/cm3 | 2nd tissue concentration in the aorta          | indicator       | (0, 1)  |       |           |
# | C_2_li   | 0.005 | mmol/cm3 | 2nd tissue concentration in the liver          | indicator       | (0, 1)  |       |           |
# | ci_1_ao  | 0.005 | mmol/mL  | 1st inlet concentration in the aorta           | indicator       |         |       |           |
# | ci_1_li  | 0.005 | mmol/mL  | 1st inlet concentration in the liver           | indicator       |         |       |           |
# | ci_2_ao  | 0.005 | mmol/mL  | 2nd inlet concentration in the aorta           | indicator       |         |       |           |
# | ci_2_li  | 0.005 | mmol/mL  | 2nd inlet concentration in the liver           | indicator       |         |       |           |
# | J_1_ao   | 1     | mmol/sec | 1st indicator flux in the aorta                | indicator       | (0, 10) |       |           |
# | J_1_la   | 1     | mmol/sec | 1st indicator flux in the liver artery         | indicator       | (0, 10) |       |           |
# | J_1_lag  | 1     | mmol/sec | 1st indicator flux in the liver artery and gut | indicator       | (0, 10) |       |           |
# | J_1_li   | 1     | mmol/sec | 1st indicator flux in the liver                | indicator       | (0, 10) |       |           |
# | J_1_or   | 1     | mmol/sec | 1st indicator flux in the organs               | indicator       | (0, 10) |       |           |
# | J_1_pv   | 1     | mmol/sec | 1st indicator flux in the portal vein          | indicator       | (0, 10) |       |           |
# | J_1_vc   | 1     | mmol/sec | 1st indicator flux in the vena cava            | indicator       | (0, 10) |       |           |
# | J_2_ao   | 1     | mmol/sec | 2nd indicator flux in the aorta                | indicator       | (0, 10) |       |           |
# | J_2_la   | 1     | mmol/sec | 2nd indicator flux in the liver artery         | indicator       | (0, 10) |       |           |
# | J_2_lag  | 1     | mmol/sec | 2nd indicator flux in the liver artery and gut | indicator       | (0, 10) |       |           |
# | J_2_li   | 1     | mmol/sec | 2nd indicator flux in the liver                | indicator       | (0, 10) |       |           |
# | J_2_or   | 1     | mmol/sec | 2nd indicator flux in the organs               | indicator       | (0, 10) |       |           |
# | J_2_pv   | 1     | mmol/sec | 2nd indicator flux in the portal vein          | indicator       | (0, 10) |       |           |
# | J_2_vc   | 1     | mmol/sec | 2nd indicator flux in the vena cava            | indicator       | (0, 10) |       |           |
# | tC_1     | 0.0   | sec      | 1st concentration time points                  | indicator       |         |       |           |
# | tC_2     | 0.0   | sec      | 2nd concentration time points                  | indicator       |         |       |           |
# +----------+-------+----------+------------------------------------------------+-----------------+---------+-------+-----------+
# | S0_1_ao  | 1.0   | a.u.     | 1st signal scaling factor in the aorta         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_1_li  | 1.0   | a.u.     | 1st signal scaling factor in the liver         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_2_ao  | 1.0   | a.u.     | 2nd signal scaling factor in the aorta         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_2_li  | 1.0   | a.u.     | 2nd signal scaling factor in the liver         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_3_ao  | 1.0   | a.u.     | 3rd signal scaling factor in the aorta         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_3_li  | 1.0   | a.u.     | 3rd signal scaling factor in the liver         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_4_ao  | 1.0   | a.u.     | 4th signal scaling factor in the aorta         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_4_li  | 1.0   | a.u.     | 4th signal scaling factor in the liver         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S_1_ao   | 1.0   | a.u.     | 1st signal in the aorta                        | signal          | (0, 5)  |       |           |
# | S_1_li   | 1.0   | a.u.     | 1st signal in the liver                        | signal          | (0, 5)  |       |           |
# | S_2_ao   | 1.0   | a.u.     | 2nd signal in the aorta                        | signal          | (0, 5)  |       |           |
# | S_2_li   | 1.0   | a.u.     | 2nd signal in the liver                        | signal          | (0, 5)  |       |           |
# | S_3_ao   | 1.0   | a.u.     | 3rd signal in the aorta                        | signal          | (0, 5)  |       |           |
# | S_3_li   | 1.0   | a.u.     | 3rd signal in the liver                        | signal          | (0, 5)  |       |           |
# | S_4_ao   | 1.0   | a.u.     | 4th signal in the aorta                        | signal          | (0, 5)  |       |           |
# | S_4_li   | 1.0   | a.u.     | 4th signal in the liver                        | signal          | (0, 5)  |       |           |
# | tS_1_ao  | 0.0   | sec      | 1st signal time points in the aorta            | signal          |         |       |           |
# | tS_1_li  | 0.0   | sec      | 1st signal time points in the liver            | signal          |         |       |           |
# | tS_2_ao  | 0.0   | sec      | 2nd signal time points in the aorta            | signal          |         |       |           |
# | tS_2_li  | 0.0   | sec      | 2nd signal time points in the liver            | signal          |         |       |           |
# | tS_3_ao  | 0.0   | sec      | 3rd signal time points in the aorta            | signal          |         |       |           |
# | tS_3_li  | 0.0   | sec      | 3rd signal time points in the liver            | signal          |         |       |           |
# | tS_4_ao  | 0.0   | sec      | 4th signal time points in the aorta            | signal          |         |       |           |
# | tS_4_li  | 0.0   | sec      | 4th signal time points in the liver            | signal          |         |       |           |
# +------------------------------------------------------------------------------------------------------------------------------+

from copy import deepcopy
from itertools import product

import numpy as np

from dcmri.core.tools import extend_varname, increment_varindex, parse_varname, group
from dcmri.core.module import Module
from dcmri.forward.aorta_liver_split import AortaLiverSplit
from dcmri.forward.aorta_liver import AortaLiver
from dcmri.bloch.functions_sequences import channels

visits, scans, rois = [1, 2], [1, 2], ['ao', 'li']

configs = deepcopy(AortaLiverSplit.configs)
defaults = deepcopy(AortaLiverSplit.defaults)

configs['inflow'].discard('inlet')

class AortaLiverSplitDrug(Module):
    """Whole-body model for the aorta and liver signal acquired over 2 separate acquisitions."""

    configs = configs
    defaults = defaults

    _all_inputs = {'B1corr_2_ao', 'k_1_e2h', 'BAT_2', 'TD_2', 'B1corr_2_li', 'TR_2', 'ki_2_e2h', 'tstart_4', 'T_b_or', 'Ti_2_h', 'PSw', 'NSR_4_li', 'Nk0_1', 'TE_2', 'E_1_li', 'TP_4', 'iScal_1_li', 'dose_1', 'FA_2', 'Nz_2', 'TE2_1', 'tacq_4', 'PA_1', 'Nph_4', 'Scal_2_ao', 'iStrig_3_ao', 'TR_1', 'B1corr_1_li', 'SA_1', 'TR_3', 'Nph_3', 'kf_2_e2h', 'BAT_1', 'Nk0_4', 'TE_1', 'TE_3', 'T_hl', 'TA_3', 'Nk0_2', 'iStrig_1_ao', 'tacq_2', 'iStrig_2_ao', 'PA_4', 'S0_1_ao', 'Nph_1', 'Ef_2_li', 'TE2_3', 'weight', 'TE2_4', 'vol_1_li', 'TE2_2', 'S0_4_ao', 'iz_2', 'dose_4', 'Scal_1_li', 'Nz_1', 'B1corr_4_ao', 'R1_h', 'v_e', 'me', 'B1corr_4_li', 'tacq_3', 'T_1_h', 'TD_1', 'TE1_1', 'S0_3_li', 'dose_2', 'BAT_4', 'S0_4_li', 'iScal_3_ao', 'NSR_1_ao', 'S0_2_ao', 'iScal_4_ao', 'B1corr_3_ao', 'NSR_3_li', 'TE1_4', 'dose_tolerance', 'iStrig_1_li', 'TE1_3', 'BAT_3', 'tstart_3', 'iStrig_4_li', 'iScal_4_li', 'tstart_1', 'Tf_1_h', 'dose_3', 'NSR_3_ao', 'ki_1_e2h', 'T_la', 'Nz_4', 'TE1_2', 'PA_3', 'S0_3_ao', 'tacq_1', 'PA_2', 'CO', 'TA_2', 'iz_1', 'TF', 'R1_li', 'iStrig_2_li', 'iScal_2_li', 'D_hl', 'Nph_2', 'Tf_2_h', 'iz_4', 'TA_1', 'tstart_2', 'iStrig_4_ao', 'Scal_1_ao', 'T_2_h', 'vr_or', 'rate_3', 'H', 'NSR_2_ao', 'dt', 'field_strength', 'SA_2', 'Scal_3_li', 'T_e_or', 'k_2_e2h', 'TD_4', 'S0_1_li', 'rate_4', 'iScal_2_ao', 'FA_1', 'Scal_4_ao', 'Scal_2_li', 'TD_3', 'Scal_4_li', 'Scal_3_ao', 'TE_4', 'iz_3', 'TA_4', 'TR_4', 'E_or', 'vol_1_ao', 'vol_2_ao', 'Nk0_3', 'iScal_1_ao', 'v_h', 'FA_3', 'T_gu', 'SA_4', 'B1corr_1_ao', 'TP_1', 'ffa', 'iScal_3_li', 'NSR_1_li', 'NSR_2_li', 'iStrig_3_li', 'rate_1', 'R1_e', 'Ti_1_h', 'SA_3', 'Nz_3', 'E_2_li', 'Ei_2_li', 'kf_1_e2h', 'rate_2', 'agent', 'NSR_4_ao', 'R1_b', 'Ei_1_li', 'B1corr_3_li', 'Ef_1_li', 'bdel', 'v_li', 'S0_2_li', 'TP_3', 'FA_4', 'TP_2', 'fCO_li', 'v_e_li', 'vol_2_li'}
    _all_outputs = {'M_3_ao', 'R1i_1_ao', 'J_1_vc', 'S_2_li', 'R1i_1_li', 'ci_2_ao', 'tS_3_ao', 'tM_3_ao', 'J_2_ao', 'tM_2_ao', 'S_1_li', 'R1_2_li', 'J_2_vc', 'C_1_ao', 'M_4_ao', 'S0_1_ao', 'R2_2_li', 'R2s_1_ao', 'S0_4_ao', 'J_1_ao', 'J_2_or', 'J_1_li', 'R1_2_ao', 'R2_1_ao', 'C_1_li', 'M_3_li', 'S0_3_li', 'tM_4_li', 'S0_4_li', 'S0_2_ao', 'J_1_la', 'tM_4_ao', 'J_1_or', 'tS_4_ao', 'C_2_li', 'tC_1', 'S0_3_ao', 'J_1_pv', 'tM_1_ao', 'R1i_2_ao', 'R1_1_li', 'R2_1_li', 'tS_4_li', 'tS_1_li', 'R2s_2_ao', 'M_1_ao', 'J_2_lag', 'S0_1_li', 'C_2_ao', 'tS_3_li', 'S_3_ao', 'ci_2_li', 'R2_2_ao', 'tS_2_li', 'R1_1_ao', 'J_2_la', 'J_2_li', 'M_2_li', 'tR_2', 'tS_2_ao', 'S_2_ao', 'R2s_2_li', 'R2s_1_li', 'tS_1_ao', 'ci_1_ao', 'ci_1_li', 'tC_2', 'S_1_ao', 'S_4_ao', 'M_2_ao', 'S_4_li', 'S_3_li', 'tR_1', 'tM_1_li', 'S0_2_li', 'M_1_li', 'M_4_li', 'R1i_2_li', 'J_1_lag', 'tM_2_li', 'J_2_pv', 'tM_3_li'}

    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)  

        for visit in visits:
            p |= self._aol[visit](p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        # Index inputs and outputs
        visit_inputs = {'BAT', 'dose', 'rate'}
        visit_inputs |= {'E_li', 'Ef_li', 'Ei_li', 'T_h', 'Tf_h', 'Ti_h', 'k_e2h', 'kf_e2h', 'ki_e2h', 'vol_ao', 'vol_li'} 

        vars = {'BAT', 'dose', 'rate'}
        vars |= {k for k in AortaLiver.all_inputs() if group(k)=='sequence' and k != 'field_strength'}
        visit_scan_inputs = {extend_varname(k, index=i) for k, i in product(vars, scans)}

        vars = {'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr'}
        visit_scan_inputs |= {extend_varname(k, roi=roi, index=i) for k, roi, i in product(vars, rois, scans)} 

        vars = AortaLiverSplit.all_outputs()
        visit_outputs = {o for o in vars if parse_varname(o)['index'] is None}
        visit_scan_outputs = {o for o in vars if parse_varname(o)['index'] is not None}

        self._aol = {}

        for visit in visits:
            imap_visit = {k: extend_varname(k, index=visit) for k in visit_inputs}
            omap_visit = {k: extend_varname(k, index=visit) for k in visit_outputs}
            if visit==2: 
                imap_visit |= {k: increment_varindex(k, 2) for k in visit_scan_inputs} 
                omap_visit |= {k: increment_varindex(k, 2) for k in visit_scan_outputs}
            
            self._aol[visit] = AortaLiverSplit(imap=imap_visit, omap=omap_visit, **self.config)

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

    def test_data(self, data: dict=None): 
        p = self.init_data()

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
                    p |= {
                        f'iScal_{idx}_{roi}': np.arange(n0, dtype=int),
                        f'Scal_{idx}_{roi}': Scal, 
                    }
        p |= {
            'tacq_1': 90,
            'tstart_2': 120,
            'tacq_2': 60,
            'tacq_3': 90,
            'tstart_4': 120,
            'tacq_4': 60,     
            'BAT_2': 120, # default is the same as BAT_1
            'BAT_4': 120, # default is the same as BAT_1      
        }

        # data['E_2_li'] /= 10 # create some effect
        return self.input_data(p, data)