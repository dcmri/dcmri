# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             InverseAortaLiverSplitDrug - all inputs (n = 172)                                                             |
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
# | pfree          | None       | a.u.       | set of free parameters                                                   | inverse problem |               |       |           |
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
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iStrig_1_ao    | None       |            | 1st indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_1_li    | None       |            | 1st indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_2_ao    | None       |            | 2nd indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_2_li    | None       |            | 2nd indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_3_ao    | None       |            | 3rd indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_3_li    | None       |            | 3rd indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_4_ao    | None       |            | 4th indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_4_li    | None       |            | 4th indices of the signal trigger in the liver                           | signal          |               |       |           |
# | nb_1           | 1          | a.u.       | 1st number of baseline time points                                       | signal          |               |       |           |
# | nb_2           | 1          | a.u.       | 2nd number of baseline time points                                       | signal          |               |       |           |
# | nb_3           | 1          | a.u.       | 3rd number of baseline time points                                       | signal          |               |       |           |
# | nb_4           | 1          | a.u.       | 4th number of baseline time points                                       | signal          |               |       |           |
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
# | S_1_ao         | 1.0        | a.u.       | 1st signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_1_li         | 1.0        | a.u.       | 1st signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | S_2_ao         | 1.0        | a.u.       | 2nd signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_2_li         | 1.0        | a.u.       | 2nd signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | S_3_ao         | 1.0        | a.u.       | 3rd signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_3_li         | 1.0        | a.u.       | 3rd signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | S_4_ao         | 1.0        | a.u.       | 4th signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_4_li         | 1.0        | a.u.       | 4th signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | tS_1_ao        | 0.0        | sec        | 1st signal time points in the aorta                                      | signal          |               |       |           |
# | tS_1_li        | 0.0        | sec        | 1st signal time points in the liver                                      | signal          |               |       |           |
# | tS_2_ao        | 0.0        | sec        | 2nd signal time points in the aorta                                      | signal          |               |       |           |
# | tS_2_li        | 0.0        | sec        | 2nd signal time points in the liver                                      | signal          |               |       |           |
# | tS_3_ao        | 0.0        | sec        | 3rd signal time points in the aorta                                      | signal          |               |       |           |
# | tS_3_li        | 0.0        | sec        | 3rd signal time points in the liver                                      | signal          |               |       |           |
# | tS_4_ao        | 0.0        | sec        | 4th signal time points in the aorta                                      | signal          |               |       |           |
# | tS_4_li        | 0.0        | sec        | 4th signal time points in the liver                                      | signal          |               |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_1_ao       | 1          | cm3        | 1st ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_1_li       | 1000       | cm3        | 1st ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | vol_2_ao       | 1          | cm3        | 2nd ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_2_li       | 1000       | cm3        | 2nd ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                              | whole-body      | (0, 300)      |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------+
# |                                 InverseAortaLiverSplitDrug - all outputs (n = 5)                                 |
# +-------+-------+------+------------------------------------------------+-----------------+--------+-------+-------+
# | Key   | Value | Unit | Name                                           | Group           | Bounds | DICOM | OSIPI |
# +-------+-------+------+------------------------------------------------+-----------------+--------+-------+-------+
# | loss  | None  | a.u. | loss value of optimized model                  | inverse problem |        |       |       |
# | pars  | None  | a.u. | dictionary with source- and derived parameters | inverse problem |        |       |       |
# | pcov  | None  | a.u. | dictionary with covariances of free parameters | inverse problem |        |       |       |
# | popt  | None  | a.u. | dictionary of optimized free parameter values  | inverse problem |        |       |       |
# | psdev | None  | a.u. | dictionary with parameter standard deviations  | inverse problem |        |       |       |
# +------------------------------------------------------------------------------------------------------------------+

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds, parse_varname, extend_varname, remove_varindex
from dcmri.utils.fit import train_brute
from dcmri.forward.aorta_liver_split_drug import AortaLiverSplitDrug as ForwardModel
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.inverse.functions import estimate_bat

configs = deepcopy(ForwardModel.configs) | {'estimate_bat': {'none', 'ttp', 'custom'}}
defaults = deepcopy(ForwardModel.defaults) | {'estimate_bat': 'custom', 'calibrate': True}

ROIS = ['ao', 'li']
SCANS = [1, 2, 3, 4]

class InverseAortaLiverSplitDrug(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'ki_2_e2h', 'FA_3', 'SA_4', 'Nk0_4', 'TP_1', 'NSR_1_ao', 'TA_2', 'NSR_2_ao', 'Ef_2_li', 'nb_1', 'PA_1', 'iz_4', 'BAT_2', 'Nk0_1', 'Ei_2_li', 'iStrig_1_li', 'TE2_1', 'iStrig_3_li', 'S_4_li', 'Ti_2_h', 'TE2_2', 'TE1_1', 'TE_2', 'D_hl', 'Nz_2', 'S0_3_li', 'SA_2', 'vol_1_li', 'S0_4_ao', 'NSR_4_ao', 'NSR_3_li', 'tS_1_li', 'iz_2', 'v_e', 'S_4_ao', 'dose_1', 'Tf_2_h', 'k_2_e2h', 'E_2_li', 'tS_3_li', 'TF', 'T_e_or', 'S0_4_li', 'Nk0_2', 'rate_2', 'TP_4', 'TD_2', 'TA_3', 'TR_1', 'Nph_3', 'iStrig_3_ao', 'T_hl', 'dose_3', 'R1_h', 'CO', 'NSR_1_li', 'weight', 'FA_2', 'TE1_2', 'vol_2_li', 'Nz_3', 'v_e_li', 'NSR_2_li', 'k_1_e2h', 'tS_4_li', 'Nk0_3', 'rate_4', 'B1corr_2_ao', 'tS_4_ao', 'S_3_li', 'FA_4', 'TR_4', 'Nz_1', 'tS_3_ao', 'Ef_1_li', 'TE2_3', 'R1_li', 'field_strength', 'vol_1_ao', 'B1corr_3_ao', 'ffa', 'S0_3_ao', 'B1corr_4_li', 'PA_3', 'T_gu', 'iStrig_2_ao', 'TE2_4', 'E_or', 'BAT_4', 'nb_2', 'TE_4', 'iz_1', 'TE1_3', 'B1corr_3_li', 'dose_2', 'dose_tolerance', 'TE_3', 'v_h', 'B1corr_4_ao', 'TD_3', 'iStrig_4_li', 'S_3_ao', 'B1corr_1_ao', 'PA_2', 'NSR_4_li', 'rate_3', 'iStrig_4_ao', 'kf_2_e2h', 'iStrig_2_li', 'S0_2_ao', 'B1corr_2_li', 'tS_2_li', 'bdel', 'Ei_1_li', 'tS_2_ao', 'vr_or', 'TA_1', 'Nph_4', 'T_2_h', 'S0_1_ao', 'S0_2_li', 'Tf_1_h', 'T_b_or', 'agent', 'TR_3', 'Nph_2', 'Ti_1_h', 'PA_4', 'S0_1_li', 'dt', 'v_li', 'S_2_ao', 'kf_1_e2h', 'TP_2', 'TD_4', 'rate_1', 'S_1_ao', 'SA_3', 'TP_3', 'NSR_3_ao', 'PSw', 'SA_1', 'TE_1', 'fCO_li', 'E_1_li', 'Nph_1', 'TD_1', 'ki_1_e2h', 'nb_3', 'T_1_h', 'BAT_3', 'TA_4', 'TE1_4', 'TR_2', 'Nz_4', 'R1_e', 'B1corr_1_li', 'BAT_1', 'H', 'FA_1', 'pfree', 'tS_1_ao', 'R1_b', 'iStrig_1_ao', 'me', 'iz_3', 'vol_2_ao', 'S_2_li', 'S_1_li', 'nb_4', 'T_la', 'dose_4'}
    _all_outputs = {'pcov', 'pars', 'popt', 'loss', 'psdev'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = ForwardModel(**self.config)
        self.map_io(imap, omap)

    def _predict(self, p):
        pred = self.forward(p)
        return (
            pred['S_1_ao'][:, :, :len(p['tS_1_ao'])].reshape(-1), 
            pred['S_2_ao'][:, :, :len(p['tS_2_ao'])].reshape(-1), 
            pred['S_1_li'][:, :, :len(p['tS_1_li'])].reshape(-1), 
            pred['S_2_li'][:, :, :len(p['tS_2_li'])].reshape(-1), 

            pred['S_3_ao'][:, :, :len(p['tS_3_ao'])].reshape(-1), 
            pred['S_4_ao'][:, :, :len(p['tS_4_ao'])].reshape(-1), 
            pred['S_3_li'][:, :, :len(p['tS_3_li'])].reshape(-1), 
            pred['S_4_li'][:, :, :len(p['tS_4_li'])].reshape(-1),
        )

    def _preproc(self, p):
        # Derive simulation timings
        for scan in SCANS:
            p[f'tstart_{scan}'] = min([p[f'tS_{scan}_{roi}'][0] for roi in ROIS])
            tmax = max([p[f'tS_{scan}_{roi}'][-1] for roi in ROIS])
            ts = p[f'tS_{scan}_ao'][1] - p[f'tS_{scan}_ao'][0]
            p[f'tacq_{scan}'] = tmax - p[f'tstart_{scan}'] + 2 * ts

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

        # Trigger signals
        if self.config['trigger']:
            for roi in ROIS:
                for scan in SCANS:
                    if p[f'iStrig_{scan}_{roi}'] is not None:
                        p[f'tS_{scan}_{roi}'] = p[f'tS_{scan}_{roi}'][p[f'iStrig_{scan}_{roi}']]  
                        p[f'S_{scan}_{roi}'] = p[f'S_{scan}_{roi}'][:, :, p[f'iStrig_{scan}_{roi}']]     

    def _pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='physiological'}
        if self.config['bolus'] in ['single', 'double']:
            pfree |= {'BAT_1', 'BAT_2'}
        else:
            pfree |= {'BAT_1', 'BAT_2', 'BAT_3', 'BAT_4'}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
            pfree |= {'S0_3_ao', 'S0_3_li', 'S0_4_ao', 'S0_4_li'}
        pfree -= {'v_li', 'H'}
        return {p: get_quantity(p)['bounds'] for p in pfree}
    

    def _pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['physiological', 'whole-body']}

        p_common = {k: v for k, v in p.items() if parse_varname(k)['index'] is None}

        p_1 = {remove_varindex(k): v for k, v in p.items() if parse_varname(k)['index'] == 1} 
        p_2 = {remove_varindex(k): v for k, v in p.items() if parse_varname(k)['index'] == 2}

        p_1 = dpars_liver(p_1 | p_common, kinetics=self.config['liver'])
        p_2 = dpars_liver(p_2 | p_common, kinetics=self.config['liver'])

        p_1 = {extend_varname(k, index=1): v for k, v in p_1.items() if k not in p_common and k not in {'v_h'}}
        p_2 = {extend_varname(k, index=2): v for k, v in p_2.items() if k not in p_common and k not in {'v_h'}}

        return p_common | p_1 | p_2

    def _call(self, p: dict=None, **kwargs) -> dict:
        self._preproc(p) 

        # Estimate BAT if needed
        if self.config['estimate_bat'] == 'custom':
            if self.config['bolus'] == 'dual':
                p['BAT_1'] = estimate_bat(p['tS_1_ao'], p['S_1_ao'])
                p['BAT_2'] = p['tS_2_ao'][np.argmax(p['S_2_ao'])]
                p['BAT_3'] = estimate_bat(p['tS_3_ao'], p['S_3_ao'])
                p['BAT_4'] = p['tS_4_ao'][np.argmax(p['S_4_ao'])]
            else:
                p['BAT_1'] = estimate_bat(p['tS_1_ao'], p['S_1_ao'])
                p['BAT_2'] = estimate_bat(p['tS_3_ao'], p['S_3_ao'])
        elif self.config['estimate_bat'] == 'ttp':
            if self.config['bolus'] == 'dual':
                p['BAT_1'] = p['tS_1_ao'][np.argmax(p['S_1_ao'])]
                p['BAT_2'] = p['tS_2_ao'][np.argmax(p['S_2_ao'])]
                p['BAT_3'] = p['tS_3_ao'][np.argmax(p['S_3_ao'])]
                p['BAT_4'] = p['tS_4_ao'][np.argmax(p['S_4_ao'])]
            else:
                p['BAT_1'] = p['tS_1_ao'][np.argmax(p['S_1_ao'])]
                p['BAT_2'] = p['tS_3_ao'][np.argmax(p['S_3_ao'])]

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # if self.config['bolus'] in ['single', 'double']:
        #     bats = ['BAT_1', 'BAT_2']
        # else:
        #     bats = ['BAT_1', 'BAT_2', 'BAT_3', 'BAT_4']

        # Compute inverse
        signal = (
            p['S_1_ao'], 
            p['S_2_ao'], 
            p['S_1_li'], 
            p['S_2_li'], 
            
            p['S_3_ao'], 
            p['S_4_ao'], 
            p['S_3_li'], 
            p['S_4_li'],
        )
        #p = train_bat(self._predict, None, signal, p, p['pfree'], bats=bats, **kwargs)
        output = train_brute(self._predict, signal, p, p['pfree'], **kwargs)
        output['pars'] = self._pder(p | output['popt'])

        return output

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'pfree'}
        for roi in ROIS:
            for scan in SCANS:
                inputs |= {f'tS_{scan}_{roi}', f'S_{scan}_{roi}'}
        if self.config['calibrate']:
            for scan in SCANS:
                inputs |= {f'nb_{scan}'}
                for roi in ROIS:
                    inputs -= {f'Scal_{scan}_{roi}', f'iScal_{scan}_{roi}'}
        if self.config['estimate_bat'] != 'none':
            inputs -= {'BAT_1', 'BAT_2', 'BAT_3', 'BAT_4'}
        for scan in SCANS:
            inputs -= {f'tstart_{scan}', f'tacq_{scan}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'pars', 'loss'}
    
    def test_data(self, data: dict=None): 
        pfree = {
            'CO': (10, 300), 
            'BAT_1': (-60, 60), 
            'BAT_2': (-60, 60),
            'BAT_3': (-60, 60), 
            'BAT_4': (-60, 60)
        }

        p = self.init_data()
        p |= self.forward.test_data()

        pred = self.forward(p)
        p |= {
            'nb_1': 5,
            'nb_2': 5,
            'pfree': self.forward.filter_data(pfree),
        }
        for roi in ROIS:  
            for scan in SCANS:      
                p |= {
                    f'tS_{scan}_{roi}': pred[f'tS_{scan}_{roi}'],
                    f'S_{scan}_{roi}': pred[f'S_{scan}_{roi}'], 
                }
            
        return self.input_data(p, data)

    def plot(self, data: dict, result:dict=None, xlim:list=None, clim:list=None, fname:str=None, show=True):
        if result is None:
            result = self.__call__(data)
        p = self.map_data(data | result['popt'])

        # Not an input parameter but required by forward model
        if self.config['estimate_bat'] != 'none':
            for bat in ['BAT_1', 'BAT_2', 'BAT_3', 'BAT_4']:
                if bat in result['popt']:
                    p[bat] = result['popt'][bat]

        self._preproc(p)
        pred = self.forward(p)
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 8))
        fig.subplots_adjust(wspace=0.3)

        ax1.set_title('Control visit')
        ax2.set_title('Treatment visit')
        ax3.set_title('Control visit')
        ax4.set_title('Treatment visit')

        def plot_data2scan(t, s, ti, si, ax, xl, yl, color):
            if xl is None: 
                xl = [0, t[-1]]
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=np.array(xl)/60, ylim=yl)
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    ax.plot(ti / 60, si[i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(t / 60, s[i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        ylim_a = [
            0.9 * min(pred[f'S_1_ao'].min(), pred[f'S_2_ao'].min(), pred[f'S_3_ao'].min(), pred[f'S_4_ao'].min(), p[f'S_1_ao'].min(), p[f'S_2_ao'].min(), p[f'S_3_ao'].min(), p[f'S_4_ao'].min()), 
            1.1 * max(pred[f'S_1_ao'].max(), pred[f'S_2_ao'].max(), pred[f'S_3_ao'].max(), pred[f'S_4_ao'].max(), p[f'S_1_ao'].max(), p[f'S_2_ao'].max(), p[f'S_3_ao'].max(), p[f'S_4_ao'].max()),
        ]
        ylim_l = [
            0.9 * min(pred[f'S_1_li'].min(), pred[f'S_2_li'].min(), pred[f'S_3_li'].min(), pred[f'S_4_li'].min(), p[f'S_1_li'].min(), p[f'S_2_li'].min(), p[f'S_3_li'].min(), p[f'S_4_li'].min()), 
            1.1 * max(pred[f'S_1_li'].max(), pred[f'S_2_li'].max(), pred[f'S_3_li'].max(), pred[f'S_4_li'].max(), p[f'S_1_li'].max(), p[f'S_2_li'].max(), p[f'S_3_li'].max(), p[f'S_4_li'].max()),
        ]

        plot_data2scan(pred[f'tS_1_ao'], pred[f'S_1_ao'], p['tS_1_ao'], p['S_1_ao'], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_2_ao'], pred[f'S_2_ao'], p['tS_2_ao'], p['S_2_ao'], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_3_ao'], pred[f'S_3_ao'], p['tS_3_ao'], p['S_3_ao'], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_4_ao'], pred[f'S_4_ao'], p['tS_4_ao'], p['S_4_ao'], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])

        plot_data2scan(pred[f'tS_1_li'], pred[f'S_1_li'], p['tS_1_li'], p['S_1_li'], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(pred[f'tS_2_li'], pred[f'S_2_li'], p['tS_2_li'], p['S_2_li'], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(pred[f'tS_3_li'], pred[f'S_3_li'], p['tS_3_li'], p['S_3_li'], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(pred[f'tS_4_li'], pred[f'S_4_li'], p['tS_4_li'], p['S_4_li'], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c[0,:], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
            ax.plot(t / 60, 10 * 1000 * c[0,:], linestyle='--', color='darkred', linewidth=2.0, label='Aorta (x10)')
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(pred['C_1_ao'].max(), pred['C_2_ao'].max())]   
        else:
            ylim = [0, clim[0] * 1000]

        plot_conc_aorta(pred['tC_1'], pred['C_1_ao'], ax3, xlim, ylim)
        plot_conc_aorta(pred['tC_2'], pred['C_2_ao'], ax4, xlim, ylim)

        def plot_conc_liver(t, C, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * C[0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax.plot(t / 60, 1000 * C[1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax.plot(t / 60, 1000 * C.sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')       
            ax.legend()

        if clim is None:
            ylim = [0, 1.1 * 1000 * max(pred['C_1_li'].max(), pred['C_2_li'].max())] 
        else:
            ylim = [0, clim[1] * 1000]

        plot_conc_liver(pred['tC_1'], pred['C_1_li'], ax7, xlim, ylim)
        plot_conc_liver(pred['tC_2'], pred['C_2_li'], ax8, xlim, ylim)

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()