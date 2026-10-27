# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                InverseAortaLiverDrug - all inputs (n = 116)                                                               |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | Key            | Value      | Unit       | Name                                                                     | Group           | Bounds        | DICOM | OSIPI     |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | B1corr_1_ao    | 1          |            | 1st B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_1_li    | 1          |            | 1st B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_ao    | 1          |            | 2nd B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_2_li    | 1          |            | 2nd B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
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
# | field_strength | 3          | T          | magnetic field strength                                                  | sequence        | (0, 20)       |       |           |
# | iz_1           | 0          |            | 1st slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | iz_2           | 0          |            | 2nd slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | Nk0_1          | 64         |            | 1st number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nk0_2          | 64         |            | 2nd number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nph_1          | 128        |            | 1st number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nph_2          | 128        |            | 2nd number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nz_1           | 64         |            | 1st number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | Nz_2           | 64         |            | 2nd number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | PA_1           | 90         | deg        | 1st preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | PA_2           | 90         | deg        | 2nd preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | SA_1           | 0          | deg        | 1st saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | SA_2           | 0          | deg        | 2nd saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | TA_1           | 2.0        | sec        | 1st acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | TA_2           | 2.0        | sec        | 2nd acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | TD_1           | 0.05       | sec        | 1st prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TD_2           | 0.05       | sec        | 2nd prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TE1_1          | 0.001      | sec        | 1st first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE1_2          | 0.001      | sec        | 2nd first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE2_1          | 0.005      | sec        | 1st second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE2_2          | 0.005      | sec        | 2nd second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TE_1           | 0.001      | sec        | 1st echo time                                                            | sequence        | (0, 10)       |       |           |
# | TE_2           | 0.001      | sec        | 2nd echo time                                                            | sequence        | (0, 10)       |       |           |
# | TP_1           | 0.05       | sec        | 1st preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TP_2           | 0.05       | sec        | 2nd preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TR_1           | 0.005      | sec        | 1st repetition time                                                      | sequence        | (0, 1)        |       |           |
# | TR_2           | 0.005      | sec        | 2nd repetition time                                                      | sequence        | (0, 1)        |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iStrig_1_ao    | None       |            | 1st indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_1_li    | None       |            | 1st indices of the signal trigger in the liver                           | signal          |               |       |           |
# | iStrig_2_ao    | None       |            | 2nd indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_2_li    | None       |            | 2nd indices of the signal trigger in the liver                           | signal          |               |       |           |
# | nb_1           | 1          | a.u.       | 1st number of baseline time points                                       | signal          |               |       |           |
# | nb_2           | 1          | a.u.       | 2nd number of baseline time points                                       | signal          |               |       |           |
# | NSR_1_ao       | 0.0        |            | 1st noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_1_li       | 0.0        |            | 1st noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | NSR_2_ao       | 0.0        |            | 2nd noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_2_li       | 0.0        |            | 2nd noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | S0_1_ao        | 1.0        | a.u.       | 1st signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | 1.0        | a.u.       | 1st signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | 1.0        | a.u.       | 2nd signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | 1.0        | a.u.       | 2nd signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S_1_ao         | 1.0        | a.u.       | 1st signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_1_li         | 1.0        | a.u.       | 1st signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | S_2_ao         | 1.0        | a.u.       | 2nd signal in the aorta                                                  | signal          | (0, 5)        |       |           |
# | S_2_li         | 1.0        | a.u.       | 2nd signal in the liver                                                  | signal          | (0, 5)        |       |           |
# | tS_1_ao        | 0.0        | sec        | 1st signal time points in the aorta                                      | signal          |               |       |           |
# | tS_1_li        | 0.0        | sec        | 1st signal time points in the liver                                      | signal          |               |       |           |
# | tS_2_ao        | 0.0        | sec        | 2nd signal time points in the aorta                                      | signal          |               |       |           |
# | tS_2_li        | 0.0        | sec        | 2nd signal time points in the liver                                      | signal          |               |       |           |
# +----------------+------------+------------+--------------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_1_ao       | 1          | cm3        | 1st ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_1_li       | 1000       | cm3        | 1st ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | vol_2_ao       | 1          | cm3        | 2nd ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_2_li       | 1000       | cm3        | 2nd ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                              | whole-body      | (0, 300)      |       |           |
# +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------+
# |                                   InverseAortaLiverDrug - all outputs (n = 5)                                    |
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
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds, parse_varname, extend_varname, remove_varindex
from dcmri.utils.fit import train_brute
from dcmri.forward.aorta_liver_drug import AortaLiverDrug as ForwardModel
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.inverse.functions import estimate_bat

configs = deepcopy(ForwardModel.configs) | {'estimate_bat': {'none', 'ttp', 'custom'}}
defaults = deepcopy(ForwardModel.defaults) | {'estimate_bat': 'custom', 'calibrate': True}

configs['bolus'].discard('dual') # Only meaningful for split protocols

ROIS = ['ao', 'li']
SCANS = [1, 2]

class InverseAortaLiverDrug(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'tS_1_ao', 'rate_4', 'Nz_1', 'TP_2', 'iStrig_2_ao', 'nb_1', 'dose_1', 'Tf_2_h', 'E_1_li', 'pfree', 'field_strength', 'TE1_1', 'nb_2', 'B1corr_2_li', 'k_2_e2h', 'Ei_1_li', 'S_1_li', 'vr_or', 'PA_2', 'BAT_1', 'T_hl', 'E_2_li', 'R1_b', 'TE1_2', 'Nk0_2', 'NSR_1_li', 'iStrig_1_li', 'PSw', 'Nph_1', 'R1_li', 'TR_2', 'TE_1', 'T_e_or', 'SA_1', 'dose_tolerance', 'TE_2', 'dose_3', 'Nph_2', 'S0_2_li', 'fCO_li', 'iz_1', 'B1corr_1_li', 'R1_e', 'iStrig_1_ao', 'TE2_1', 'TA_1', 'v_h', 'Tf_1_h', 'E_or', 'T_2_h', 'S_2_li', 'dose_2', 'Ef_2_li', 'NSR_1_ao', 'PA_1', 'ki_1_e2h', 'TA_2', 'tS_2_ao', 'ki_2_e2h', 'TP_1', 'FA_1', 'k_1_e2h', 'ffa', 'T_la', 'B1corr_2_ao', 'rate_1', 'T_1_h', 'TR_1', 'dose_4', 'Ti_1_h', 'NSR_2_li', 'bdel', 'vol_1_li', 'Nz_2', 'vol_2_li', 'me', 'v_e_li', 'v_e', 'Ti_2_h', 'v_li', 'weight', 'S0_2_ao', 'D_hl', 'S_2_ao', 'vol_1_ao', 'tS_1_li', 'NSR_2_ao', 'agent', 'tS_2_li', 'kf_2_e2h', 'TD_2', 'iz_2', 'TD_1', 'FA_2', 'TE2_2', 'rate_3', 'kf_1_e2h', 'S0_1_ao', 'vol_2_ao', 'T_gu', 'BAT_2', 'dt', 'Ef_1_li', 'iStrig_2_li', 'CO', 'rate_2', 'S0_1_li', 'TF', 'Ei_2_li', 'T_b_or', 'B1corr_1_ao', 'R1_h', 'Nk0_1', 'S_1_ao', 'SA_2', 'H'}
    _all_outputs = {'popt', 'loss', 'pars', 'psdev', 'pcov'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self._forward = ForwardModel(**self.config)
        self.map_io(imap, omap)

    def _predict(self, p):
        pred = self._forward(p)
        return ( 
            pred['S_1_ao'][:, :, :len(p['tS_1_ao'])].reshape(-1), 
            pred['S_2_ao'][:, :, :len(p['tS_2_ao'])].reshape(-1), 
            pred['S_1_li'][:, :, :len(p['tS_1_li'])].reshape(-1), 
            pred['S_2_li'][:, :, :len(p['tS_2_li'])].reshape(-1), 
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
        inputs = self._forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='physiological'}
        pfree |= {'BAT_1', 'BAT_2'}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
        pfree -= {'H', 'v_li'}
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
            p['BAT_1'] = estimate_bat(p['tS_1_ao'], p['S_1_ao'])
            p['BAT_2'] = estimate_bat(p['tS_2_ao'], p['S_2_ao'])
        elif self.config['estimate_bat'] == 'ttp':
            p['BAT_1'] = p['tS_1_ao'][np.argmax(p['S_1_ao'])]
            p['BAT_2'] = p['tS_2_ao'][np.argmax(p['S_2_ao'])]

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        signal = (p['S_1_ao'], p['S_2_ao'], p['S_1_li'], p['S_2_li'])
        #output = train_bat(self._predict, time, signal, p, p['pfree'], bats=['BAT_1', 'BAT_2'], **kwargs)
        output = train_brute(self._predict, signal, p, p['pfree'], **kwargs)
        output['pars'] = self._pder(p | output['popt'])

        return output

    def inputs(self) -> set:
        inputs = self._forward.mapped_inputs()
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
            inputs -= {'BAT_1', 'BAT_2'}
        for scan in SCANS:
            inputs -= {f'tstart_{scan}', f'tacq_{scan}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'pars', 'loss'}
    
    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self._forward.test_data()

        pfree = {'CO': (10, 300), 'BAT_1': (-60, 60), 'BAT_2': (-60, 60)}

        pred = self._forward(p)
        p |= {
            'nb': 5,
            'pfree': self._forward.filter_data(pfree),
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
            for bat in ['BAT_1', 'BAT_2']:
                if bat in result['popt']:
                    p[bat] = result['popt'][bat]

        self._preproc(p)
        pred = self._forward(p)
        
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
            0.9 * min(pred[f'S_1_ao'].min(), pred[f'S_2_ao'].min(), p[f'S_1_ao'].min(), p[f'S_2_ao'].min()), 
            1.1 * max(pred[f'S_1_ao'].max(), pred[f'S_2_ao'].max(), p[f'S_1_ao'].max(), p[f'S_2_ao'].max()),
        ]
        ylim_l = [
            0.9 * min(pred[f'S_1_li'].min(), pred[f'S_2_li'].min(), p[f'S_1_li'].min(), p[f'S_2_li'].min()), 
            1.1 * max(pred[f'S_1_li'].max(), pred[f'S_2_li'].max(), p[f'S_1_li'].max(), p[f'S_2_li'].max()),
        ]

        plot_data2scan(pred[f'tS_1_ao'], pred[f'S_1_ao'], p['tS_1_ao'], p['S_1_ao'], ax1, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_1_li'], pred[f'S_1_li'], p['tS_1_li'], p['S_1_li'], ax5, xlim, ylim_l, ['cornflowerblue', 'darkblue'])
        plot_data2scan(pred[f'tS_2_ao'], pred[f'S_2_ao'], p['tS_2_ao'], p['S_2_ao'], ax2, xlim, ylim_a, ['lightcoral', 'darkred'])
        plot_data2scan(pred[f'tS_2_li'], pred[f'S_2_li'], p['tS_2_li'], p['S_2_li'], ax6, xlim, ylim_l, ['cornflowerblue', 'darkblue'])

        def plot_conc_aorta(t, c, ax, xl, yl):
            if xl is None: 
                xl = [t[0], t[-1]]
            ax.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=np.array(xl)/60, ylim=yl)
            ax.plot(t / 60, 0 * t, color='gray')
            ax.plot(t / 60, 1000 * c[0,:], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
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
