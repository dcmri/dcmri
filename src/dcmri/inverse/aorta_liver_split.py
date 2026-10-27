# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             InverseAortaLiverSplit - all inputs (n = 104)                                                             |
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
# | pfree          | None       | a.u.       | set of free parameters                                               | inverse problem |               |       |           |
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
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iStrig_1_ao    | None       |            | 1st indices of the signal trigger in the aorta                       | signal          |               |       |           |
# | iStrig_1_li    | None       |            | 1st indices of the signal trigger in the liver                       | signal          |               |       |           |
# | iStrig_2_ao    | None       |            | 2nd indices of the signal trigger in the aorta                       | signal          |               |       |           |
# | iStrig_2_li    | None       |            | 2nd indices of the signal trigger in the liver                       | signal          |               |       |           |
# | nb_1           | 1          | a.u.       | 1st number of baseline time points                                   | signal          |               |       |           |
# | nb_2           | 1          | a.u.       | 2nd number of baseline time points                                   | signal          |               |       |           |
# | NSR_1_ao       | 0.0        |            | 1st noise-to-signal ratio in the aorta                               | signal          | (0, 100000.0) |       |           |
# | NSR_1_li       | 0.0        |            | 1st noise-to-signal ratio in the liver                               | signal          | (0, 100000.0) |       |           |
# | NSR_2_ao       | 0.0        |            | 2nd noise-to-signal ratio in the aorta                               | signal          | (0, 100000.0) |       |           |
# | NSR_2_li       | 0.0        |            | 2nd noise-to-signal ratio in the liver                               | signal          | (0, 100000.0) |       |           |
# | S0_1_ao        | 1.0        | a.u.       | 1st signal scaling factor in the aorta                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_1_li        | 1.0        | a.u.       | 1st signal scaling factor in the liver                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_ao        | 1.0        | a.u.       | 2nd signal scaling factor in the aorta                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_2_li        | 1.0        | a.u.       | 2nd signal scaling factor in the liver                               | signal          | (0, 5)        |       | Q.MS1.010 |
# | S_1_ao         | 1.0        | a.u.       | 1st signal in the aorta                                              | signal          | (0, 5)        |       |           |
# | S_1_li         | 1.0        | a.u.       | 1st signal in the liver                                              | signal          | (0, 5)        |       |           |
# | S_2_ao         | 1.0        | a.u.       | 2nd signal in the aorta                                              | signal          | (0, 5)        |       |           |
# | S_2_li         | 1.0        | a.u.       | 2nd signal in the liver                                              | signal          | (0, 5)        |       |           |
# | tS_1_ao        | 0.0        | sec        | 1st signal time points in the aorta                                  | signal          |               |       |           |
# | tS_1_li        | 0.0        | sec        | 1st signal time points in the liver                                  | signal          |               |       |           |
# | tS_2_ao        | 0.0        | sec        | 2nd signal time points in the aorta                                  | signal          |               |       |           |
# | tS_2_li        | 0.0        | sec        | 2nd signal time points in the liver                                  | signal          |               |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_ao         | 1          | cm3        | ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_li         | 1000       | cm3        | ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                          | whole-body      | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------+
# |                                   InverseAortaLiverSplit - all outputs (n = 5)                                   |
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

import matplotlib.pyplot as plt
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import get_quantity, update_bounds
from dcmri.utils.fit import train_brute
from dcmri.forward.aorta_liver_split import AortaLiverSplit as ForwardModel
from dcmri.kinetics.functions_liver import dpars_liver
from dcmri.inverse.functions import estimate_bat

configs = deepcopy(ForwardModel.configs) | {'estimate_bat': {'none', 'ttp', 'custom'}}
defaults = deepcopy(ForwardModel.defaults) | {'estimate_bat': 'custom', 'calibrate': True}

ROIS = ['ao', 'li']
SCANS = [1, 2]

class InverseAortaLiverSplit(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'FA_1', 'dose_2', 'B1corr_2_ao', 'R1_e', 'iStrig_2_li', 'Ei_li', 'agent', 'nb_1', 'dt', 'Nz_2', 'R1_li', 'TE1_2', 'ffa', 'field_strength', 'PSw', 'tS_2_ao', 'S_2_ao', 'TE2_1', 'FA_2', 'TF', 'iStrig_1_ao', 'v_e', 'iStrig_2_ao', 'TP_1', 'vol_ao', 'S0_1_ao', 'v_e_li', 'Ef_li', 'T_b_or', 'B1corr_2_li', 'TD_2', 'kf_e2h', 'TR_1', 'Nk0_2', 'SA_2', 'TA_2', 'vr_or', 'S0_1_li', 'Nph_1', 'B1corr_1_li', 'nb_2', 'TE2_2', 'BAT_2', 'k_e2h', 'S_2_li', 'me', 'rate_1', 'T_h', 'TE1_1', 'rate_2', 'TR_2', 'dose_1', 'NSR_1_li', 'S0_2_li', 'BAT_1', 'NSR_2_ao', 'PA_1', 'bdel', 'CO', 'R1_h', 'Ti_h', 'ki_e2h', 'B1corr_1_ao', 'iStrig_1_li', 'v_li', 'tS_1_ao', 'SA_1', 'tS_2_li', 'iz_1', 'TP_2', 'D_hl', 'T_e_or', 'TA_1', 'Nph_2', 'T_gu', 'NSR_2_li', 'T_hl', 'NSR_1_ao', 'S_1_li', 'E_li', 'rate', 'iz_2', 'dose_tolerance', 'T_la', 'weight', 'fCO_li', 'BAT', 'dose', 'tS_1_li', 'v_h', 'TD_1', 'S_1_ao', 'vol_li', 'pfree', 'TE_1', 'S0_2_ao', 'PA_2', 'R1_b', 'Tf_h', 'TE_2', 'Nk0_1', 'Nz_1', 'E_or', 'H'}
    _all_outputs = {'popt', 'psdev', 'pcov', 'loss', 'pars'}

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

        # trigger signals
        if self.config['trigger']:
            for roi in ROIS:
                for scan in SCANS:
                    if p[f'iStrig_{scan}_{roi}'] is not None:
                        p[f'tS_{scan}_{roi}'] = p[f'tS_{scan}_{roi}'][p[f'iStrig_{scan}_{roi}']]  
                        p[f'S_{scan}_{roi}'] = p[f'S_{scan}_{roi}'][:, :, p[f'iStrig_{scan}_{roi}']]     

    def _pfree(self):
        inputs = self._forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='physiological'}
        pfree |= {p for p in ['BAT', 'BAT_1', 'BAT_2'] if p in inputs}
        if not self.config['calibrate']:
            pfree |= {'S0_1_ao', 'S0_1_li', 'S0_2_ao', 'S0_2_li'}
        pfree -= {'v_li', 'H'}
        return {p: get_quantity(p)['bounds'] for p in pfree}

    def _pder(self, data:dict):
        p = {k: v for k, v in data.items() if get_quantity(k)['group'] in ['physiological', 'whole-body']}
        p = dpars_liver(p, kinetics=self.config['liver'])
        return p
    
    def _call(self, p: dict=None, **kwargs) -> dict:
        self._preproc(p)

        # Estimate BAT if needed
        if self.config['estimate_bat'] == 'custom':
            if self.config['bolus'] == 'dual':
                p['BAT_1'] = estimate_bat(p['tS_1_ao'], p['S_1_ao'])
                p['BAT_2'] = p['tS_2_ao'][np.argmax(p['S_2_ao'])]
            else:
                p['BAT'] = estimate_bat(p['tS_1_ao'], p['S_1_ao'])
        elif self.config['estimate_bat'] == 'ttp':
            if self.config['bolus'] == 'dual':
                p['BAT_1'] = p['tS_1_ao'][np.argmax(p['S_1_ao'])]
                p['BAT_2'] = p['tS_2_ao'][np.argmax(p['S_2_ao'])]
            else:
                p['BAT'] = p['tS_1_ao'][np.argmax(p['S_1_ao'])]

        # Initialize pfree if needed
        if p['pfree'] is None:
            p['pfree'] = self._pfree() 

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # if self.config['bolus'] == 'dual':
        #     bats = ['BAT_1', 'BAT_2']
        # else:
        #     bats = ['BAT']

        # Compute inverse
        signal = (p['S_1_ao'], p['S_2_ao'], p['S_1_li'], p['S_2_li'])

        # output = train_bat(self._predict, time, signal, p, p['pfree'], bats=bats, **kwargs)
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
            inputs -= {'BAT', 'BAT_1', 'BAT_2'}
        for scan in SCANS:
            inputs -= {f'tstart_{scan}', f'tacq_{scan}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'pars', 'loss'}
    
    def test_data(self, data: dict=None): 
        pfree = {
            'CO': (10, 300), 
            'BAT': (-60, 60),  
            'BAT_1': (-60, 60), 
            'BAT_2': (-60, 60)
        }

        p = self.init_data()
        p |= self._forward.test_data()

        pred = self._forward(p)
        p |= {
            'nb_1': 5,
            'nb_2': 5,
            'pfree': self._forward.filter_data(pfree)
        }
        for roi in ROIS:  
            for scan in SCANS:      
                p |= {
                    f'tS_{scan}_{roi}': pred[f'tS_{scan}_{roi}'],
                    f'S_{scan}_{roi}': pred[f'S_{scan}_{roi}'], 
                }
        
        return self.input_data(p, data)

    def plot(self, data: dict, result:dict=None, xlim:list=None, fname:str=None, show=True):
        if result is None:
            result = self.__call__(data)
        p = self.map_data(data | result['popt'])

        # Not an input parameter but required by forward model
        if self.config['estimate_bat'] != 'none':
            for bat in ['BAT', 'BAT_1', 'BAT_2']:
                if bat in result['popt']:
                    p[bat] = result['popt'][bat]

        self._preproc(p)
        pred = self._forward(p)

        if xlim is None: 
            xlim = [pred['tR'][0], pred['tR'][-1]]
        xlim = np.array(xlim)/60

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.3)

        # Plot signals
        def _plot_data2scan(roi, ts, s, ax, color):
            ax.set(xlabel='Time (min)', ylabel='MR Signal (a.u.)', xlim=xlim)
            for i in range(s[0].shape[0]):
                for j in range(s[0].shape[1]):
                    ax.plot(ts[0] / 60, s[0][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(ts[1] / 60, s[1][i, j, :], marker='o', color=color[0], label='fitted data', linestyle='None')
                    ax.plot(pred[f'tS_1_{roi}'] / 60, pred[f'S_1_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
                    ax.plot(pred[f'tS_2_{roi}'] / 60, pred[f'S_2_{roi}'][i, j, :], linestyle='-', color=color[1], linewidth=3.0, label='fit')
            ax.legend()

        _plot_data2scan('ao',(p['tS_1_ao'], p['tS_2_ao']), (p['S_1_ao'], p['S_2_ao']), ax1, ['lightcoral', 'darkred'])
        _plot_data2scan('li',(p['tS_1_li'], p['tS_2_li']), (p['S_1_li'], p['S_2_li']), ax3, ['cornflowerblue', 'darkblue'])

        # Plot concentrations
        ax2.set(xlabel='Time (min)', ylabel='Concentration (mM)', xlim=xlim)
        ax2.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        ax2.plot(pred['tC'] / 60, 1000 * pred['C_ao'][0], linestyle='-', color='darkred', linewidth=2.0, label='Aorta')
        ax2.legend()

        ax4.set(xlabel='Time (min)', ylabel='Tissue concentration (mM)', xlim=xlim)
        ax4.plot(pred['tC'] / 60, 0 * pred['tC'], color='gray')
        if pred['C_li'].shape[0]==2:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][0, :], linestyle='-.', color='darkblue', linewidth=2.0, label='Extracellular')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'][1, :], linestyle='--', color='darkblue', linewidth=2.0, label='Hepatocytes')
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'].sum(axis=0), linestyle='-', color='darkblue', linewidth=2.0, label='Tissue')
        else:
            ax4.plot(pred['tC'] / 60, 1000 * pred['C_li'], linestyle='-', color='darkblue', linewidth=2.0, label='Liver')
        ax4.legend()

        if fname is not None: 
            plt.savefig(fname=fname)
        if show: 
            plt.show()
        else: 
            plt.close()