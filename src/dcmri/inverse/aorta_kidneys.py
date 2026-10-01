# +--------------------------------------------------------------------------------------------------+
# |                            InverseAortaKidneys - all configs (n = 21)                            |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | inflow            | none, pool                                                      | none       |
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
# | bolus             | double, single                                                  | single     |
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
# |                                                         InverseAortaKidneys - all inputs (n = 85)                                                         |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |                |       |           |
# | bdel           | sec        | delay in a double injection                             | Indicator       | 30         | (-30, 30)      |       |           |
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
# | S_ao           | a.u.       | signal in the aorta                                     | Signal          | 1.0        | (0, 5)         |       |           |
# | S_lk           | a.u.       | signal in the left kidney                               | Signal          | 1.0        | (0, 5)         |       |           |
# | S_rk           | a.u.       | signal in the right kidney                              | Signal          | 1.0        | (0, 5)         |       |           |
# | iStrig_ao      |            | indices of the signal trigger in the aorta              | Signal          | None       |                |       |           |
# | iStrig_lk      |            | indices of the signal trigger in the left kidney        | Signal          | None       |                |       |           |
# | iStrig_rk      |            | indices of the signal trigger in the right kidney       | Signal          | None       |                |       |           |
# | nb             | a.u.       | number of baseline time points                          | Signal          | 1          |                |       |           |
# | pfree          | a.u.       | set of free parameters                                  | Signal          | 1          |                |       |           |
# | tS_ao          | sec        | signal time points in the aorta                         | Signal          | 0.0        |                |       |           |
# | tS_lk          | sec        | signal time points in the left kidney                   | Signal          | 0.0        |                |       |           |
# | tS_rk          | sec        | signal time points in the right kidney                  | Signal          | 0.0        |                |       |           |
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
# | h_u_lk         | Hz         | transit time distribution in tubuli of the left kidney  | Physiological   | 1.0        | (0.1, 60)      |       |           |
# | h_u_rk         | Hz         | transit time distribution in tubuli of the right kidney | Physiological   | 1.0        | (0.1, 60)      |       |           |
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

# +-----------------------------------------------------------------------------------------------------------------+
# |                                    InverseAortaKidneys - all outputs (n = 4)                                    |
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
from dcmri.forward.aorta_kidneys import ForwardAortaKidneys as Forward

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

configs['bolus'].discard('dual') # Only meaningful for split protocols

ROIS = ['ao', 'lk', 'rk']

class InverseAortaKidneys(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'nb', 'pfree', 'Nz', 'S0_lk', 'dose_1', 'F_u_lk', 'PA', 'F_u_rk', 'H', 'h_u_rk', 'NSR_rk', 'iStrig_ao', 'S_rk', 'TD', 'B1corr_lk', 'TA', 'T_hl', 'v_b', 'iStrig_rk', 'S0_rk', 'bdel', 'NSR_lk', 'T_b_or', 'S0_ao', 'S_ao', 'agent', 'rate', 'TF', 'iStrig_lk', 'tS_lk', 'R1_ki', 'T_gu', 'rate_1', 'vol_lk', 'T_u_lk', 'D_hl', 'TP', 'tstart', 'tS_ao', 'TR', 'B1corr_ao', 'T_ar', 'dose_2', 'DRPF', 'S_lk', 'tS_rk', 'T_e_or', 'dose_tolerance', 'NSR_ao', 'v_ki', 'dose', 'dt', 'v_p_lk', 'rate_2', 'Nph', 'TE', 'F_u', 'vol_rk', 'TE1', 'TE2', 'E_or', 'FF_lk', 'R1_c', 'iz', 'FA', 'E_li', 'PSw', 'v_p_rk', 'R1_b', 'Nk0', 'tacq', 'FF_rk', 'weight', 'vol_ao', 'B1corr_rk', 'T_u_rk', 'h_u_lk', 'CO', 'fCO_ki', 'R1_u', 'v_u', 'SA', 'me', 'field_strength', 'v_c'}
    _all_outputs = {'psdev', 'pcov', 'popt', 'loss'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        return tuple([pred[f'S_{roi}'].reshape(-1) for roi in ROIS])

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  

        # Set calibration signal
        if self.config['calibrate']:
            for roi in ROIS:
                p[f"Scal_{roi}"] = p[f"S_{roi}"][..., :p['nb']]
                p[f'iScal_{roi}'] = np.arange(p['nb'])

        # Estimate bat from data
        bat = estimate_bat(p['tS_ao'], p['S_ao'], p['nb'])
        p['BAT'] = max(bat - p['T_hl'], 0)

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        signal = tuple([data[f'S_{roi}'].reshape(-1) for roi in ROIS])
        p = train_bat(self._predict, None, signal, p, p['pfree'], **kwargs)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        inputs |= {'nb', 'pfree'}
        for roi in ROIS:
            inputs |= {f'tS_{roi}', f'S_{roi}'}
        inputs -= {'BAT'}
        for roi in ROIS:
            inputs -= {f'Scal_{roi}', f'iScal_{roi}'}
        return inputs  
    
    def outputs(self):
        return {'popt', 'psdev', 'pcov', 'loss'}
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': {'CO': (10, 300), 'BAT': (-60, 60)},
        }
        for roi in ROIS:        
            p |= {
                f'tS_{roi}': pred[f'tS_{roi}'],
                f'S_{roi}': pred[f'S_{roi}'], 
            }
               
        return self.input_data(p, data)

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        pfree |= {'BAT'}
        return {p: get_quantity(p)['bounds'] for p in pfree}