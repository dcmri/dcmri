# +--------------------------------------------------------------------------------------------------+
# |                          InverseAortaPortalLiver - all configs (n = 22)                          |
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
# | liver             | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF                                | 1I-EC      |
# | non_stationary    | E, None, U, UE                                                  | None       |
# | t1_relaxation_ao  | None, lin                                                       | lin        |
# | t1_relaxation_pv  | None, lin                                                       | lin        |
# | t1_relaxation_li  | None, lin                                                       | lin        |
# | t2_relaxation_ao  | None, lin                                                       | None       |
# | t2_relaxation_pv  | None, lin                                                       | None       |
# | t2_relaxation_li  | None, lin                                                       | None       |
# | t2s_relaxation_ao | None, lin, quad                                                 | lin        |
# | t2s_relaxation_pv | None, lin, quad                                                 | lin        |
# | t2s_relaxation_li | None, lin, quad                                                 | lin        |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             InverseAortaPortalLiver - all inputs (n = 82)                                                             |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                 | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | agent          |            | contrast agent generic name                                          | Indicator       | gadoterate |               |       |           |
# | bdel           | sec        | delay in a double injection                                          | Indicator       | 30         | (-30, 30)     |       |           |
# | dose           | mL/kg      | contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s       | injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_ao         |            | noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_li         |            | noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_pv         |            | noise-to-signal ratio in the portal vein                             | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_ao          | a.u.       | signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_li          | a.u.       | signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_pv          | a.u.       | signal scaling factor in the portal vein                             | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S_ao           | a.u.       | signal in the aorta                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_li           | a.u.       | signal in the liver                                                  | Signal          | 1.0        | (0, 5)        |       |           |
# | S_pv           | a.u.       | signal in the portal vein                                            | Signal          | 1.0        | (0, 5)        |       |           |
# | iStrig_ao      |            | indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_li      |            | indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
# | iStrig_pv      |            | indices of the signal trigger in the portal vein                     | Signal          | None       |               |       |           |
# | nb             | a.u.       | number of baseline time points                                       | Signal          | 1          |               |       |           |
# | pfree          | a.u.       | set of free parameters                                               | Signal          | 1          |               |       |           |
# | tS_ao          | sec        | signal time points in the aorta                                      | Signal          | 0.0        |               |       |           |
# | tS_li          | sec        | signal time points in the liver                                      | Signal          | 0.0        |               |       |           |
# | tS_pv          | sec        | signal time points in the portal vein                                | Signal          | 0.0        |               |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                                           | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space              | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space                            | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                        | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                         | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg        | saturation Slab Flip Angle                                           | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                                     | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                                            | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                             | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                            | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                                    | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                                      | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                              | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition                            | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq           | sec        | acquisition duration                                                 | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart         | sec        | start of the acquisition                                             | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr_ao      |            | B1-correction factor in the aorta                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_li      |            | B1-correction factor in the liver                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | B1corr_pv      |            | B1-correction factor in the portal vein                              | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                               | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_e           | Hz         | tissue R1 in extracellular space                                     | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_h           | Hz         | tissue R1 in hepatocytes                                             | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                            | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                                       | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs                       | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_li           |            | extraction fraction in the liver                                     | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | E_or           |            | extraction fraction in the organs                                    | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | Ef_li          |            | final extraction fraction in the liver                               | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_li          |            | initial extraction fraction in the liver                             | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | GFR            | mL/sec     | glomerular filtration rate                                           | Physiological   | 2          | (0, 10)       |       |           |
# | H              |            | hematocrit                                                           | Physiological   | 0.45       | (0, 1)        |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                              | Physiological   | 0.03       | (0, 100)      |       |           |
# | TF             | sec        | inflow time                                                          | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                             | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_or         | sec        | mean transit time in extracellular space of the organs               | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec        | mean transit time in the gut                                         | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_h            | sec        | mean transit time in hepatocytes                                     | Physiological   | 1800       | (600, 36000)  |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                             | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec        | mean transit time in the liver artery                                | Physiological   | 30         | (0.1, 60)     |       |           |
# | Tf_h           | sec        | final mean transit time in hepatocytes                               | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_h           | sec        | initial mean transit time in hepatocytes                             | Physiological   | 1800       | (600, 36000)  |       |           |
# | fCO_li         |            | fraction of the cardiac output in the liver                          | Physiological   | 0.1        | (0, 0.5)      |       |           |
# | ffa            |            | arterial flow fraction                                               | Physiological   | 0.2        | (0, 1)        |       |           |
# | k_e2h          | mL/sec/cm3 | tissue transfer rate from extracellular space to hepatocytes         | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_e2h         | mL/sec/cm3 | final tissue transfer rate from extracellular space to hepatocytes   | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_e2h         | mL/sec/cm3 | initial tissue transfer rate from extracellular space to hepatocytes | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | v_e_li         | mL/cm3     | volume fraction in extracellular space of the liver                  | Physiological   | 0.3        | (0.01, 0.6)   |       |           |
# | v_h            | mL/cm3     | volume fraction in hepatocytes                                       | Physiological   | 1          | (0, 1)        |       |           |
# | v_li           | mL/cm3     | volume fraction in the liver                                         | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                                       | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec        | pseudo-continuous time step                                          | Hyperparameters | 0.5        |               |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_ao         | cm3        | ROI volume in the aorta                                              | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | vol_la         | cm3        | ROI volume in the liver artery                                       | Whole-body      | 150        | (0.0, 1000)   |       |           |
# | vol_li         | cm3        | ROI volume in the liver                                              | Whole-body      | 1000       | (0, 10000)    |       |           |
# | vol_pv         | cm3        | ROI volume in the portal vein                                        | Whole-body      | 150        | (0.0, 1000)   |       |           |
# | weight         | kg         | body weight                                                          | Whole-body      | 70         | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------+
# |                                  InverseAortaPortalLiver - all outputs (n = 4)                                  |
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
from dcmri.forward.aorta_portal_liver import ForwardAortaPortalLiver as Forward

configs = deepcopy(Forward.configs)
defaults = deepcopy(Forward.defaults)

configs['bolus'].discard('dual') # Only meaningful for split protocols

ROIS = ['ao', 'pv', 'li']

class InverseAortaPortalLiver(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'bdel', 'S0_ao', 'vol_ao', 'ki_e2h', 'nb', 'NSR_pv', 'TD', 'agent', 'kf_e2h', 'dose_2', 'NSR_li', 'TR', 'k_e2h', 'pfree', 'Tf_h', 'T_hl', 'B1corr_li', 'dose_tolerance', 'S0_pv', 'Nk0', 'T_la', 'E_li', 'v_h', 'fCO_li', 'TF', 'FA', 'H', 'R1_b', 'rate_2', 'iStrig_pv', 'tS_ao', 'dose', 'PA', 'T_b_or', 'iz', 'B1corr_pv', 'TP', 'Ei_li', 'dose_1', 'TA', 'R1_h', 'v_e_li', 'vol_li', 'TE', 'R1_e', 'dt', 'S_li', 'TE2', 'vol_la', 'Ti_h', 'tS_li', 'TE1', 'rate_1', 'tstart', 'CO', 'B1corr_ao', 'Nph', 'tacq', 'GFR', 'field_strength', 'S_pv', 'v_li', 'T_gu', 'Ef_li', 'rate', 'ffa', 'S_ao', 'vol_pv', 'iStrig_ao', 'weight', 'me', 'T_h', 'tS_pv', 'NSR_ao', 'SA', 'iStrig_li', 'D_hl', 'S0_li', 'PSw', 'Nz', 'E_or', 'T_e_or'}
    _all_outputs = {'popt', 'psdev', 'loss', 'pcov'}

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