# +--------------------------------------------------------------------------------------------------+
# |                                AortaLiver - all configs (n = 20)                                 |
# +-------------------+-----------------------------------------------------------------+------------+
# | Key               | Values                                                          | Default    |
# +-------------------+-----------------------------------------------------------------+------------+
# | inflow            | none, pool                                                      | none       |
# | sequence          | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS,           | 3D-SPGR-SS |
# |                   | 2D-SR-SPGR, 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS,    |            |
# |                   | 3D-IR-SS, 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI,       |            |
# |                   | 3D-SPGR, 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,       |            |
# |                   | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS, ZTE-3D-SPGR-SSI              |            |
# | tof_corr          | False, True                                                     | False      |
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

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                    AortaLiver - all inputs (n = 77)                                                                   |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | Key            | Value      | Unit       | Name                                                                 | Group           | Bounds        | DICOM | OSIPI     |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | B1corr_ao      | 1          |            | B1-correction factor in the aorta                                    | electromagnetic | (0, 5)        |       |           |
# | B1corr_li      | 1          |            | B1-correction factor in the liver                                    | electromagnetic | (0, 5)        |       |           |
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
# | FA             | 15         | deg        | flip angle                                                           | sequence        | (0, 180)      |       |           |
# | field_strength | 3          | T          | magnetic field strength                                              | sequence        | (0, 20)       |       |           |
# | iz             | 0          |            | slice number in a multi-slice acquisition                            | sequence        | (0, 1000)     |       |           |
# | Nk0            | 64         |            | number of acquired phase lines to the center of k-space              | sequence        | (0, 1000)     |       |           |
# | Nph            | 128        |            | number of acquired phase lines in k-space                            | sequence        | (0, 1000)     |       |           |
# | Nz             | 64         |            | number of slices in a multi-slice acquisition                        | sequence        | (0, 1000)     |       |           |
# | PA             | 90         | deg        | preparation Pulse Flip Angle                                         | sequence        | (0, 180)      |       |           |
# | SA             | 0          | deg        | saturation Slab Flip Angle                                           | sequence        | (0, 180)      |       |           |
# | TA             | 2.0        | sec        | acquisition time                                                     | sequence        | (0, 30)       |       |           |
# | tacq           | 240        | sec        | acquisition duration                                                 | sequence        | (0, 10000.0)  |       |           |
# | TD             | 0.05       | sec        | prepulse delay                                                       | sequence        | (0, 1)        |       |           |
# | TE             | 0.001      | sec        | echo time                                                            | sequence        | (0, 10)       |       |           |
# | TE1            | 0.001      | sec        | first echo time in a multi-echo sequence                             | sequence        | (0, 1)        |       |           |
# | TE2            | 0.005      | sec        | second echo time in a multi-echo sequence                            | sequence        | (0, 1)        |       |           |
# | TP             | 0.05       | sec        | preparation delay                                                    | sequence        | (0, 1)        |       |           |
# | TR             | 0.005      | sec        | repetition time                                                      | sequence        | (0, 1)        |       |           |
# | tstart         | 0          | sec        | start of the acquisition                                             | sequence        | (0, 10000.0)  |       |           |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | iScal_ao       | 0          |            | indices of calibration signal in the aorta                           | signal          |               |       |           |
# | iScal_li       | 0          |            | indices of calibration signal in the liver                           | signal          |               |       |           |
# | iStrig_ao      | None       |            | indices of the signal trigger in the aorta                           | signal          |               |       |           |
# | iStrig_li      | None       |            | indices of the signal trigger in the liver                           | signal          |               |       |           |
# | NSR_ao         | 0.0        |            | noise-to-signal ratio in the aorta                                   | signal          | (0, 100000.0) |       |           |
# | NSR_li         | 0.0        |            | noise-to-signal ratio in the liver                                   | signal          | (0, 100000.0) |       |           |
# | S0_ao          | 1.0        | a.u.       | signal scaling factor in the aorta                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | S0_li          | 1.0        | a.u.       | signal scaling factor in the liver                                   | signal          | (0, 5)        |       | Q.MS1.010 |
# | Scal_ao        | 1.0        | a.u.       | calibration signal in the aorta                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# | Scal_li        | 1.0        | a.u.       | calibration signal in the liver                                      | signal          | (0, 5)        |       | Q.MS1.002 |
# +----------------+------------+------------+----------------------------------------------------------------------+-----------------+---------------+-------+-----------+
# | vol_ao         | 1          | cm3        | ROI volume in the aorta                                              | whole-body      | (0.0, 1000)   |       |           |
# | vol_li         | 1000       | cm3        | ROI volume in the liver                                              | whole-body      | (0, 10000)    |       |           |
# | weight         | 70         | kg         | body weight                                                          | whole-body      | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------+
# |                                           AortaLiver - all outputs (n = 31)                                            |
# +--------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | Key    | Value | Unit     | Name                                       | Group           | Bounds  | DICOM | OSIPI     |
# +--------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | M_ao   | 1     | A/cm     | magnetization in the aorta                 | electromagnetic | (0, 5)  |       |           |
# | M_li   | 1     | A/cm     | magnetization in the liver                 | electromagnetic | (0, 5)  |       |           |
# | R1_ao  | 0.65  | Hz       | tissue R1 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R1_li  | 0.65  | Hz       | tissue R1 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R1i_ao | 0.65  | Hz       | inlet R1 in the aorta                      | electromagnetic | (0, 5)  |       |           |
# | R1i_li |
# 0.65  | Hz       | inlet R1 in the liver                      | electromagnetic | (0, 5)  |       |           |
# | R2_ao  | 2.0   | Hz       | tissue R2 in the aorta                     | electromagnetic | (0, 5)  |       |           |
# | R2_li  | 2.0   | Hz       | tissue R2 in the liver                     | electromagnetic | (0, 5)  |       |           |
# | R2s_ao | 20    | Hz       | tissue R2* in the aorta                    | electromagnetic | (0, 5)  |       |           |
# | R2s_li | 20    | Hz       | tissue R2* in the liver                    | electromagnetic | (0, 5)  |       |           |
# | tM_ao  | 0.0   | sec      | magnetization time points in the aorta     | electromagnetic |         |       |           |
# | tM_li  | 0.0   | sec      | magnetization time points in the liver     | electromagnetic |         |       |           |
# | tR     | 0.0   | sec      | relaxation rate time points                | electromagnetic |         |       |           |
# +--------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | C_ao   | 0.005 | mmol/cm3 | tissue concentration in the aorta          | indicator       | (0, 1)  |       |           |
# | C_li   | 0.005 | mmol/cm3 | tissue concentration in the liver          | indicator       | (0, 1)  |       |           |
# | ci_ao  | 0.005 | mmol/mL  | inlet concentration in the aorta           | indicator       |         |       |           |
# | ci_li  | 0.005 | mmol/mL  | inlet concentration in the liver           | indicator       |         |       |           |
# | J_ao   | 1     | mmol/sec | indicator flux in the aorta                | indicator       | (0, 10) |       |           |
# | J_la   | 1     | mmol/sec | indicator flux in the liver artery         | indicator       | (0, 10) |       |           |
# | J_lag  | 1     | mmol/sec | indicator flux in the liver artery and gut | indicator       | (0, 10) |       |           |
# | J_li   | 1     | mmol/sec | indicator flux in the liver                | indicator       | (0, 10) |       |           |
# | J_or   | 1     | mmol/sec | indicator flux in the organs               | indicator       | (0, 10) |       |           |
# | J_pv   | 1     | mmol/sec | indicator flux in the portal vein          | indicator       | (0, 10) |       |           |
# | J_vc   | 1     | mmol/sec | indicator flux in the vena cava            | indicator       | (0, 10) |       |           |
# | tC     | 0.0   | sec      | concentration time points                  | indicator       |         |       |           |
# +--------+-------+----------+--------------------------------------------+-----------------+---------+-------+-----------+
# | S0_ao  | 1.0   | a.u.     | signal scaling factor in the aorta         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S0_li  | 1.0   | a.u.     | signal scaling factor in the liver         | signal          | (0, 5)  |       | Q.MS1.010 |
# | S_ao   | 1.0   | a.u.     | signal in the aorta                        | signal          | (0, 5)  |       |           |
# | S_li   | 1.0   | a.u.     | signal in the liver                        | signal          | (0, 5)  |       |           |
# | tS_ao  | 0.0   | sec      | signal time points in the aorta            | signal          |         |       |           |
# | tS_li  | 0.0   | sec      | signal time points in the liver            | signal          |         |       |           |
# +------------------------------------------------------------------------------------------------------------------------+


from copy import deepcopy
import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import extend_varname
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityLiver
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeLiver
from dcmri.bloch.functions_sequences import channels
from dcmri.signal.modules_tissue import ConcToSignal

rois = ['ao', 'li']
tissue_rel = {'ao': RelaxivityArtery, 'li': RelaxivityLiver}
tissue_wex = {'ao': WaterExchangeArtery, 'li': WaterExchangeLiver}

# ROI-specific configurations
roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

CONFIGS = deepcopy(ConcToSignal.configs | WaterExchangeArtery.configs | WaterExchangeLiver.configs | RelaxivityArtery.configs | RelaxivityLiver.configs | ConcAortaLiver.configs)
DEFAULTS = deepcopy(ConcToSignal.defaults | WaterExchangeArtery.defaults | WaterExchangeLiver.defaults | RelaxivityArtery.defaults | RelaxivityLiver.defaults | ConcAortaLiver.defaults)
CMAP = {roi: {} for roi in rois}

for key in roi_configs:
    config = CONFIGS.pop(key)
    default = DEFAULTS.pop(key)
    for roi in rois:
        CONFIGS[f'{key}_{roi}'] = config
        DEFAULTS[f'{key}_{roi}'] = default
        CMAP[roi] |= {key: f'{key}_{roi}'}

CONFIGS['inflow'].discard('inlet')


class AortaLiver(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = CONFIGS
    defaults = DEFAULTS

    _all_inputs = {'rate_1', 'dose_tolerance', 'vol_li', 'E_or', 'vr_or', 'H', 'v_e', 'Nz', 'vol_ao', 'tstart', 'BAT_2', 'field_strength', 'B1corr_ao', 'rate_2', 'PA', 'Ei_li', 'S0_li', 'v_li', 'T_h', 'NSR_li', 'tacq', 'SA', 'iStrig_li', 'D_hl', 'v_e_li', 'TP', 'iz', 'TE1', 'Ef_li', 'ki_e2h', 'agent', 'Ti_h', 'E_li', 'Scal_ao', 'BAT_1', 'iStrig_ao', 'T_la', 'B1corr_li', 'PSw', 'v_h', 'iScal_li', 'TF', 'fCO_li', 'FA', 'weight', 'dose_2', 'ffa', 'dt', 'S0_ao', 'TE2', 'Scal_li', 'R1_e', 'R1_h', 'NSR_ao', 'TR', 'TD', 'T_e_or', 'me', 'iScal_ao', 'Nk0', 'CO', 'T_b_or', 'k_e2h', 'kf_e2h', 'TE', 'dose_1', 'TA', 'T_gu', 'T_hl', 'rate', 'dose', 'bdel', 'Nph', 'BAT', 'R1_b', 'R1_li', 'Tf_h'}
    _all_outputs = {'J_ao', 'tS_li', 'R2_li', 'R2s_ao', 'S_li', 'R2s_li', 'S0_ao', 'J_or', 'C_li', 'M_ao', 'tM_ao', 'S0_li', 'J_vc', 'tC', 'ci_ao', 'J_lag', 'S_ao', 'R1i_li', 'R2_ao', 'J_la', 'tS_ao', 'R1i_ao', 'R1_ao', 'J_li', 'tM_li', 'ci_li', 'J_pv', 'C_ao', 'R1_li', 'M_li', 'tR'}

    def __call__(self, data: dict=None) -> dict:
        p = self.map_data(data)  

        p['tmax'] = p['tstart'] + p['tacq'] + p['dt']

        p |= self._conc(p)
        for roi in rois:
            p |= self._tissue_rel[roi](p) 
            p |= self._tissue_wex[roi](p) 
            p |= self._conc_to_signal[roi](p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)

        # Liver never has tof_corr
        config = {
            'ao': self.config,
            'li': self.config | {'tof_corr': False}
        }
        self._conc = ConcAortaLiver(**self.config)
        self._tissue_rel = {}
        self._tissue_wex = {}
        self._conc_to_signal = {}

        for roi in rois:
            iomap_roi = {'F_b_ar': 'F_b_ao'}
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in tissue_rel[roi].all_outputs() | tissue_wex[roi].all_outputs()} 
            self._tissue_rel[roi] = tissue_rel[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])
            self._tissue_wex[roi] = tissue_wex[roi](iomap=iomap_roi, cmap=CMAP[roi], **config[roi])

            iomap_roi |= {k: extend_varname(k, roi=roi) for k in {'C', 'ci', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr', 'S', 'M', 'R1', 'R1i', 'R2', 'R2s', 'tM', 'tS'}}
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
            # inputs -= self._conc_to_signal[roi].new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        for roi in rois:
            outputs |= self._conc_to_signal[roi].mapped_outputs() 
            outputs -= {f'F_b_{roi}'}
        return outputs
    
    def test_data(self, data: dict=None): 
        p = self.init_data()
        p |= self._conc.test_data()

        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        for roi in rois:
            p |= {
                f'iScal_{roi}': np.arange(n0, dtype=int),
                f'Scal_{roi}': Scal, 
                'BAT_2': 90, # default is the same as BAT_1
            }

        return self.input_data(p, data)