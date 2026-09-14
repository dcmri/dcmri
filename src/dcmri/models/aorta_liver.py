import numpy as np

from dcmri.core.module import Module
from dcmri.core.tools import extend_varname
from dcmri.kinetics.modules_conc import ConcAortaLiver
from dcmri.relaxivity.modules_rois import RelaxivityArtery, RelaxivityLiver
from dcmri.bloch.modules_rois import WaterExchangeArtery, WaterExchangeLiver
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels


# +--------------------------------------------------------------------------------------------------+
# |                              AortaLiverModel - all configs (n = 20)                              |
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

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                 AortaLiverModel - all inputs (n = 74)                                                                 |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                 | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT            | sec        | bolus arrival time                                                   | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_1          | sec        | 1st bolus arrival time                                               | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                               | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                                          | Indicator       | gadoterate |               |       |           |
# | dose           | mL/kg      | contrast agent dose                                                  | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                              | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s       | injection rate                                                       | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s       | 1st injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                                   | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+------------+----------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR_ao         |            | noise-to-signal ratio in the aorta                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | NSR_li         |            | noise-to-signal ratio in the liver                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0_ao          | a.u.       | signal scaling factor in the aorta                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | S0_li          | a.u.       | signal scaling factor in the liver                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal_ao        | a.u.       | calibration signal in the aorta                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | Scal_li        | a.u.       | calibration signal in the liver                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal_ao       |            | indices of calibration signal in the aorta                           | Signal          | 0          |               |       |           |
# | iScal_li       |            | indices of calibration signal in the liver                           | Signal          | 0          |               |       |           |
# | iStrig_ao      |            | indices of the signal trigger in the aorta                           | Signal          | None       |               |       |           |
# | iStrig_li      |            | indices of the signal trigger in the liver                           | Signal          | None       |               |       |           |
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
# | vol_li         | cm3        | ROI volume in the liver                                              | Whole-body      | 1000       | (0, 10000)    |       |           |
# | weight         | kg         | body weight                                                          | Whole-body      | 70         | (0, 300)      |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------+
# |                                         AortaLiverModel - all outputs (n = 31)                                         |
# +--------+----------+--------------------------------------------+-----------------+-------+---------+-------+-----------+
# | Key    | Unit     | Name                                       | Group           | Init  | Bounds  | DICOM | OSIPI     |
# +--------+----------+--------------------------------------------+-----------------+-------+---------+-------+-----------+
# | C_ao   | mmol/cm3 | tissue concentration in the aorta          | Indicator       | 0.005 | (0, 1)  |       |           |
# | C_li   | mmol/cm3 | tissue concentration in the liver          | Indicator       | 0.005 | (0, 1)  |       |           |
# | J_ao   | mmol/sec | indicator flux in the aorta                | Indicator       | 1     | (0, 10) |       |           |
# | J_la   | mmol/sec | indicator flux in the liver artery         | Indicator       | 1     | (0, 10) |       |           |
# | J_lag  | mmol/sec | indicator flux in the liver artery and gut | Indicator       | 1     | (0, 10) |       |           |
# | J_li   | mmol/sec | indicator flux in the liver                | Indicator       | 1     | (0, 10) |       |           |
# | J_or   | mmol/sec | indicator flux in the organs               | Indicator       | 1     | (0, 10) |       |           |
# | J_pv   | mmol/sec | indicator flux in the portal vein          | Indicator       | 1     | (0, 10) |       |           |
# | J_ve   | mmol/sec | indicator flux in the vein                 | Indicator       | 1     | (0, 10) |       |           |
# | ci_ao  | mmol/mL  | inlet concentration in the aorta           | Indicator       | 0.005 |         |       |           |
# | ci_li  | mmol/mL  | inlet concentration in the liver           | Indicator       | 0.005 |         |       |           |
# | tC     | sec      | concentration time points                  | Indicator       | 0.0   |         |       |           |
# +--------+----------+--------------------------------------------+-----------------+-------+---------+-------+-----------+
# | S0_ao  | a.u.     | signal scaling factor in the aorta         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S0_li  | a.u.     | signal scaling factor in the liver         | Signal          | 1.0   | (0, 5)  |       | Q.MS1.010 |
# | S_ao   | a.u.     | signal in the aorta                        | Signal          | 1.0   | (0, 5)  |       |           |
# | S_li   | a.u.     | signal in the liver                        | Signal          | 1.0   | (0, 5)  |       |           |
# +--------+----------+--------------------------------------------+-----------------+-------+---------+-------+-----------+
# | M_ao   | A/cm     | magnetization in the aorta                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | M_li   | A/cm     | magnetization in the liver                 | Electromagnetic | 1     | (0, 5)  |       |           |
# | R1_ao  | Hz       | tissue R1 in the aorta                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1_li  | Hz       | tissue R1 in the liver                     | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_ao | Hz       | inlet R1 in the aorta                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R1i_li | Hz       | inlet R1 in the liver                      | Electromagnetic | 0.65  | (0, 5)  |       |           |
# | R2_ao  | Hz       | tissue R2 in the aorta                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2_li  | Hz       | tissue R2 in the liver                     | Electromagnetic | 2.0   | (0, 5)  |       |           |
# | R2s_ao | Hz       | tissue R2* in the aorta                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | R2s_li | Hz       | tissue R2* in the liver                    | Electromagnetic | 20    | (0, 5)  |       |           |
# | tM_ao  | sec      | magnetization time points in the aorta     | Electromagnetic | 0.0   |         |       |           |
# | tM_li  | sec      | magnetization time points in the liver     | Electromagnetic | 0.0   |         |       |           |
# | tR     | sec      | relaxation rate time points                | Electromagnetic | 0.0   |         |       |           |
# | tS_ao  | sec      | signal time points in the aorta            | Electromagnetic | 0.0   |         |       |           |
# | tS_li  | sec      | signal time points in the liver            | Electromagnetic | 0.0   |         |       |           |
# +------------------------------------------------------------------------------------------------------------------------+



rois = ['ao', 'li']
tissue_rel = {'ao': RelaxivityArtery, 'li': RelaxivityLiver}
tissue_wex = {'ao': WaterExchangeArtery, 'li': WaterExchangeLiver}

# ROI-specific configurations
roi_configs = ['t1_relaxation', 't2_relaxation', 't2s_relaxation']

CONFIGS = ConcToSignal.configs | WaterExchangeArtery.configs | WaterExchangeLiver.configs | RelaxivityArtery.configs | RelaxivityLiver.configs | ConcAortaLiver.configs
DEFAULTS = ConcToSignal.defaults | WaterExchangeArtery.defaults | WaterExchangeLiver.defaults | RelaxivityArtery.defaults | RelaxivityLiver.defaults | ConcAortaLiver.defaults
CMAP = {roi: {} for roi in rois}

for key in roi_configs:
    config = CONFIGS.pop(key)
    default = DEFAULTS.pop(key)
    for roi in rois:
        CONFIGS[f'{key}_{roi}'] = config
        DEFAULTS[f'{key}_{roi}'] = default
        CMAP[roi] |= {key: f'{key}_{roi}'}


class AortaLiverModel(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = CONFIGS
    defaults = DEFAULTS

    _all_inputs = {'TF', 'Scal_li', 'Ti_h', 'CO', 'TA', 'Scal_ao', 'T_la', 'Nz', 'B1corr_ao', 'E_li', 'GFR', 'Nk0', 'v_li', 'tacq', 'Ef_li', 'iScal_ao', 'dose_tolerance', 'T_h', 'Nph', 'iz', 'R1_b', 'TE1', 'H', 'dt', 'tstart', 'SA', 'BAT_1', 'dose_2', 'TR', 'R1_e', 'S0_li', 'BAT_2', 'TE', 'NSR_li', 'vol_ao', 'T_gu', 'me', 'Ei_li', 'BAT', 'B1corr_li', 'iScal_li', 'rate_1', 'v_h', 'k_e2h', 'kf_e2h', 'S0_ao', 'T_b_or', 'rate', 'ffa', 'weight', 'R1_h', 'FA', 'vol_li', 'field_strength', 'Tf_h', 'agent', 'NSR_ao', 'T_e_or', 'TE2', 'PA', 'rate_2', 'fCO_li', 'TP', 'E_or', 'PSw', 'dose', 'T_hl', 'D_hl', 'iStrig_li', 'dose_1', 'TD', 'v_e_li', 'ki_e2h', 'iStrig_ao'}
    _all_outputs = {'S0_ao', 'R1_ao', 'J_la', 'ci_ao', 'J_li', 'R1i_ao', 'S_ao', 'tS_li', 'J_ao', 'tC', 'tM_li', 'R2s_li', 'R2_li', 'tR', 'C_ao', 'R1i_li', 'tS_ao', 'J_lag', 'J_or', 'tM_ao', 'R2_ao', 'R2s_ao', 'S0_li', 'J_pv', 'R1_li', 'ci_li', 'M_ao', 'J_ve', 'S_li', 'M_li', 'C_li'}

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
            iomap_roi |= {k: extend_varname(k, roi=roi) for k in tissue_rel[roi].all_outputs() | tissue_wex[roi].all_outputs() |{'v_e'}} 
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