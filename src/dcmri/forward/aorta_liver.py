# +--------------------------------------------------------------------------------------------------+
# |                             ForwardAortaLiver - all configs (n = 20)                             |
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


class ForwardAortaLiver(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = CONFIGS
    defaults = DEFAULTS

    _all_inputs = {'dt', 'PSw', 'GFR', 'H', 'T_e_or', 'TF', 'tacq', 'iStrig_ao', 'NSR_ao', 'TA', 'Nph', 'Nz', 'PA', 'agent', 'rate', 'B1corr_li', 'TR', 'v_li', 'E_or', 'Scal_ao', 'T_gu', 'T_h', 'BAT_1', 'field_strength', 'TE1', 'iScal_li', 'ki_e2h', 'TP', 'bdel', 'S0_li', 'R1_li', 'Tf_h', 'Ei_li', 'iz', 'ffa', 'TE', 'R1_h', 'FA', 'T_hl', 'R1_e', 'TD', 'T_b_or', 'k_e2h', 'rate_2', 'R1_b', 'dose_tolerance', 'v_e_li', 'Scal_li', 'dose_1', 'iStrig_li', 'iScal_ao', 'E_li', 'CO', 'Ef_li', 'BAT', 'NSR_li', 'TE2', 'dose', 'kf_e2h', 'vol_ao', 'me', 'Ti_h', 'vol_li', 'fCO_li', 'tstart', 'dose_2', 'SA', 'v_h', 'weight', 'S0_ao', 'BAT_2', 'Nk0', 'D_hl', 'T_la', 'rate_1', 'B1corr_ao'}
    _all_outputs = {'R1_ao', 'R2s_ao', 'C_li', 'tS_ao', 'J_or', 'R1i_ao', 'J_li', 'M_ao', 'tC', 'R2_li', 'S_ao', 'M_li', 'R1i_li', 'tR', 'J_ao', 'ci_li', 'C_ao', 'ci_ao', 'R2_ao', 'J_pv', 'J_vc', 'J_lag', 'S_li', 'R2s_li', 'tM_li', 'S0_li', 'R1_li', 'J_la', 'tM_ao', 'S0_ao', 'tS_li'}

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