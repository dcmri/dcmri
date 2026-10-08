# +--------------------------------------------------------------------------------------------------+
# |                               ForwardKidney - all configs (n = 11)                               |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | inflow         | none, pool                                                         | none       |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | water_exchange | F, N, R                                                            | F          |
# | baseline       | literature, measured                                               | literature |
# | kinetics       | 2CF, 2CFU, 2PF, 2PFU, CPF, FN, HF, HFU                             | 2CF        |
# +--------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            ForwardKidney - all inputs (n = 41)                                                            |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                             | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR_ki         |            | noise-to-signal ratio in the kidney                     | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0_ki          | a.u.       | signal scaling factor in the kidney                     | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | Scal_ki        | a.u.       | calibration signal in the kidney                        | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | iScal_ki       |            | indices of calibration signal in the kidney             | Signal          | 0          |                |       |           |
# | iStrig_ki      |            | indices of the signal trigger in the kidney             | Signal          | None       |                |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                              | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)      |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr_ki      |            | B1-correction factor in the kidney                      | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_u           | Hz         | tissue R1 in tubuli                                     | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FF             |            | filtration fraction                                     | Physiological   | 0.1        | (0, 0.5)       |       |           |
# | F_b_ki         | mL/sec/cm3 | flow per unit tissue in blood of the kidney             | Physiological   | 0.02       | (0, 1)         |       |           |
# | F_p_ki         | mL/sec/cm3 | flow per unit tissue in plasma of the kidney            | Physiological   | 0.02       | (0, 1)         |       |           |
# | F_u            | mL/sec/cm3 | flow per unit tissue in tubuli                          | Physiological   | 0.005      | (0, 0.05)      |       |           |
# | PSw            | mL/sec/cm3 | water permeability-surface area product                 | Physiological   | 0.03       | (0, 100)       |       |           |
# | T_ar           | sec        | mean transit time in the artery                         | Physiological   | 30         | (0.1, 60)      |       |           |
# | T_u            | sec        | mean transit time in tubuli                             | Physiological   | 120        | (0, 600)       |       |           |
# | h_u            | Hz         | transit time distribution in tubuli                     | Physiological   | 1.0        | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                            | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_ki           | mL/cm3     | volume fraction in the kidney                           | Physiological   | 1          | (0, 1)         |       |           |
# | v_p_ki         | mL/cm3     | volume fraction in plasma of the kidney                 | Physiological   | 0.15       | (0, 0.3)       |       |           |
# | v_u            | mL/cm3     | volume fraction in tubuli                               | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |                |       |           |
# +-----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +---------------------------------------------------------------------------------------------------------------------------+
# |                                            ForwardKidney - all outputs (n = 13)                                           |
# +--------+------------+-------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | Key    | Unit       | Name                                      | Group           | Init  | Bounds    | DICOM | OSIPI     |
# +--------+------------+-------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | C_ki   | mmol/cm3   | tissue concentration in the kidney        | Indicator       | 0.005 | (0, 1)    |       |           |
# | tC_ki  | sec        | concentration time points in the kidney   | Indicator       | 0.0   |           |       |           |
# +--------+------------+-------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | S0_ki  | a.u.       | signal scaling factor in the kidney       | Signal          | 1.0   | (0, 5)    |       | Q.MS1.010 |
# | S_ki   | a.u.       | signal in the kidney                      | Signal          | 1.0   | (0, 5)    |       |           |
# | tS_ki  | sec        | signal time points in the kidney          | Signal          | 0.0   |           |       |           |
# +--------+------------+-------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | M_ki   | A/cm       | magnetization in the kidney               | Electromagnetic | 1     | (0, 5)    |       |           |
# | R1_ki  | Hz         | tissue R1 in the kidney                   | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R1i_ki | Hz         | inlet R1 in the kidney                    | Electromagnetic | 0.65  | (0, 5)    |       |           |
# | R2_ki  | Hz         | tissue R2 in the kidney                   | Electromagnetic | 2.0   | (0, 5)    |       |           |
# | R2s_ki | Hz         | tissue R2* in the kidney                  | Electromagnetic | 20    | (0, 5)    |       |           |
# | tM_ki  | sec        | magnetization time points in the kidney   | Electromagnetic | 0.0   |           |       |           |
# | tR_ki  | sec        | relaxation rate time points in the kidney | Electromagnetic | 0.0   |           |       |           |
# +--------+------------+-------------------------------------------+-----------------+-------+-----------+-------+-----------+
# | F_u    | mL/sec/cm3 | flow per unit tissue in tubuli            | Physiological   | 0.005 | (0, 0.05) |       |           |
# +---------------------------------------------------------------------------------------------------------------------------+

from copy import deepcopy
import numpy as np

from dcmri.core.tools import extend_varname
from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcKidney
from dcmri.relaxivity.modules_rois import RelaxivityKidney
from dcmri.bloch.modules_rois import WaterExchangeKidney
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

roi = 'ki'

tissue_rel = RelaxivityKidney
tissue_wex = WaterExchangeKidney

configs = deepcopy(ConcToSignal.configs | tissue_wex.configs | tissue_rel.configs | ConcKidney.configs)
defaults = deepcopy(ConcToSignal.defaults | tissue_wex.defaults | tissue_rel.defaults | ConcKidney.defaults)

configs['inflow'].discard('inlet')
configs.pop('tof_corr')
defaults.pop('tof_corr')


class ForwardKidney(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {'TR', 'TE2', 'c_ar', 'R1_c', 'v_u', 'S0_ki', 'v_ki', 'NSR_ki', 'FF', 'Nk0', 'v_b', 'tstart', 'F_u', 'iz', 'TA', 'iStrig_ki', 'TE', 'v_c', 'v_p_ki', 'T_ar', 'Nz', 'F_p_ki', 'B1corr_ki', 'Scal_ki', 'TE1', 'h_u', 'me', 'TP', 'FA', 'agent', 'dt', 'PSw', 'PA', 'R1_b', 'R1_u', 'T_u', 'F_b_ki', 'field_strength', 'iScal_ki', 'TD', 'Nph'}
    _all_outputs = {'tR_ki', 'R1i_ki', 'tM_ki', 'S0_ki', 'C_ki', 'F_u', 'tS_ki', 'R1_ki', 'R2_ki', 'M_ki', 'R2s_ki', 'tC_ki', 'S_ki'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._conc(p)
        p |= self._tissue_rel(p) 
        p |= self._tissue_wex(p) 

        p['tacq'] = p['dt'] * (p['ci_ki_in'].size - 1)
        p |= self._conc_to_signal(p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        omap = {'ci_ki': 'ci_ki_in'}
        self._conc = ConcKidney(omap=omap, **self.config)

        iomap_roi = {k: extend_varname(k, roi=roi) for k in tissue_rel.all_outputs() | tissue_wex.all_outputs()}

        self._tissue_rel = tissue_rel(iomap=iomap_roi, **self.config)
        self._tissue_wex = tissue_wex(iomap=iomap_roi, **self.config)

        iomap_roi |= {'ci': 'ci_ki_in'}
        iomap_roi |= {k: extend_varname(k, roi=roi) for k in {'tC', 'C', 'NSR', 'S0', 'Scal', 'iScal', 'iStrig', 'B1corr', 'S', 'M', 'R1', 'R1i', 'R2', 'R2s', 'tR', 'tM', 'tS'}}

        self._conc_to_signal = ConcToSignal(iomap=iomap_roi, **self.config)

        self.map_io(imap, omap)

    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        inputs |= self._tissue_rel.mapped_inputs() 
        inputs |= self._tissue_wex.mapped_inputs() 
        inputs |= self._conc_to_signal.mapped_inputs()

        inputs -= {'tacq'}
        inputs -= self._conc.new_mapped_outputs()
        inputs -= self._tissue_rel.new_mapped_outputs()
        inputs -= self._tissue_wex.new_mapped_outputs()
        # inputs -= self._conc_to_signal.new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._conc_to_signal.mapped_outputs()
        outputs -= {'ci_ki_in'} 
        return outputs
    
    def dummy_data(self, data:dict=None): 
        p = self.init_data()

        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        p |= {
            'iScal_ki': np.arange(n0, dtype=int),
            'Scal_ki': Scal,
        }
        nt = 180
        ci = np.ones(nt)
        p['c_ar'] = ci

        return self.input_data(p, data)