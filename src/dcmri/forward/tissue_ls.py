# +--------------------------------------------------------------------------------------------------+
# |                               ForwardTissueLS - all configs (n = 9)                                |
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
# | baseline       | literature, measured                                               | literature |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                           ForwardTissueLS - all inputs (n = 29)                                                            |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |               |       |           |
# | ci             | mmol/mL    | inlet concentration                                     | Indicator       | 0.005      |               |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0             | a.u.       | signal scaling factor                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal           | a.u.       | calibration signal                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal          |            | indices of calibration signal                           | Signal          | 0          |               |       |           |
# | iStrig         |            | indices of the signal trigger                           | Signal          | None       |               |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                              | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)     |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr         |            | B1-correction factor                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_t           | Hz         | tissue R1 in tissue                                     | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | F_b            | mL/sec/cm3 | flow per unit tissue in the blood                       | Physiological   | 0.02       | (0, 1)        |       |           |
# | irf            | mL/sec/cm3 | Impulse response function                               | Physiological   | 0.02       | (0, 10)       |       |           |
# | v_t            | mL/cm3     | volume fraction in tissue                               | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |               |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------+
# |                                      ForwardTissueLS - all outputs (n = 14)                                      |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | Key | Unit     | Name                                   | Group           | Init  | Bounds | DICOM | OSIPI     |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | C   | mmol/cm3 | tissue concentration                   | Indicator       | 0.005 | (0, 1) |       |           |
# | tC  | sec      | concentration time points              | Indicator       | 0.0   |        |       |           |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | S   | a.u.     | signal                                 | Signal          | 1.0   | (0, 5) |       |           |
# | S0  | a.u.     | signal scaling factor                  | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# | tS  | sec      | signal time points                     | Signal          | 0.0   |        |       |           |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | M   | A/cm     | magnetization                          | Electromagnetic | 1     | (0, 5) |       |           |
# | Mz  | A/cm     | longitudinal magnetization             | Electromagnetic | 1     | (0, 5) |       |           |
# | R1  | Hz       | tissue R1                              | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1i | Hz       | inlet R1                               | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R2  | Hz       | tissue R2                              | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2s | Hz       | tissue R2*                             | Electromagnetic | 20    | (0, 5) |       |           |
# | tM  | sec      | magnetization time points              | Electromagnetic | 0.0   |        |       |           |
# | tMz | sec      | longitudinal magnetization time points | Electromagnetic | 0.0   |        |       |           |
# | tR  | sec      | relaxation rate time points            | Electromagnetic | 0.0   |        |       |           |
# +----------------------------------------------------------------------------------------------------------------+


import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcTissueLS
from dcmri.relaxivity.modules_rois import RelaxivityGeneric
from dcmri.bloch.modules_rois import WaterExchangeGeneric
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels


configs = ConcToSignal.configs | WaterExchangeGeneric.configs | RelaxivityGeneric.configs | ConcTissueLS.configs
defaults = ConcToSignal.defaults | WaterExchangeGeneric.defaults | RelaxivityGeneric.defaults | ConcTissueLS.defaults

configs['inflow'].discard('inlet')
for discard in ['tof_corr', 'water_exchange']:
    configs.pop(discard, None)
    defaults.pop(discard, None)


class ForwardTissueLS(Module):
    """Whole-body model for linear and stationary tissues."""

    configs = configs
    defaults = defaults

    _all_inputs = {'Nz', 'TR', 'irf', 'F_b', 'TE2', 'TP', 'iStrig', 'B1corr', 'iScal', 'field_strength', 'tstart', 'Scal', 'me', 'ci', 'R1_t', 'S0', 'TE1', 'FA', 'Nk0', 'TD', 'v_t', 'TE', 'dt', 'TA', 'Nph', 'PA', 'NSR', 'iz', 'agent'}
    _all_outputs = {'Mz', 'S', 'R1i', 'tS', 'tMz', 'R2s', 'tR', 'R1', 'tM', 'R2', 'M', 'C', 'tC', 'S0'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._conc = ConcTissueLS(**self.config)
        self._tissue_rel = RelaxivityGeneric(**self.config)
        self._tissue_wex = WaterExchangeGeneric(**self.config)
        self._conc_to_signal = ConcToSignal(**self.config)

        self.map_io(imap, omap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._conc(p)

        p['tacq'] = p['dt'] * (p['ci'].size - 1)
        p['v_t'] = [p['v_t']]
        p['R1_t'] = [p['R1_t']]

        p |= self._tissue_rel(p) 
        p |= self._tissue_wex(p) 
        p |= self._conc_to_signal(p)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = {'v_t', 'R1_t'}
        inputs |= self._conc.mapped_inputs()
        inputs |= self._tissue_rel.mapped_inputs() 
        inputs |= self._tissue_wex.mapped_inputs() 
        inputs |= self._conc_to_signal.mapped_inputs()

        inputs -= {'tacq'}
        inputs -= self._conc.new_mapped_outputs()
        inputs -= self._tissue_rel.new_mapped_outputs()
        inputs -= self._tissue_wex.new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._conc_to_signal.mapped_outputs() 
        return outputs
    
    def dummy_data(self): 
        n0, nt = 1, 180
        
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        p = self.init_data() 
        p |= {
            'iScal': np.arange(n0, dtype=int),
            'Scal': Scal, 
            'ci': np.ones(nt),
            'irf': p['irf'] * np.ones(nt)
        }
        return p