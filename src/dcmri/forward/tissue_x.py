# +--------------------------------------------------------------------------------------------------+
# |                              ForwardTissueX - all configs (n = 11)                               |
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
# | water_exchange | FF, FN, FR, NF, NN, NR, RF, RN, RR                                 | FF         |
# | kinetics       | 2CU, 2CX, FX, HF, HFU, NX, NXP, U, WV                              | 2CX        |
# | baseline       | literature, measured                                               | literature |
# +--------------------------------------------------------------------------------------------------+

# +------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            ForwardTissueX - all inputs (n = 46)                                                            |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                     | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                              | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                              | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                    | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S0             | a.u.       | signal scaling factor                                    | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | Scal           | a.u.       | calibration signal                                       | Signal          | 1.0        | (0, 5)         |       | Q.MS1.002 |
# | iScal          |            | indices of calibration signal                            | Signal          | 0          |                |       |           |
# | iStrig         |            | indices of the signal trigger                            | Signal          | None       |                |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | FA             | deg        | flip angle                                               | Sequence        | 15         | (0, 180)       |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space  | Sequence        | 64         | (0, 1000)      |       |           |
# | Nph            |            | number of acquired phase lines in k-space                | Sequence        | 128        | (0, 1000)      |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition            | Sequence        | 64         | (0, 1000)      |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                             | Sequence        | 90         | (0, 180)       |       |           |
# | TA             | sec        | acquisition time                                         | Sequence        | 2.0        | (0, 30)        |       |           |
# | TD             | sec        | prepulse delay                                           | Sequence        | 0.05       | (0, 1)         |       |           |
# | TE             | sec        | echo time                                                | Sequence        | 0.001      | (0, 10)        |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                 | Sequence        | 0.001      | (0, 1)         |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                | Sequence        | 0.005      | (0, 1)         |       |           |
# | TP             | sec        | preparation delay                                        | Sequence        | 0.05       | (0, 1)         |       |           |
# | TR             | sec        | repetition time                                          | Sequence        | 0.005      | (0, 1)         |       |           |
# | field_strength | T          | magnetic field strength                                  | Sequence        | 3          | (0, 20)        |       |           |
# | iz             |            | slice number in a multi-slice acquisition                | Sequence        | 0          | (0, 1000)      |       |           |
# | tstart         | sec        | start of the acquisition                                 | Sequence        | 0          | (0, 10000.0)   |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | B1corr         |            | B1-correction factor                                     | Electromagnetic | 1          | (0, 5)         |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                   | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_bi          | Hz         | tissue R1 in blood and interstitium                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_c           | Hz         | tissue R1 in cells                                       | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_i           | Hz         | tissue R1 in interstitium                                | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ic          | Hz         | tissue R1 in interstitium and cells                      | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | R1_ti          | Hz         | tissue R1 in the tissue                                  | Electromagnetic | 0.65       | (0, 5)         |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                | Electromagnetic | 1          | (0, 5)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | F_b            | mL/sec/cm3 | flow per unit tissue in the blood                        | Physiological   | 0.02       | (0, 1)         |       |           |
# | H              |            | hematocrit                                               | Physiological   | 0.45       | (0, 1)         |       |           |
# | Ktrans         | mL/sec/cm3 | plasma clearance                                         | Physiological   | 0.015      | (0.0, 0.1)     |       |           |
# | P              |            | porosity                                                 | Physiological   | 0.3        | (0, 1)         |       |           |
# | PS             | mL/sec/cm3 | permeability-surface area product                        | Physiological   | 0.003      | (0, 1)         |       |           |
# | PSc            | mL/sec/cm3 | transcytolemmal water permeability-surface area product  | Physiological   | 0.03       | (0, 100)       |       |           |
# | PSe            | mL/sec/cm3 | transendothelial water permeability-surface area product | Physiological   | 0.03       | (0, 100)       |       |           |
# | T_ar           | sec        | mean transit time in the artery                          | Physiological   | 30         | (0.1, 60)      |       |           |
# | v_b            | mL/cm3     | volume fraction in the blood                             | Physiological   | 0.1        | (0.001, 0.999) |       |           |
# | v_bi           | mL/cm3     | volume fraction in blood and interstitium                | Physiological   | 1          | (0, 1)         |       |           |
# | v_c            | mL/cm3     | volume fraction in cells                                 | Physiological   | 0.6        | (0.001, 0.999) |       |           |
# | v_e            | mL/cm3     | volume fraction in extracellular space                   | Physiological   | 0.3        | (0.01, 0.6)    |       |           |
# | v_i            | mL/cm3     | volume fraction in interstitium                          | Physiological   | 0.3        | (0.001, 0.999) |       |           |
# | v_ic           | mL/cm3     | volume fraction in interstitium and cells                | Physiological   | 1          | (0, 1)         |       |           |
# | v_ti           | mL/cm3     | volume fraction in the tissue                            | Physiological   | 1          | (0, 1)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                              | Hyperparameters | 0.5        |                |       |           |
# +------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------+
# |                                ForwardTissueX - all outputs (n = 13)                                |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | Key | Unit     | Name                        | Group           | Init  | Bounds | DICOM | OSIPI     |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | C   | mmol/cm3 | tissue concentration        | Indicator       | 0.005 | (0, 1) |       |           |
# | ci  | mmol/mL  | inlet concentration         | Indicator       | 0.005 |        |       |           |
# | tC  | sec      | concentration time points   | Indicator       | 0.0   |        |       |           |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | S   | a.u.     | signal                      | Signal          | 1.0   | (0, 5) |       |           |
# | S0  | a.u.     | signal scaling factor       | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# | tS  | sec      | signal time points          | Signal          | 0.0   |        |       |           |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | M   | A/cm     | magnetization               | Electromagnetic | 1     | (0, 5) |       |           |
# | R1  | Hz       | tissue R1                   | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1i | Hz       | inlet R1                    | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R2  | Hz       | tissue R2                   | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2s | Hz       | tissue R2*                  | Electromagnetic | 20    | (0, 5) |       |           |
# | tM  | sec      | magnetization time points   | Electromagnetic | 0.0   |        |       |           |
# | tR  | sec      | relaxation rate time points | Electromagnetic | 0.0   |        |       |           |
# +-----------------------------------------------------------------------------------------------------+

from copy import deepcopy
import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcTissueX
from dcmri.relaxivity.modules_rois import RelaxivityTissueX
from dcmri.bloch.modules_rois import WaterExchangeTissueX
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

tissue_rel = RelaxivityTissueX
tissue_wex = WaterExchangeTissueX

configs = deepcopy(ConcToSignal.configs | tissue_wex.configs | tissue_rel.configs | ConcTissueX.configs)
defaults = deepcopy(ConcToSignal.defaults | tissue_wex.defaults | tissue_rel.defaults | ConcTissueX.defaults)

configs['inflow'].discard('inlet')
configs.pop('tof_corr')
defaults.pop('tof_corr')

class ForwardTissueX(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {'dt', 'v_bi', 'PSc', 'TA', 'iStrig', 'R1_b', 'B1corr', 'v_b', 'TE2', 'T_ar', 'TP', 'v_c', 'PA', 'TD', 'agent', 'me', 'Nph', 'FA', 'F_b', 'v_e', 'R1_ic', 'c_ar', 'TE', 'Nk0', 'R1_i', 'field_strength', 'Nz', 'R1_bi', 'tstart', 'iScal', 'TE1', 'P', 'v_ic', 'TR', 'v_ti', 'PS', 'NSR', 'R1_ti', 'Ktrans', 'Scal', 'iz', 'PSe', 'v_i', 'H', 'R1_c', 'S0'}
    _all_outputs = {'tR', 'tC', 'M', 'ci', 'R1i', 'tM', 'R2', 'C', 'S', 'R2s', 'tS', 'R1', 'S0'}

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._conc(p)
        p |= self._tissue_rel(p) 
        p |= self._tissue_wex(p) 

        p['tacq'] = p['dt'] * (p['ci'].size - 1)
        p |= self._conc_to_signal(p)

        return self.map_results(p)

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._conc = ConcTissueX(**self.config)
        self._tissue_rel = RelaxivityTissueX(**self.config)
        self._tissue_wex = WaterExchangeTissueX(**self.config)
        self._conc_to_signal = ConcToSignal(**self.config)

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
        return outputs
    
    def dummy_data(self, data:dict=None): 
        p = self.init_data()

        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        p |= {
            'iScal': np.arange(n0, dtype=int),
            'Scal': Scal, 
        }
        nt = 180
        ci = np.ones(nt)
        p['c_ar'] = ci

        return self.input_data(p, data)