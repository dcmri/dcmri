# +--------------------------------------------------------------------------------------------------+
# |                              InverseTissueX - all configs (n = 11)                               |
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
# |                                                            InverseTissueX - all inputs (n = 48)                                                            |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | Key            | Unit       | Name                                                     | Group           | Init       | Bounds         | DICOM | OSIPI     |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | agent          |            | contrast agent generic name                              | Indicator       | gadoterate |                |       |           |
# | c_ar           | mmol/mL    | concentration in the artery                              | Indicator       | 0.005      | (0, 1)         |       |           |
# +----------------+------------+----------------------------------------------------------+-----------------+------------+----------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                    | Signal          | 0.0        | (0, 100000.0)  |       |           |
# | S              | a.u.       | signal                                                   | Signal          | 1.0        | (0, 5)         |       |           |
# | S0             | a.u.       | signal scaling factor                                    | Signal          | 1.0        | (0, 5)         |       | Q.MS1.010 |
# | iStrig         |            | indices of the signal trigger                            | Signal          | None       |                |       |           |
# | nb             | a.u.       | number of baseline time points                           | Signal          | 1          |                |       |           |
# | pfree          | a.u.       | set of free parameters                                   | Signal          | 1          |                |       |           |
# | tS             | sec        | signal time points                                       | Signal          | 0.0        |                |       |           |
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

# +-----------------------------------------------------------------------------------------------------------------+
# |                                       InverseTissueX - all outputs (n = 4)                                      |
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
from dcmri.utils.fit import train
from dcmri.forward.tissue_x import ForwardTissueX as Forward

configs = deepcopy(Forward.configs) 
defaults = deepcopy(Forward.defaults)

class InverseTissueX(Module):

    configs = configs
    defaults = defaults

    _all_inputs = {'R1_b', 'T_ar', 'v_e', 'Nph', 'pfree', 'tS', 'PSe', 'TE', 'Nz', 'TP', 'agent', 'TE1', 'TD', 'Nk0', 'R1_bi', 'TR', 'B1corr', 'field_strength', 'H', 'v_bi', 'v_b', 'NSR', 'R1_ti', 'R1_ic', 'PA', 'S', 'tstart', 'R1_c', 'PS', 'Ktrans', 'P', 'v_ti', 'TA', 'v_c', 'FA', 'iStrig', 'iz', 'R1_i', 'v_i', 'S0', 'nb', 'v_ic', 'PSc', 'TE2', 'dt', 'F_b', 'c_ar', 'me'}
    _all_outputs = {'pcov', 'psdev', 'popt', 'loss'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)
        self.forward = Forward(**self.config)
        self.map_io(imap, omap)

    def _predict(self, time):
        pred = self.forward(self._pars)
        nt = len(time)
        return pred['S'][:, :, :nt].reshape(-1)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data)  

        if self.config['calibrate']:
            p['Scal'] = p['S'][..., :p['nb']]
            p['iScal'] = np.arange(p['nb'])

        p['pfree'] = update_bounds(p['pfree'], value=p)

        # Compute inverse
        self._pars = p
        time = data['tS']
        signal = data['S']
        p |= train(self._predict, time, signal, p, p['pfree'], **kwargs)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self.forward.mapped_inputs()
        if self.config['calibrate']:
            inputs |= {'S', 'nb'}
            inputs -= {'Scal', 'iScal'}
        inputs |= {'tS', 'S', 'pfree'}
        return inputs  
    
    def outputs(self):
        outputs = {'popt', 'psdev', 'pcov', 'loss'}
        return outputs
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self.forward.dummy_data()

        pred = self.forward(p)
        p |= {
            'nb': 5,
            'pfree': self.forward.filter_data({'v_b': (0, 1), 'v_i': (0, 1), 'v_e': (0, 1), 'F_b': (0, 1)}),
            'tS': pred['tS'],
            'S': pred['S'],
        }
        return self.input_data(p, data)

    def pfree(self):
        inputs = self.forward.mapped_inputs()
        pfree = {p for p in inputs if get_quantity(p)['group']=='phys'}
        if not self.config['calibrate']:
            pfree |= {'S0'}
        return {p: get_quantity(p)['bounds'] for p in pfree}