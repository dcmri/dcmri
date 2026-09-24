import numpy as np

from dcmri.core.tools import extend_varname
from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcLiver
from dcmri.relaxivity.modules_rois import RelaxivityLiver
from dcmri.bloch.modules_rois import WaterExchangeLiver
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

# +--------------------------------------------------------------------------------------------------+
# |                                ForwardLiver - all configs (n = 12)                                 |
# +----------------+--------------------------------------------------------------------+------------+
# | Key            | Values                                                             | Default    |
# +----------------+--------------------------------------------------------------------+------------+
# | t1_relaxation  | None, lin                                                          | lin        |
# | t2_relaxation  | None, lin                                                          | None       |
# | t2s_relaxation | None, lin, quad                                                    | lin        |
# | inflow         | False, True                                                        | False      |
# | sequence       | 2D-DE-EPI, 2D-GE-EPI, 2D-SE-EPI, 2D-SPGR, 2D-SPGR-SS, 2D-SR-SPGR,  | 3D-SPGR-SS |
# |                | 3D-DE-EPI, 3D-GE-EPI, 3D-IR-SPGR, 3D-IR-SPGR-SS, 3D-IR-SS,         |            |
# |                | 3D-PR-SPGR, 3D-PR-SPGR-SS, 3D-PR-SS, 3D-SE-EPI, 3D-SPGR,           |            |
# |                | 3D-SPGR-SS, 3D-SR-SPGR, 3D-SR-SPGR-SS, 3D-SR-SS,                   |            |
# |                | ZTE-3D-IR-SPGR-SS, ZTE-3D-SPGR-SS                                  |            |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | compartments   | ('e', 'h'), ('li',)                                                | ('li',)    |
# | baseline       | literature, measured                                               | literature |
# | kinetics       | 1I-EC, 1I-EC-HF, 1I-IC, 1I-IC-HF, 2I-EC, 2I-EC-HF, 2I-IC,          | 2I-EC      |
# |                | 2I-IC-HF, 2I-IC-U                                                  |            |
# | non_stationary | E, None, U, UE                                                     | None       |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                                         ForwardLiver - all inputs (n = 45)                                                                         |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                                            | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | agent          |            | contrast agent generic name                                                     | Indicator       | gadoterate |               |       |           |
# | ci_li          | mmol/mL    | inlet concentration in the liver                                                | Indicator       | 0.005      |               |       |           |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR            |            | noise-to-signal ratio                                                           | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0             | a.u.       | signal scaling factor                                                           | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal           | a.u.       | calibration signal                                                              | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal          |            | indices of calibration signal                                                   | Signal          | 0          |               |       |           |
# | iStrig         |            | indices of the signal trigger                                                   | Signal          | None       |               |       |           |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg        | flip angle                                                                      | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |            | number of acquired phase lines to the center of k-space                         | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |            | number of acquired phase lines in k-space                                       | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |            | number of slices in a multi-slice acquisition                                   | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg        | preparation Pulse Flip Angle                                                    | Sequence        | 90         | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                                                | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                                                  | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                                                       | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                                        | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence                                       | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                                               | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                                                 | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                                         | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition                                       | Sequence        | 0          | (0, 1000)     |       |           |
# | tstart         | sec        | start of the acquisition                                                        | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr         |            | B1-correction factor                                                            | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_e           | Hz         | tissue R1 in extracellular space                                                | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_h           | Hz         | tissue R1 in hepatocytes                                                        | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | R1_li          | Hz         | tissue R1 in the liver                                                          | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                                                       | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | E_li           |            | extraction fraction in the liver                                                | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ef_li          |            | final extraction fraction in the liver                                          | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | Ei_li          |            | initial extraction fraction in the liver                                        | Physiological   | 0.1        | (0.0, 1.0)    |       |           |
# | F_b_li         | mL/sec/cm3 | flow per unit tissue in blood of the liver                                      | Physiological   | 0.02       | (0, 1)        |       |           |
# | F_p_li         | mL/sec/cm3 | flow per unit tissue in plasma of the liver                                     | Physiological   | 0.01       | (0, 0.05)     |       |           |
# | PSw_e2h        | mL/sec/cm3 | water permeability-surface area product from extracellular space to hepatocytes | Physiological   | 0.03       | (0, 100)      |       |           |
# | PSw_h2e        | mL/sec/cm3 | water permeability-surface area product from hepatocytes to extracellular space | Physiological   | 0.03       | (0, 100)      |       |           |
# | T_h            | sec        | mean transit time in hepatocytes                                                | Physiological   | 1800       | (600, 36000)  |       |           |
# | Tf_h           | sec        | final mean transit time in hepatocytes                                          | Physiological   | 1800       | (600, 36000)  |       |           |
# | Ti_h           | sec        | initial mean transit time in hepatocytes                                        | Physiological   | 1800       | (600, 36000)  |       |           |
# | ffa            |            | arterial flow fraction                                                          | Physiological   | 0.2        | (0, 1)        |       |           |
# | k_e2h          | mL/sec/cm3 | tissue transfer rate from extracellular space to hepatocytes                    | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | kf_e2h         | mL/sec/cm3 | final tissue transfer rate from extracellular space to hepatocytes              | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | ki_e2h         | mL/sec/cm3 | initial tissue transfer rate from extracellular space to hepatocytes            | Physiological   | 0.003      | (0.0, 0.1)    |       |           |
# | v_e_li         | mL/cm3     | volume fraction in extracellular space of the liver                             | Physiological   | 0.3        | (0.01, 0.6)   |       |           |
# | v_h            | mL/cm3     | volume fraction in hepatocytes                                                  | Physiological   | 1          | (0, 1)        |       |           |
# | v_li           | mL/cm3     | volume fraction in the liver                                                    | Physiological   | 1          | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dt             | sec        | pseudo-continuous time step                                                     | Hyperparameters | 0.5        |               |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-----------------------------------------------------------------------------------------------------+
# |                                  ForwardLiver - all outputs (n = 13)                                  |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | Key | Unit     | Name                        | Group           | Init  | Bounds | DICOM | OSIPI     |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | C   | mmol/cm3 | tissue concentration        | Indicator       | 0.005 | (0, 1) |       |           |
# | ci  | mmol/mL  | inlet concentration         | Indicator       | 0.005 |        |       |           |
# | tC  | sec      | concentration time points   | Indicator       | 0.0   |        |       |           |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | S   | a.u.     | signal                      | Signal          | 1.0   | (0, 5) |       |           |
# | S0  | a.u.     | signal scaling factor       | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# +-----+----------+-----------------------------+-----------------+-------+--------+-------+-----------+
# | M   | A/cm     | magnetization               | Electromagnetic | 1     | (0, 5) |       |           |
# | R1  | Hz       | tissue R1                   | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1i | Hz       | inlet R1                    | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R2  | Hz       | tissue R2                   | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2s | Hz       | tissue R2*                  | Electromagnetic | 20    | (0, 5) |       |           |
# | tM  | sec      | magnetization time points   | Electromagnetic | 0.0   |        |       |           |
# | tR  | sec      | relaxation rate time points | Electromagnetic | 0.0   |        |       |           |
# | tS  | sec      | signal time points          | Electromagnetic | 0.0   |        |       |           |
# +-----------------------------------------------------------------------------------------------------+

roi = 'li'

configs = ConcToSignal.configs | WaterExchangeLiver.configs | RelaxivityLiver.configs | ConcLiver.configs
defaults = ConcToSignal.defaults | WaterExchangeLiver.defaults | RelaxivityLiver.defaults | ConcLiver.defaults

configs['inflow'].discard('inlet')
configs.pop('tof_corr')
defaults.pop('tof_corr')


class ForwardLiver(Module):
    """Whole-body model for the aorta and liver signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {}
    _all_outputs = {}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        omap = {extend_varname(k, roi=roi): k for k in {'C', 'tC', 'ci'}}
        self._conc = ConcLiver(omap=omap, **self.config)

        imap = {'v_e': 'v_e_li'}
        self._tissue_rel = RelaxivityLiver(imap=imap, **self.config)
        self._tissue_wex = WaterExchangeLiver(imap=imap, **self.config)
        self._conc_to_signal = ConcToSignal(**self.config)

        self.map_io(imap, omap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p |= self._conc(p)
        p |= self._tissue_rel(p) 
        p |= self._tissue_wex(p)

        p['tacq'] = p['dt'] * (p['ci'].size - 1)
        p |= self._conc_to_signal(p)

        return self.map_results(p)

    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        inputs |= self._tissue_rel.mapped_inputs() 
        inputs |= self._tissue_wex.mapped_inputs() 
        inputs |= self._conc_to_signal.mapped_inputs()

        inputs -= {'tacq'}
        inputs -= self._conc.new_mapped_outputs()
        inputs -= self._tissue_rel.new_mapped_outputs()
        inputs -= self._tissue_wex.new_mapped_outputs()
        inputs -= self._conc_to_signal.new_mapped_outputs()
        return inputs 
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._conc_to_signal.mapped_outputs() 
        return outputs
    
    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            f'iScal': np.arange(n0, dtype=int),
            f'Scal': Scal, 
        }
        nt = 180
        ci = np.ones(nt)
        data[f'ci_li'] = ci if '1I' in self.config['kinetics'] else (ci, ci)
        return data