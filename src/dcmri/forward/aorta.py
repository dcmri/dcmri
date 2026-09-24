# +--------------------------------------------------------------------------------------------------+
# |                                ForwardAorta - all configs (n = 16)                                 |
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
# | tof_corr       | False, True                                                        | False      |
# | magnitude      | False, True                                                        | True       |
# | trigger        | False, True                                                        | False      |
# | calibrate      | False, True                                                        | False      |
# | baseline       | literature, measured                                               | literature |
# | heartlung      | chain, comp, pfcomp                                                | pfcomp     |
# | organs         | 2cxm, comp                                                         | comp       |
# | kidneys        | None, comp, pass, plug                                             | None       |
# | liver          | None, comp, pass, plug                                             | None       |
# | lagut          | None, comp, pass, plucom                                           | None       |
# | bolus          | double, dual, single                                               | single     |
# +--------------------------------------------------------------------------------------------------+

# +--------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                            ForwardAorta - all inputs (n = 57)                                                            |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit     | Name                                                    | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT            | sec      | bolus arrival time                                      | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_1          | sec      | 1st bolus arrival time                                  | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec      | 2nd bolus arrival time                                  | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |          | contrast agent generic name                             | Indicator       | gadoterate |               |       |           |
# | bdel           | sec      | delay in a double injection                             | Indicator       | 30         | (-30, 30)     |       |           |
# | dose           | mL/kg    | contrast agent dose                                     | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg    | 1st contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg    | 2nd contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s     | injection rate                                          | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s     | 1st injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s     | 2nd injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | NSR            |          | noise-to-signal ratio                                   | Signal          | 0.0        | (0, 100000.0) |       |           |
# | S0             | a.u.     | signal scaling factor                                   | Signal          | 1.0        | (0, 5)        |       | Q.MS1.010 |
# | Scal           | a.u.     | calibration signal                                      | Signal          | 1.0        | (0, 5)        |       | Q.MS1.002 |
# | iScal          |          | indices of calibration signal                           | Signal          | 0          |               |       |           |
# | iStrig         |          | indices of the signal trigger                           | Signal          | None       |               |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | FA             | deg      | flip angle                                              | Sequence        | 15         | (0, 180)      |       |           |
# | Nk0            |          | number of acquired phase lines to the center of k-space | Sequence        | 64         | (0, 1000)     |       |           |
# | Nph            |          | number of acquired phase lines in k-space               | Sequence        | 128        | (0, 1000)     |       |           |
# | Nz             |          | number of slices in a multi-slice acquisition           | Sequence        | 64         | (0, 1000)     |       |           |
# | PA             | deg      | preparation Pulse Flip Angle                            | Sequence        | 90         | (0, 180)      |       |           |
# | SA             | deg      | saturation Slab Flip Angle                              | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec      | acquisition time                                        | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec      | prepulse delay                                          | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec      | echo time                                               | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec      | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec      | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec      | preparation delay                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec      | repetition time                                         | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T        | magnetic field strength                                 | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |          | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq           | sec      | acquisition duration                                    | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart         | sec      | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr         |          | B1-correction factor                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz       | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec   | cardiac output                                          | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |          | transit time dispersion in the heart and Lungs          | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_or           |          | extraction fraction in the organs                       | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | TF             | sec      | inflow time                                             | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_b_or         | sec      | mean transit time in blood of the organs                | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_li         | sec      | mean transit time in extracellular space of the liver   | Physiological   | 30.0       | (0.1, 60)     |       |           |
# | T_e_or         | sec      | mean transit time in extracellular space of the organs  | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec      | mean transit time in the gut                            | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_hl           | sec      | mean transit time in the heart and Lungs                | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec      | mean transit time in the liver artery                   | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_p_lk         | sec      | mean transit time in plasma of the left kidney          | Physiological   | 5          | (0, 30)       |       |           |
# | T_p_rk         | sec      | mean transit time in plasma of the right kidney         | Physiological   | 5          | (0, 30)       |       |           |
# | ffa            |          | arterial flow fraction                                  | Physiological   | 0.2        | (0, 1)        |       |           |
# | vr_li          |          | Venous return in the liver                              | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_lk          |          | Venous return in the left kidney                        | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_or          |          | Venous return in the organs                             | Physiological   | 0.9        | (0, 1)        |       |           |
# | vr_rk          |          | Venous return in the right kidney                       | Physiological   | 0.15       | (0, 1)        |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |          | dose tolerance                                          | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec      | pseudo-continuous time step                             | Hyperparameters | 0.5        |               |       |           |
# +----------------+----------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_ao         | cm3      | ROI volume in the aorta                                 | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | weight         | kg       | body weight                                             | Whole-body      | 70         | (0, 300)      |       |           |
# +--------------------------------------------------------------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------+
# |                                       ForwardAorta - all outputs (n = 15)                                        |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | Key | Unit     | Name                                   | Group           | Init  | Bounds | DICOM | OSIPI     |
# +-----+----------+----------------------------------------+-----------------+-------+--------+-------+-----------+
# | C   | mmol/cm3 | tissue concentration                   | Indicator       | 0.005 | (0, 1) |       |           |
# | ci  | mmol/mL  | inlet concentration                    | Indicator       | 0.005 |        |       |           |
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
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_rois import RelaxivityArtery
from dcmri.bloch.modules_rois import WaterExchangeArtery
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

configs = ConcToSignal.configs | WaterExchangeArtery.configs | RelaxivityArtery.configs | ConcAorta.configs
defaults = ConcToSignal.defaults | WaterExchangeArtery.defaults | RelaxivityArtery.defaults | ConcAorta.defaults

configs['inflow'].discard('inlet')

class ForwardAorta(Module):
    """Whole-body model for the aorta signal."""

    configs = configs
    defaults = defaults

    _all_inputs = {'E_or', 'T_p_lk', 'vr_rk', 'NSR', 'iz', 'T_p_rk', 'dt', 'vr_lk', 'rate_1', 'Scal', 'T_b_or', 'dose', 'Nk0', 'iScal', 'dose_2', 'BAT', 'TP', 'TE2', 'tstart', 'Nph', 'TR', 'FA', 'TA', 'CO', 'B1corr', 'field_strength', 'T_e_li', 'SA', 'dose_tolerance', 'T_gu', 'Nz', 'me', 'agent', 'R1_b', 'TD', 'iStrig', 'dose_1', 'ffa', 'rate_2', 'D_hl', 'TE', 'BAT_2', 'bdel', 'tacq', 'vr_or', 'BAT_1', 'T_e_or', 'TF', 'weight', 'vol_ao', 'rate', 'PA', 'TE1', 'vr_li', 'S0', 'T_hl', 'T_la'}
    _all_outputs = {'tC', 'R2s', 'Mz', 'R1', 'tR', 'S', 'tM', 'tMz', 'R1i', 'C', 'ci', 'tS', 'S0', 'R2', 'M'}

    def __init__(self, imap:dict=None, omap:dict=None, iomap:dict=None, cmap:dict=None, **config):
        self.set_config(config, cmap)

        omap = {'C_ao':'C', 'ci_ao':'ci', 'F_b_ao':'F_b_ar'}
        self._conc = ConcAorta(omap=omap, **self.config)

        self._tissue_rel = RelaxivityArtery(**self.config)
        self._tissue_wex = WaterExchangeArtery(**self.config)
        self._conc_to_signal = ConcToSignal(**self.config) 

        self.map_io(imap, omap, iomap)

    def __call__(self, data: dict=None, **kwargs) -> dict:
        p = self.map_data(data, kwargs)  

        p['tmax'] = p['tstart'] + p['tacq'] + p['dt']

        p |= self._conc(p)
        p |= self._tissue_rel(p)
        p |= self._tissue_wex(p)
        p |= self._conc_to_signal(p)

        return self.map_results(p)
        
    def inputs(self) -> set:
        inputs = self._conc.mapped_inputs()
        inputs |= self._tissue_rel.mapped_inputs()
        inputs |= self._tissue_wex.mapped_inputs()
        inputs |= self._conc_to_signal.mapped_inputs()

        inputs -= {'tmax'}
        inputs -= self._conc.new_mapped_outputs()
        inputs -= self._tissue_rel.new_mapped_outputs()
        inputs -= self._tissue_wex.new_mapped_outputs()
        return inputs  
    
    def outputs(self):
        outputs = self._conc.mapped_outputs()
        outputs |= self._conc_to_signal.mapped_outputs() 
        outputs -= {'F_b_ar'}
        return outputs
    
    def dummy_data(self, data: dict=None): 
        p = self.init_data()
        p |= self._conc.dummy_data()

        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        p |= {
            'iScal': np.arange(n0, dtype=int),
            'Scal': Scal, 
            'BAT_2': 90, # default is the same as BAT_1
        }
        if data is not None:
            p |= data
            
        return self.input_data(p)