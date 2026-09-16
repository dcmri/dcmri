
# +--------------------------------------------------------------------------------------------------+
# |                                AortaModel - all configs (n = 16)                                 |
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
# | bolus          | dual, single                                                       | single     |
# +--------------------------------------------------------------------------------------------------+

# +----------------------------------------------------------------------------------------------------------------------------------------------------------+
# |                                                             AortaModel - all inputs (n = 57)                                                             |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | Key            | Unit       | Name                                                    | Group           | Init       | Bounds        | DICOM | OSIPI     |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | BAT            | sec        | bolus arrival time                                      | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_1          | sec        | 1st bolus arrival time                                  | Indicator       | 30         | (-30, 30)     |       |           |
# | BAT_2          | sec        | 2nd bolus arrival time                                  | Indicator       | 30         | (-30, 30)     |       |           |
# | agent          |            | contrast agent generic name                             | Indicator       | gadoterate |               |       |           |
# | dose           | mL/kg      | contrast agent dose                                     | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_1         | mL/kg      | 1st contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | dose_2         | mL/kg      | 2nd contrast agent dose                                 | Indicator       | 0.1        | (0, 0.2)      |       |           |
# | rate           | mL/s       | injection rate                                          | Indicator       | 1          | (0, 10)       |       |           |
# | rate_1         | mL/s       | 1st injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
# | rate_2         | mL/s       | 2nd injection rate                                      | Indicator       | 1          | (0, 10)       |       |           |
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
# | SA             | deg        | saturation Slab Flip Angle                              | Sequence        | 0          | (0, 180)      |       |           |
# | TA             | sec        | acquisition time                                        | Sequence        | 2.0        | (0, 30)       |       |           |
# | TD             | sec        | prepulse delay                                          | Sequence        | 0.05       | (0, 1)        |       |           |
# | TE             | sec        | echo time                                               | Sequence        | 0.001      | (0, 10)       |       |           |
# | TE1            | sec        | first echo time in a multi-echo sequence                | Sequence        | 0.001      | (0, 1)        |       |           |
# | TE2            | sec        | second echo time in a multi-echo sequence               | Sequence        | 0.005      | (0, 1)        |       |           |
# | TP             | sec        | preparation delay                                       | Sequence        | 0.05       | (0, 1)        |       |           |
# | TR             | sec        | repetition time                                         | Sequence        | 0.005      | (0, 1)        |       |           |
# | field_strength | T          | magnetic field strength                                 | Sequence        | 3          | (0, 20)       |       |           |
# | iz             |            | slice number in a multi-slice acquisition               | Sequence        | 0          | (0, 1000)     |       |           |
# | tacq           | sec        | acquisition duration                                    | Sequence        | 240        | (0, 10000.0)  |       |           |
# | tstart         | sec        | start of the acquisition                                | Sequence        | 0          | (0, 10000.0)  |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | B1corr         |            | B1-correction factor                                    | Electromagnetic | 1          | (0, 5)        |       |           |
# | R1_b           | Hz         | tissue R1 in the blood                                  | Electromagnetic | 0.65       | (0, 5)        |       |           |
# | me             | A cm2/mL   | equilibrium magnetization                               | Electromagnetic | 1          | (0, 5)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | CO             | mL/sec     | cardiac output                                          | Physiological   | 100        | (0, 500)      |       |           |
# | D_hl           |            | transit time dispersion in the heart and Lungs          | Physiological   | 0.2        | (0.01, 0.99)  |       |           |
# | E_or           |            | extraction fraction in the organs                       | Physiological   | 0.15       | (0, 0.5)      |       |           |
# | F_b_ar         | mL/sec/cm3 | flow per unit tissue in blood of the artery             | Physiological   | 0.02       | (0, 1)        |       |           |
# | TF             | sec        | inflow time                                             | Physiological   | 0.5        | (0, 10)       |       |           |
# | T_b_or         | sec        | mean transit time in blood of the organs                | Physiological   | 20         | (0, 60)       |       |           |
# | T_e_li         | sec        | mean transit time in extracellular of the liver         | Physiological   | 30.0       | (0.1, 60)     |       |           |
# | T_e_or         | sec        | mean transit time in extracellular of the organs        | Physiological   | 120        | (0, 800)      |       |           |
# | T_gu           | sec        | mean transit time in the gut                            | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_hl           | sec        | mean transit time in the heart and Lungs                | Physiological   | 10         | (0, 30)       |       |           |
# | T_la           | sec        | mean transit time in the liver artery                   | Physiological   | 30         | (0.1, 60)     |       |           |
# | T_p_lk         | sec        | mean transit time in plasma of the left kidney          | Physiological   | 5          | (0, 30)       |       |           |
# | T_p_rk         | sec        | mean transit time in plasma of the right kidney         | Physiological   | 5          | (0, 30)       |       |           |
# | ffa            |            | arterial flow fraction                                  | Physiological   | 0.2        | (0, 1)        |       |           |
# | vr_li          |            | Venous return in the liver                              | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_lk          |            | Venous return in the left kidney                        | Physiological   | 0.15       | (0, 1)        |       |           |
# | vr_or          |            | Venous return in the organs                             | Physiological   | 0.9        | (0, 1)        |       |           |
# | vr_rk          |            | Venous return in the right kidney                       | Physiological   | 0.15       | (0, 1)        |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | dose_tolerance |            | dose tolerance                                          | Hyperparameters | 0.1        |               |       |           |
# | dt             | sec        | pseudo-continuous time step                             | Hyperparameters | 0.5        |               |       |           |
# +----------------+------------+---------------------------------------------------------+-----------------+------------+---------------+-------+-----------+
# | vol_ao         | cm3        | ROI volume in the aorta                                 | Whole-body      | 10         | (0.0, 1000)   |       |           |
# | weight         | kg         | body weight                                             | Whole-body      | 70         | (0, 300)      |       |           |
# +----------------------------------------------------------------------------------------------------------------------------------------------------------+

# +-------------------------------------------------------------------------------------------------------------+
# |                                      AortaModel - all outputs (n = 13)                                      |
# +-------+----------+-----------------------------------+-----------------+-------+--------+-------+-----------+
# | Key   | Unit     | Name                              | Group           | Init  | Bounds | DICOM | OSIPI     |
# +-------+----------+-----------------------------------+-----------------+-------+--------+-------+-----------+
# | C_ao  | mmol/cm3 | tissue concentration in the aorta | Indicator       | 0.005 | (0, 1) |       |           |
# | ci_ao | mmol/mL  | inlet concentration in the aorta  | Indicator       | 0.005 |        |       |           |
# | tC    | sec      | concentration time points         | Indicator       | 0.0   |        |       |           |
# +-------+----------+-----------------------------------+-----------------+-------+--------+-------+-----------+
# | S     | a.u.     | signal                            | Signal          | 1.0   | (0, 5) |       |           |
# | S0    | a.u.     | signal scaling factor             | Signal          | 1.0   | (0, 5) |       | Q.MS1.010 |
# +-------+----------+-----------------------------------+-----------------+-------+--------+-------+-----------+
# | M     | A/cm     | magnetization                     | Electromagnetic | 1     | (0, 5) |       |           |
# | R1    | Hz       | tissue R1                         | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R1i   | Hz       | inlet R1                          | Electromagnetic | 0.65  | (0, 5) |       |           |
# | R2    | Hz       | tissue R2                         | Electromagnetic | 2.0   | (0, 5) |       |           |
# | R2s   | Hz       | tissue R2*                        | Electromagnetic | 20    | (0, 5) |       |           |
# | tM    | sec      | magnetization time points         | Electromagnetic | 0.0   |        |       |           |
# | tR    | sec      | relaxation rate time points       | Electromagnetic | 0.0   |        |       |           |
# | tS    | sec      | signal time points                | Electromagnetic | 0.0   |        |       |           |
# +-------------------------------------------------------------------------------------------------------------+


import numpy as np

from dcmri.core.module import Module
from dcmri.kinetics.modules_conc import ConcAorta
from dcmri.relaxivity.modules_rois import RelaxivityArtery
from dcmri.bloch.modules_rois import WaterExchangeArtery
from dcmri.signal.modules_tissue import ConcToSignal
from dcmri.bloch.functions_sequences import channels

roi_quantities = {'F_b', 'ci', 'C'}
iomap = {k:f'{k}_ao' for k in roi_quantities}


class AortaModel(Module):
    """Whole-body model for the aorta signal."""

    configs = ConcToSignal.configs | WaterExchangeArtery.configs | RelaxivityArtery.configs | ConcAorta.configs
    defaults = ConcToSignal.defaults | WaterExchangeArtery.defaults | RelaxivityArtery.defaults | ConcAorta.defaults

    _all_inputs = {'PA', 'dose', 'Nk0', 'E_or', 'T_b_or', 'D_hl', 'dose_2', 'tstart', 'TR', 'T_p_lk', 'BAT_1', 'field_strength', 'B1corr', 'BAT', 'dt', 'iz', 'iStrig', 'TF', 'TA', 'T_la', 'weight', 'rate', 'TE2', 'Nz', 'agent', 'ffa', 'TD', 'F_b_ar', 'T_gu', 'iScal', 'Scal', 'T_p_rk', 'FA', 'vr_lk', 'TE1', 'me', 'tacq', 'rate_1', 'TE', 'S0', 'T_e_li', 'R1_b', 'vr_rk', 'T_hl', 'rate_2', 'CO', 'T_e_or', 'SA', 'BAT_2', 'NSR', 'dose_1', 'TP', 'dose_tolerance', 'vr_or', 'Nph', 'vol_ao', 'vr_li'}
    _all_outputs = {'tC', 'R2s', 'R1', 'M', 'tM', 'R2', 'C_ao', 'ci_ao', 'tR', 'S', 'R1i', 'S0', 'tS'}

    def __init__(self, imap:dict=None, omap:dict=None, **config):
        self.set_config(config)

        self._conc = ConcAorta(**self.config)
        self._tissue_rel = RelaxivityArtery(imap=iomap, **self.config)
        self._tissue_wex = WaterExchangeArtery(imap=iomap, **self.config)
        self._conc_to_signal = ConcToSignal(imap=iomap, **self.config) 

        self.map_io(imap, omap)

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
        outputs -= {'F_b_ao'}
        return outputs
    
    def dummy_data(self): 
        data = self.init_data()
        n_channels = channels(self.config['sequence'])
        components = 1 if self.config['magnitude'] else 2
        n0 = 1
        Scal = np.zeros((n_channels, components, n0))
        Scal[:, 0, :] = 1

        data |= {
            'iScal': np.arange(n0, dtype=int),
            'Scal': Scal, 
        }
        return data